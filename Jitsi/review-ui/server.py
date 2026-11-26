#!/usr/bin/env python3
"""
Interview Review UI — a tiny, dependency-free web app to browse recorded Jitsi
interviews. For each meeting it shows the meeting name, who joined, the video
recording, and the transcript. Clicking a transcript line seeks the video to
that moment.

It scans the live Jitsi storage folders on every /api/meetings request, so a new
meeting simply appears when you refresh — nothing to rebuild:

    transcripts :  <storage>/transcripts/**/transcript_*.txt   (auto-saved by jigasi)
                   <storage>/transcripts/Interview Transcripts/*.txt (collected copies)
    recordings  :  <storage>/jibri/Video Recordings/*.mp4       (+ *.info.json sidecars)

Transcript<->video time alignment uses the sidecar written by finalize.sh
(start_epoch + duration); if a sidecar is missing it falls back to parsing the
mp4 duration and using the file mtime, so older recordings still work.

Stdlib only (no pip installs). Run:  python3 server.py  [--port 8080]
Override the storage path with $JITSI_STORAGE or --storage.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import struct
import unicodedata
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
DEFAULT_STORAGE = (
    HERE.parent / "docker-jitsi-meet" / ".jitsi-meet-cfg" / "storage"
).resolve()

STORAGE = Path(os.environ.get("JITSI_STORAGE", str(DEFAULT_STORAGE))).resolve()

# Names that are the bots, never real interview participants.
BOT_NAMES = {"recorder", "fellow jitser", "transcriber"}

# ----------------------------------------------------------------------------
# Transcript parsing
# ----------------------------------------------------------------------------
_LINE_RE = re.compile(r"^<\s*(\d{1,2}:\d{2}:\d{2}\s*[AaPp][Mm])\s*>\s*(.*)$")
_HEADER_ROOM_RE = re.compile(r"in room\s+([^@\s]+)")
_HEADER_DATE_RE = re.compile(r"held at\s+(.+?)\s+in room")
_HEADER_START_RE = re.compile(r"started at\s+(\d{1,2}:\d{2}:\d{2}\s*[AaPp][Mm])")
# ISO stamp at the front of a raw transcript folder name (…_<uuid>), with or
# without colons in the time: 2026-11-17T06:55:54…Z  or  2026-11-17T065554…Z
_FOLDER_ISO_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):?(\d{2}):?(\d{2})")


def _slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _naive_utc(date_str: str, time_str: str):
    """('Nov 16, 2026', '9:11:27 AM') -> epoch, reading the wall-clock as UTC.

    The real timezone is corrected later via the folder's UTC anchor, so this
    only needs to be a consistent reference, not the true local time.
    """
    for fmt in ("%b %d, %Y %I:%M:%S %p", "%B %d, %Y %I:%M:%S %p"):
        try:
            dt = datetime.strptime(f"{date_str} {time_str.upper()}", fmt)
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def _folder_anchor_utc(folder_name: str):
    """True UTC epoch from a raw transcript folder's leading ISO stamp (the 'Z'
    makes it unambiguous), or None for collected copies that lack it."""
    m = _FOLDER_ISO_RE.search(folder_name or "")
    if not m:
        return None
    y, mo, d, h, mi, s = (int(x) for x in m.groups())
    try:
        return datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def parse_transcript(path: Path):
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = text.splitlines()
    if not lines or "Transcript of conference" not in lines[0]:
        return None

    room = "meeting"
    m = _HEADER_ROOM_RE.search(lines[0])
    if m:
        room = m.group(1)
    date_str = None
    m = _HEADER_DATE_RE.search(lines[0])
    if m:
        date_str = m.group(1).strip()

    # Timezone correction: jigasi writes line times in the CONTAINER's timezone
    # (often UTC), which may differ from this host. The raw transcript folder
    # name carries the same instant as an explicit UTC stamp. Aligning the
    # header "started at" time (read as UTC) with that anchor yields a constant
    # correction we add to every line, so epochs match the recording's UTC
    # start_epoch regardless of either timezone. Collected copies lack the
    # anchor -> correction 0 (assumes the common UTC-container case).
    correction = 0.0
    anchor = _folder_anchor_utc(path.parent.name)
    if anchor is not None and date_str:
        mh = _HEADER_START_RE.search(text)
        header_naive = _naive_utc(date_str, mh.group(1)) if mh else None
        if header_naive is not None:
            correction = anchor - header_naive

    entries = []        # spoken lines: {time, speaker, text, epoch}
    participants = []    # join order, de-duplicated, bots removed
    seen = set()
    start_epoch = None
    end_epoch = None

    for ln in lines[1:]:
        m = _LINE_RE.match(ln.strip())
        if not m:
            continue
        tstr, content = m.group(1).strip(), m.group(2).strip()
        naive = _naive_utc(date_str, tstr) if date_str else None
        epoch = (naive + correction) if naive is not None else None
        if epoch is not None:
            start_epoch = epoch if start_epoch is None else min(start_epoch, epoch)
            end_epoch = epoch if end_epoch is None else max(end_epoch, epoch)

        if content.endswith("joined the conference"):
            name = content[: -len("joined the conference")].strip()
            if name and name.lower() not in BOT_NAMES and name not in seen:
                seen.add(name)
                participants.append(name)
            continue
        if content.endswith("left the conference"):
            continue
        # spoken line "Speaker: words"
        if ":" in content:
            speaker, said = content.split(":", 1)
            speaker, said = speaker.strip(), said.strip()
            if speaker.lower() in BOT_NAMES or not said:
                continue
            entries.append({"time": tstr, "speaker": speaker, "text": said, "epoch": epoch})
            if speaker not in seen:
                seen.add(speaker)
                participants.append(speaker)

    # Display label straight from the transcript's own clock (matches the line
    # times the viewer reads), e.g. "Nov 17, 2026 · 6:56 AM".
    when_label = None
    if date_str:
        label_time = None
        mh = _HEADER_START_RE.search(text)
        if mh:
            label_time = mh.group(1)
        elif entries:
            label_time = entries[0]["time"]
        if label_time:
            lm = re.match(r"(\d{1,2}:\d{2}):\d{2}\s*([AaPp][Mm])", label_time.strip())
            pretty = f"{lm.group(1)} {lm.group(2).upper()}" if lm else label_time
            when_label = f"{date_str} · {pretty}"

    return {
        "room": room,
        "date": date_str,
        "when_label": when_label,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "participants": participants,
        "entries": entries,
        "source": str(path),
    }


def find_transcripts():
    root = STORAGE / "transcripts"
    out = []
    if not root.is_dir():
        return out
    for p in root.rglob("transcript_*.txt"):
        t = parse_transcript(p)
        if t:
            out.append(t)
    # also collected copies (Interview Transcripts/<room>_<date>.txt)
    coll = root / "Interview Transcripts"
    if coll.is_dir():
        for p in coll.glob("*.txt"):
            t = parse_transcript(p)
            if t:
                out.append(t)
    # de-dup: same room + same start second + same #entries => same meeting
    uniq = {}
    for t in out:
        key = (_slug(t["room"]), int(t["start_epoch"] or 0), len(t["entries"]))
        uniq.setdefault(key, t)
    return list(uniq.values())


# ----------------------------------------------------------------------------
# Recording parsing
# ----------------------------------------------------------------------------
_REC_NAME_RE = re.compile(r"^(?P<room>.+?)_(?P<ts>\d{8}-\d{6})_")


def _mp4_duration(path: Path):
    """Duration in seconds from the mp4 mvhd atom; None if unparseable."""
    try:
        with path.open("rb") as f:
            def walk(end):
                while f.tell() < end:
                    hdr = f.read(8)
                    if len(hdr) < 8:
                        return None
                    size, typ = struct.unpack(">I4s", hdr)
                    start = f.tell() - 8
                    if size == 1:
                        size = struct.unpack(">Q", f.read(8))[0]
                    if size < 8:
                        return None
                    box_end = start + size
                    if typ in (b"moov", b"trak", b"mdia"):
                        r = walk(box_end)
                        if r is not None:
                            return r
                    elif typ in (b"mvhd", b"mdhd"):
                        ver = f.read(1)[0]
                        f.read(3)  # flags
                        if ver == 1:
                            f.read(16)
                            timescale = struct.unpack(">I", f.read(4))[0]
                            duration = struct.unpack(">Q", f.read(8))[0]
                        else:
                            f.read(8)
                            timescale = struct.unpack(">I", f.read(4))[0]
                            duration = struct.unpack(">I", f.read(4))[0]
                        if timescale:
                            return duration / timescale
                    f.seek(box_end)
                return None

            size = path.stat().st_size
            return walk(size)
    except (OSError, struct.error):
        return None


def find_recordings():
    root = STORAGE / "jibri" / "Video Recordings"
    out = []
    if not root.is_dir():
        return out
    for mp4 in sorted(root.glob("*.mp4")):
        info = root / f"{mp4.name}.info.json"
        room = None
        duration = None
        start_epoch = None
        end_epoch = None
        participants = []
        if info.is_file():
            try:
                d = json.loads(info.read_text(encoding="utf-8"))
                room = d.get("room")
                duration = float(d.get("duration") or 0) or None
                start_epoch = float(d["start_epoch"]) if d.get("start_epoch") is not None else None
                end_epoch = float(d["end_epoch"]) if d.get("end_epoch") is not None else None
                participants = [p for p in (d.get("participants") or []) if p.lower() not in BOT_NAMES]
            except (ValueError, OSError, KeyError):
                pass
        if room is None:
            m = _REC_NAME_RE.match(mp4.name)
            room = m.group("room") if m else mp4.stem
        if duration is None:
            duration = _mp4_duration(mp4)
        if end_epoch is None:
            end_epoch = mp4.stat().st_mtime
        if start_epoch is None and duration is not None:
            start_epoch = end_epoch - duration
        out.append({
            "room": room,
            "file": mp4.name,
            "duration": duration,
            "start_epoch": start_epoch,
            "end_epoch": end_epoch,
            "participants": participants,
        })
    return out


# ----------------------------------------------------------------------------
# Pair transcripts with recordings and build the meeting list
# ----------------------------------------------------------------------------
def build_meetings():
    transcripts = find_transcripts()
    recordings = find_recordings()
    used_rec = set()

    def best_recording(t):
        best, best_score = None, None
        for i, r in enumerate(recordings):
            if i in used_rec:
                continue
            if _slug(r["room"]) != _slug(t["room"]):
                continue
            if None in (t["start_epoch"], t["end_epoch"], r["start_epoch"], r["end_epoch"]):
                score = 1e18
            else:
                # overlap of [t.start,t.end] and [r.start,r.end]; more overlap = better
                overlap = min(t["end_epoch"], r["end_epoch"]) - max(t["start_epoch"], r["start_epoch"])
                score = -overlap if overlap > 0 else abs(t["start_epoch"] - r["start_epoch"])
            if best_score is None or score < best_score:
                best, best_score = i, score
        return best

    meetings = []
    for t in sorted(transcripts, key=lambda x: x["start_epoch"] or 0, reverse=True):
        ri = best_recording(t)
        rec = None
        if ri is not None:
            used_rec.add(ri)
            rec = recordings[ri]
        meetings.append(_meeting_obj(t, rec))

    # recordings with no transcript -> video-only cards
    for i, r in enumerate(recordings):
        if i in used_rec:
            continue
        meetings.append(_meeting_obj(None, r))

    meetings.sort(key=lambda m: m["sort_epoch"], reverse=True)
    return meetings


def _fmt_when(epoch):
    if not epoch:
        return ""
    return datetime.fromtimestamp(epoch).strftime("%b %d, %Y · %I:%M %p")


def _meeting_obj(t, rec):
    room = (t or rec)["room"]
    start_epoch = (t and t["start_epoch"]) or (rec and rec["start_epoch"]) or 0
    participants = []
    if t:
        participants = t["participants"]
    elif rec:
        participants = rec["participants"]

    video = None
    rec_start = None
    duration = None
    if rec:
        video = "/media/rec/" + rec["file"]
        rec_start = rec["start_epoch"]
        duration = rec["duration"]

    entries = []
    if t:
        for e in t["entries"]:
            offset = None
            if rec_start is not None and e["epoch"] is not None:
                offset = e["epoch"] - rec_start
                if offset < 0:
                    offset = 0.0
                if duration and offset > duration:
                    offset = duration
            entries.append({
                "time": e["time"],
                "speaker": e["speaker"],
                "text": e["text"],
                "offset": round(offset, 2) if offset is not None else None,
            })

    when = (t and t.get("when_label")) or _fmt_when(start_epoch)
    return {
        "room": room,
        "title": room,
        "when": when,
        "sort_epoch": start_epoch,
        "participants": participants,
        "video": video,
        "duration": duration,
        "has_video": video is not None,
        "has_transcript": t is not None,
        "entries": entries,
    }


# ----------------------------------------------------------------------------
# HTTP server (with Range support for video seeking)
# ----------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "InterviewReviewUI/1.0"

    def log_message(self, fmt, *args):
        pass  # quiet

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            return self._send_file(HERE / "index.html", "text/html; charset=utf-8")
        if path == "/app.js":
            return self._send_file(HERE / "app.js", "application/javascript; charset=utf-8")
        if path == "/api/meetings":
            return self._send_json(build_meetings())
        if path.startswith("/media/rec/"):
            name = unquote(path[len("/media/rec/"):])
            return self._send_media(name)
        self.send_error(HTTPStatus.NOT_FOUND)

    # -- helpers --
    def _send_json(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, p: Path, ctype):
        if not p.is_file():
            return self.send_error(HTTPStatus.NOT_FOUND)
        body = p.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_media(self, name):
        # basename only — never allow path traversal out of the recordings dir
        if "/" in name or "\\" in name or ".." in name:
            return self.send_error(HTTPStatus.FORBIDDEN)
        root = STORAGE / "jibri" / "Video Recordings"
        fpath = (root / name).resolve()
        if root.resolve() not in fpath.parents or not fpath.is_file():
            return self.send_error(HTTPStatus.NOT_FOUND)
        size = fpath.stat().st_size
        ctype = "video/mp4"
        rng = self.headers.get("Range")
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            start = int(m.group(1)) if m and m.group(1) else 0
            end = int(m.group(2)) if m and m.group(2) else size - 1
            end = min(end, size - 1)
            if start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            length = end - start + 1
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            self._stream(fpath, start, length)
        else:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self._stream(fpath, 0, size)

    def _stream(self, fpath: Path, start, length):
        chunk = 64 * 1024
        try:
            with fpath.open("rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    data = f.read(min(chunk, remaining))
                    if not data:
                        break
                    self.wfile.write(data)
                    remaining -= len(data)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client seeked/closed — normal for video


def _lan_ip():
    """Best-effort primary LAN IP of this machine (no packets sent)."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("REVIEW_UI_PORT", "8080")))
    # Bind to all interfaces by default so the page is reachable at this
    # machine's LAN IP (e.g. http://192.168.7.174:8080), not just localhost.
    # Set REVIEW_UI_HOST=127.0.0.1 to restrict it back to this machine only.
    ap.add_argument("--host", default=os.environ.get("REVIEW_UI_HOST", "0.0.0.0"))
    ap.add_argument("--storage", default=None)
    args = ap.parse_args()
    global STORAGE
    if args.storage:
        STORAGE = Path(args.storage).resolve()

    print(f"[review-ui] storage : {STORAGE}", flush=True)
    if args.host in ("0.0.0.0", "::"):
        ip = _lan_ip()
        print(f"[review-ui] open on this machine : http://localhost:{args.port}", flush=True)
        print(f"[review-ui] open from the network : http://{ip}:{args.port}", flush=True)
    else:
        print(f"[review-ui] serving : http://{args.host}:{args.port}", flush=True)
    if not STORAGE.is_dir():
        print(f"[review-ui] WARNING: storage path does not exist yet: {STORAGE}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
