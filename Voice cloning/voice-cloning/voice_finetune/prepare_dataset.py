"""Turns long recordings of the reading script into training clips for train_lora.py.

Read the script (reading_script.txt) aloud in as many recordings as you like, pausing
between sentences. Mistakes are fine: just pause and read the sentence again.

For every recording, Whisper transcribes the speech with word timings, and it is split at
pauses and sentence ends. Each piece is matched to the script:
  - a piece holding several sentences read without a pause is matched to a run of
    consecutive script lines and split at the word timings between them;
  - a sentence split in two by a pause is put back together from neighbouring pieces;
  - pieces that don't match the script (false starts, mistakes, noise) are dropped.
Every clip is cut with a little silence at each edge (never into the next sentence),
converted to 24 kHz mono, set to the same loudness, and transcribed again by Whisper as a
check. Its transcript is the script line when the words match exactly, otherwise what was
actually said (Whisper's text), so text and audio always agree.

Writes <out_dir>/clips/*.wav, <out_dir>/ref.wav (one clip used as every sample's reference,
as Qwen recommends), <out_dir>/train_raw.jsonl (Qwen's format) and <out_dir>/report.csv
(every piece, kept or dropped, and why).

    "%USERPROFILE%\\.venvs\\qwen3tts\\Scripts\\python.exe" voice_finetune\\prepare_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from voice_finetune.common import DATA_DIR, SAMPLE_RATE  # noqa: E402

AUDIO_TYPES = {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".aac", ".webm"}
PAUSE_SPLIT_SECONDS = 0.55    # a gap this long between words ends a piece
SENTENCE_SPLIT_SECONDS = 0.2  # ...or this long after a word ending a sentence (. ? !)
MATCH_THRESHOLD = 0.75        # how closely speech must match the script to be kept
MAX_MERGE_PIECES = 3          # a sentence may be split by up to 2 pauses
MAX_RUN_LINES = 12            # a piece may hold up to this many sentences read together
# Lines said one after another with hardly a pause (and Whisper often drops the punctuation
# there) make one long piece that matches no run of lines: split pieces longer than this at
# their largest gaps between words; a sentence cut in two is joined again when matching.
LONG_PIECE_SECONDS = 12.0
SEARCH_BACK, SEARCH_AHEAD = 15, 60  # recordings follow the script, so look near the last match
MIN_SECONDS, MAX_SECONDS = 0.8, 15.0
LEAD_SECONDS, TAIL_SECONDS = 0.15, 0.25
TARGET_RMS_DB = -20.0
MAX_PEAK = 0.95
CLIPPED_LEVEL = 0.995
REF_TARGET_SECONDS = 10.0
# How Whisper spells names that are spoken correctly.
WHISPER_SPELLINGS = {"ritika": "rithika", "rithica": "rithika"}

ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
        "sixteen seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
ORDINALS = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
            "nine": "ninth", "twelve": "twelfth"}
# For matching only: "I'm" and "I am" are the same words (people often say them in full).
# The training transcript is still exactly what was said (see choose_text).
WHOLE_CONTRACTIONS = {"won't": "will not", "can't": "can not", "cannot": "can not", "shan't": "shall not",
                      "let's": "let us", "ain't": "is not"}
CONTRACTION_ENDINGS = [("n't", " not"), ("'m", " am"), ("'re", " are"), ("'ve", " have"), ("'ll", " will"),
                       ("'d", " would"), ("'s", " is")]


def expand_contraction(word: str) -> str:
    if word in WHOLE_CONTRACTIONS:
        return WHOLE_CONTRACTIONS[word]
    for ending, full in CONTRACTION_ENDINGS:
        if word.endswith(ending) and len(word) > len(ending):
            return word[: -len(ending)] + full
    return word


def number_words(n: int) -> str:
    """12 -> "twelve", 2002 -> "two thousand two": Whisper writes numbers as digits."""
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + number_words(n % 100))
    if n < 1_000_000:
        return number_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + number_words(n % 1000))
    return str(n)


def ordinal_words(n: int) -> str:
    """23 -> "twenty third": Whisper writes "23rd"."""
    *head, last = number_words(n).split()
    last = ORDINALS.get(last, last[:-1] + "ieth" if last.endswith("y") else last + "th")
    return " ".join([*head, last])


def normalize_word(w: str, expand: bool = False) -> list[str]:
    """One written word as the spoken words used for matching (may be several, or none).
    expand: contractions in full ("I'm" -> "i am"), for matching only."""
    w = w.lower().replace("’", "'").replace("%", " percent")
    out: list[str] = []
    for part in re.sub(r"[^a-z0-9' ]+", " ", w).split():
        ordinal = re.fullmatch(r"(\d+)(st|nd|rd|th)", part)
        if ordinal:
            part = ordinal_words(int(ordinal.group(1)))
        elif part.isdigit():
            part = number_words(int(part))
        else:
            part = WHISPER_SPELLINGS.get(part, part)
            if expand:
                part = expand_contraction(part)
        out += part.split()
    return [p for p in out if p != "and"]  # "two hundred and fifty" vs "250"


def words_of(text: str, expand: bool = False) -> list[str]:
    return [n for w in text.split() for n in normalize_word(w, expand)]


def similarity(a: str, b: str) -> float:
    """How closely two texts match as spoken words (contractions count as the full words)."""
    return difflib.SequenceMatcher(None, words_of(a, True), words_of(b, True), autojunk=False).ratio()


def load_script(path: Path) -> list[str]:
    lines = [re.sub(r"^\s*\d+[.)]\s*", "", line).strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def load_audio(path: Path) -> np.ndarray:
    """Any audio file as 24 kHz mono float32 (other formats go through ffmpeg)."""
    if path.suffix.lower() in {".wav", ".flac", ".ogg"}:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
    else:
        import imageio_ffmpeg

        with tempfile.TemporaryDirectory() as tmp:
            wav = os.path.join(tmp, "audio.wav")
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                            "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE), wav], check=True)
            audio, sr = sf.read(wav, dtype="float32")
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio.astype(np.float32)


def load_whisper():
    if sys.platform == "win32":  # make pip's cuBLAS/cuDNN DLLs visible to CTranslate2
        import nvidia.cublas
        import nvidia.cudnn

        for pkg in (nvidia.cublas, nvidia.cudnn):
            bin_dir = os.path.join(list(pkg.__path__)[0], "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    from faster_whisper import WhisperModel

    return WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")


def transcribe(whisper, audio: np.ndarray, words: bool):
    audio16 = librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=16000)
    segments, _ = whisper.transcribe(audio16, language="en", word_timestamps=words, beam_size=5,
                                     condition_on_previous_text=False, vad_filter=words)
    return list(segments)


def split_pieces(words: list) -> list[tuple[int, int]]:
    """Word index ranges [a, b] split at pauses and sentence ends."""
    out: list[list[int]] = []
    for i, w in enumerate(words):
        if out:
            prev = words[out[-1][1]]
            gap = w.start - prev.end
            sentence_ended = prev.word.rstrip().endswith((".", "?", "!"))
            if gap < PAUSE_SPLIT_SECONDS and not (sentence_ended and gap >= SENTENCE_SPLIT_SECONDS):
                out[-1][1] = i
                continue
        out.append([i, i])
    return [p for a, b in out for p in split_long(words, a, b)]


def split_long(words: list, a: int, b: int) -> list[tuple[int, int]]:
    """words[a..b] in pieces of at most LONG_PIECE_SECONDS: split at the largest gap in the
    middle half (Whisper often gives no gap at all between words there, so ties go to the
    word nearest the middle, never peeling off one word at a time)."""
    start, end = words[a].start, words[b].end
    if a == b or end - start <= LONG_PIECE_SECONDS:
        return [(a, b)]
    middle = (start + end) / 2
    inner = [i for i in range(a, b) if abs(words[i].end - middle) <= (end - start) / 4] or list(range(a, b))
    k = max(inner, key=lambda i: (round(words[i + 1].start - words[i].end, 2), -abs(words[i].end - middle)))
    return split_long(words, a, k) + split_long(words, k + 1, b)


def span_text(words: list, a: int, b: int) -> str:
    return "".join(w.word for w in words[a:b + 1]).strip()


def best_run(text: str, script: list[str], near: int) -> tuple[int, int, float]:
    """(first line, number of lines, score) of the run of script lines that best matches text."""
    target = words_of(text, True)
    if not target:
        return 0, 1, 0.0

    def search(lo: int, hi: int) -> tuple[int, int, float]:
        best = (0, 1, 0.0)
        for j in range(max(0, lo), min(len(script), hi)):
            run: list[str] = []
            for k in range(1, MAX_RUN_LINES + 1):
                if j + k > len(script):
                    break
                run += words_of(script[j + k - 1], True)
                if len(run) > 2 * len(target) + 4:
                    break
                score = difflib.SequenceMatcher(None, run, target, autojunk=False).ratio()
                if score > best[2]:
                    best = (j, k, score)
        return best

    best = search(near - SEARCH_BACK, near + SEARCH_AHEAD)
    if best[2] < MATCH_THRESHOLD:  # the reader jumped around: look everywhere
        best = max(best, search(0, len(script)), key=lambda r: r[2])
    return best


def split_run(words: list, a: int, b: int, lines: list[str]) -> list[tuple[int, int]]:
    """Splits words[a..b], which hold several script lines read together, at the words
    where each next line starts (aligning the script's words to Whisper's)."""
    heard, source = [], []
    for i in range(a, b + 1):
        for n in normalize_word(words[i].word, True):
            heard.append(n)
            source.append(i)
    script_words, starts = [], []
    for line in lines:
        starts.append(len(script_words))
        script_words += words_of(line, True)
    to_heard: dict[int, int] = {}
    for blk in difflib.SequenceMatcher(None, script_words, heard, autojunk=False).get_matching_blocks():
        for t in range(blk.size):
            to_heard[blk.a + t] = blk.b + t
    bounds = [a]
    for s in starts[1:]:
        # The first script word of this line that was heard, minus the words before it.
        matched = next((p for p in range(s, len(script_words)) if p in to_heard), None)
        pos = source[to_heard[matched]] if matched is not None else bounds[-1] + 1
        bounds.append(min(max(pos, bounds[-1] + 1), b))
    bounds.append(b + 1)
    return [(bounds[m], bounds[m + 1] - 1) for m in range(len(lines)) if bounds[m] <= bounds[m + 1] - 1]


def spell_digits(text: str, line: str) -> str:
    """Whisper's digits written as the words that were spoken: "20" -> "twenty", "23rd" ->
    "twenty third", and a string of digits read one by one when the script line reads
    digits that way ("984" -> "nine eight four" for "nine, eight, four")."""
    digit_by_digit = re.search(r"\b(zero|one|two|three|four|five|six|seven|eight|nine)\b"
                               r"(,?\s+(zero|one|two|three|four|five|six|seven|eight|nine)\b){2,}", line.lower())

    def words(m: re.Match) -> str:
        digits, suffix = m.group(1), m.group(2)
        if suffix:
            return ordinal_words(int(digits))
        if digit_by_digit and len(digits) >= 3:
            return " ".join(ONES[int(d)] for d in digits)
        return number_words(int(digits))

    return re.sub(r"\b(\d+)(st|nd|rd|th)?\b", words, text)


def _changes(expected: list[str], heard: list[str]) -> list[tuple]:
    return [o for o in difflib.SequenceMatcher(None, expected, heard, autojunk=False).get_opcodes() if o[0] != "equal"]


def _at_edge(expected: list[str], heard: list[str]) -> bool:
    """A word missing or added at the very start or end."""
    return any(i1 == 0 or i2 == len(expected) or j1 == 0 or j2 == len(heard)
               for tag, i1, i2, j1, j2 in _changes(expected, heard) if tag != "replace")


def _sound_alike_swaps(expected: list[str], heard: list[str]) -> bool:
    """Every difference is one word heard as a similar one ("plan"/"plant", "grab"/"graph")."""
    return all(tag == "replace" and i2 - i1 == 1 and j2 - j1 == 1
               and difflib.SequenceMatcher(None, expected[i1], heard[j1]).ratio() >= 0.6
               for tag, i1, i2, j1, j2 in _changes(expected, heard))


def choose_text(line: str, in_recording: str, in_clip: str) -> tuple[str | None, str]:
    """The transcript for a clip, from the script line and Whisper's two hearings of it
    (in the whole recording, and in the cut clip alone), or None and why it was dropped.
    The script is trusted, except where both hearings agree the reader said something else."""
    s, r, c = words_of(line), words_of(in_recording), words_of(in_clip)
    if c == s:
        return line, ""
    if _at_edge(s, c) and not _at_edge(s, r):
        return None, "cut clipped a word at the edge"
    if _sound_alike_swaps(s, c):
        return line, ""  # Whisper mishearing a word in a short clip
    if c == r:
        return spell_digits(in_clip, line), ""  # both hearings agree: the reader's own wording
    if similarity(in_recording, line) >= 0.85:
        return line, ""
    return None, "unclear what was said"


def cut(audio: np.ndarray, words: list, a: int, b: int) -> np.ndarray:
    """words[a..b] with a little silence around them, never reaching into the next words."""
    start, end = words[a].start, words[b].end
    prev_end = words[a - 1].end if a > 0 else 0.0
    next_start = words[b + 1].start if b + 1 < len(words) else len(audio) / SAMPLE_RATE
    lo = max(start - LEAD_SECONDS, (prev_end + start) / 2, 0.0)
    hi = min(end + TAIL_SECONDS, (end + next_start) / 2, len(audio) / SAMPLE_RATE)
    clip = audio[int(lo * SAMPLE_RATE):int(hi * SAMPLE_RATE)].copy()
    fade = int(0.01 * SAMPLE_RATE)
    clip[:fade] *= np.linspace(0, 1, fade)
    clip[-fade:] *= np.linspace(1, 0, fade)
    return clip


def level(clip: np.ndarray) -> np.ndarray:
    speech = clip[np.abs(clip) > 1e-3]
    rms_db = 20 * np.log10(np.sqrt(np.mean(np.square(speech))) + 1e-9) if len(speech) else TARGET_RMS_DB
    clip = clip * 10 ** ((TARGET_RMS_DB - rms_db) / 20)
    peak = np.abs(clip).max()
    return clip * (MAX_PEAK / peak) if peak > MAX_PEAK else clip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings", type=Path, default=DATA_DIR / "recordings")
    parser.add_argument("--script", type=Path, default=Path(__file__).resolve().parent / "reading_script.txt")
    parser.add_argument("--out_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--ref_audio", type=Path, default=None,
                        help="reference clip for every sample (default: a clean ~10 s clip from the recordings)")
    args = parser.parse_args()

    script = load_script(args.script)
    files = sorted((p for p in args.recordings.iterdir() if p.suffix.lower() in AUDIO_TYPES),
                   key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.stem)])
    if not files:
        raise SystemExit(f"no recordings in {args.recordings}")
    clips_dir = args.out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for old in clips_dir.glob("*.wav"):
        old.unlink()

    whisper = load_whisper()
    kept, report, covered, near = [], [], set(), 0
    for f in files:
        audio = load_audio(f)
        if np.mean(np.abs(audio) >= CLIPPED_LEVEL) > 1e-4:
            print(f"warning: {f.name} is clipping (too loud): record with the input level lower")
        words = [w for s in transcribe(whisper, audio, words=True) for w in s.words]
        pieces = split_pieces(words)
        n_kept = 0
        i = 0
        while i < len(pieces):
            # A sentence split by pauses: try joining this piece with the next one or two.
            options = []
            for m in range(1, MAX_MERGE_PIECES + 1):
                if i + m > len(pieces):
                    break
                a, b = pieces[i][0], pieces[i + m - 1][1]
                j, k, score = best_run(span_text(words, a, b), script, near)
                options.append((score, -m, m, a, b, j, k))
            score, _, m, a, b, j, k = max(options)
            if score < MATCH_THRESHOLD:
                report.append({"recording": f.name, "start": round(words[a].start, 2), "end": round(words[b].end, 2),
                               "heard": span_text(words, *pieces[i]), "script_line": "", "match": round(score, 2),
                               "kept": False, "reason": "doesn't match the script", "text": "", "clip": ""})
                i += 1
                continue
            lines = script[j:j + k]
            spans = split_run(words, a, b, lines) if k > 1 else [(a, b)]
            for line_no, (sa, sb) in zip(range(j, j + k), spans):
                line = script[line_no]
                raw = cut(audio, words, sa, sb)
                clipped = np.mean(np.abs(raw) >= CLIPPED_LEVEL) > 1e-4
                clip = level(raw)
                seconds = len(clip) / SAMPLE_RATE
                heard = " ".join(s.text.strip() for s in transcribe(whisper, clip, words=False))
                check = similarity(heard, line)
                text, reason = choose_text(line, span_text(words, sa, sb), heard)
                if not MIN_SECONDS <= seconds <= MAX_SECONDS:
                    reason = f"{seconds:.1f}s long"
                elif check < MATCH_THRESHOLD:
                    reason = "cut clip doesn't match its sentence"
                elif clipped:
                    reason = "clipping"
                name = f"{f.stem}_{len(report):04d}.wav"
                if not reason:
                    sf.write(clips_dir / name, clip, SAMPLE_RATE)
                    kept.append({"audio": str(clips_dir / name), "text": text, "seconds": seconds,
                                 "exact": text == line})
                    covered.add(line_no)
                    n_kept += 1
                report.append({"recording": f.name, "start": round(words[sa].start, 2), "end": round(words[sb].end, 2),
                               "heard": heard, "script_line": line_no + 1, "match": round(check, 2),
                               "kept": not reason, "reason": reason, "text": text if not reason else "",
                               "clip": name if not reason else ""})
            near = j + k
            i += m
        print(f"{f.name}: {len(pieces)} pieces, {n_kept} clips kept", flush=True)

    ref = args.out_dir / "ref.wav"
    if args.ref_audio:
        sf.write(ref, level(load_audio(args.ref_audio)), SAMPLE_RATE)
        print(f"reference clip: {args.ref_audio}")
    else:  # the exactly matching clip closest to 10 s, from the same recordings
        pick = min((c for c in kept if c["exact"]), key=lambda c: abs(c["seconds"] - REF_TARGET_SECONDS))
        sf.write(ref, sf.read(pick["audio"], dtype="float32")[0], SAMPLE_RATE)
        print(f"reference clip: {Path(pick['audio']).name} ({pick['seconds']:.1f}s): {pick['text']}")

    with open(args.out_dir / "train_raw.jsonl", "w", encoding="utf-8") as out:
        for c in kept:
            out.write(json.dumps({"audio": c["audio"], "text": c["text"], "ref_audio": str(ref),
                                  "language": "English"}, ensure_ascii=False) + "\n")
    with open(args.out_dir / "report.csv", "w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    minutes = sum(c["seconds"] for c in kept) / 60
    print(f"kept {len(kept)} clips, {minutes:.1f} minutes ({sum(not c['exact'] for c in kept)} with your own wording); "
          f"{len(covered)} of {len(script)} script sentences covered")
    print(f"wrote {args.out_dir / 'train_raw.jsonl'} and {args.out_dir / 'report.csv'}")


if __name__ == "__main__":
    main()
