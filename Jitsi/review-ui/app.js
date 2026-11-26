"use strict";
// Interview Review UI — client logic.
// Fetches /api/meetings, lists meetings, and for the selected one shows the
// recording + transcript. Clicking a transcript line seeks the video; the line
// under the current playhead is highlighted as the video plays.

let MEETINGS = [];
let selectedId = null;

// Transcript timestamps are stamped when the ASR *returned* the text, which on a
// slow (CPU) ASR is a few seconds AFTER the words were actually spoken. So a
// click would otherwise land just past the moment. Seek this many seconds
// EARLIER so you land on (or just before) the speech instead of after it.
// Lower it toward 0 once the ASR is fast (e.g. running on a GPU).
const SEEK_LEAD_SECONDS = 3;

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function meetingId(m, i) { return `${i}:${m.room}:${Math.round(m.sort_epoch)}`; }

async function load() {
  try {
    const res = await fetch("/api/meetings", { cache: "no-store" });
    MEETINGS = await res.json();
  } catch (e) {
    MEETINGS = [];
  }
  MEETINGS.forEach((m, i) => (m._id = meetingId(m, i)));
  $("#count").textContent = MEETINGS.length
    ? `${MEETINGS.length} meeting${MEETINGS.length > 1 ? "s" : ""}`
    : "";
  // keep selection if still present, else pick first
  if (!MEETINGS.some((m) => m._id === selectedId)) {
    selectedId = MEETINGS.length ? MEETINGS[0]._id : null;
  }
  renderPicker();
  renderDetail();
}

function renderPicker() {
  const sel = $("#picker");
  if (!MEETINGS.length) {
    sel.innerHTML = `<option>No meetings</option>`;
    sel.disabled = true;
    return;
  }
  sel.disabled = false;
  sel.innerHTML = MEETINGS.map((m) => {
    const tags = [m.has_video ? "video" : "", m.has_transcript ? "transcript" : ""]
      .filter(Boolean).join(" + ");
    const label = `${m.title} — ${m.when || ""}${tags ? "  [" + tags + "]" : ""}`;
    return `<option value="${esc(m._id)}" ${m._id === selectedId ? "selected" : ""}>${esc(label)}</option>`;
  }).join("");
}

function fmtDur(s) {
  if (s == null) return "";
  s = Math.max(0, Math.floor(s));
  const m = Math.floor(s / 60), sec = s % 60;
  const h = Math.floor(m / 60), mm = m % 60;
  return h ? `${h}:${String(mm).padStart(2, "0")}:${String(sec).padStart(2, "0")}`
           : `${m}:${String(sec).padStart(2, "0")}`;
}

function renderDetail() {
  const main = $("#detail");
  const m = MEETINGS.find((x) => x._id === selectedId);
  if (!m) {
    main.innerHTML = `<div class="empty"><div>
      <div style="font-size:15px;margin-bottom:6px">No meeting selected</div>
      <div class="muted">Recorded interviews will appear here. Record a meeting and refresh.</div>
    </div></div>`;
    return;
  }

  const whoChips = (m.participants || []).length
    ? m.participants.map((p) => `<span class="chip">${esc(p)}</span>`).join("")
    : `<span class="chip muted">unknown</span>`;

  const videoBlock = m.has_video
    ? `<div class="videocard"><video id="player" controls preload="metadata" src="${esc(m.video)}"></video></div>`
    : `<div class="novideo">No video recording for this meeting.<br>
       <span style="font-size:12px">(Transcript only — start a recording during the call to capture video.)</span></div>`;

  const nLines = (m.entries || []).length;
  const speakers = [...new Set((m.entries || []).map((e) => e.speaker))];
  const speakerChips = speakers.length
    ? speakers.map((s) => `<span class="chip">${esc(s)}</span>`).join("")
    : `<span class="chip muted">—</span>`;
  const videoFile = m.video ? decodeURIComponent(m.video.split("/").pop()) : null;
  const detailsPanel = `
      <div class="panel">
        <h2 class="mhead">${esc(m.title)}</h2>
        <div class="mwhen">${esc(m.when || "")}</div>
        <div class="drow"><span class="dl">Meeting</span>
          <div class="dv">${esc(m.title)}</div></div>
        <div class="drow"><span class="dl">Who joined (${(m.participants || []).length})</span>
          <div class="who">${whoChips}</div></div>
        <div class="drow"><span class="dl">Spoke in transcript (${speakers.length})</span>
          <div class="who">${speakerChips}</div></div>
        <div class="drow"><span class="dl">Recording length</span>
          <div class="dv">${m.has_video ? fmtDur(m.duration) : "— (no recording)"}</div></div>
        <div class="drow"><span class="dl">Transcript</span>
          <div class="dv">${m.has_transcript ? nLines + " line" + (nLines === 1 ? "" : "s") : "— (none)"}</div></div>
        ${videoFile ? `<div class="drow"><span class="dl">Video file</span>
          <div class="dv" style="word-break:break-all;font-size:12px;color:var(--muted)">${esc(videoFile)}</div></div>` : ""}
      </div>`;

  let transcriptBlock;
  if (m.has_transcript && m.entries.length) {
    const seekable = m.has_video;
    const hint = seekable
      ? "Click any line to jump the video to that moment."
      : "No video to sync — record the meeting to enable click-to-jump.";
    const rows = m.entries.map((e, idx) => {
      const canSeek = seekable && e.offset != null;
      const jump = canSeek ? `<span class="jump">▶ ${fmtDur(e.offset)}</span>` : "";
      return `<div class="trow ${canSeek ? "clickable" : ""}" data-idx="${idx}" ${
        canSeek ? `data-offset="${e.offset}"` : ""}>
        <div class="head"><span class="tt">${esc(e.time)}${jump}</span><span class="tsp">${esc(e.speaker)}</span></div>
        <div class="ttx">${esc(e.text)}</div>
      </div>`;
    }).join("");
    transcriptBlock = `
      <div class="thead"><h3>Transcript</h3><span class="hint">${hint}</span></div>
      <div class="transcript" id="transcript">${rows}</div>`;
  } else {
    transcriptBlock = `
      <div class="thead"><h3>Transcript</h3></div>
      <div class="novideo">No transcript for this meeting.</div>`;
  }

  main.innerHTML = `
    <aside class="vnav">
      ${videoBlock}
      ${transcriptBlock}
    </aside>
    <section class="maincol">
      ${detailsPanel}
    </section>`;

  wireSync();
}

function wireSync() {
  const player = $("#player");
  const transcript = $("#transcript");
  if (!transcript) return;

  const rows = Array.from(transcript.querySelectorAll(".trow"));
  // click-to-seek
  rows.forEach((row) => {
    if (!row.classList.contains("clickable")) return;
    row.addEventListener("click", () => {
      if (!player) return;
      const off = parseFloat(row.dataset.offset);
      if (!isNaN(off)) {
        player.currentTime = Math.max(0, off - SEEK_LEAD_SECONDS);
        player.play().catch(() => {});
      }
    });
  });

  if (!player) return;
  // highlight the line under the playhead
  const offsets = rows.map((r) => r.dataset.offset != null ? parseFloat(r.dataset.offset) : null);
  let activeIdx = -1;
  player.addEventListener("timeupdate", () => {
    const t = player.currentTime;
    let idx = -1;
    for (let i = 0; i < offsets.length; i++) {
      if (offsets[i] != null && offsets[i] <= t + 0.25) idx = i; else if (offsets[i] != null) break;
    }
    if (idx === activeIdx) return;
    if (activeIdx >= 0) rows[activeIdx].classList.remove("active");
    activeIdx = idx;
    if (idx >= 0) {
      rows[idx].classList.add("active");
      const r = rows[idx].getBoundingClientRect();
      const host = transcript.getBoundingClientRect();
      if (r.top < host.top || r.bottom > host.bottom) {
        rows[idx].scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    }
  });
}

$("#refresh").addEventListener("click", load);
$("#picker").addEventListener("change", (e) => { selectedId = e.target.value; renderDetail(); });
load();
