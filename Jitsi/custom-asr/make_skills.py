"""
Generate per-candidate biasing skills for the Jitsi Moonshine ASR.

This is the SKILLS-GENERATION half of D:\\Realtime-ASR\\dynamic_interview_asr.py,
copied here and trimmed to drop the live-mic / report code (Jigasi + the ASR
server already do the live part). It reads a job-description JSON and a resume
JSON, extracts skill terms, tokenizes them with the fine-tuned model's tokenizer,
builds the biasing trie, and writes them to a SEPARATE output folder:

    custom-asr/generated_skills/<out-name>_skills.json
    custom-asr/generated_skills/<out-name>_trie.json

These are NOT used by run-asr.ps1 — the server keeps its fixed default (the
benchmark set in ./data/). To use a generated set, point ASR_VOCAB / ASR_TRIE at
the files above before starting run-asr.ps1.

Usage (with the Realtime-ASR venv Python):
    python make_skills.py
        -> paste the JD JSON, type END, paste the resume JSON, type END.
    python make_skills.py --jd jd.json --resume resume.json
        -> read them from files instead of pasting.
Options:
    --out-name candidate   base filename (default 'candidate' -> candidate_skills.json)
    --keep-dated           also save a timestamped copy alongside the default

Standalone: this tool does not import or depend on the ASR server code
(asr_engine / asr_ws_server). It only loads the model's tokenizer, chosen with
--checkpoint (default: ./model) or the ASR_CHECKPOINT env var.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

from transformers import AutoProcessor

HERE = Path(__file__).resolve().parent
# Generated per-candidate skills go here — a SEPARATE folder from ./data/.
# The ASR server never reads this folder automatically; its fixed default stays
# the benchmark set in ./data/. To use a generated set, point ASR_VOCAB/ASR_TRIE
# at the files written here.
OUT_DIR = HERE / "generated_skills"

# Standalone: this tool does NOT import the ASR server code (asr_engine). It only
# needs the model's tokenizer. By default it uses the local checkpoint copy; the
# tokenizer is identical across the moonshine checkpoints, so any works. Override
# with --checkpoint or the ASR_CHECKPOINT env var.
DEFAULT_CHECKPOINT = Path(os.environ.get("ASR_CHECKPOINT", str(HERE / "model")))

# ---- extraction config (copied verbatim from dynamic_interview_asr.py) ------
SKILL_KEY_HINTS = [
    "skill", "technolog", "tech_stack", "techstack", "tool", "competenc",
    "keyword", "stack", "framework", "language", "platform", "certif",
    "proficien", "expertise", "software", "systems", "domain",
    "must_have", "must have", "nice_to_have", "nice to have",
    "requirement", "qualification",
]

NATURAL_LANGUAGES = {
    "english", "hindi", "telugu", "tamil", "kannada", "malayalam", "marathi",
    "bengali", "gujarati", "punjabi", "odia", "assamese", "urdu", "sanskrit",
    "spanish", "french", "german", "mandarin", "chinese", "cantonese",
    "japanese", "korean", "arabic", "portuguese", "russian", "italian",
    "dutch", "turkish", "vietnamese", "thai", "indonesian", "polish",
    "native", "mother tongue", "bilingual", "fluent",
}

STOPWORDS = {
    "experience", "years", "year", "strong", "good", "excellent", "knowledge",
    "understanding", "ability", "team", "teamwork", "communication",
    "communications", "etc", "various", "using", "work", "working", "skills",
    "skill", "proficiency", "expertise", "familiarity", "hands on", "hands-on",
    "plus", "preferred", "required", "must", "nice to have", "responsible",
    "responsibilities", "including", "related", "other", "self", "detail",
    "oriented", "written", "verbal", "problem solving", "problem-solving",
}

MAX_TERM_WORDS = 4

LEADING_FILLER = re.compile(
    r"^(?:experience (?:with|in|of)|knowledge of|familiar(?:ity)? with|"
    r"proficien(?:t|cy) (?:with|in)|expert(?:ise)? (?:with|in)|"
    r"understanding of|working knowledge of|hands[- ]on(?: with| in)?|"
    r"ability to|strong|good|solid|excellent|advanced|basic|deep)\s+",
    re.I,
)


def _clean_term(t: str) -> str:
    t = t.strip().strip("-–—*:.•()[]\"' ")
    t = re.sub(r"\s+", " ", t)
    while True:
        stripped = LEADING_FILLER.sub("", t)
        if stripped == t:
            break
        t = stripped.strip()
    return t


def _split_terms(s: str):
    parts = re.split(r"[,;|\n\t•·]", s)
    out = []
    for p in parts:
        for q in re.split(r"\s(?:and|&)\s", p, flags=re.I):
            c = _clean_term(q)
            if c:
                out.append(c)
    return out


def _acceptable(term: str) -> bool:
    if len(term) < 2:
        return False
    if term.lower() in STOPWORDS:
        return False
    if term.lower() in NATURAL_LANGUAGES:
        return False
    if term.replace(".", "").isdigit():
        return False
    if len(term.split()) > MAX_TERM_WORDS:
        return False
    if not re.search(r"[A-Za-z]", term):
        return False
    return True


def extract_terms(obj, under_skill_key=False, out=None):
    if out is None:
        out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).strip().lower()
            flag = under_skill_key or any(h in kl for h in SKILL_KEY_HINTS)
            extract_terms(v, flag, out)
    elif isinstance(obj, list):
        for it in obj:
            extract_terms(it, under_skill_key, out)
    elif isinstance(obj, str) and under_skill_key:
        for t in _split_terms(obj):
            if _acceptable(t):
                out.append(t)
    return out


def dedupe_terms(terms):
    seen, result = set(), []
    for t in terms:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            result.append(t)
    return result


def guess_name(resume):
    if isinstance(resume, dict):
        for k in ("name", "full_name", "candidate_name", "candidate"):
            v = resume.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        basics = resume.get("basics")
        if isinstance(basics, dict) and isinstance(basics.get("name"), str):
            return basics["name"].strip()
    return None


def build_vocabulary(terms, tokenizer):
    tokenized = []
    for term in terms:
        ids = tokenizer.encode(term, add_special_tokens=False)
        if not ids:
            continue
        tokenized.append({
            "term": term,
            "classification": "strong",
            "strength": "strong",
            "decoder_target": True,
            "valid": True,
            "token_ids": [int(i) for i in ids],
            "token_count": len(ids),
        })
    return tokenized


def build_trie(tokenized):
    root = {"children": {}}
    for t in tokenized:
        node = root
        for tid in t["token_ids"]:
            node = node["children"].setdefault(str(int(tid)), {"children": {}})
        node["term"] = t["term"]
    return {"trie": root}


def read_json_stdin(label: str):
    """Read a JSON document pasted into the terminal (ended by a line 'END')."""
    print(f"\n>>> Paste the {label} JSON below.")
    print(">>> When done, type a line containing only:  END")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().upper() == "END":
            break
        lines.append(line)
    text = "\n".join(lines).strip()
    if not text:
        raise SystemExit(f"No {label} JSON provided.")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"{label} is not valid JSON: {e}")


def get_input(path_opt, label):
    """From a file if --<label> was given, else pasted into the terminal."""
    if path_opt:
        p = Path(path_opt)
        if not p.exists():
            raise SystemExit(f"File not found: {p}")
        return json.loads(p.read_text(encoding="utf-8")), str(p)
    return read_json_stdin(label), "terminal paste"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jd", default=None,
                    help="Job-description JSON file. Omit to paste it in the terminal.")
    ap.add_argument("--resume", default=None,
                    help="Resume JSON file. Omit to paste it in the terminal.")
    ap.add_argument("--name", default=None, help="Candidate name (for the dated copy).")
    ap.add_argument("--out-name", default="candidate",
                    help="Base name for the output files (default: candidate -> "
                         "generated_skills/candidate_skills.json + _trie.json).")
    ap.add_argument("--keep-dated", action="store_true",
                    help="Also save a timestamped copy in generated_skills/.")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                    help="Model checkpoint whose tokenizer to use "
                         f"(default: {DEFAULT_CHECKPOINT}).")
    args = ap.parse_args()

    jd, jd_src = get_input(args.jd, "JD (job description)")
    resume, resume_src = get_input(args.resume, "resume")

    terms = dedupe_terms(extract_terms(jd) + extract_terms(resume))
    if not terms:
        raise SystemExit(
            "No skills found. Ensure skills live under fields like 'skills', "
            "'technologies', 'tools', 'requirements', 'qualifications'...")
    print(f"Extracted {len(terms)} skill terms:")
    print("  " + ", ".join(terms))

    print(f"\nLoading tokenizer from: {args.checkpoint}")
    processor = AutoProcessor.from_pretrained(str(args.checkpoint))

    tokenized = build_vocabulary(terms, processor.tokenizer)
    trie = build_trie(tokenized)
    print(f"Tokenized {len(tokenized)} terms into the biasing trie.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    skills_payload = {
        "generated": ts,
        "jd_source": jd_src,
        "resume_source": resume_src,
        "term_count": len(tokenized),
        "model": str(args.checkpoint),
        "terms": tokenized,
    }

    # Written to generated_skills/ — the server does NOT read these automatically.
    skills_path = OUT_DIR / f"{args.out_name}_skills.json"
    trie_path = OUT_DIR / f"{args.out_name}_trie.json"
    skills_path.write_text(json.dumps(skills_payload, indent=2), encoding="utf-8")
    trie_path.write_text(json.dumps(trie), encoding="utf-8")
    print(f"\nSkills file : {skills_path}")
    print(f"Trie file   : {trie_path}")

    if args.keep_dated:
        base = re.sub(r"[^A-Za-z0-9_-]+", "_",
                      (args.name or guess_name(resume) or "candidate")).strip("_") or "candidate"
        (OUT_DIR / f"{base}_{ts}_skills.json").write_text(
            json.dumps(skills_payload, indent=2), encoding="utf-8")
        (OUT_DIR / f"{base}_{ts}_trie.json").write_text(
            json.dumps(trie), encoding="utf-8")
        print(f"Dated copy    : {base}_{ts}_skills.json / _trie.json")

    print("\nDone. These files are NOT used by run-asr.ps1 (it keeps the default")
    print("benchmark set). To use them, point the server at them, e.g. in PowerShell:")
    print(f'  $env:ASR_VOCAB="{skills_path}"; $env:ASR_TRIE="{trie_path}"; .\\run-asr.ps1')


if __name__ == "__main__":
    main()
