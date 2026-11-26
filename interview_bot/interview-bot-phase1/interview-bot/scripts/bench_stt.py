"""CPU speech-to-text benchmark on your own recorded voice.

    python scripts/record_stt.py --device N         # first: record data/stt_bench/
    python scripts/bench_stt.py                     # all models, CPU, 8 threads
    python scripts/bench_stt.py --models parakeet-v2 moonshine-medium --threads 6
    python scripts/bench_stt.py --list

Everything runs on CPU only. For each model it reports:
  WER        word error rate over the clips that have a reference (lower is better)
  terms      tech terms / numbers from the reference that came out right
             (LoRA, BM25, Neo4j, 0.87 ...): this is what the interview bot cares about
  answer s   median seconds to transcribe a whole answer (clips of 10 s or more)
  RTF        compute seconds per second of audio (must stay well below 1)
  after s    estimated wait after the candidate stops talking:
               whisper / parakeet: time for the last 6 s of audio (the live bot
                 transcribes VAD segments, so only the last one is left at the end)
               moonshine streaming: time of the final pass after the audio is fed
                 in 0.1 s chunks (it transcribes while the candidate talks)
  short s    median time for the short command clips (under 4 s)

Whisper gets the same vocab prompt the live bot uses (candidate name + plan vocab)
from --plan. The other models have no prompt option.

Extra installs (only for the models you run):
  parakeet-*   pip install onnx-asr
  moonshine-*  pip install moonshine-voice
Model files download to the normal Hugging Face / Moonshine caches, like Whisper.
The report is saved to logs/bench_stt_*.txt.
"""

import argparse
import os
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "data" / "stt_bench"
SR = 16000

MODELS = {
    # name: (backend, model id)
    "whisper-turbo": ("whisper", "deepdml/faster-whisper-large-v3-turbo-ct2"),  # what the bot uses now
    "distil-large-v3": ("whisper", "Systran/faster-distil-whisper-large-v3"),
    "whisper-small.en": ("whisper", "small.en"),
    "parakeet-v2": ("parakeet", "nemo-parakeet-tdt-0.6b-v2"),
    "parakeet-v3": ("parakeet", "nemo-parakeet-tdt-0.6b-v3"),
    "moonshine-medium": ("moonshine", "MEDIUM_STREAMING"),
    "moonshine-small": ("moonshine", "SMALL_STREAMING"),
    "moonshine-base": ("moonshine", "BASE"),
}
DEFAULT = ["whisper-turbo", "distil-large-v3", "parakeet-v2", "moonshine-medium", "moonshine-small"]

# Terms checked when they appear in a reference. Matching ignores case, spaces and
# hyphens ("Neo 4j" == "Neo4j"), but not spelling ("Laura" != "LoRA").
TERMS = [
    "Hardhik", "Hyderabad", "Whisper", "Pipecat", "autoencoder", "anomaly detection",
    "reconstruction error", "95th percentile", "Neo4j", "knowledge graph", "Cypher", "LoRA",
    "QLoRA", "16", "32", "8 GB", "4 bit", "quantization", "epochs", "RAG", "500", "embeddings",
    "ChromaDB", "BM25", "cosine similarity", "LLM", "Ollama", "overfitting", "dropout",
    "early stopping", "validation error", "3 percent", "precision", "recall", "F1 score", "0.87",
    "Label Studio", "AimStack", "MCP", "SQL Server", "Docker", "Django", "SVM", "decision trees",
    "feature engineering", "XGBoost", "BERT", "Llama", "GPT", "summarization", "repeat",
    "ready", "not finished",
]

# ---------------------------------------------------------------------------
# Text normalisation (applied to reference and hypothesis alike)
# ---------------------------------------------------------------------------

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
        "eighth": 8, "ninth": 9, "tenth": 10, "twelfth": 12, "twentieth": 20, "thirtieth": 30,
        "fortieth": 40, "fiftieth": 50, "hundredth": 100}
_FILLERS = {"uh", "um", "umm", "uhh", "hmm", "ah", "er", "erm", "mm"}
_SUFFIX = {1: "st", 2: "nd", 3: "rd"}


def _ordinal(n: int) -> str:
    s = "th" if 10 <= n % 100 <= 20 else _SUFFIX.get(n % 10, "th")
    return f"{n}{s}"


def _words_to_digits(tokens: list[str]) -> list[str]:
    """'ninety fifth' -> '95th', 'zero point eight seven' -> '0.87', 'thirty two' -> '32'."""
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if t not in _UNITS and t not in _TENS:
            out.append(t)
            i += 1
            continue
        total, cur, j, ordinal = 0, 0, i, False
        while j < len(tokens):
            w = tokens[j]
            if w in _UNITS:
                cur += _UNITS[w]
            elif w in _TENS:
                cur += _TENS[w]
            elif w in _ORD and cur % 10 == 0 and (cur > 0 or j == i):
                cur += _ORD[w]
                ordinal = True
                j += 1
                break
            elif w == "hundred" and cur:
                cur *= 100
            elif w == "thousand":
                total += max(cur, 1) * 1000
                cur = 0
            else:
                break
            j += 1
        num = str(total + cur)
        if not ordinal and j + 1 < len(tokens) and tokens[j] == "point" and tokens[j + 1] in _UNITS:
            digits = []
            j += 1
            while j < len(tokens) and tokens[j] in _UNITS and _UNITS[tokens[j]] < 10:
                digits.append(str(_UNITS[tokens[j]]))
                j += 1
            num += "." + "".join(digits)
        out.append(_ordinal(int(num)) if ordinal else num)
        i = j
    return out


def normalize(text: str) -> list[str]:
    t = text.lower().replace("%", " percent ").replace("’", "'")
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)          # 10,000 -> 10000
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)         # keep 0.87, drop sentence dots
    t = re.sub(r"[-_/]", " ", t)
    t = re.sub(r"[^\w\s.']", " ", t)
    t = re.sub(r"(\d)(gb|bit)\b", r"\1 \2", t)        # 8gb -> 8 gb
    toks = [w.strip("'") for w in t.split()]
    toks = [w for w in toks if w and w not in _FILLERS]
    return _words_to_digits(toks)


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def has_term(term: list[str], hyp: list[str]) -> bool:
    joined = "".join(term)
    n = len(term)
    for i in range(len(hyp)):
        if hyp[i : i + n] == term:
            return True
        for k in (1, 2, 3):  # "neo 4j" vs "neo4j", "chroma db" vs "chromadb"
            if "".join(hyp[i : i + k]) == joined:
                return True
    return False


# ---------------------------------------------------------------------------
# Backends: each returns an object with .transcribe(audio) -> str
# ---------------------------------------------------------------------------


class WhisperBackend:
    def __init__(self, model_id: str, threads: int, prompt: str | None):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_id, device="cpu", compute_type="int8", cpu_threads=threads, num_workers=1)
        self.prompt = prompt

    def transcribe(self, audio: np.ndarray) -> str:
        segs, _ = self.model.transcribe(audio, language="en", beam_size=1, initial_prompt=self.prompt,
                                        condition_on_previous_text=False, vad_filter=False)
        return " ".join(s.text.strip() for s in segs)


class ParakeetBackend:
    def __init__(self, model_id: str, threads: int, prompt: str | None):
        import onnx_asr
        import onnxruntime as rt

        so = rt.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.model = onnx_asr.load_model(model_id, quantization="int8", sess_options=so,
                                         providers=["CPUExecutionProvider"])

    def transcribe(self, audio: np.ndarray) -> str:
        return str(self.model.recognize(audio, sample_rate=SR))


class MoonshineBackend:
    def __init__(self, arch_name: str, threads: int, prompt: str | None):
        from moonshine_voice import ModelArch, get_model_for_language
        from moonshine_voice.transcriber import Transcriber

        path, arch = get_model_for_language("en", getattr(ModelArch, arch_name))
        self.t = Transcriber(model_path=path, model_arch=arch)
        self.streaming = "STREAMING" in arch_name

    @staticmethod
    def _text(tr) -> str:
        return " ".join(l.text.strip() for l in (tr.lines if tr else []) if l.text)

    def transcribe(self, audio: np.ndarray) -> str:
        return self._text(self.t.transcribe_without_streaming(audio.tolist(), SR))

    def stream(self, audio: np.ndarray) -> tuple[str, float, float]:
        """Feed 0.1 s chunks as the live bot would. Returns (text, total compute s, final-pass s)."""
        s = self.t.create_stream(update_interval=0.5)
        s.start()
        step = SR // 10
        t0 = time.perf_counter()
        for i in range(0, len(audio), step):
            s.add_audio(audio[i : i + step].tolist(), SR)
        t1 = time.perf_counter()
        tr = s.stop()
        t2 = time.perf_counter()
        try:
            s.close()
        except Exception:
            pass
        return self._text(tr), t2 - t0, t2 - t1


BACKENDS = {"whisper": WhisperBackend, "parakeet": ParakeetBackend, "moonshine": MoonshineBackend}
INSTALL = {"whisper": "pip install faster-whisper", "parakeet": "pip install onnx-asr",
           "moonshine": "pip install moonshine-voice"}

# ---------------------------------------------------------------------------


def load_clips() -> list[tuple[str, np.ndarray, str | None]]:
    refs_file = BENCH_DIR / "refs.txt"
    if not refs_file.exists():
        sys.exit(f"No {refs_file}. Record first: python scripts/record_stt.py --device N")
    clips = []
    for line in refs_file.read_text(encoding="utf-8").splitlines():
        if "|" not in line:
            continue
        name, ref = (x.strip() for x in line.split("|", 1))
        path = BENCH_DIR / name
        if not path.exists():
            print(f"  missing {path}, skipped")
            continue
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if rate != SR:
            import soxr

            audio = soxr.resample(audio, rate, SR).astype(np.float32)
        clips.append((name, audio, None if ref.upper() == "TODO" else ref))
    return clips


def plan_prompt(plan_path: str | None) -> str | None:
    if not plan_path or not Path(plan_path).exists():
        return None
    import json

    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    words = [plan.get("candidate", "")] + list(plan.get("vocab", []))
    return ", ".join(w for w in words if w) or None


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def p90(xs):
    return sorted(xs)[min(len(xs) - 1, int(0.9 * len(xs)))] if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT)
    ap.add_argument("--threads", type=int, default=8, help="CPU threads per model (match the server box)")
    ap.add_argument("--plan", default=str(ROOT / "data/candidates/rithika/plan.json"),
                    help="plan.json whose vocab Whisper gets as its prompt ('' for none)")
    ap.add_argument("--tail", type=float, default=6.0, help="seconds of audio for the 'after' estimate")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, (b, m) in MODELS.items():
            print(f"  {k:18s} {b:10s} {m}{'  (default)' if k in DEFAULT else ''}")
        return

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only, everywhere
    clips = load_clips()
    prompt = plan_prompt(args.plan)
    total_audio = sum(len(a) for _, a, _ in clips) / SR
    out = []

    def say(s=""):
        print(s)
        out.append(s)

    say(f"STT benchmark {datetime.now():%Y-%m-%d %H:%M} | CPU only, {args.threads} threads | "
        f"{len(clips)} clips, {total_audio:.0f} s audio | {sum(r is not None for *_, r in clips)} with reference")
    say(f"Whisper prompt: {prompt[:100] + '...' if prompt and len(prompt) > 100 else prompt}")
    summary = []

    for key in args.models:
        if key not in MODELS:
            say(f"\n=== {key}: unknown (see --list)")
            continue
        backend, model_id = MODELS[key]
        say(f"\n=== {key} ({model_id})")
        try:
            t0 = time.perf_counter()
            m = BACKENDS[backend](model_id, args.threads, prompt if backend == "whisper" else None)
            load_s = time.perf_counter() - t0
            m.transcribe(clips[0][1][: 3 * SR])  # warm-up
        except ImportError as e:
            say(f"  skipped, missing package ({e}). Run: {INSTALL[backend]}")
            continue
        except Exception as e:
            say(f"  failed to load: {type(e).__name__}: {e}")
            continue
        say(f"  load {load_s:.1f}s")

        errs = ref_words = term_hit = term_all = 0
        ans_s, rtf, after_s, short_s, missed = [], [], [], [], []
        for name, audio, ref in clips:
            dur = len(audio) / SR
            try:
                t0 = time.perf_counter()
                hyp = m.transcribe(audio)
                full = time.perf_counter() - t0
                if getattr(m, "streaming", False):
                    _, _, after = m.stream(audio)
                else:
                    tail = audio[-int(args.tail * SR):] if dur > args.tail else audio
                    t0 = time.perf_counter()
                    m.transcribe(tail)
                    after = time.perf_counter() - t0
            except Exception as e:
                say(f"  {name}: error {type(e).__name__}: {e}")
                continue
            rtf.append(full / dur)
            after_s.append(after)
            (short_s if dur < 4 else ans_s if dur >= 10 else []).append(full)
            line = f"  {name} {dur:5.1f}s  full {full:5.2f}s  after {after:5.2f}s"
            if ref is not None:
                r, h = normalize(ref), normalize(hyp)
                e = edit_distance(r, h)
                errs += e
                ref_words += len(r)
                miss = [t for t in TERMS if has_term(normalize(t), r) and not has_term(normalize(t), h)]
                n_terms = sum(has_term(normalize(t), r) for t in TERMS)
                term_all += n_terms
                term_hit += n_terms - len(miss)
                line += f"  WER {e / max(1, len(r)):4.0%}" + (f"  missed: {', '.join(miss)}" if miss else "")
                if e:
                    missed.append((name, hyp))
            say(line)
            say(f"      heard: {hyp}")

        wer = errs / max(1, ref_words)
        summary.append((key, wer, term_hit, term_all, med(ans_s), p90(ans_s), med(rtf), med(after_s),
                        p90(after_s), med(short_s)))
        say(f"  -> WER {wer:.1%} | terms {term_hit}/{term_all} | answer {med(ans_s):.2f}s (p90 {p90(ans_s):.2f}) "
            f"| RTF {med(rtf):.2f} | after {med(after_s):.2f}s (p90 {p90(after_s):.2f}) | short {med(short_s):.2f}s")
        del m

    say("\n=== summary (CPU)")
    say(f"  {'model':18s} {'WER':>6s} {'terms':>7s} {'answer s':>9s} {'p90':>6s} {'RTF':>5s} "
        f"{'after s':>8s} {'p90':>6s} {'short s':>8s}")
    for k, wer, th, ta, a, ap90, r, af, afp90, sh in summary:
        say(f"  {k:18s} {wer:6.1%} {f'{th}/{ta}':>7s} {a:9.2f} {ap90:6.2f} {r:5.2f} {af:8.2f} {afp90:6.2f} {sh:8.2f}")
    say("Target: 'after' under ~1.5 s with WER and terms close to whisper-turbo.")

    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    path = logs / f"bench_stt_{datetime.now():%Y%m%d_%H%M%S}.txt"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
