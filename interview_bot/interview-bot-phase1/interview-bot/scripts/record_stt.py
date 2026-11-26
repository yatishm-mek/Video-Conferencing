"""Record your own voice for the speech-to-text benchmark (scripts/bench_stt.py).

    pip install sounddevice
    python scripts/record_stt.py --list              # show microphones
    python scripts/record_stt.py --device 3          # record every clip with mic 3
    python scripts/record_stt.py --device 3 --only 5 7   # re-record clips 5 and 7

For each clip it shows a text. Press Enter, say it the way you would answer in an
interview (normal speed, normal accent), then press Enter to stop.
- Clips 1-15 are read aloud. If you slip, re-record, or edit that line in refs.txt
  to what you really said.
- Clips 16-17 are free answers (spontaneous speech is harder than reading). Their
  reference is "TODO": type what you said into refs.txt, or leave TODO and the
  benchmark will only time them.

Saves data/stt_bench/clipNN.wav (16 kHz mono, 16-bit) and data/stt_bench/refs.txt
(one line per clip: clipNN.wav|reference text).
"""

import argparse
import sys
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "stt_bench"
REFS = OUT_DIR / "refs.txt"
SR = 16000

CLIPS = [
    # 1-12: interview-style answers with the tech terms the bot must get right
    "Hi, my name is Hardhik. I am an AI engineer based in Hyderabad. For the last six months "
    "I have been working on speech recognition and voice agents, mainly fine tuning Whisper "
    "models and building live transcription pipelines with Pipecat.",
    "In my anomaly detection project I trained an autoencoder only on normal transactions. "
    "If the reconstruction error was above a threshold, we marked the transaction as suspicious. "
    "We picked the threshold from the 95th percentile of the validation errors.",
    "We stored the entities in Neo4j as a knowledge graph. Each candidate, skill and company was "
    "a node, and the relationships were edges. Then I wrote Cypher queries to find which skills "
    "were connected to which job roles.",
    "To fine tune the model on our data, I used LoRA with a rank of 16 and an alpha of 32. "
    "Because the GPU had only 8 GB of memory, I also tried QLoRA with 4 bit quantization, and the "
    "training loss became stable after about 2 epochs.",
    "For the chatbot I built a RAG pipeline. First we split the documents into chunks of 500 tokens, "
    "then created embeddings and stored them in ChromaDB. For retrieval we combined BM25 with cosine "
    "similarity, and the top 5 chunks were passed to the LLM running on Ollama.",
    "Overfitting means the model does very well on the training data but badly on new data. So the "
    "training error is low and the validation error is high. To reduce it I used dropout, early "
    "stopping and more data augmentation.",
    "Our dataset was imbalanced, only around 3 percent of the samples were fraud, so accuracy was not "
    "useful. I looked at precision, recall and F1 score, and our final F1 score was 0.87 on the test set.",
    "We used Label Studio for data annotation. Two people labelled every document, and when they did "
    "not agree, a third person checked it. For experiment tracking we used AimStack so that we could "
    "compare the runs.",
    "I built an AI agent that can call tools through MCP. The agent reads the user's request, decides "
    "which tool to call, for example a SQL Server query or a web search, and then writes the final "
    "answer. I deployed it with Docker and a Django backend.",
    "Before deep learning, I tried classical models like SVM and decision trees with some feature "
    "engineering. XGBoost gave the best result, but the autoencoder was better at finding new types "
    "of anomalies that were not in the labelled data.",
    "Encoder only models like BERT are good for classification and extraction, because they see the "
    "whole sentence at once. Decoder only models like Llama or GPT generate text one token at a time, "
    "so we use them for summarization and chat.",
    "Actually, I am not fully sure about that. I think we used a batch size of 32, but I need to check. "
    "Can I explain the other part first?",
    # 13-15: short turns (commands must be fast and exact)
    "Yes, I'm ready. Let's start.",
    "Sorry, can you repeat the question?",
    "No, wait, I'm not finished yet.",
    # 16-17: free answers
    "FREE ANSWER (30-60 s): Tell me about a project you are proud of and your role in it.",
    "FREE ANSWER (30-60 s): What was the hardest bug you fixed, and how did you find it?",
]


def load_refs() -> dict[str, str]:
    refs = {}
    if REFS.exists():
        for line in REFS.read_text(encoding="utf-8").splitlines():
            if "|" in line:
                name, text = line.split("|", 1)
                refs[name.strip()] = text.strip()
    return refs


def save_refs(refs: dict[str, str]) -> None:
    REFS.write_text("".join(f"{k}|{refs[k]}\n" for k in sorted(refs)), encoding="utf-8")


def record_until_enter(device, rate: int) -> np.ndarray:
    import sounddevice as sd

    chunks, stop = [], threading.Event()

    def cb(indata, frames, t, status):
        if not stop.is_set():
            chunks.append(indata[:, 0].copy())

    with sd.InputStream(samplerate=rate, channels=1, dtype="float32", device=device, callback=cb):
        input("  recording... press Enter to stop ")
        stop.set()
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    if rate != SR:
        import soxr

        audio = soxr.resample(audio, rate, SR)
    return audio.astype(np.float32)


def pick_rate(device) -> int:
    import sounddevice as sd

    try:
        sd.check_input_settings(device=device, samplerate=SR, channels=1)
        return SR
    except Exception:
        return int(sd.query_devices(device, "input")["default_samplerate"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list microphones and exit")
    ap.add_argument("--device", type=int, default=None, help="mic index from --list (default: system mic)")
    ap.add_argument("--only", type=int, nargs="*", help="clip numbers to (re)record")
    args = ap.parse_args()

    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("Missing sounddevice. Run: pip install sounddevice")

    if args.list:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                print(f"{i:3d}  {d['name']}  ({int(d['default_samplerate'])} Hz)")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rate = pick_rate(args.device)
    name = sd.query_devices(args.device, "input")["name"]
    print(f"Mic: {name} @ {rate} Hz -> saved as 16 kHz mono in {OUT_DIR}")
    refs = load_refs()
    todo = args.only or list(range(1, len(CLIPS) + 1))

    for n in todo:
        text = CLIPS[n - 1]
        fname = f"clip{n:02d}.wav"
        print(f"\n[{n}/{len(CLIPS)}] {text}")
        while True:
            input("  press Enter to start ")
            audio = record_until_enter(args.device, rate)
            secs = len(audio) / SR
            peak = float(np.abs(audio).max()) if len(audio) else 0.0
            print(f"  {secs:.1f} s, peak {peak:.2f}" + ("  (very quiet - check the mic)" if peak < 0.05 else ""))
            if input("  keep it? [Enter = yes, r = redo] ").strip().lower() != "r":
                break
        sf.write(OUT_DIR / fname, audio, SR, subtype="PCM_16")
        refs[fname] = "TODO" if text.startswith("FREE ANSWER") else text
        save_refs(refs)

    print(f"\nDone. {len(refs)} clips in {REFS}")
    print("Next: python scripts/bench_stt.py")


if __name__ == "__main__":
    main()
