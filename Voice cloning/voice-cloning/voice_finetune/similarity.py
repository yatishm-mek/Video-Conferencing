"""How close each sample sounds to the real voice, as a number.

Every recording is turned into a voice fingerprint (a speaker embedding, from Qwen3-TTS's
own speaker encoder) and compared with the average fingerprint of the real recordings
(cosine similarity: 1.0 = identical). Two reference points make the numbers readable:
  - real vs real: each real clip against the average of the others, as high as it gets;
  - a different person: another voice's zero-shot sample (qwen3tts_1_7b.wav, if present), which should score clearly lower.
The encoder is also what zero-shot cloning listens to, so treat small differences with
care and trust your ears for the final choice.

    "%USERPROFILE%\\.venvs\\qwen3tts\\Scripts\\python.exe" voice_finetune\\similarity.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from voice_finetune.common import BASE_MODEL, RAW_JSONL  # noqa: E402

SAMPLES_DIR = ROOT / "output" / "voice_samples"
OTHER_PERSON = "qwen3tts_1_7b"   # another person's voice, if present
REAL_CLIPS = 60
CHUNK_SECONDS = 6.0               # long samples are scored in pieces and averaged


def main() -> None:
    from qwen_tts import Qwen3TTSModel

    qwen = Qwen3TTSModel.from_pretrained(str(BASE_MODEL), device_map="cuda:0", dtype=torch.bfloat16,
                                         attn_implementation="sdpa")
    model = qwen.model
    sr = model.speaker_encoder_sample_rate

    def embed(audio: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            e = model.extract_speaker_embedding(audio=audio.astype(np.float32), sr=sr)
        e = e.float().cpu().numpy().reshape(-1)
        return e / np.linalg.norm(e)

    def chunk_embeddings(path: Path) -> list[np.ndarray]:
        """One fingerprint per ~6 s, so long samples are judged like the short real clips."""
        audio, _ = librosa.load(path, sr=sr, mono=True)
        n = int(CHUNK_SECONDS * sr)
        chunks = [audio[i:i + n] for i in range(0, len(audio), n) if len(audio[i:i + n]) > n // 2] or [audio]
        return [embed(c) for c in chunks]

    def embed_file(path: Path) -> np.ndarray:
        e = np.mean(chunk_embeddings(path), axis=0)
        return e / np.linalg.norm(e)

    rows = [json.loads(x) for x in RAW_JSONL.read_text(encoding="utf-8").splitlines() if x]
    random.seed(0)
    real = np.stack([embed_file(Path(r["audio"])) for r in random.sample(rows, min(REAL_CLIPS, len(rows)))])
    center = real.mean(axis=0)
    center /= np.linalg.norm(center)
    leave_one_out = []
    for i in range(len(real)):
        others = np.delete(real, i, axis=0).mean(axis=0)
        leave_one_out.append(float(real[i] @ (others / np.linalg.norm(others))))

    print(f"{'recording':32} similarity to the real voice")
    print(f"{'real vs real (one clip)':32} {np.mean(leave_one_out):.3f}  (range {min(leave_one_out):.3f}-{max(leave_one_out):.3f})")
    results = {}
    for wav in sorted(SAMPLES_DIR.glob("*.wav")):
        score = float(np.mean([e @ center for e in chunk_embeddings(wav)]))
        results[wav.stem] = score
        note = "  <- a different person" if wav.stem == OTHER_PERSON else ""
        print(f"{wav.stem:32} {score:.3f}{note}")
    (SAMPLES_DIR / "similarity.json").write_text(json.dumps(
        {"real_vs_real": float(np.mean(leave_one_out)), "samples": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
