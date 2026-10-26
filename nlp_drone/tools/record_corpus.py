"""Record yourself saying each corpus command (push-to-talk) for tools/eval_stt.py.

  python tools/record_corpus.py --speaker ian            # saves recordings/ian/01.wav ... 30.wav
  python tools/eval_stt.py --audio recordings            # then evaluate on your own voice
Already-recorded ids are skipped, so you can stop (Ctrl+C) and resume later.
"""
import argparse, sys, wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from nlp_drone import stt
from nlp_drone.corpus import load


def save_wav(path, audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(stt.SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", required=True, help="folder name for this speaker, e.g. your name")
    ap.add_argument("--out", type=Path, default=ROOT / "recordings")
    args = ap.parse_args()
    d = args.out / args.speaker
    d.mkdir(parents=True, exist_ok=True)
    items = load()
    for it in items:
        f = d / f"{it['id']:02d}.wav"
        if f.exists():
            continue
        print(f"\n[{it['id']}/{len(items)}] say:  \"{it['text']}\"")
        while True:
            audio = stt.record_push_to_talk()
            if len(audio) < stt.SAMPLE_RATE * 0.3 or np.abs(audio).max() < 0.01:
                print("  too short or silent - check microphone permission, then try again"); continue
            save_wav(f, audio)
            print(f"  saved {f.relative_to(ROOT)} ({len(audio) / stt.SAMPLE_RATE:.1f} s)")
            break
    print(f"\nDone. Evaluate with:  python tools/eval_stt.py --audio {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
