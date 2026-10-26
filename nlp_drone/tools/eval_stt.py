"""Evaluate speech-to-text engines on the command corpus.

  python tools/eval_stt.py                      # synthesise audio (macOS 'say') if missing, run all engines
  python tools/eval_stt.py --audio my_recordings # use your own recordings: <dir>/<speaker>/<id>.wav

Metrics per engine: word error rate (WER), command accuracy (transcript parses to exactly the
expected commands), false activations (non-command phrases that produced a command), mean latency.
"""
import argparse, csv, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import jiwer
from nlp_drone import stt
from nlp_drone.corpus import load, matches
from nlp_drone.parser import parse
from nlp_drone.validator import validate, State

VOICES = ["Daniel", "Eddy (English (US))", "Flo (English (UK))"]
ENGINES = ["whisper:tiny.en", "whisper:base.en", "vosk", "vosk:free"]


def synthesise(audio_dir, items):
    for v in VOICES:
        d = audio_dir / ("tts_" + re.sub(r"\W+", "_", v).strip("_"))
        d.mkdir(parents=True, exist_ok=True)
        for it in items:
            f = d / f"{it['id']:02d}.wav"
            if not f.exists():
                subprocess.run(["say", "-v", v, "--data-format=LEI16@16000", "-o", str(f), it["text"]], check=True)


def norm(text):
    """Comparable word string: parser normalisation (numbers -> digits) + US/UK spelling folded."""
    t = parse(text).normalized
    t = re.sub(r"\bmeters?\b", "metres", t).replace("centimeter", "centimetre")
    return re.sub(r"\.0\b", "", t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, default=ROOT / "corpus" / "audio")
    ap.add_argument("--engines", nargs="+", default=ENGINES)
    args = ap.parse_args()
    items = {it["id"]: it for it in load()}
    if args.audio == ROOT / "corpus" / "audio":
        synthesise(args.audio, items.values())
    clips = sorted((d.name, int(f.stem), f) for d in args.audio.iterdir() if d.is_dir() for f in d.glob("*.wav")
                   if int(f.stem) in items)
    print(f"{len(clips)} clips from {len({c[0] for c in clips})} speakers\n")

    out = ROOT / "results"; out.mkdir(exist_ok=True)
    rows, summary = [], []
    for eng in args.engines:
        model = stt.make(eng)
        stt.timed(model, clips[0][2])                                      # warm-up, not timed
        refs, hyps, ok, false_act, n_reject, lat, wrong_exec = [], [], 0, 0, 0, [], 0
        for speaker, cid, f in clips:
            text, dt = stt.timed(model, f)
            exp = items[cid]["expected"]
            res = parse(text)
            cmds = res.commands
            hit = matches(cmds, exp) and not res.errors
            start = State() if exp[:1] and exp[0]["intent"] == "takeoff" else State(airborne=True, altitude_m=1.0)
            executes = not res.errors and validate(cmds, start)[0]
            wrong_exec += (not hit) and executes and bool(exp)
            ok += hit; lat.append(dt)
            if not exp:
                n_reject += 1; false_act += bool(cmds)
            refs.append(norm(items[cid]["text"])); hyps.append(norm(text) or "<empty>")
            rows.append([model.name, speaker, cid, items[cid]["text"], text, " | ".join(map(str, cmds)), int(hit), f"{dt:.3f}"])
        wer = jiwer.wer(refs, hyps)
        summary.append((model.name, wer, ok / len(clips), false_act, n_reject, sum(lat) / len(lat), wrong_exec))

    stamp = time.strftime("%Y%m%d_%H%M")
    with open(out / f"stt_eval_{stamp}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["engine", "speaker", "id", "reference", "transcript", "parsed", "command_correct", "latency_s"])
        w.writerows(rows)
    lines = ["| Engine | WER | Command accuracy | Wrong command executed | False activations | Mean latency |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {n} | {wer:.1%} | {acc:.1%} | {we}/{len(clips) - nr} | {fa}/{nr} | {lt:.2f} s |"
              for n, wer, acc, fa, nr, lt, we in summary]
    (out / f"stt_eval_{stamp}.md").write_text(f"{len(clips)} clips\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nper-clip detail: results/stt_eval_{stamp}.csv")


if __name__ == "__main__":
    main()
