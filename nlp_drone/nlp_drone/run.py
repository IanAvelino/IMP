"""Voice or text -> commands -> safety check -> drone (simulated or real).

  python -m nlp_drone.run --sitl --text "take off then move forward one metre then land"
  python -m nlp_drone.run --sitl --mic                  # push-to-talk loop, Whisper by default
  python -m nlp_drone.run --wav clip.wav --dry-run      # transcribe + parse + check only
  python -m nlp_drone.run --connect udp:<bridge-ip>:<port> --mic   # real drone via the ESP32 bridge (check DroneBridge settings)
"""
import argparse
import time

from . import sim, stt
from .parser import parse
from .validator import validate, State


def handle(text, drone, log=print):
    """Parse, check and (unless drone is None) fly one utterance. Returns a list of step results."""
    r = parse(text)
    log(f"  heard   : {text!r}")
    log(f"  parsed  : {[str(c) for c in r.commands] or '-'}")
    if r.errors:
        log(f"  REJECTED: {'; '.join(r.errors)}")
        return None
    ok, why = validate(r.commands, drone.state() if drone else State())
    if not ok:
        log(f"  REJECTED: {'; '.join(why)}")
        return None
    if drone is None:
        log("  (dry run - nothing sent)")
        return []
    results = []
    try:
        for c in r.commands:
            t0 = time.time()
            res = drone.execute(c)
            results.append((str(c), res, round(time.time() - t0, 1)))
            log(f"  done    : {c}  {res}  [{results[-1][2]} s]")
    except KeyboardInterrupt:
        drone.hold()
        log("  INTERRUPTED: holding position; remaining steps cancelled")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text")
    src.add_argument("--wav", help="audio file to transcribe")
    src.add_argument("--mic", action="store_true", help="push-to-talk from the microphone")
    ap.add_argument("--engine", default="whisper:base.en", help="whisper:base.en | whisper:tiny.en | vosk | vosk:free")
    ap.add_argument("--sitl", action="store_true", help="start a local ArduPilot simulator and fly it")
    ap.add_argument("--speedup", type=int, default=1, help="SITL speed multiplier")
    ap.add_argument("--connect", default=None, help="MAVLink connection string for a real/remote vehicle")
    ap.add_argument("--dry-run", action="store_true", help="do not connect to any vehicle")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt on a real vehicle")
    args = ap.parse_args()

    model = stt.make(args.engine) if (args.wav or args.mic) else None
    proc, drone = None, None
    try:
        if not args.dry_run:
            from .vehicle import Drone
            conn = args.connect
            if args.sitl:
                proc, conn, log = sim.start(args.speedup)
                print(f"SITL started (log: {log})")
            drone = Drone(conn or sim.CONNECTION)
            print(f"connected: mode {drone.mode}, armed {drone.armed}")
        real = drone is not None and not args.sitl

        def one(text):
            if real and not args.yes:
                r = parse(text)
                if input(f"  send {[str(c) for c in r.commands]} to the REAL drone? [y/N] ").lower() != "y":
                    print("  cancelled"); return
            handle(text, drone)

        if args.text:
            one(args.text)
        elif args.wav:
            text, dt = stt.timed(model, args.wav)
            print(f"  ({model.name}: {dt:.2f} s)")
            one(text)
        else:
            print("Push-to-talk. Ctrl+C at the prompt to quit.")
            while True:
                audio = stt.record_push_to_talk()
                text, dt = stt.timed(model, audio)
                print(f"  ({model.name}: {dt:.2f} s)")
                one(text)
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        if proc:
            sim.stop(proc)


if __name__ == "__main__":
    main()
