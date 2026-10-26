"""Integration tests against ArduPilot SITL (skipped if SITL isn't built). ~2 min."""
import shutil
import subprocess
import pytest

from nlp_drone import sim, stt
from nlp_drone.run import handle

pytestmark = pytest.mark.skipif(not sim.available(), reason="ArduPilot SITL not built")


@pytest.fixture(scope="module")
def drone():
    from nlp_drone.vehicle import Drone
    proc, conn, _ = sim.start(speedup=1)
    try:
        yield Drone(conn, log=lambda *a: None)
    finally:
        sim.stop(proc)


def test_command_before_takeoff_is_refused_and_nothing_moves(drone):
    assert handle("move forward one metre", drone, log=lambda *a: None) is None
    assert not drone.state().airborne


def test_text_mission_accuracy(drone):
    steps = handle("take off to one metre, move forward one metre, turn right ninety degrees, "
                   "move left one metre, go up half a metre and land", drone, log=print)
    assert steps and len(steps) == 6
    res = {name: r for name, r, _ in steps}
    assert res["takeoff(altitude_m=1)"]["achieved_alt_m"] >= 0.9
    for name, r, _ in steps:
        if name.startswith("move"):
            assert r["final_error_m"] < 0.15, name
            assert r["heading_drift_deg"] < 5, name          # translations must not turn the drone
    assert abs(res["rotate(degrees=90)"]["net_heading_change_deg"] - 90) < 5
    assert not drone.state().airborne


@pytest.mark.skipif(not shutil.which("say"), reason="needs macOS 'say' to synthesise speech")
def test_voice_to_simulated_flight(drone, tmp_path):
    """Gate G3: a spoken command drives the simulated drone end to end."""
    wav = tmp_path / "cmd.wav"
    subprocess.run(["say", "-v", "Daniel", "--data-format=LEI16@16000", "-o", str(wav),
                    "take off, then move forward one metre, then land"], check=True)
    text, latency = stt.timed(stt.make("whisper:base.en"), wav)
    print(f"\nwhisper heard {text!r} in {latency:.2f} s")
    steps = handle(text, drone, log=print)
    assert [s[0] for s in steps] == ["takeoff(altitude_m=1)", "move(direction=forward, distance_m=1)", "land()"]
    assert not drone.state().airborne
