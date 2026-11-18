"""SITL with the real drone's settings (real_drone.parm): optical flow only, no GPS/compass/RC,
link-loss failsafe. Skipped if SITL isn't built. ~1.5 min."""
import time
import pytest
from pymavlink import mavutil

from nlp_drone import sim
from nlp_drone.run import handle

pytestmark = pytest.mark.skipif(not sim.available(), reason="ArduPilot SITL not built")


@pytest.fixture
def sitl():
    proc, conn, _ = sim.start(speedup=1, real_drone=True)
    try:
        yield conn
    finally:
        sim.stop(proc)


def test_mission_on_optical_flow_only(sitl):
    from nlp_drone.vehicle import Drone
    drone = Drone(sitl, log=lambda *a: None)
    steps = handle("take off to one metre, move forward one metre, turn right ninety degrees, "
                   "move left half a metre and land", drone, log=print)
    assert steps and len(steps) == 5
    res = {name: r for name, r, _ in steps}
    assert res["takeoff(altitude_m=1)"]["achieved_alt_m"] >= 0.9
    for name, r, _ in steps:
        if name.startswith("move"):
            assert r["final_error_m"] < 0.15, name
    assert not drone.state().airborne


def test_link_loss_lands_the_drone(sitl):
    """FS_GCS_ENABLE=5: when the program's heartbeats stop, the drone must land and disarm by itself."""
    from nlp_drone.vehicle import Drone
    drone = Drone(sitl, log=lambda *a: None)
    drone.takeoff(1.0)
    watch = mavutil.mavlink_connection("tcp:127.0.0.1:5762", source_system=200)   # observer only
    assert watch.wait_heartbeat(timeout=10)
    drone.m.close()                                          # simulate the WiFi dropping
    t0, modes = time.time(), []
    while time.time() - t0 < 40:
        msg = watch.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if msg is None or msg.type == mavutil.mavlink.MAV_TYPE_GCS:
            continue
        modes.append(mavutil.mode_string_v10(msg))
        if not msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            break
    else:
        pytest.fail(f"still armed 40 s after link loss (modes seen: {sorted(set(modes))})")
    assert "LAND" in modes
    print(f"\nlanded and disarmed {time.time() - t0:.1f} s after the link dropped")
