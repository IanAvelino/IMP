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


def test_slow_takeoff_hands_over_to_guided(sitl):
    """The real drone has no flow position on the ground (sensor 3 mm from the floor): take off level in
    GUIDED_NOGPS, switch to GUIDED once the flow gives a position, then hold, move and land as normal."""
    from nlp_drone.vehicle import Drone, HANDOVER_MIN_ALT_M, HANDOVER_MAX_ALT_M
    drone = Drone(sitl, log=lambda *a: None, slow_takeoff=True)
    steps = handle("take off to half a metre, move forward half a metre and land", drone, log=print)
    assert steps and len(steps) == 3
    takeoff = steps[0][1]
    assert takeoff["slow_takeoff"] and HANDOVER_MIN_ALT_M <= takeoff["handover_alt_m"] < HANDOVER_MAX_ALT_M
    assert takeoff["achieved_alt_m"] >= 0.45
    assert steps[1][1]["final_error_m"] < 0.15
    assert not drone.state().airborne


def test_slow_takeoff_lands_if_flow_never_comes(sitl):
    """Safety branch: with no flow at all, the slow take-off must stop by HANDOVER_MAX_ALT_M and land."""
    from nlp_drone.vehicle import Drone, HANDOVER_MAX_ALT_M
    drone = Drone(sitl, log=lambda *a: None, slow_takeoff=True)
    drone.m.mav.param_set_send(drone.m.target_system, drone.m.target_component, b"SIM_FLOW_ENABLE", 0,
                               mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    drone._update(8.0)                                        # let the EKF drop its flow position
    highest = 0.0
    original_update = drone._update
    def tracking_update(duration=0.1):
        nonlocal highest
        original_update(duration)
        highest = max(highest, drone.rel_alt)
    drone._update = tracking_update
    with pytest.raises(RuntimeError, match="no optical-flow position"):
        drone.takeoff(0.5)
    assert not drone.armed
    # the lift-off itself can hop past HANDOVER_MAX_ALT_M (SITL's heavier quad hops to ~0.65 m) - what
    # matters is that it turns round and lands, well below the 1.5 m fence
    assert highest < 1.0, f"climbed to {highest:.2f} m without a position"


def test_altitude_fence_blocks_climb_above_ceiling(sitl):
    """FENCE_ALT_MAX=1.5: with an EKF origin set, GUIDED refuses a target above the ceiling, so the drone
    stays below 1.5 m. (An uncommanded climb past it would switch to LAND - FENCE_ACTION=2.)"""
    from nlp_drone.vehicle import Drone
    drone = Drone(sitl, log=lambda *a: None)
    drone.takeoff(0.5)                                       # the fence must not stop a normal take-off
    drone._goto(drone.x, drone.y, -2.2, drone.yaw)           # bypasses the validator on purpose
    highest, end = drone.rel_alt, time.time() + 15
    while time.time() < end:
        drone._update(0.5)
        highest = max(highest, drone.rel_alt)
    assert highest < 1.5, f"climbed to {highest:.2f} m despite the 1.5 m fence"
    drone.land()


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
