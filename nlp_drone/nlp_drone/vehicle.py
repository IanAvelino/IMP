"""ArduCopter control over MAVLink (GUIDED mode). Works the same against SITL or the real drone.

Moves are body-relative (forward = where the nose points) and keep the current heading.
"""
import math
import threading
import time
from pymavlink import mavutil

from .validator import State

ML = mavutil.mavlink
POS_TOL_M = 0.10
YAW_TOL_DEG = 3.0
CLIMB_FRACTION = 0.95
SETTLE_S = 1.0
HEARTBEAT_S = 1.0       # ArduPilot's GCS failsafe (FS_GCS_ENABLE) fires when these stop for FS_GCS_TIMEOUT

# Slow take-off (GUIDED_NOGPS): the MTF-01's flow camera can't see the floor from its 3 mm ground
# clearance, so on the ground there is no position estimate and GUIDED refuses to arm. Instead climb
# level and slowly with no position hold, and hand over to GUIDED once the flow gives a position.
SLOW_CLIMB_MS = 0.25        # climb rate before the hand-over
HANDOVER_MIN_ALT_M = 0.12   # don't trust a flow position below this
HANDOVER_MAX_ALT_M = 0.40   # still no position by here -> land
FLIP_LIMIT_DEG = 35.0       # tilt beyond this at any point while armed -> stop the motors
SLOW_TAKEOFF_TIMEOUT_S = 20
EKF_HAS_POS_REL = ML.EKF_POS_HORIZ_REL | ML.EKF_POS_HORIZ_ABS
EKF_HAS_POS_PRED = EKF_HAS_POS_REL | ML.EKF_PRED_POS_HORIZ_REL | ML.EKF_PRED_POS_HORIZ_ABS


class Drone:
    def __init__(self, connection="tcp:127.0.0.1:5760", log=print, connect_timeout=30, slow_takeoff=None):
        """slow_takeoff: None = use it only when the EKF has no position on the ground (the real drone),
        True = always (tests), False = never."""
        self.log = log
        self.slow_takeoff = slow_takeoff
        deadline = time.time() + connect_timeout
        while True:                                   # SITL may still be starting up
            try:
                self.m = mavutil.mavlink_connection(connection, source_system=255, source_component=190)
                break
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(0.5)
        # Wait for the flight controller itself: over WiFi, DroneBridge also sends heartbeats (as a
        # telemetry radio), and locking onto those would send every command to the ESP32 instead.
        hb, deadline = None, time.time() + connect_timeout
        while hb is None and time.time() < deadline:
            msg = self.m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
            if msg is not None and msg.autopilot != ML.MAV_AUTOPILOT_INVALID:
                hb = msg
        if hb is None:
            raise TimeoutError(f"no flight-controller heartbeat on {connection}")
        self.m.target_system, self.m.target_component = hb.get_srcSystem(), hb.get_srcComponent()
        # Heartbeats go out from a background thread so the link-loss failsafe stays armed while the
        # program waits for the next voice command; the lock stops two threads interleaving bytes.
        tx_lock, raw_write = threading.Lock(), self.m.write
        def locked_write(buf):
            with tx_lock:
                raw_write(buf)
        self.m.write = locked_write
        threading.Thread(target=self._heartbeats, daemon=True).start()
        self.x = self.y = self.z = self.yaw = self.roll = self.pitch = self.rel_alt = 0.0
        self.armed, self.mode, self.last_text, self.ekf_flags, self.no_gps = False, None, "", 0, False
        self.range_m, self.range_t = None, 0.0           # downward rangefinder (MTF-01) and when it last reported
        for msg_id in (ML.MAVLINK_MSG_ID_LOCAL_POSITION_NED, ML.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
                       ML.MAVLINK_MSG_ID_ATTITUDE, ML.MAVLINK_MSG_ID_EKF_STATUS_REPORT,
                       ML.MAVLINK_MSG_ID_DISTANCE_SENSOR):
            self._cmd(ML.MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, 100_000)          # 10 Hz
        self._update(1.0)

    # ---------- plumbing ----------
    def _heartbeats(self):
        while True:
            try:
                self.m.mav.heartbeat_send(ML.MAV_TYPE_GCS, ML.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            except OSError:                            # link down: keep trying, the failsafe will act
                pass
            time.sleep(HEARTBEAT_S)

    def _cmd(self, command, *p):
        p = list(p) + [0] * (7 - len(p))
        self.m.mav.command_long_send(self.m.target_system, self.m.target_component, command, 0, *p)

    def _update(self, duration=0.1):
        end = time.time() + duration
        while True:
            msg = self.m.recv_match(blocking=True, timeout=max(0.0, end - time.time()))
            if msg is None:
                return
            t = msg.get_type()
            if t == "LOCAL_POSITION_NED":
                self.x, self.y, self.z = msg.x, msg.y, msg.z
            elif t == "ATTITUDE":
                self.roll, self.pitch, self.yaw = msg.roll, msg.pitch, msg.yaw
            elif t == "EKF_STATUS_REPORT":
                self.ekf_flags = msg.flags
            elif t == "DISTANCE_SENSOR" and msg.orientation == ML.MAV_SENSOR_ROTATION_PITCH_270:
                self.range_m, self.range_t = msg.current_distance / 100.0, time.time()
            elif t == "GLOBAL_POSITION_INT":
                self.rel_alt = msg.relative_alt / 1000.0
            elif t == "HEARTBEAT" and msg.autopilot != ML.MAV_AUTOPILOT_INVALID:   # FC only, not GCS/radio
                self.armed = bool(msg.base_mode & ML.MAV_MODE_FLAG_SAFETY_ARMED)
                self.mode = mavutil.mode_string_v10(msg)
            elif t == "STATUSTEXT":
                self.last_text = msg.text
            if time.time() >= end:
                return

    def _range_fresh(self):
        return self.range_m is not None and time.time() - self.range_t < 0.5

    def _check_upright(self):
        """Flip protection for every step: a hovering/moving quad never tilts this far, so stop the motors
        (the first real take-off flipped and still 'reached' its altitude on the way over)."""
        tilt = math.degrees(max(abs(self.roll), abs(self.pitch)))
        if self.armed and tilt > FLIP_LIMIT_DEG:
            self.kill()
            raise RuntimeError(f"tilted {tilt:.0f} deg - motors stopped")

    def _wait(self, cond, timeout, what):
        end = time.time() + timeout
        while not cond():
            if time.time() > end:
                raise TimeoutError(f"{what} (last vehicle message: {self.last_text!r})")
            self._update(0.2)
            self._check_upright()
        self._check_upright()

    def _goto(self, x, y, z, yaw):
        mask = (ML.POSITION_TARGET_TYPEMASK_VX_IGNORE | ML.POSITION_TARGET_TYPEMASK_VY_IGNORE |
                ML.POSITION_TARGET_TYPEMASK_VZ_IGNORE | ML.POSITION_TARGET_TYPEMASK_AX_IGNORE |
                ML.POSITION_TARGET_TYPEMASK_AY_IGNORE | ML.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
                ML.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE)
        self.m.mav.set_position_target_local_ned_send(0, self.m.target_system, self.m.target_component,
                                                      ML.MAV_FRAME_LOCAL_NED, mask, x, y, z,
                                                      0, 0, 0, 0, 0, 0, yaw, 0)

    def _dist(self, x, y, z):
        return math.dist((self.x, self.y, self.z), (x, y, z))

    # ---------- public ----------
    def state(self):
        self._update(0.3)
        return State(airborne=self.armed and self.rel_alt > 0.15, altitude_m=max(self.rel_alt, 0.0))

    def set_mode(self, name):
        self.m.set_mode(self.m.mode_mapping()[name])
        self._wait(lambda: self.mode == name, 10, f"mode change to {name}")

    def ensure_origin(self):
        """Without GPS the EKF has no origin, and ArduPilot then refuses GUIDED flight. Give it one;
        the location is arbitrary indoors (only local positions are used). A no-op when one exists,
        and never done when a GPS is fitted: an origin set before GPS lock would fight the GPS."""
        self.no_gps = self._param("GPS1_TYPE", None) == 0
        self._cmd(ML.MAV_CMD_REQUEST_MESSAGE, ML.MAVLINK_MSG_ID_GPS_GLOBAL_ORIGIN)
        if self.m.recv_match(type="GPS_GLOBAL_ORIGIN", blocking=True, timeout=2) is not None:
            return
        if not self.no_gps:
            return                                     # GPS fitted (or unknown): the EKF sets its own origin on lock
        for _ in range(5):
            self.m.mav.set_gps_global_origin_send(self.m.target_system, int(1.3521e7), int(103.8198e7), 0)
            self._cmd(ML.MAV_CMD_REQUEST_MESSAGE, ML.MAVLINK_MSG_ID_GPS_GLOBAL_ORIGIN)
            if self.m.recv_match(type="GPS_GLOBAL_ORIGIN", blocking=True, timeout=2) is not None:
                self.log("  set EKF origin (no GPS)")
                return
        raise TimeoutError("the drone did not accept an EKF origin")

    def arm(self, timeout=120):
        """GUIDED + arm, retrying: both are refused until the EKF has a position estimate."""
        self.ensure_origin()
        end = time.time() + timeout
        while not (self.armed and self.mode == "GUIDED"):
            if time.time() > end:
                raise TimeoutError(f"could not enter GUIDED and arm (last vehicle message: {self.last_text!r})")
            if self.mode != "GUIDED":
                self.m.set_mode(self.m.mode_mapping()["GUIDED"])
            else:
                self._cmd(ML.MAV_CMD_COMPONENT_ARM_DISARM, 1)
            self._update(2.0)

    def takeoff(self, altitude_m):
        self.ensure_origin()
        self._update(1.0)
        if self.slow_takeoff or (self.slow_takeoff is None and self.no_gps and not self.ekf_flags & EKF_HAS_POS_PRED):
            return self._slow_takeoff(altitude_m)
        self.arm()
        self._cmd(ML.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, altitude_m)
        self._wait(lambda: self.rel_alt >= CLIMB_FRACTION * altitude_m, 30, "takeoff")
        return {"target_alt_m": altitude_m, "achieved_alt_m": round(self.rel_alt, 3)}

    def _param(self, name, default):
        for _ in range(3):
            self.m.mav.param_request_read_send(self.m.target_system, self.m.target_component, name.encode(), -1)
            p = self.m.recv_match(type="PARAM_VALUE", blocking=True, timeout=1.5)
            if p is not None and p.param_id == name:
                return p.param_value
        return default

    def _abort_slow_takeoff(self, why, level=None):
        if level is not None:                          # stop the climb at once, before the mode change lands
            for _ in range(5):
                self.m.mav.set_attitude_target_send(0, self.m.target_system, self.m.target_component, 7,
                                                    level, 0, 0, 0, 0.3)
                self._update(0.1)
        self.m.set_mode(self.m.mode_mapping()["LAND"])
        self._wait(lambda: not self.armed, 30, "landing after an aborted slow take-off")
        raise RuntimeError(f"slow take-off aborted and landed: {why}")

    def _slow_takeoff(self, altitude_m):
        """Arm in GUIDED_NOGPS, climb level at SLOW_CLIMB_MS, hand over to GUIDED once the EKF has a
        flow position above HANDOVER_MIN_ALT_M, then finish the climb with position hold."""
        self.log("  slow take-off: no position on the ground, climbing level until the flow sees the floor")
        self.set_mode("GUIDED_NOGPS")
        end = time.time() + 30
        while not self.armed:
            if time.time() > end:
                raise TimeoutError(f"could not arm in GUIDED_NOGPS (last vehicle message: {self.last_text!r})")
            self._cmd(ML.MAV_CMD_COMPONENT_ARM_DISARM, 1)
            self._update(2.0)
        # in GUIDED_NOGPS the thrust field is a climb-rate request: 0.5 = hold, 1.0 = +WP_SPD_UP m/s
        thrust = 0.5 + SLOW_CLIMB_MS / (2 * self._param("WP_SPD_UP", 2.5))
        yaw = self.yaw
        level = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]        # roll 0, pitch 0, current heading
        ignore_rates = (ML.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE | ML.ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE |
                        ML.ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE)
        # Height for the hand-over/abort decisions: the rangefinder measures the real gap to the floor; the
        # barometer is fooled by prop wash near the ground (it read +0.74 m on the floor in a real attempt)
        ground_range = self.range_m if self._range_fresh() else None
        def height():
            if ground_range is not None and self._range_fresh():
                return self.range_m - ground_range
            return self.rel_alt
        end = time.time() + SLOW_TAKEOFF_TIMEOUT_S
        while True:
            self.m.mav.set_attitude_target_send(0, self.m.target_system, self.m.target_component, ignore_rates,
                                                level, 0, 0, 0, thrust)
            self._update(0.1)
            self._check_upright()
            if not self.armed:
                raise RuntimeError(f"disarmed during the slow take-off (last vehicle message: {self.last_text!r})")
            if height() >= HANDOVER_MIN_ALT_M and self.ekf_flags & EKF_HAS_POS_REL:
                break
            if height() >= HANDOVER_MAX_ALT_M:
                self._abort_slow_takeoff(f"no optical-flow position by {HANDOVER_MAX_ALT_M} m", level)
            if time.time() > end:
                self._abort_slow_takeoff(f"still below {HANDOVER_MIN_ALT_M} m or without a position after {SLOW_TAKEOFF_TIMEOUT_S} s", level)
        handover_alt = height()
        self.log(f"  flow position at {handover_alt:.2f} m - switching to GUIDED")
        self.m.set_mode(self.m.mode_mapping()["GUIDED"])
        self._wait(lambda: self.mode == "GUIDED", 5, "switch to GUIDED after the slow take-off")
        target_z = self.z - (altitude_m - self.rel_alt)                   # NED: further up is more negative
        self._goto(self.x, self.y, target_z, self.yaw)
        self._wait(lambda: self.rel_alt >= CLIMB_FRACTION * altitude_m, 30, "climb after the slow take-off")
        return {"target_alt_m": altitude_m, "achieved_alt_m": round(self.rel_alt, 3),
                "slow_takeoff": True, "handover_alt_m": round(handover_alt, 3)}

    def move(self, direction, distance_m):
        fwd, right, down = {"forward": (1, 0, 0), "back": (-1, 0, 0), "right": (0, 1, 0),
                            "left": (0, -1, 0), "up": (0, 0, -1), "down": (0, 0, 1)}[direction]
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        tx = self.x + distance_m * (fwd * c - right * s)
        ty = self.y + distance_m * (fwd * s + right * c)
        tz = self.z + distance_m * down
        start, yaw = (self.x, self.y, self.z), self.yaw
        self._goto(tx, ty, tz, yaw)
        self._wait(lambda: self._dist(tx, ty, tz) < POS_TOL_M, 15 + 8 * distance_m, f"move {direction}")
        self._update(SETTLE_S)                         # let it settle before measuring accuracy
        return {"target_m": distance_m, "achieved_m": round(math.dist(start, (self.x, self.y, self.z)), 3),
                "final_error_m": round(self._dist(tx, ty, tz), 3),
                "heading_drift_deg": round(math.degrees(abs(math.atan2(math.sin(self.yaw - yaw), math.cos(self.yaw - yaw)))), 1)}

    def rotate(self, degrees):
        start = math.degrees(self.yaw)
        remaining = degrees
        while abs(remaining) > 0.5:                    # chunks of <=180 deg so 270/360 turn the asked way
            step = max(-180.0, min(180.0, remaining))
            target = math.radians(math.degrees(self.yaw) + step)
            self._cmd(ML.MAV_CMD_CONDITION_YAW, abs(step), 45, 1 if step > 0 else -1, 1)
            err = lambda: abs(math.degrees(math.atan2(math.sin(self.yaw - target), math.cos(self.yaw - target))))
            self._wait(lambda: err() < YAW_TOL_DEG, 10 + abs(step) / 20, "rotate")
            remaining -= step
        turned = (math.degrees(self.yaw) - start + 540) % 360 - 180
        return {"target_deg": degrees, "net_heading_change_deg": round(turned, 1)}

    def hold(self, seconds=None):
        self._goto(self.x, self.y, self.z, self.yaw)
        end = time.time() + (seconds or 0)
        while time.time() < end:                       # in short slices so flip protection keeps running
            self._update(min(0.2, max(0.0, end - time.time())))
            self._check_upright()
        return {"held_s": seconds or 0}

    def land(self):
        self.set_mode("LAND")
        self._wait(lambda: not self.armed, 60, "landing and disarm")
        return {"landed": True}

    def kill(self):
        """Emergency stop: force-disarm immediately (motors stop - the drone will fall)."""
        self._cmd(ML.MAV_CMD_COMPONENT_ARM_DISARM, 0, 21196)

    def execute(self, command):
        p = command.params
        if command.intent == "takeoff":
            return self.takeoff(p["altitude_m"])
        if command.intent == "move":
            return self.move(p["direction"], p["distance_m"])
        if command.intent == "rotate":
            return self.rotate(p["degrees"])
        if command.intent == "hold":
            return self.hold(p.get("seconds"))
        if command.intent == "land":
            return self.land()
        raise ValueError(command.intent)
