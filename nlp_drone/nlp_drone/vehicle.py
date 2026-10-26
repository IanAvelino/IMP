"""ArduCopter control over MAVLink (GUIDED mode). Works the same against SITL or the real drone.

Moves are body-relative (forward = where the nose points) and keep the current heading.
"""
import math
import time
from pymavlink import mavutil

from .validator import State

ML = mavutil.mavlink
POS_TOL_M = 0.10
YAW_TOL_DEG = 3.0
CLIMB_FRACTION = 0.95
SETTLE_S = 1.0


class Drone:
    def __init__(self, connection="tcp:127.0.0.1:5760", log=print, connect_timeout=30):
        self.log = log
        deadline = time.time() + connect_timeout
        while True:                                   # SITL may still be starting up
            try:
                self.m = mavutil.mavlink_connection(connection, source_system=255, source_component=190)
                break
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(0.5)
        if self.m.wait_heartbeat(timeout=connect_timeout) is None:
            raise TimeoutError(f"no heartbeat on {connection}")
        self.x = self.y = self.z = self.yaw = self.rel_alt = 0.0
        self.armed, self.mode, self.last_text = False, None, ""
        for msg_id in (ML.MAVLINK_MSG_ID_LOCAL_POSITION_NED, ML.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
                       ML.MAVLINK_MSG_ID_ATTITUDE):
            self._cmd(ML.MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, 100_000)          # 10 Hz
        self._update(1.0)

    # ---------- plumbing ----------
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
                self.yaw = msg.yaw
            elif t == "GLOBAL_POSITION_INT":
                self.rel_alt = msg.relative_alt / 1000.0
            elif t == "HEARTBEAT" and msg.type != ML.MAV_TYPE_GCS:
                self.armed = bool(msg.base_mode & ML.MAV_MODE_FLAG_SAFETY_ARMED)
                self.mode = mavutil.mode_string_v10(msg)
            elif t == "STATUSTEXT":
                self.last_text = msg.text
            if time.time() >= end:
                return

    def _wait(self, cond, timeout, what):
        end = time.time() + timeout
        while not cond():
            if time.time() > end:
                raise TimeoutError(f"{what} (last vehicle message: {self.last_text!r})")
            self._update(0.2)

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

    def arm(self, timeout=120):
        """GUIDED + arm, retrying: both are refused until the EKF has a position estimate."""
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
        self.arm()
        self._cmd(ML.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, altitude_m)
        self._wait(lambda: self.rel_alt >= CLIMB_FRACTION * altitude_m, 30, "takeoff")
        return {"target_alt_m": altitude_m, "achieved_alt_m": round(self.rel_alt, 3)}

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
        if seconds:
            self._update(seconds)
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
