"""Safety check for a parsed command sequence before anything is sent to the vehicle.

All-or-nothing: if any command in an utterance breaks a limit, the whole utterance is rejected,
so a misheard instruction never half-executes.
"""
from dataclasses import dataclass

# Indoor test-space limits (edit to suit the room)
MAX_ALTITUDE_M = 2.0
MIN_ALTITUDE_M = 0.3          # below this, use 'land' instead of 'move down'
MAX_STEP_M = 2.0              # largest single horizontal/vertical move
MAX_ROTATE_DEG = 360.0
MAX_HOLD_S = 30.0


@dataclass
class State:
    airborne: bool = False
    altitude_m: float = 0.0


def validate(commands, state=None):
    """Return (ok, reasons). Simulates the sequence on a copy of `state`."""
    s = State(**vars(state)) if state else State()
    reasons = []
    if not commands:
        return False, ["no command recognised"]
    for n, c in enumerate(commands, 1):
        p, tag = c.params, f"step {n} ({c})"
        if c.intent == "takeoff":
            if s.airborne:
                reasons.append(f"{tag}: already airborne")
            elif not (MIN_ALTITUDE_M <= p["altitude_m"] <= MAX_ALTITUDE_M):
                reasons.append(f"{tag}: altitude must be {MIN_ALTITUDE_M}-{MAX_ALTITUDE_M} m")
            s.airborne, s.altitude_m = True, p["altitude_m"]
        elif not s.airborne:
            reasons.append(f"{tag}: drone is not flying - take off first")
        elif c.intent == "land":
            s.airborne, s.altitude_m = False, 0.0
        elif c.intent == "move":
            d, dist = p["direction"], p["distance_m"]
            if not (0 < dist <= MAX_STEP_M):
                reasons.append(f"{tag}: distance must be >0 and <= {MAX_STEP_M} m")
            if d == "up":
                s.altitude_m += dist
            elif d == "down":
                s.altitude_m -= dist
            if s.altitude_m > MAX_ALTITUDE_M:
                reasons.append(f"{tag}: would climb to {s.altitude_m:.2f} m (max {MAX_ALTITUDE_M} m)")
            if s.altitude_m < MIN_ALTITUDE_M:
                reasons.append(f"{tag}: would descend to {s.altitude_m:.2f} m (min {MIN_ALTITUDE_M} m) - use 'land'")
        elif c.intent == "rotate":
            if abs(p["degrees"]) > MAX_ROTATE_DEG:
                reasons.append(f"{tag}: rotation limited to {MAX_ROTATE_DEG:g} degrees")
        elif c.intent == "hold":
            if p.get("seconds", 0) > MAX_HOLD_S:
                reasons.append(f"{tag}: hold limited to {MAX_HOLD_S:g} s")
    return not reasons, reasons
