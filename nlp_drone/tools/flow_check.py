"""Hand-held optical-flow check (props OFF): is the flow fused, and is the sensor mounted the right way?

  python tools/flow_check.py                      # over DroneBridge WiFi (tcp:192.168.2.1:5760)
  python tools/flow_check.py <connection>

Hold the drone level ~50 cm above a textured, well-lit floor, nose pointing away from you, and
follow the prompts. Read-only: sends no commands apart from asking for message rates.
"""
import math
import sys
import time

from pymavlink import mavutil

CONNECTION = sys.argv[1] if len(sys.argv) > 1 else "tcp:192.168.2.1:5760"
ML = mavutil.mavlink
PHASES = [("HOLD STILL", 8), ("Slowly move FORWARD ~50 cm, then back", 14),
          ("Slowly move RIGHT ~50 cm, then back", 14), ("HOLD STILL", 5)]


def main():
    m = mavutil.mavlink_connection(CONNECTION, source_system=254, source_component=191)
    if m.wait_heartbeat(timeout=10) is None:
        print("FAIL: no heartbeat from the drone")
        return 1
    for msg_id in (ML.MAVLINK_MSG_ID_LOCAL_POSITION_NED, ML.MAVLINK_MSG_ID_DISTANCE_SENSOR,
                   ML.MAVLINK_MSG_ID_OPTICAL_FLOW, ML.MAVLINK_MSG_ID_EKF_STATUS_REPORT, ML.MAVLINK_MSG_ID_ATTITUDE):
        m.mav.command_long_send(m.target_system, m.target_component, ML.MAV_CMD_SET_MESSAGE_INTERVAL,
                                0, msg_id, 50_000, 0, 0, 0, 0, 0)
    # of_fwd/of_right: distance integrated straight from the flow readings (flow rate x height), so the
    # sensor's direction can be judged even when the EKF rejects the data. EKF sign convention: forward
    # speed = +comp_y * range, right speed = -comp_x * range (AP_NavEKF3 writeOptFlowMeas/FuseOptFlow).
    s = {"x": None, "y": None, "yaw": 0.0, "range": None, "q": None, "rel": False, "texts": [],
         "of_fwd": 0.0, "of_right": 0.0, "of_t": None}

    def pump(seconds):
        end = time.time() + seconds
        while time.time() < end:
            msg = m.recv_match(blocking=True, timeout=0.1)
            if msg is None:
                continue
            t = msg.get_type()
            if t == "LOCAL_POSITION_NED":
                s["x"], s["y"] = msg.x, msg.y
            elif t == "ATTITUDE":
                s["yaw"] = msg.yaw
            elif t == "DISTANCE_SENSOR":
                s["range"] = msg.current_distance / 100
            elif t == "OPTICAL_FLOW":
                s["q"] = msg.quality
                now = time.time()
                if s["of_t"] is not None and s["range"]:
                    dt = min(now - s["of_t"], 0.2)
                    s["of_fwd"] += msg.flow_comp_m_y * s["range"] * dt
                    s["of_right"] += -msg.flow_comp_m_x * s["range"] * dt
                s["of_t"] = now
            elif t == "EKF_STATUS_REPORT":
                s["rel"] = bool(msg.flags & (ML.EKF_POS_HORIZ_REL | ML.EKF_PRED_POS_HORIZ_REL))
            elif t == "STATUSTEXT":
                s["texts"].append(msg.text)
                print(f"      drone says: {msg.text}")

    print("Hold the drone level ~50 cm above the floor, nose pointing away from you.")
    for i in (3, 2, 1):
        print(f"   starting in {i} ...")
        pump(1)
    results = []
    for label, seconds in PHASES:
        print(f"\n>>> {label}  ({seconds} s)")
        x0, y0 = s["x"], s["y"]
        of0 = (s["of_fwd"], s["of_right"])
        best = {"fwd": 0.0, "right": 0.0, "of_fwd": 0.0, "of_right": 0.0, "range": [], "q": [], "rel": []}
        for _ in range(seconds):
            pump(1)
            if None not in (x0, s["x"]):
                dn, de = s["x"] - x0, s["y"] - y0                 # NED displacement -> body frame
                fwd = dn * math.cos(s["yaw"]) + de * math.sin(s["yaw"])
                right = -dn * math.sin(s["yaw"]) + de * math.cos(s["yaw"])
                for k, v in (("fwd", fwd), ("right", right)):
                    if abs(v) > abs(best[k]):
                        best[k] = v
            else:
                fwd = right = float("nan")
            for k, v in (("of_fwd", s["of_fwd"] - of0[0]), ("of_right", s["of_right"] - of0[1])):
                if abs(v) > abs(best[k]):
                    best[k] = v
            best["range"].append(s["range"]); best["q"].append(s["q"]); best["rel"].append(s["rel"])
            print(f"   range {s['range'] if s['range'] is not None else '-':>5} m | flow quality {s['q'] if s['q'] is not None else '-':>3}"
                  f" | EKF flow position {'yes' if s['rel'] else 'no '} | EKF fwd {fwd:+.2f} right {right:+.2f}"
                  f" | flow-only fwd {s['of_fwd'] - of0[0]:+.2f} right {s['of_right'] - of0[1]:+.2f} m")
        results.append((label, best))

    print("\n=== summary ===")
    ranges = [r for _, b in results for r in b["range"] if r is not None]
    quals = [q for _, b in results for q in b["q"] if q is not None]
    fused = any("optical flow" in t.lower() for t in s["texts"]) or any(r for _, b in results for r in b["rel"])
    print(f"height seen        : {min(ranges):.2f}-{max(ranges):.2f} m" if ranges else "height seen        : none (rangefinder silent)")
    print(f"flow quality       : {min(quals)}-{max(quals)} (0-255)" if quals else "flow quality       : none (flow silent)")
    print(f"EKF using the flow : {'yes' if fused else 'NO'}")
    fwd_move, right_move = results[1][1], results[2][1]
    print(f"forward phase      : largest forward {fwd_move['fwd']:+.2f} m, sideways {fwd_move['right']:+.2f} m")
    print(f"right phase        : largest right   {right_move['right']:+.2f} m, forward/back {right_move['fwd']:+.2f} m")
    print(f"flow-only, forward : forward {fwd_move['of_fwd']:+.2f} m, sideways {fwd_move['of_right']:+.2f} m")
    print(f"flow-only, right   : right   {right_move['of_right']:+.2f} m, forward/back {right_move['of_fwd']:+.2f} m")
    ok_fwd = fwd_move["fwd"] > 0.15 and abs(fwd_move["fwd"]) > 2 * abs(fwd_move["right"])
    ok_right = right_move["right"] > 0.15 and abs(right_move["right"]) > 2 * abs(right_move["fwd"])
    of_fwd_ok = fwd_move["of_fwd"] > 0.15 and abs(fwd_move["of_fwd"]) > 2 * abs(fwd_move["of_right"])
    of_right_ok = right_move["of_right"] > 0.15 and abs(right_move["of_right"]) > 2 * abs(right_move["of_fwd"])
    print(f"ORIENTATION (EKF)  : {'OK' if ok_fwd and ok_right else 'CHECK - see the numbers above'}")
    print(f"ORIENTATION (flow) : {'OK' if of_fwd_ok and of_right_ok else 'CHECK - see the numbers above'}")
    m.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
