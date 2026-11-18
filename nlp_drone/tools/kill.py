"""Emergency motor stop over the WiFi link: force-disarms the drone, which then FALLS.

  python tools/kill.py                       # waits for Enter, then stops the motors
  python tools/kill.py tcp:<host>:<port>

Prefer Ctrl+\\ in the flight program itself (nlp_drone.run), which stops the motors over its own link:
a second WiFi connection like this one was dropped repeatedly while the flight program was running.
Use this only when the flight program has died, and only when landing is not safe.
"""
import sys
import time

from pymavlink import mavutil

CONNECTION = sys.argv[1] if len(sys.argv) > 1 else "tcp:192.168.2.1:5760"
ML = mavutil.mavlink
FORCE = 21196                      # ArduPilot magic number: disarm even in flight


def main():
    m = mavutil.mavlink_connection(CONNECTION, source_system=253, source_component=190)
    fc, deadline = None, time.time() + 10
    while fc is None and time.time() < deadline:
        hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if hb is not None and hb.autopilot != ML.MAV_AUTOPILOT_INVALID:
            fc = hb
    if fc is None:
        print("FAIL: no flight-controller heartbeat - cannot reach the drone")
        return 1
    sysid, compid = fc.get_srcSystem(), fc.get_srcComponent()
    input(f"Connected to the drone (system {sysid}). Press ENTER to STOP THE MOTORS ... ")
    for _ in range(5):                 # repeat in case a packet is lost on WiFi
        m.mav.command_long_send(sysid, compid, ML.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, FORCE, 0, 0, 0, 0, 0)
        ack = m.recv_match(type="COMMAND_ACK", blocking=True, timeout=0.3)
        if ack is not None and ack.command == ML.MAV_CMD_COMPONENT_ARM_DISARM and ack.result == ML.MAV_RESULT_ACCEPTED:
            print("Motors stopped (disarm accepted).")
            return 0
    print("WARNING: no confirmation from the drone - pull the battery or the tether if it is still running")
    return 1


if __name__ == "__main__":
    sys.exit(main())
