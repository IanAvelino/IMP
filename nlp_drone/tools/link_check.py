"""Check the WiFi link to the drone through the ESP32 DroneBridge.

  python tools/link_check.py                      # DroneBridge access point (192.168.2.1, TCP 5760)
  python tools/link_check.py tcp:<esp32-ip>:5760  # DroneBridge joined to another WiFi network

Read-only: listens for a heartbeat and a few status messages, sends no commands.
"""
import sys
import time

from pymavlink import mavutil

CONNECTION = sys.argv[1] if len(sys.argv) > 1 else "tcp:192.168.2.1:5760"


def main():
    print(f"connecting to {CONNECTION} ...")
    try:
        m = mavutil.mavlink_connection(CONNECTION, source_system=255, source_component=190)
    except OSError as e:
        print(f"FAIL: cannot reach the ESP32 ({e}). Are you on the DroneBridge WiFi?")
        return 1
    # DroneBridge sends heartbeats of its own; only an autopilot's heartbeat proves the FC is heard
    hb, others, t0 = None, set(), time.time()
    while hb is None and time.time() - t0 < 10:
        msg = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if msg is None:
            continue
        if msg.autopilot != mavutil.mavlink.MAV_AUTOPILOT_INVALID and msg.get_srcSystem() != 0:
            hb = msg
            m.target_system, m.target_component = msg.get_srcSystem(), msg.get_srcComponent()
        else:
            others.add((msg.get_srcSystem(), msg.get_srcComponent(), msg.type))
    for sysid, compid, mav_type in sorted(others):
        print(f"   (heartbeat from a non-autopilot: system {sysid}, component {compid}, type {mav_type})")
    if hb is None:
        print("FAIL: reached the ESP32 but no heartbeat from the flight controller.")
        print("      Check DroneBridge TX/RX GPIOs (25/26, swap if needed), baud 115200, protocol MAVLink,")
        print("      and SERIALn_PROTOCOL = 2, SERIALn_BAUD = 115 on the FC port it is wired to (SERIAL1 = T1/R1).")
        return 1
    print(f"OK: heartbeat from system {m.target_system}, mode {mavutil.mode_string_v10(hb)}  (FC -> laptop works)")
    # a reply to a request proves the laptop -> FC direction as well
    # (ask by index: parameter names change between ArduPilot versions, index 0 always exists)
    reply = None
    for _ in range(5):
        m.mav.param_request_read_send(m.target_system, m.target_component, b"", 0)
        reply = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=1.5)
        if reply is not None:
            break
    if reply is None:
        print("FAIL: the FC never answered a request, so commands are not reaching it (ESP32 TX -> FC RX).")
        return 1
    print("OK: the FC answered a request  (laptop -> FC works)")
    m.mav.request_data_stream_send(m.target_system, m.target_component,
                                   mavutil.mavlink.MAV_DATA_STREAM_ALL, 2, 1)
    seen, t0 = set(), time.time()
    while time.time() - t0 < 8:
        msg = m.recv_match(type=["SYS_STATUS", "DISTANCE_SENSOR", "OPTICAL_FLOW"], blocking=True, timeout=1)
        if msg is None or msg.get_type() in seen:
            continue
        seen.add(msg.get_type())
        if msg.get_type() == "SYS_STATUS":
            print(f"   battery      {msg.voltage_battery / 1000:.2f} V")
        elif msg.get_type() == "DISTANCE_SENSOR":
            print(f"   rangefinder  {msg.current_distance} cm")
        else:
            print(f"   optical flow quality {msg.quality}")
    for name in ("SYS_STATUS", "DISTANCE_SENSOR", "OPTICAL_FLOW"):
        if name not in seen:
            print(f"   (no {name} received)")
    m.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
