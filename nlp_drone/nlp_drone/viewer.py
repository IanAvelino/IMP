"""Live 3D viewer for a MAVLink vehicle (SITL or real), served to the browser.

  python -m nlp_drone.run --sitl --mic --viewer      # usual way: opens alongside a voice session
  python -m nlp_drone.viewer                         # standalone, watching SITL's second port (5762)

Uses its own MAVLink connection, so it never interferes with the link that flies the drone.
"""
import argparse
import collections
import json
import math
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pymavlink import mavutil

from .validator import MAX_ALTITUDE_M

HTML = Path(__file__).with_name("viewer.html")
ML = mavutil.mavlink
STREAM_HZ = 20


class Viewer:
    def __init__(self, connection="tcp:127.0.0.1:5762", port=8765):
        self.connection, self.port = connection, port
        self.lock = threading.Lock()
        self.log = collections.deque(maxlen=10)
        self.state = {"connected": False, "have_pos": False, "x": 0.0, "y": 0.0, "z": 0.0,
                      "vx": 0.0, "vy": 0.0, "vz": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                      "alt": 0.0, "armed": False, "mode": "-", "status": "", "ceiling": MAX_ALTITUDE_M}
        self.server = None

    # ---- data in ----
    def publish(self, line):
        """Show a line (heard / parsed / done / REJECTED ...) in the viewer's log panel."""
        with self.lock:
            self.log.append(line.strip())

    def _set(self, **kv):
        with self.lock:
            self.state.update(kv)

    def _telemetry(self):
        while True:
            try:
                m = mavutil.mavlink_connection(self.connection, source_system=255, source_component=191)
                m.wait_heartbeat()
                for msg_id in (ML.MAVLINK_MSG_ID_LOCAL_POSITION_NED, ML.MAVLINK_MSG_ID_ATTITUDE,
                               ML.MAVLINK_MSG_ID_GLOBAL_POSITION_INT):
                    m.mav.command_long_send(m.target_system, m.target_component,
                                            ML.MAV_CMD_SET_MESSAGE_INTERVAL, 0, msg_id, 50_000, 0, 0, 0, 0, 0)
                self._set(connected=True)
                while True:
                    msg = m.recv_match(blocking=True, timeout=3)
                    if msg is None:
                        self._set(connected=False)
                        continue
                    t = msg.get_type()
                    if t == "LOCAL_POSITION_NED":
                        self._set(have_pos=True, x=msg.x, y=msg.y, z=msg.z, vx=msg.vx, vy=msg.vy, vz=msg.vz, connected=True)
                    elif t == "ATTITUDE":
                        self._set(roll=msg.roll, pitch=msg.pitch, yaw=msg.yaw, connected=True)
                    elif t == "GLOBAL_POSITION_INT":
                        self._set(alt=msg.relative_alt / 1000.0)
                    elif t == "HEARTBEAT" and msg.type != ML.MAV_TYPE_GCS:
                        self._set(armed=bool(msg.base_mode & ML.MAV_MODE_FLAG_SAFETY_ARMED),
                                  mode=mavutil.mode_string_v10(msg), connected=True)
                    elif t == "STATUSTEXT":
                        self._set(status=msg.text)
            except Exception as e:                     # link dropped: show it and retry
                self._set(connected=False, status=f"telemetry: {e}")
                time.sleep(1)

    # ---- data out ----
    def snapshot(self):
        with self.lock:
            s = dict(self.state)
            s["log"] = list(self.log)
        s["speed"] = round(math.hypot(s["vx"], s["vy"]), 2)
        return s

    def _handler(self):
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    body = HTML.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    try:
                        while True:
                            self.wfile.write(f"data: {json.dumps(viewer.snapshot())}\n\n".encode())
                            self.wfile.flush()
                            time.sleep(1 / STREAM_HZ)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                else:
                    self.send_error(404)

            def log_message(self, *args):
                pass

        return Handler

    def start(self, open_browser=True):
        threading.Thread(target=self._telemetry, daemon=True).start()
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), self._handler())
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{self.port}/"
        if open_browser:
            webbrowser.open(url)
        return url

    def stop(self):
        if self.server:
            self.server.shutdown()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--connect", default="tcp:127.0.0.1:5762", help="MAVLink connection to watch")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    url = Viewer(args.connect, args.port).start(open_browser=not args.no_browser)
    print(f"3D viewer: {url}  (watching {args.connect}) - Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
