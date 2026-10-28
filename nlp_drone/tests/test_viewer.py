import json
import urllib.request
from nlp_drone.viewer import Viewer


def test_viewer_serves_page_and_live_stream():
    v = Viewer(connection="udpin:127.0.0.1:14599", port=8799)       # no vehicle: telemetry just waits
    url = v.start(open_browser=False)
    try:
        html = urllib.request.urlopen(url, timeout=5).read().decode()
        assert "Drone 3D Viewer" in html and "EventSource" in html
        v.publish("heard   : 'take off'")
        with urllib.request.urlopen(url + "stream", timeout=5) as r:
            line = r.readline().decode()
        state = json.loads(line.removeprefix("data: "))
        assert state["connected"] is False
        assert state["log"] == ["heard   : 'take off'"]
        assert state["ceiling"] == 2.0
    finally:
        v.stop()
