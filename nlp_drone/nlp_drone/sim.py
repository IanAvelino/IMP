"""Start/stop an ArduCopter SITL instance (built from ~/ardupilot, or $ARDUPILOT_DIR)."""
import os
import subprocess
import tempfile
from pathlib import Path

ARDUPILOT = Path(os.environ.get("ARDUPILOT_DIR", Path.home() / "ardupilot"))
BINARY = ARDUPILOT / "build" / "sitl" / "bin" / "arducopter"
DEFAULTS = ARDUPILOT / "Tools" / "autotest" / "default_params" / "copter.parm"
REAL_DRONE = Path(__file__).with_name("real_drone.parm")   # no RC/compass/GPS, flow navigation, failsafes
CONNECTION = "tcp:127.0.0.1:5760"


def available():
    return BINARY.exists() and DEFAULTS.exists()


def start(speedup=1, workdir=None, real_drone=False):
    """Launch SITL; returns (process, connection string, log path).

    real_drone=True loads real_drone.parm on top of the defaults, so the simulated copter navigates
    like the real one (optical flow, no GPS or compass) and has the same failsafes.
    """
    if not available():
        raise FileNotFoundError(f"SITL not built: expected {BINARY}")
    workdir = Path(workdir or tempfile.mkdtemp(prefix="sitl_"))
    log = workdir / "sitl.log"
    defaults = f"{DEFAULTS},{REAL_DRONE}" if real_drone else str(DEFAULTS)
    proc = subprocess.Popen([str(BINARY), "--model", "quad", "--speedup", str(speedup), "-w",
                             "--defaults", defaults, "-I0"],
                            cwd=workdir, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    return proc, CONNECTION, log


def stop(proc):
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
