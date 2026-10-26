"""Start/stop an ArduCopter SITL instance (built from ~/ardupilot, or $ARDUPILOT_DIR)."""
import os
import subprocess
import tempfile
from pathlib import Path

ARDUPILOT = Path(os.environ.get("ARDUPILOT_DIR", Path.home() / "ardupilot"))
BINARY = ARDUPILOT / "build" / "sitl" / "bin" / "arducopter"
DEFAULTS = ARDUPILOT / "Tools" / "autotest" / "default_params" / "copter.parm"
CONNECTION = "tcp:127.0.0.1:5760"


def available():
    return BINARY.exists() and DEFAULTS.exists()


def start(speedup=1, workdir=None):
    """Launch SITL; returns (process, connection string, log path)."""
    if not available():
        raise FileNotFoundError(f"SITL not built: expected {BINARY}")
    workdir = Path(workdir or tempfile.mkdtemp(prefix="sitl_"))
    log = workdir / "sitl.log"
    proc = subprocess.Popen([str(BINARY), "--model", "quad", "--speedup", str(speedup), "-w",
                             "--defaults", str(DEFAULTS), "-I0"],
                            cwd=workdir, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    return proc, CONNECTION, log


def stop(proc):
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
