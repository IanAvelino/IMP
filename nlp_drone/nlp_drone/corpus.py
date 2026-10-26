import json
import math
from pathlib import Path

CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "commands.json"


def load():
    return json.loads(CORPUS.read_text())


def matches(commands, expected, tol=0.01):
    """True if a parsed command list equals the expected list (intent + every parameter)."""
    if len(commands) != len(expected):
        return False
    for c, e in zip(commands, expected):
        exp = dict(e)
        if c.intent != exp.pop("intent") or set(c.params) != set(exp):
            return False
        for k, v in exp.items():
            got = c.params[k]
            same = math.isclose(got, v, abs_tol=tol) if isinstance(v, (int, float)) else got == v
            if not same:
                return False
    return True
