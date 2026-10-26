"""Rule-based intent parser: transcript text -> ordered list of drone commands.

Command set: takeoff, land, hold, move (forward/back/left/right/up/down), rotate.
Sentences can chain several commands: "take off, move forward one metre then land".
"""
import re
from dataclasses import dataclass, field

DEFAULT_TAKEOFF_M = 1.0
DEFAULT_MOVE_M = 0.5
DEFAULT_ROTATE_DEG = 90.0


@dataclass
class Command:
    intent: str
    params: dict = field(default_factory=dict)

    def __str__(self):
        p = ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in self.params.items())
        return f"{self.intent}({p})"


@dataclass
class ParseResult:
    commands: list
    errors: list
    normalized: str


# ---- vocabulary ----
ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
        "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
FRACTIONS = {"half": 0.5, "quarter": 0.25}

UNITS = {  # word -> (kind, multiplier to SI unit)
    "m": ("dist", 1.0), "meter": ("dist", 1.0), "meters": ("dist", 1.0), "metre": ("dist", 1.0), "metres": ("dist", 1.0),
    "cm": ("dist", 0.01), "centimeter": ("dist", 0.01), "centimeters": ("dist", 0.01),
    "centimetre": ("dist", 0.01), "centimetres": ("dist", 0.01),
    "ft": ("dist", 0.3048), "foot": ("dist", 0.3048), "feet": ("dist", 0.3048),
    "degree": ("angle", 1.0), "degrees": ("angle", 1.0), "deg": ("angle", 1.0), "°": ("angle", 1.0),
    "second": ("time", 1.0), "seconds": ("time", 1.0), "sec": ("time", 1.0), "secs": ("time", 1.0), "s": ("time", 1.0),
}
# speech-to-text often writes these homophones in place of numbers, but only right before a unit
HOMOPHONES = {"to": 2, "too": 2, "for": 4, "won": 1}

PHRASES = [  # multi-word expressions collapsed to one token before parsing
    (r"\btake ?off\b", "takeoff"), (r"\blift ?off\b", "takeoff"), (r"\btouch ?down\b", "land"),
    (r"\b(turn|spin) around\b", "turnaround"), (r"\bback up\b", "backward"),
    (r"\b(counter|anti) ?clockwise\b", "ccw"), (r"\bclockwise\b", "cw"),
]
TAKEOFF = {"takeoff", "launch", "liftoff"}
LAND = {"land", "landing"}
HOLD = {"hold", "hover", "stop", "stay", "wait", "freeze", "pause", "halt"}
ROTATE = {"turn", "rotate", "yaw", "spin", "turnaround"}
MOVE = {"move", "go", "fly", "head", "travel", "shift", "slide", "come"}
DIRECTION = {"forward": "forward", "forwards": "forward", "ahead": "forward", "front": "forward",
             "back": "back", "backward": "back", "backwards": "back", "reverse": "back",
             "left": "left", "right": "right",
             "up": "up", "upward": "up", "upwards": "up", "higher": "up",
             "climb": "up", "ascend": "up", "rise": "up", "raise": "up",
             "down": "down", "downward": "down", "downwards": "down", "lower": "down",
             "descend": "down", "drop": "down", "sink": "down"}
SLOT_KIND = {"takeoff": "dist", "move": "dist", "rotate": "angle", "hold": "time"}


def _tokens(text):
    t = text.lower().replace("-", " ")
    t = re.sub(r"(\d),(\d)", r"\1\2", t)
    t = re.sub(r"(\d)([a-z°])", r"\1 \2", t)               # "50cm" -> "50 cm", "90°" -> "90 °"
    for pat, rep in PHRASES:
        t = re.sub(pat, rep, t)
    return re.findall(r"\d+(?:\.\d+)?|[a-z°]+", t)


def _numbers(tokens):
    """Collapse number words/digits into floats: 'one and a half' -> 1.5, 'half a' -> 0.5."""
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if re.fullmatch(r"\d+(?:\.\d+)?", t):
            out.append(float(t)); i += 1; continue
        if t in FRACTIONS:                                          # "half a metre"
            out.append(FRACTIONS[t]); i += 2 if nxt in ("a", "an") else 1; continue
        if t in ("a", "an") and nxt in UNITS:                         # "a metre"
            out.append(1.0); i += 1; continue
        if t in HOMOPHONES and nxt in UNITS and UNITS[nxt][0] == "dist":
            out.append(float(HOMOPHONES[t])); i += 1; continue
        if t in ONES or t in TENS:
            val, j = 0.0, i
            while j < len(tokens) and (tokens[j] in ONES or tokens[j] in TENS or tokens[j] == "hundred"):
                w = tokens[j]
                val = (val or 1) * 100 if w == "hundred" else val + ONES.get(w, TENS.get(w, 0))
                j += 1
            if j + 1 < len(tokens) and tokens[j] == "point" and tokens[j + 1] in ONES:     # "one point five"
                j += 1; digits = ""
                while j < len(tokens) and tokens[j] in ONES and ONES[tokens[j]] < 10:
                    digits += str(ONES[tokens[j]]); j += 1
                val += float("0." + digits)
            if tokens[j:j + 3] in (["and", "a", "half"], ["and", "a", "quarter"]):         # "one and a half"
                val += FRACTIONS[tokens[j + 2]]; j += 3
            out.append(val); i = j; continue
        out.append(t); i += 1
    return out


def parse(text):
    toks = _numbers(_tokens(text))
    clauses, errors, cur, pending, pending_unclear = [], [], None, None, None

    def start(intent, **params):
        nonlocal cur, pending, pending_unclear
        cur = {"intent": intent, **params}
        clauses.append(cur)
        if pending is not None:                                       # "two metres forward"
            _attach(cur, pending); pending = None
        if pending_unclear is not None:                               # "through metres forward"
            cur["unclear"] = pending_unclear; pending_unclear = None

    i = 0
    while i < len(toks):
        t = toks[i]
        if isinstance(t, float):
            unit = toks[i + 1] if i + 1 < len(toks) and isinstance(toks[i + 1], str) and toks[i + 1] in UNITS else None
            kind, mult = UNITS[unit] if unit else (None, 1.0)
            num = (t * mult, kind)
            if cur is None or not _attach(cur, num):
                pending = num
            i += 2 if unit else 1
            continue
        if t in TAKEOFF:
            start("takeoff")
        elif t in LAND:
            start("land")
        elif t in HOLD:
            start("hold")
        elif t in ROTATE:
            start("rotate", **({"dir": "right", "deg": 180.0} if t == "turnaround" else {}))
        elif t in ("cw", "ccw"):
            d = "right" if t == "cw" else "left"
            if cur and cur["intent"] == "rotate" and "dir" not in cur:
                cur["dir"] = d
            else:
                start("rotate", dir=d)
        elif t in MOVE:
            start("move")
        elif t in UNITS and UNITS[t][0] in ("dist", "angle"):         # unit with its number lost
            if cur is not None:
                cur["unclear"] = t
            else:
                pending_unclear = t
        elif t in DIRECTION:
            d = DIRECTION[t]
            if cur and cur["intent"] == "rotate" and d in ("left", "right") and "dir" not in cur:
                cur["dir"] = d
            elif cur and cur["intent"] == "move" and "dir" not in cur:
                cur["dir"] = d
            else:
                start("move", dir=d)
        i += 1

    commands = []
    for c in clauses:
        it = c["intent"]
        if "unclear" in c:
            errors.append(f"heard '{c['unclear']}' without a clear number - please repeat")
            continue
        if it == "takeoff":
            commands.append(Command("takeoff", {"altitude_m": c.get("dist", DEFAULT_TAKEOFF_M)}))
        elif it == "land":
            commands.append(Command("land"))
        elif it == "hold":
            commands.append(Command("hold", {"seconds": c["time"]} if "time" in c else {}))
        elif it == "rotate":
            sign = -1.0 if c.get("dir") == "left" else 1.0
            commands.append(Command("rotate", {"degrees": sign * c.get("deg", DEFAULT_ROTATE_DEG)}))
        elif it == "move":
            if "dir" not in c:
                errors.append("'move' without a direction (forward/back/left/right/up/down)")
                continue
            commands.append(Command("move", {"direction": c["dir"], "distance_m": c.get("dist", DEFAULT_MOVE_M)}))
    return ParseResult(commands, errors, " ".join(str(t) for t in toks))


def _attach(clause, num):
    """Give a number to the clause slot it fits; return False if it doesn't fit."""
    value, kind = num
    slot = SLOT_KIND.get(clause["intent"])
    if slot is None:
        return False
    key = {"dist": "dist", "angle": "deg", "time": "time"}[slot]
    if key in clause or (kind is not None and kind != slot):
        return False
    clause[key] = round(value, 4)
    return True
