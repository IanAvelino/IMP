import pytest
from nlp_drone.parser import parse
from nlp_drone.corpus import load, matches


def test_corpus_accuracy_meets_target():
    items = load()
    failures = [(it["id"], it["text"], [str(c) for c in parse(it["text"]).commands])
                for it in items if not matches(parse(it["text"]).commands, it["expected"])]
    acc = 1 - len(failures) / len(items)
    print(f"\nparser accuracy on corpus: {acc:.1%} ({len(items) - len(failures)}/{len(items)})")
    assert acc >= 0.90, f"below 90% target; failures: {failures}"


@pytest.mark.parametrize("text, expected", [
    ("move forward to metres", [("move", {"direction": "forward", "distance_m": 2.0})]),      # STT homophone
    ("go left 50cm", [("move", {"direction": "left", "distance_m": 0.5})]),
    ("climb 1.5m", [("move", {"direction": "up", "distance_m": 1.5})]),
    ("turn right 90°", [("rotate", {"degrees": 90.0})]),
    ("turn anticlockwise", [("rotate", {"degrees": -90.0})]),
    ("back up one metre", [("move", {"direction": "back", "distance_m": 1.0})]),
    ("Take-off to 1 meter.", [("takeoff", {"altitude_m": 1.0})]),
    ("stop and land", [("hold", {}), ("land", {})]),
    ("move two feet forward", [("move", {"direction": "forward", "distance_m": 0.6096})]),
])
def test_phrasings(text, expected):
    got = [(c.intent, c.params) for c in parse(text).commands]
    assert len(got) == len(expected)
    for (gi, gp), (ei, ep) in zip(got, expected):
        assert gi == ei
        assert gp == pytest.approx(ep) if all(isinstance(v, float) for v in ep.values()) else gp == ep


def test_move_without_direction_is_an_error():
    r = parse("move one metre")
    assert r.commands == [] and r.errors


@pytest.mark.parametrize("text", ["go back through metres", "move forward a few metres", "turn left some degrees", "through metres forward"])
def test_unit_without_number_is_unclear_not_defaulted(text):
    r = parse(text)
    assert r.errors and not any(c.intent in ("move", "rotate") for c in r.commands)
