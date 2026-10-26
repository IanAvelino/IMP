from nlp_drone.parser import parse
from nlp_drone.validator import validate, State


def check(text, state=None):
    return validate(parse(text).commands, state)


def test_full_mission_from_ground_is_valid():
    ok, why = check("take off then move forward one metre then land")
    assert ok, why


def test_move_before_takeoff_is_rejected():
    ok, why = check("move forward one metre")
    assert not ok and "take off first" in why[0]


def test_climb_above_ceiling_is_rejected():
    ok, why = check("go up two metres", State(airborne=True, altitude_m=1.0))
    assert not ok and "max" in why[0]


def test_descend_below_floor_suggests_land():
    ok, why = check("go down one metre", State(airborne=True, altitude_m=1.0))
    assert not ok and "use 'land'" in why[0]


def test_step_too_large_is_rejected():
    ok, why = check("move forward five metres", State(airborne=True, altitude_m=1.0))
    assert not ok


def test_second_takeoff_is_rejected():
    ok, _ = check("take off", State(airborne=True, altitude_m=1.0))
    assert not ok


def test_one_bad_step_rejects_whole_utterance():
    ok, why = check("take off then move forward nine metres then land")
    assert not ok and len(why) == 1


def test_nothing_recognised_is_rejected():
    assert check("hello there") == (False, ["no command recognised"])
