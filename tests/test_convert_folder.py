"""Finding a folder's sessions: each subject and day is one session unless the user chooses otherwise."""

import numpy as np
import pytest

from tank_lab_to_nwb.virmen import convert_folder as folder
from tank_lab_to_nwb.virmen.convert_folder import SessionSelectionError, parse_choice, select_logs
from tests.synthetic_log import make_block, make_log, make_trial


def _log(root, gate, subject="user_mouse", day="20260101", name="task"):
    where = root / "user" / subject / f"{day}_{gate}"
    where.mkdir(parents=True, exist_ok=True)
    path = where / f"{name}_{gate}.mat"
    path.write_bytes(b"")
    return path


def test_one_gate_a_day_is_the_session(tmp_path):
    log = _log(tmp_path, "g0")
    assert [item.path for item in select_logs(tmp_path)] == [log]


def test_a_lone_g1_is_the_session(tmp_path):
    # Someone can start a session, delete the first attempt and keep the second.
    log = _log(tmp_path, "g1")
    selected = select_logs(tmp_path)
    assert [item.path for item in selected] == [log]
    assert selected[0].session_suffix == ""


def test_several_gates_on_a_day_stop_with_what_to_choose(tmp_path):
    _log(tmp_path, "g0")
    _log(tmp_path, "g1")
    with pytest.raises(SessionSelectionError) as error:
        select_logs(tmp_path)
    message = str(error.value)
    assert "user_mouse/20260101" in message and "g0" in message and "g1" in message and "--choose" in message


def test_choosing_one_gate_converts_only_it_with_the_plain_session_id(tmp_path):
    _log(tmp_path, "g0")
    second = _log(tmp_path, "g1")
    selected = select_logs(tmp_path, choices={("user_mouse", "20260101"): ["g1"]})
    assert [item.path for item in selected] == [second]
    assert selected[0].session_suffix == ""


def test_choosing_several_gates_keeps_their_session_ids_apart(tmp_path):
    first = _log(tmp_path, "g0")
    second = _log(tmp_path, "g1")
    selected = select_logs(tmp_path, choices={("user_mouse", "20260101"): ["g0", "g1"]})
    assert [(item.path, item.session_suffix) for item in selected] == [(first, "-g0"), (second, "-g1")]


def test_same_day_for_different_subjects_is_no_conflict(tmp_path):
    _log(tmp_path, "g0", subject="user_a")
    _log(tmp_path, "g0", subject="user_b")
    assert len(select_logs(tmp_path)) == 2


def test_a_gate_that_does_not_exist_cannot_be_chosen(tmp_path):
    _log(tmp_path, "g0")
    with pytest.raises(SessionSelectionError, match="g7"):
        select_logs(tmp_path, choices={("user_mouse", "20260101"): ["g7"]})


def test_a_choice_for_a_session_that_does_not_exist_is_an_error(tmp_path):
    _log(tmp_path, "g0")
    with pytest.raises(SessionSelectionError, match="user_other/20260101"):
        select_logs(tmp_path, choices={("user_other", "20260101"): ["g0"]})


def test_a_choice_can_be_made_when_nothing_conflicts(tmp_path):
    log = _log(tmp_path, "g1")
    assert [item.path for item in select_logs(tmp_path, choices={("user_mouse", "20260101"): ["g1"]})] == [log]


def test_two_logs_in_one_gate_folder_stop_the_conversion(tmp_path):
    _log(tmp_path, "g0")
    _log(tmp_path, "g0", name="other")
    with pytest.raises(SessionSelectionError, match="more than one log"):
        select_logs(tmp_path)


def test_only_names_the_exact_file_so_it_counts_as_the_choice(tmp_path):
    _log(tmp_path, "g0")
    second = _log(tmp_path, "g1")
    assert [item.path for item in select_logs(tmp_path, only=[second.name])] == [second]


def test_no_logs_is_an_error(tmp_path):
    with pytest.raises(SessionSelectionError, match="no logs"):
        select_logs(tmp_path)


def test_folders_not_named_for_a_day_and_gate_are_not_logs(tmp_path):
    where = tmp_path / "user" / "user_mouse" / "notes"
    where.mkdir(parents=True)
    (where / "x.mat").write_bytes(b"")
    with pytest.raises(SessionSelectionError, match="no logs"):
        select_logs(tmp_path)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("user_mouse/20260101=g1", (("user_mouse", "20260101"), ["g1"])),
        ("user_mouse/20260101=g0,g1", (("user_mouse", "20260101"), ["g0", "g1"])),
    ],
)
def test_parse_choice(text, expected):
    assert parse_choice(text) == expected


@pytest.mark.parametrize("text", ["user_mouse=g1", "user_mouse/20260101", "user_mouse/20260101=", "user_mouse/20260101=first", "/20260101=g0", "user_mouse/2026=g0"])
def test_a_badly_written_choice_is_an_error(text):
    with pytest.raises(SessionSelectionError, match="--choose"):
        parse_choice(text)


class TestValidation:
    """What the log itself says must agree with the folder it's in."""

    def _session(self, **changes):
        session = type("Session", (), dict(subject_id="user_mouse", session_id="20260101"))()
        for key, value in changes.items():
            setattr(session, key, value)
        return session

    def _selected(self, tmp_path):
        return select_logs(tmp_path)[0]

    def test_a_matching_log_passes(self, tmp_path):
        _log(tmp_path, "g0")
        folder.check_session(self._selected(tmp_path), self._session())

    def test_a_log_from_another_day_is_refused(self, tmp_path):
        _log(tmp_path, "g0")
        with pytest.raises(SessionSelectionError, match="20260102"):
            folder.check_session(self._selected(tmp_path), self._session(session_id="20260102"))

    def test_a_log_from_another_subject_is_refused(self, tmp_path):
        _log(tmp_path, "g0")
        with pytest.raises(SessionSelectionError, match="other_mouse"):
            folder.check_session(self._selected(tmp_path), self._session(subject_id="other_mouse"))


def _fake_load(_path):
    log = make_log([make_block([make_trial()], start=(2026, 1, 1, 12, 0, 1.0))])
    log["session"]["start"] = np.array([[2026, 1, 1, 11, 59, 50.0]])
    log["session"]["end"] = np.array([[2026, 1, 1, 12, 0, 11.0]])
    log["animal"]["name"] = "user_mouse"
    return log


def test_converting_gives_each_chosen_gate_its_own_session_id(tmp_path, monkeypatch):
    _log(tmp_path / "in", "g0")
    _log(tmp_path / "in", "g1")
    monkeypatch.setattr(folder, "load_log", _fake_load)
    paths = folder.convert_folder(tmp_path / "in", tmp_path / "out", choices={("user_mouse", "20260101"): ["g0", "g1"]})
    assert sorted(path.name for path in paths) == ["sub-user-mouse_ses-20260101-g0_behavior.nwb", "sub-user-mouse_ses-20260101-g1_behavior.nwb"]


def test_changing_the_choice_replaces_the_other_gates_files(tmp_path, monkeypatch):
    _log(tmp_path / "in", "g0")
    _log(tmp_path / "in", "g1")
    monkeypatch.setattr(folder, "load_log", _fake_load)
    choose = {("user_mouse", "20260101"): ["g0"]}
    folder.convert_folder(tmp_path / "in", tmp_path / "out", choices=choose)
    choose = {("user_mouse", "20260101"): ["g1"]}
    paths = folder.convert_folder(tmp_path / "in", tmp_path / "out", choices=choose)
    assert [path.name for path in (tmp_path / "out" / "staging").glob("*.nwb")] == ["task_g1.nwb"]
    assert [path.name for path in (tmp_path / "out" / "dandiset").rglob("*.nwb")] == [path.name for path in paths]


def test_keeping_both_then_choosing_one_leaves_one_file(tmp_path, monkeypatch):
    _log(tmp_path / "in", "g0")
    _log(tmp_path / "in", "g1")
    monkeypatch.setattr(folder, "load_log", _fake_load)
    folder.convert_folder(tmp_path / "in", tmp_path / "out", choices={("user_mouse", "20260101"): ["g0", "g1"]})
    paths = folder.convert_folder(tmp_path / "in", tmp_path / "out", choices={("user_mouse", "20260101"): ["g1"]})
    assert [path.name for path in paths] == ["sub-user-mouse_ses-20260101_behavior.nwb"]
    assert [path.name for path in (tmp_path / "out" / "dandiset").rglob("*.nwb")] == ["sub-user-mouse_ses-20260101_behavior.nwb"]


def test_a_log_whose_date_disagrees_with_its_folder_stops_the_conversion(tmp_path, monkeypatch):
    _log(tmp_path / "in", "g0", day="20260105")
    monkeypatch.setattr(folder, "load_log", _fake_load)
    with pytest.raises(SessionSelectionError, match="20260101"):
        folder.convert_folder(tmp_path / "in", tmp_path / "out")
