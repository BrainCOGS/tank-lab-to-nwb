"""Every real log: the registry covers it, nothing is lost, and the NWB file holds exactly the normalized values.

Set VIRMEN_CORPUS to the folder of logs to run these; the logs aren't in this repository.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from pynwb import validate
from scipy.io import loadmat

from tank_lab_to_nwb.virmen.convert import convert_log
from tank_lab_to_nwb.virmen.loader import load_log, records
from tank_lab_to_nwb.virmen.session import normalize_log

CORPUS = os.environ.get("VIRMEN_CORPUS")
LOGS = sorted(Path(CORPUS).glob("*/*/*_g0/*.mat")) if CORPUS else []
pytestmark = [pytest.mark.corpus, pytest.mark.skipif(not LOGS, reason="VIRMEN_CORPUS is not set")]
# Index ids keep subject names out of test output.
each_log = pytest.mark.parametrize("path", LOGS, ids=[f"log{i:02d}" for i in range(len(LOGS))])


def test_all_37_logs_are_found():
    assert len(LOGS) == 37


@each_log
def test_every_field_is_registered(path):
    session = normalize_log(load_log(path), file=path)
    assert session.coverage.unregistered == []


@each_log
def test_round_trip_is_exact_and_valid(path, tmp_path):
    # convert_log raises RoundTripError unless every value on disk equals the normalized source, bit for bit.
    output = tmp_path / "out.nwb"
    convert_log(path, output)
    errors = validate(path=str(output))
    errors = errors[0] if isinstance(errors, tuple) else errors
    assert list(errors) == []


def _pwm_trials(log):
    return [trial for block in records(log["block"]) for trial in records(block["trial"])]


@each_log
def test_dropped_pwm_histories_are_exactly_derivable(path):
    log = load_log(path)
    trials = _pwm_trials(log)
    if not trials or "hitHistory" not in trials[0]:
        pytest.skip("not a PWM session")
    pairs, correct = [], []
    for number, trial in enumerate(trials, start=1):
        pairs.append(float(trial["pairNum"].item()))
        correct.append(float(trial["choice"].name == trial["trialType"].name))
        assert trial["classHistory"].ravel().tolist() == pairs
        assert trial["hitHistory"].ravel().astype(float).tolist() == correct
        assert trial["trialNum"].item() == number


def _scipy_leaves(value, path, out):
    """Numeric and text leaves of plain structs and cells, as scipy reads them. MATLAB objects are opaque to scipy."""
    if isinstance(value, np.ndarray) and value.dtype.names:
        for element_index, element in enumerate(value.flat):
            for name in value.dtype.names:
                _scipy_leaves(element[name], f"{path}[{element_index}].{name}", out)
    elif isinstance(value, np.ndarray) and value.dtype == object:
        for element_index, element in enumerate(value.flat):
            _scipy_leaves(element, f"{path}{{{element_index}}}", out)
    elif isinstance(value, np.ndarray) and value.dtype.kind in "fiubU":
        out[path] = value


def _ours(value, path, out):
    if isinstance(value, dict):
        for name, member in value.items():
            _ours(member, f"{path}[0].{name}", out)
    elif isinstance(value, list) and value and isinstance(value[0], dict):
        for element_index, element in enumerate(value):
            for name, member in element.items():
                _ours(member, f"{path}[{element_index}].{name}", out)
    elif isinstance(value, np.ndarray) and value.dtype == object:
        for element_index, element in enumerate(value.flat):
            _ours(element, f"{path}{{{element_index}}}", out)
    else:
        out[path] = value


@each_log
def test_loader_agrees_with_scipy_on_every_plain_value(path):
    # scipy can't decode MATLAB objects, so enums, handles and tables have no counterpart. Their values are checked
    # against MATLAB separately. Every value scipy can read must match ours.
    with_scipy = {}
    _scipy_leaves(loadmat(path, struct_as_record=True, squeeze_me=False)["log"], "log", with_scipy)
    with_ours = {}
    _ours(load_log(path), "log[0]".removesuffix("[0]"), with_ours)
    with_ours = {key.replace("log[0].", "log[0].", 1): value for key, value in with_ours.items()}
    compared = 0
    for key, ours in with_ours.items():
        theirs = with_scipy.get(key)
        if theirs is None or isinstance(ours, str) and theirs.dtype.kind != "U":
            continue
        compared += 1
        if isinstance(ours, str):
            assert "".join(map(str, theirs.ravel())) == ours, key
        elif isinstance(ours, np.ndarray) and ours.dtype.kind in "fiub":
            # scipy stores logicals and small doubles in smaller integer types; the values are what must agree.
            # mat-io gives every canonical empty the shape 0x0, so an empty's orientation (1x0) doesn't compare.
            if theirs.size or ours.size:
                np.testing.assert_array_equal(np.asarray(theirs, dtype="float64"), np.asarray(ours, dtype="float64"), err_msg=key)
    assert compared > 100
