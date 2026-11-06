"""Tying ScanImage time to ViRMEn's elapsed clock from the I2C packets."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from tank_lab_to_nwb.imaging.alignment import (
    AlignmentError,
    fit_clock,
    frame_labels,
    in_progress_series,
    iteration_clock,
    match_packets,
    session_start_shift,
    shifted_session,
)
from tank_lab_to_nwb.virmen.session import normalize_log
from tests.synthetic_log import make_block, make_log, make_trial, scalar


def _packets(block, trial, iteration, timestamp=None):
    block, trial, iteration = (np.asarray(v, dtype="int64") for v in (block, trial, iteration))
    return dict(block=block, trial=trial, iteration=iteration, timestamp=np.zeros(block.size) if timestamp is None else np.asarray(timestamp, dtype="float64"), frame_index=np.zeros(block.size, dtype="int64"))


def _line(n=2000, slope=1 - 5e-6, intercept=11.0):
    elapsed = np.arange(n) * 0.012
    return slope * elapsed + intercept, elapsed, np.arange(n) % 500 + 2


def test_fit_recovers_an_exact_line():
    times, elapsed, iterations = _line()
    fit = fit_clock(times, elapsed, iterations)
    assert fit.slope == pytest.approx(1 - 5e-6, abs=1e-12)
    assert fit.intercept == pytest.approx(11.0, abs=1e-9)
    assert fit.n_packets == 2000
    assert fit.n_rejected == 0
    assert abs(fit.residual_ms["p99"]) < 1e-6


def test_late_packets_are_rejected_and_do_not_move_the_fit():
    times, elapsed, iterations = _line()
    times = times.copy()
    late = np.arange(50, 2000, 100)
    times[late] += 0.3
    fit = fit_clock(times, elapsed, iterations)
    assert fit.n_rejected == late.size
    assert fit.slope == pytest.approx(1 - 5e-6, abs=1e-10)
    assert fit.intercept == pytest.approx(11.0, abs=1e-7)
    assert fit.residual_ms["max"] == pytest.approx(300.0, abs=0.01)


def test_first_iterations_of_a_trial_are_not_fit():
    times, elapsed, iterations = _line()
    iterations = iterations.copy()
    iterations[::100] = 1
    times = times.copy()
    times[iterations == 1] += 0.008  # below the rejection limit, but known to be late
    fit = fit_clock(times, elapsed, iterations)
    assert fit.intercept == pytest.approx(11.0, abs=1e-7)
    assert fit.n_first_iteration == 20


@pytest.mark.parametrize("n", [0, 1, 50])
def test_too_few_packets_is_an_error(n):
    times, elapsed, iterations = _line(n)
    with pytest.raises(AlignmentError, match="packets"):
        fit_clock(times, elapsed, iterations)


def test_clocks_that_run_at_different_rates_are_an_error():
    times, elapsed, iterations = _line(slope=1.01)
    with pytest.raises(AlignmentError, match="ppm"):
        fit_clock(times, elapsed, iterations)


def test_a_step_in_the_clock_is_an_error():
    times, elapsed, iterations = _line()
    times = times.copy()
    times[1000:] += 0.05
    with pytest.raises(AlignmentError):
        fit_clock(times, elapsed, iterations)


def test_noise_within_the_limits_is_accepted_and_reported():
    times, elapsed, iterations = _line()
    jitter = np.random.default_rng(0).uniform(-0.002, 0.002, times.size)
    fit = fit_clock(times + jitter, elapsed, iterations)
    assert 1.5 < fit.residual_ms["p99"] < 2.5
    assert -2.5 < fit.residual_ms["p1"] < -1.5


def test_a_mapped_time_inverts_the_line():
    times, elapsed, iterations = _line()
    fit = fit_clock(times, elapsed, iterations)
    np.testing.assert_allclose(fit.to_elapsed(times), elapsed, atol=1e-8)


def _two_trials_log(**extra):
    first = make_trial(start=10.0, n_frames=3, n_iti=2)
    second = make_trial(start=20.0, n_frames=2, n_iti=1)
    return make_log([make_block([first, second])], **extra)


def _in_progress(start=30.0, n=2, allocated=6):
    time = np.zeros((allocated, 1))
    time[:n, 0] = np.arange(n) * 0.25
    position = np.zeros((allocated, 3))
    position[:n] = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]][:n]
    return dict(
        currentIt=scalar(n),
        currentTrial=dict(
            start=scalar(start), time=time, position=position, velocity=position * 2, collision=np.zeros((allocated, 1), dtype=bool),
            sensorDots=np.zeros((allocated, 5), dtype="int16"), viStart=np.array([[9]], dtype="uint32"), choice=np.zeros((0, 0)), comment="",
        ),
    )


def test_iteration_clock_looks_up_logged_iterations():
    clock = iteration_clock(_two_trials_log())
    elapsed = match_packets(_packets([1, 1, 1], [1, 1, 2], [1, 5, 3]), clock)
    assert elapsed.tolist() == [10.0, 12.0, 21.0]


def test_iteration_clock_includes_the_trial_in_progress():
    clock = iteration_clock(_two_trials_log(**_in_progress()))
    assert match_packets(_packets([1], [3], [2]), clock).tolist() == [30.25]


def test_an_iteration_past_the_saved_count_of_the_trial_in_progress_is_unmatched():
    clock = iteration_clock(_two_trials_log(**_in_progress()))
    with pytest.raises(AlignmentError, match=r"block 1, trial 3, iteration 3"):
        match_packets(_packets([1], [3], [3]), clock)


@pytest.mark.parametrize(("block", "trial", "iteration"), [(1, 1, 0), (1, 1, 6), (1, 9, 1), (2, 1, 1), (0, 0, 0)])
def test_packets_the_log_does_not_hold_are_errors(block, trial, iteration):
    with pytest.raises(AlignmentError, match="wrong log"):
        match_packets(_packets([block], [trial], [iteration]), iteration_clock(_two_trials_log()))


def test_a_log_with_no_trial_in_progress():
    clock = iteration_clock(_two_trials_log(currentIt=scalar(0), currentTrial=dict(start=scalar(30.0), time=np.zeros((4, 1)))))
    with pytest.raises(AlignmentError):
        match_packets(_packets([1], [3], [1]), clock)


def test_packets_out_of_order_are_an_error():
    clock = iteration_clock(_two_trials_log())
    with pytest.raises(AlignmentError, match="order"):
        match_packets(_packets([1, 1], [2, 1], [1, 1]), clock)


def test_shift_is_zero_when_imaging_starts_after_the_session():
    assert session_start_shift(np.array([0.0, 1.0])) == 0
    assert session_start_shift(np.array([2.5, 3.0])) == 0


def test_shift_rounds_up_to_whole_microseconds_so_no_time_is_negative():
    shift = session_start_shift(np.array([-10.2500001, 1.0]))
    assert shift == 10250001
    assert -10.2500001 + shift / 1e6 >= 0


def test_shift_of_no_frames_is_zero():
    assert session_start_shift(np.zeros(0)) == 0


def _session():
    block = make_block([make_trial(start=0.5, n_frames=3, n_iti=2), make_trial(start=9.0, n_frames=2, n_iti=1)])
    return normalize_log(make_log([block]), file="session.mat")


def _column(columns, name):
    return next(c for c in columns if c.name == name).data


def test_shifted_session_moves_every_time_and_keeps_intervals():
    session = _session()
    moved = shifted_session(session, 2_500_000)
    assert moved.session_start_time == session.session_start_time - timedelta(seconds=2.5)
    for name in ("start_time", "stop_time"):
        np.testing.assert_array_equal(_column(moved.trials, name), _column(session.trials, name) + 2.5)
        np.testing.assert_array_equal(_column(moved.epochs, name), _column(session.epochs, name) + 2.5)
    np.testing.assert_array_equal(_column(moved.epochs, "wall_clock_start"), _column(session.epochs, "wall_clock_start") + 2.5)
    for before, after in zip(session.series, moved.series):
        np.testing.assert_array_equal(after.timestamps, before.timestamps + 2.5)
        assert after.data is before.data


def test_shifted_session_keeps_the_raw_clock_recoverable():
    session = _session()
    moved = shifted_session(session, 2_500_000)
    settings = {key: value for key, value, _ in moved.settings}
    before = {key: value for key, value, _ in session.settings}
    assert settings["derived.clock_offset"] != before["derived.clock_offset"]
    assert settings["derived.session_start_shift_seconds"] == "2.5"
    assert settings["derived.virmen_session_start"] == session.session_start_time.isoformat()


def test_a_zero_shift_changes_nothing_but_says_so():
    session = _session()
    moved = shifted_session(session, 0)
    assert moved.session_start_time == session.session_start_time
    np.testing.assert_array_equal(_column(moved.trials, "start_time"), _column(session.trials, "start_time"))
    assert {key: value for key, value, _ in moved.settings}["derived.session_start_shift_seconds"] == "0.0"


def test_shift_does_not_change_the_original_session():
    session = _session()
    before = _column(session.trials, "start_time").copy()
    n_settings = len(session.settings)
    shifted_session(session, 1_000_000)
    np.testing.assert_array_equal(_column(session.trials, "start_time"), before)
    assert len(session.settings) == n_settings


def test_in_progress_series_cuts_preallocated_rows_at_full_precision():
    log = _two_trials_log(**_in_progress())
    series, record = in_progress_series(log, offset=1.0)
    by_name = {item.name: item for item in series}
    assert set(by_name) == {"collision", "position", "sensorDots", "time", "velocity"}
    np.testing.assert_array_equal(by_name["position"].data, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    assert by_name["position"].data.dtype == np.float64
    assert by_name["sensorDots"].data.dtype == np.int16
    np.testing.assert_array_equal(by_name["position"].timestamps, [31.0, 31.25])
    assert record["block"] == 1 and record["trial"] == 3 and record["saved_iterations"] == 2
    assert record["viStart"] == 9


def test_no_in_progress_series_when_nothing_was_saved():
    log = _two_trials_log(currentIt=scalar(0), currentTrial=dict(start=scalar(1.0), time=np.zeros((4, 1))))
    assert in_progress_series(log, offset=1.0) == ([], None)


def test_no_in_progress_series_when_the_log_has_no_such_fields():
    log = _two_trials_log()
    del log["currentIt"], log["currentTrial"]
    assert in_progress_series(log, offset=1.0) == ([], None)
    assert (1, 3) not in iteration_clock(log)


def test_in_progress_keeps_its_own_iterations_field():
    log = _two_trials_log(**_in_progress())
    log["currentTrial"]["iterations"] = np.array([[0]], dtype="uint32")
    _, record = in_progress_series(log, offset=1.0)
    assert record["iterations"] == 0 and record["saved_iterations"] == 2


def test_in_progress_data_past_the_saved_count_is_an_error():
    log = _two_trials_log(**_in_progress())
    log["currentTrial"]["position"][4, 0] = 1.0
    with pytest.raises(AlignmentError, match="position"):
        in_progress_series(log, offset=1.0)


def test_session_time_is_utc_safe():
    start = datetime(2026, 7, 15, 12, tzinfo=timezone.utc)
    assert (start - timedelta(microseconds=session_start_shift(np.array([-1.0])))).second == 59


def _acquisition_with(n_frames, frame_index, block, trial, iteration):
    packets = _packets(block, trial, iteration)
    packets["frame_index"] = np.asarray(frame_index, dtype="int64")
    return SimpleNamespace(frames=dict(frame_number=np.arange(1, n_frames + 1)), packets=packets)


def test_labels_take_the_first_packet_of_a_frame():
    acquisition = _acquisition_with(3, [1, 1, 2], [1, 1, 1], [1, 1, 1], [5, 7, 9])
    assert frame_labels(acquisition, [5])["iteration"].tolist() == [0, 5, 9]


def test_labels_repeat_across_interior_gaps_only():
    acquisition = _acquisition_with(7, [1, 4], [1, 1], [1, 2], [3, 8])
    labels = frame_labels(acquisition, [5])
    assert labels["iteration"].tolist() == [0, 3, 3, 3, 8, 0, 0]
    assert labels["trial"].tolist() == [0, 1, 1, 1, 2, 0, 0]
    assert labels["block"].tolist() == [0, 1, 1, 1, 1, 0, 0]


def test_labels_clear_the_trial_missing_from_the_log():
    acquisition = _acquisition_with(4, [0, 1, 2], [1, 1, 1], [5, 6, 6], [9, 1, 3])
    labels = frame_labels(acquisition, [5])
    assert labels["trial"].tolist() == [5, 0, 0, 0]
    assert labels["iteration"].tolist() == [9, 0, 0, 0]
    assert labels["block"].tolist() == [1, 0, 0, 0]


def test_labels_refuse_trials_far_past_the_log():
    with pytest.raises(AlignmentError, match="only 5 trials"):
        frame_labels(_acquisition_with(2, [0], [1], [7], [1]), [5])


def test_labels_with_no_packets_are_all_zero():
    acquisition = _acquisition_with(3, [], [], [], [])
    assert frame_labels(acquisition, [5])["block"].tolist() == [0, 0, 0]


def test_labels_keep_each_block():
    acquisition = _acquisition_with(3, [0, 2], [1, 2], [5, 1], [9, 1])
    labels = frame_labels(acquisition, [5, 4])
    assert labels["block"].tolist() == [1, 1, 2]
    assert labels["trial"].tolist() == [5, 5, 1]
