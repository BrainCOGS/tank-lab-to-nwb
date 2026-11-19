"""The lab's synchronization rule: each iteration at its own pulse, spikes at their sample, on the recording's clock.

Written before the converter moved to this rule (it used to fit a line to ViRMEn's clock). See docs/ephys-timing.md.
"""

import numpy as np
import pytest

from tank_lab_to_nwb.ephys import pipeline_sync as ps
from tank_lab_to_nwb.ephys.clock import (
    EphysClock,
    behavior_timestamps,
    identity_sync,
    iteration_edges,
    iteration_times,
    probe_sync,
)
from tests.synthetic_ephys import IMEC_RATE, NIDQ_RATE, iteration_schedule, sync_word


def _behavior_time(trials):
    out = np.empty(len(trials), dtype=object)
    for i, (_, times) in enumerate(trials):
        out[i] = times.reshape(-1, 1)
    return out


def _pipeline(word, trials):
    trial_line, iteration_line = ps.pipeline_lines(word, "nidq", NIDQ_RATE, word.size / NIDQ_RATE)
    result = ps.pipeline_sync(trial_line, iteration_line, NIDQ_RATE, _behavior_time(trials))
    return result, iteration_edges(result.iteration_idx_vector, trial_line, iteration_line, result.mode, NIDQ_RATE)


def test_each_iteration_takes_its_own_pulse():
    trials = iteration_schedule(n_trials=8)
    word, _ = sync_word(trials, n_samples=int(NIDQ_RATE * 60))
    result, edges = _pipeline(word, trials)
    times = iteration_times(result.iteration_idx_vector, edges, NIDQ_RATE, "pipeline_index")
    for got, index in zip(times, result.iteration_idx_vector):
        assert np.array_equal(got, np.asarray(index) / NIDQ_RATE)
    times = iteration_times(result.iteration_idx_vector, edges, NIDQ_RATE, "edge")
    for got, edge in zip(times, edges):
        assert np.array_equal(got, (edge - 0.5) / NIDQ_RATE)


def test_frame_late_pulses_need_no_tolerance():
    """A pulse a display frame late keeps its own time: nothing is fitted, so nothing is rejected."""
    trials = iteration_schedule(n_trials=8, period=0.020)
    rng = np.random.default_rng(1)
    late = {(t, i) for t, (_, times) in enumerate(trials) for i in range(1, times.size) if rng.random() < 0.08}
    word, truth = sync_word(trials, n_samples=int(NIDQ_RATE * 80), late={key: 0.0167 * 0.6 for key in late})
    result, edges = _pipeline(word, trials)
    assert result.regular_sync_status == 1
    times = iteration_times(result.iteration_idx_vector, edges, NIDQ_RATE, "edge")
    for got, true in zip(times, truth):
        assert np.array_equal(got, (true - 0.5) / NIDQ_RATE)
    flat = behavior_timestamps(times, [times_.size for _, times_ in trials])
    assert flat.size == sum(t.size for _, t in trials) and np.all(np.diff(flat) > 0)


def test_unsynced_iterations_are_refused():
    trials = iteration_schedule(n_trials=8)
    word, _ = sync_word(trials, n_samples=int(NIDQ_RATE * 60))
    result, edges = _pipeline(word, trials)
    times = iteration_times(result.iteration_idx_vector, edges, NIDQ_RATE, "pipeline_index")
    with pytest.raises(ValueError, match="trial 9"):
        behavior_timestamps(times, [t.size for _, t in trials] + [100])
    counts = [t.size for _, t in trials]
    counts[3] += 1
    with pytest.raises(ValueError, match="trial 4: 1 logged iteration"):
        behavior_timestamps(times, counts)


def _drifting_clock(probe_clock):
    nidq = np.ceil((np.arange(0.3, 3600, 1.0)) * NIDQ_RATE).astype(np.int64)
    probe_seconds = lambda s: (s - 0.03) * (1 - 10.4e-6)  # noqa: E731
    probe = np.ceil(probe_seconds(np.arange(0.3, 3600, 1.0)) * IMEC_RATE).astype(np.int64)[::30]
    fit = probe_sync(probe, IMEC_RATE, 0, nidq, NIDQ_RATE, 0)
    return EphysClock("nidq", NIDQ_RATE, probe_clock, {"imec0": fit, "nidq": identity_sync()}, {"imec0": IMEC_RATE}), probe_seconds


def test_meta_rates_is_the_sample_over_imsamprate():
    clock, _ = _drifting_clock("meta_rates")
    samples = np.array([0, 1000, 10_000_000, 100_000_000])
    assert np.array_equal(clock.probe_samples_to_session("imec0", samples), samples / IMEC_RATE)
    assert np.array_equal(clock.probe_samples_to_session("imec0", samples, rate=30000.0), samples / 30000.0)


def test_sync_edges_removes_the_drift():
    clock, probe_seconds = _drifting_clock("sync_edges")
    nidq_seconds = np.array([10.0, 1800.0, 3500.0])
    got = clock.probe_samples_to_session("imec0", probe_seconds(nidq_seconds) * IMEC_RATE)
    assert np.max(np.abs(got - nidq_seconds)) < 40e-6
    plain, _ = _drifting_clock("meta_rates")
    # Uncorrected, the probe drifts 10.4 ppm, 36 ms over the hour, plus its 30 ms later start.
    assert abs((nidq_seconds[-1] - plain.probe_samples_to_session("imec0", [probe_seconds(3500.0) * IMEC_RATE])[0]) - (0.03 + 3500 * 10.4e-6)) < 1e-4


def test_unknown_choices_are_refused():
    with pytest.raises(ValueError, match="probe_clock"):
        EphysClock("nidq", NIDQ_RATE, "line_fit", {}, {})
    with pytest.raises(ValueError, match="iteration_time"):
        iteration_times([], [], NIDQ_RATE, "fit")


def test_curated_clusters_iteration_times_are_one_sample_before_the_index():
    """U19's CuratedClustersIteration expands the vectors per sample and takes their transitions, one sample early."""
    trials = iteration_schedule(n_trials=5)
    word, _ = sync_word(trials, n_samples=int(NIDQ_RATE * 40))
    result, _ = _pipeline(word, trials)
    times = ps.curated_clusters_iteration_times(result.iteration_idx_vector, NIDQ_RATE, word.size)
    flat = np.concatenate([np.asarray(v) for v in result.iteration_idx_vector])
    # Every iteration but the session's first, at its index minus one sample.
    assert np.array_equal(times, (flat[1:] - 1) / NIDQ_RATE)
