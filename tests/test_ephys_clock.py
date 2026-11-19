"""Probe sync maps and the pipeline's iteration matching. The lab's rule built on them is in test_ephys_rule.py."""

import numpy as np
import pytest

from tank_lab_to_nwb.ephys import pipeline_sync as ps
from tank_lab_to_nwb.ephys.clock import (
    EphysClock,
    SyncError,
    fit_sync,
    identity_sync,
    iteration_edges,
    pair_sync_edges,
    probe_sync,
)
from tests.synthetic_ephys import IMEC_RATE, NIDQ_RATE, iteration_schedule, sync_word


def drifting_probe(seconds_sync):
    """Probe seconds of a sync-stream time: -10 ppm, a 30 ms start offset, and 150 us of slow wander, like S3's imec0."""
    s = np.asarray(seconds_sync, dtype="float64")
    return (s - 0.030) * (1 - 10.4e-6) + 150e-6 * np.sin(2 * np.pi * s / 3000.0)


def edges_at(rate, first, last, seconds_to_stream, phase=0.3, period=1.0):
    """Rising-edge sample indices of the sync wave in a stream, given a map from sync seconds to its seconds."""
    sync_times = np.arange(phase, last, period)
    stream_seconds = seconds_to_stream(sync_times)
    keep = (stream_seconds >= first) & (stream_seconds < last)
    return np.ceil(stream_seconds[keep] * rate).astype(np.int64)


def test_sync_map_follows_drift_and_wander_to_microseconds():
    nidq = edges_at(NIDQ_RATE, 0, 3970, lambda s: s)
    probe = edges_at(IMEC_RATE, 0, 3970, drifting_probe)
    probe = probe[::30]  # only windows of the AP file are read
    fit = probe_sync(probe, IMEC_RATE, 0, nidq, NIDQ_RATE, 0)
    check = np.linspace(10, 3900, 500)
    error = fit(drifting_probe(check)) - check
    assert np.max(np.abs(error)) < 40e-6
    assert fit.linear_residual_us["max"] > 120  # a single line misses the wander
    assert abs((fit.linear_slope - 1) * 1e6 - 10.4) < 0.5
    assert fit.residual_us["std"] < 70  # what's left is the 5 kHz sampling step, 200 us / sqrt(12)


def test_sync_map_is_exact_without_quantization():
    x = np.linspace(0, 3600, 61)
    y = x * (1 + 3e-6) + 0.02
    fit = fit_sync(x, y)
    assert np.allclose(fit(x), y, atol=1e-9, rtol=0)
    assert np.allclose(fit(np.array([-100.0, 4000.0])), np.array([-100.0, 4000.0]) * (1 + 3e-6) + 0.02, atol=1e-9, rtol=0)


def test_missing_edges_still_pair():
    nidq = edges_at(NIDQ_RATE, 0, 600, lambda s: s)
    probe = edges_at(IMEC_RATE, 0, 600, drifting_probe)
    nidq = np.delete(nidq, [5, 6, 7, 300])
    probe_index, sync_index, offset = pair_sync_edges(probe, IMEC_RATE, 0, nidq, NIDQ_RATE, 0)
    assert np.all(np.abs(offset) < 0.05)
    assert probe.size - probe_index.size == 4
    fit = fit_sync(probe[probe_index] / IMEC_RATE, nidq[sync_index] / NIDQ_RATE)
    assert fit.residual_us["max"] < 250


def test_a_glitch_edge_is_refused():
    x = np.arange(0.3, 600, 1.0)
    y = x.copy()
    y[100] += 0.002
    with pytest.raises(SyncError, match="off the fitted map"):
        fit_sync(x, y)


def test_streams_started_apart_are_refused():
    nidq = edges_at(NIDQ_RATE, 0, 100, lambda s: s)
    probe = edges_at(IMEC_RATE, 0, 100, lambda s: s - 0.4)
    with pytest.raises(SyncError, match="paired safely"):
        pair_sync_edges(probe, IMEC_RATE, 0, nidq, NIDQ_RATE, 0)


def test_rate_mismatch_is_refused():
    x = np.linspace(0, 1000, 50)
    with pytest.raises(SyncError, match="ppm"):
        fit_sync(x, x * (1 + 500e-6))


def _lines(word):
    return ps.pipeline_lines(word, "nidq", NIDQ_RATE, word.size / NIDQ_RATE)


def _behavior_time(trials):
    out = np.empty(len(trials), dtype=object)
    for i, (_, times) in enumerate(trials):
        out[i] = times.reshape(-1, 1)
    return out


def test_pipeline_index_is_six_samples_before_the_edge():
    trials = iteration_schedule()
    word, truth = sync_word(trials, n_samples=int(NIDQ_RATE * 70))
    trial_line, iteration_line = _lines(word)
    result = ps.pipeline_sync(trial_line, iteration_line, NIDQ_RATE, _behavior_time(trials))
    assert (result.regular_sync_status, result.mode) == (1, "counter_bit0")
    for stored, true in zip(result.iteration_idx_vector, truth):
        assert stored[0] == true[0] - 1
        assert np.array_equal(stored[1:], true[1:] - 6)
    assert ps.pipeline_index_lag("counter_bit0", NIDQ_RATE) == (1, 6)
    edges = iteration_edges(result.iteration_idx_vector, trial_line, iteration_line, result.mode, NIDQ_RATE)
    for found, true in zip(edges, truth):
        assert np.array_equal(found, true)


def test_virmen_vectors_follow_the_pipeline_formula():
    trials = iteration_schedule()
    word, truth = sync_word(trials, n_samples=int(NIDQ_RATE * 70))
    result = ps.pipeline_sync(*_lines(word), NIDQ_RATE, _behavior_time(trials))
    for stored, true, (_, times) in zip(result.iteration_idx_vector_from_virmen, truth, trials):
        assert np.array_equal(stored, (true[0] - 1) + np.int64(times * NIDQ_RATE))


def test_missed_pulses_take_the_fix_path_and_borrowed_indices_are_not_edges():
    trials = iteration_schedule(n_trials=20)
    skip = {(4, 50), (4, 51), (9, 120)}
    word, truth = sync_word(trials, n_samples=int(NIDQ_RATE * 110), skip=skip)
    trial_line, iteration_line = _lines(word)
    result = ps.pipeline_sync(trial_line, iteration_line, NIDQ_RATE, _behavior_time(trials))
    assert result.regular_sync_status == 0
    assert result.fixed_sync_status == 1
    edges = iteration_edges(result.iteration_idx_vector, trial_line, iteration_line, result.mode, NIDQ_RATE)
    for t, (found, true) in enumerate(zip(edges, truth)):
        assert found.size == true.size
        measured = found >= 0
        # Every index called an edge is the true edge of that iteration.
        assert np.array_equal(found[measured], true[measured]), t
    assert all(edges[t][i] == -1 for t, i in skip)


def test_clock_chains_probe_samples_to_the_recording():
    nidq = edges_at(NIDQ_RATE, 0, 110, lambda s: s)
    probe = edges_at(IMEC_RATE, 0, 110, drifting_probe)
    fits = {"imec0": probe_sync(probe, IMEC_RATE, 0, nidq, NIDQ_RATE, 0), "nidq": identity_sync()}
    clock = EphysClock("nidq", NIDQ_RATE, "sync_edges", fits, {"imec0": IMEC_RATE, "nidq": NIDQ_RATE})
    sync_seconds = np.array([20.0, 55.5, 90.25])
    assert np.max(np.abs(clock.probe_samples_to_session("imec0", drifting_probe(sync_seconds) * IMEC_RATE) - sync_seconds)) < 40e-6
    assert np.array_equal(clock.sync_samples_to_session(np.array([0, 5000])), np.array([0, 5000]) / NIDQ_RATE)
    assert np.allclose(clock.shifted(2.0).sync_samples_to_session([0]), [2.0])
    with pytest.raises(KeyError, match="imec3"):
        clock.probe_samples_to_session("imec3", [0])
