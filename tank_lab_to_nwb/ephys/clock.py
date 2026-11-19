"""Put every ephys stream and the behavior on the recording's clock. Pure functions of arrays. See docs/ephys-timing.md.

The lab's rule (U19 BehaviorSync, SYNC_DATA_PLAN.md): the file's clock is the recording's, t = 0 at the first sample of
the sync stream (NIDQ or OneBox). Each logged ViRMEn iteration takes the time of its own pulse on that stream,
`iteration_idx_vector / rate`, one for one: nothing is fitted. Spikes keep their sample's time.

Two departures from that rule can be switched on, each recorded in the file with its size:

- `probe_clock`: "sync_edges" maps each probe's seconds to the sync stream's through a piecewise-linear fit to the 1 Hz
  sync edges, which removes the drift between crystals that the calibrated rates leave. "meta_rates" is the plain
  `sample / imSampRate`. "params_rate" is `sample / params.py sample_rate`, what the DB's CuratedClustering stores.
- `iteration_time`: "pipeline_index" is the pipeline's stored index (6 samples before the edge in counter_bit0 mode);
  "edge" is the edge itself, timed half a sample before its first high sample.
"""

from dataclasses import dataclass, field

import numpy as np

from .pipeline_sync import pipeline_index_lag

# The defaults, until the lab decides. Each is one line.
PROBE_CLOCK = "sync_edges"
ITERATION_TIME = "pipeline_index"
PROBE_CLOCKS = ("sync_edges", "meta_rates", "params_rate")
ITERATION_TIMES = ("pipeline_index", "edge")

# Sync edges are paired by the streams' firstSample counts, which start together within milliseconds. Further apart than
# this, an edge could pair with its neighbour a second away, which no residual would show.
MAX_COARSE_OFFSET_SECONDS = 0.25
MIN_PAIRED_FRACTION = 0.9
# One segment of the piecewise-linear sync map per this many seconds of recording.
KNOT_SPACING_SECONDS = 300.0
MAX_SYNC_RATE_DIFFERENCE_PPM = 200.0
# Beyond this an edge is off the map: a glitch or a wrong pairing, so the session needs a person to look at it.
MAX_SYNC_RESIDUAL_SECONDS = 0.0005

def edge_seconds(samples, rate):
    """When an edge happened, from the first sample at its new level: half a sample earlier, on average.

    The change happened somewhere in the sample period before that sample. Without the half sample, NIDQ edges (5 kHz)
    would read 100 us late and imec edges (30 kHz) 17 us late, and the map between them would carry the 83 us difference.
    """
    return (np.asarray(samples, dtype="float64") - 0.5) / rate


class SyncError(ValueError):
    """The sync edges don't tie this probe to the sync stream."""


@dataclass(frozen=True)
class SyncFit:
    """A continuous piecewise-linear map from a probe's seconds (sample / imSampRate) to sync-stream seconds."""

    knots: np.ndarray  # probe seconds
    values: np.ndarray  # sync-stream seconds at the knots
    n_edges: int
    linear_slope: float  # of the best single line, for the record
    linear_residual_us: dict  # of that line: how much the piecewise map adds
    residual_us: dict  # of the piecewise map, over every paired edge
    coarse_offset_ms: dict  # between the paired edges before fitting, from firstSample

    def __call__(self, seconds):
        seconds = np.asarray(seconds, dtype="float64")
        out = np.interp(seconds, self.knots, self.values)
        if self.knots.size > 1:
            low, high = seconds < self.knots[0], seconds > self.knots[-1]
            first = (self.values[1] - self.values[0]) / (self.knots[1] - self.knots[0])
            last = (self.values[-1] - self.values[-2]) / (self.knots[-1] - self.knots[-2])
            out[low] = self.values[0] + (seconds[low] - self.knots[0]) * first
            out[high] = self.values[-1] + (seconds[high] - self.knots[-1]) * last
        return out


def identity_sync(n_edges=0):
    """The sync stream's own map onto itself."""
    zero = dict(max=0.0, std=0.0)
    return SyncFit(np.array([0.0, 1.0]), np.array([0.0, 1.0]), n_edges, 1.0, zero, zero, dict(min=0.0, max=0.0))


def pair_sync_edges(probe_edges, probe_rate, probe_first_sample, sync_edges, sync_rate, sync_first_sample):
    """(probe edge indices, sync edge indices, coarse offsets in seconds) of the same sync pulses.

    Each probe edge pairs with the sync edge nearest it on the run's timeline, `(firstSample + index) / rate`. A pair
    further apart than MAX_COARSE_OFFSET_SECONDS is dropped. Raises SyncError if under MIN_PAIRED_FRACTION of the probe
    edges pair, or if two probe edges claim one sync edge.
    """
    probe_edges, sync_edges = np.asarray(probe_edges, dtype=np.int64), np.asarray(sync_edges, dtype=np.int64)
    if probe_edges.size < 2 or sync_edges.size < 2:
        raise SyncError(f"only {probe_edges.size} probe and {sync_edges.size} sync-stream sync edges, too few to pair")
    probe_time = (probe_first_sample + probe_edges) / probe_rate
    sync_time = (sync_first_sample + sync_edges) / sync_rate
    after = np.clip(np.searchsorted(sync_time, probe_time), 1, sync_time.size - 1)
    nearest = np.where(np.abs(sync_time[after - 1] - probe_time) <= np.abs(sync_time[after] - probe_time), after - 1, after)
    offset = sync_time[nearest] - probe_time
    # An edge with no partner (a sync-stream edge lost, or a probe edge outside the sync file) is left out.
    keep = np.abs(offset) <= MAX_COARSE_OFFSET_SECONDS
    if keep.sum() < max(2, int(np.ceil(MIN_PAIRED_FRACTION * keep.size))):
        raise SyncError(f"only {keep.sum()} of {keep.size} probe sync edges are within {MAX_COARSE_OFFSET_SECONDS * 1e3:.0f} ms of a sync-stream edge, so the edges can't be paired safely; were the streams started apart?")
    probe_index, nearest, offset = np.flatnonzero(keep), nearest[keep], offset[keep]
    if np.unique(nearest).size != nearest.size:
        raise SyncError("two probe sync edges pair with one sync-stream edge; the sync wave isn't 1 Hz on one of the streams")
    return probe_index, nearest, offset


def _hat_basis(x, knots):
    basis = np.zeros((x.size, knots.size))
    if knots.size == 1:
        basis[:, 0] = 1
        return basis
    position = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, knots.size - 2)
    fraction = (x - knots[position]) / (knots[position + 1] - knots[position])
    rows = np.arange(x.size)
    basis[rows, position] = 1 - fraction
    basis[rows, position + 1] = fraction
    return basis


def _summary(residual):
    return dict(max=float(np.max(np.abs(residual)) * 1e6), std=float(np.std(residual) * 1e6))


def fit_sync(probe_seconds, sync_seconds, coarse_offset=None, knot_spacing=KNOT_SPACING_SECONDS):
    """Least-squares continuous piecewise-linear map from paired probe edge times to sync-stream edge times.

    Knots are evenly spaced, about `knot_spacing` apart, from the first paired edge to the last. A single line leaves
    slow wander of a few hundred microseconds over an hour; the segments follow it, and averaging the edges within each
    one beats down the sync stream's sampling step (200 us at 5 kHz).
    """
    x, y = np.asarray(probe_seconds, dtype="float64"), np.asarray(sync_seconds, dtype="float64")
    if x.size < 2 or np.any(np.diff(x) <= 0):
        raise SyncError(f"need at least 2 increasing paired edges, got {x.size}")
    slope, intercept = np.polyfit(x - x[0], y, 1)
    ppm = abs(slope - 1) * 1e6
    if ppm > MAX_SYNC_RATE_DIFFERENCE_PPM:
        raise SyncError(f"the probe and sync-stream clocks differ in rate by {ppm:.0f} ppm, more than the {MAX_SYNC_RATE_DIFFERENCE_PPM:.0f} ppm allowed")
    linear_residual = y - (slope * (x - x[0]) + intercept)
    n_segments = max(1, int(round((x[-1] - x[0]) / knot_spacing)))
    # Each segment needs a few edges, or it would chase single edges' quantization.
    n_segments = max(1, min(n_segments, x.size // 4))
    knots = np.linspace(x[0], x[-1], n_segments + 1)
    values = np.linalg.lstsq(_hat_basis(x, knots), y, rcond=None)[0]
    fitted = SyncFit(knots, values, int(x.size), float(slope), _summary(linear_residual), {}, {})
    residual = y - fitted(x)
    if np.max(np.abs(residual)) > MAX_SYNC_RESIDUAL_SECONDS:
        raise SyncError(f"a sync edge is {np.max(np.abs(residual)) * 1e3:.2f} ms off the fitted map, more than the {MAX_SYNC_RESIDUAL_SECONDS * 1e3:.1f} ms allowed")
    coarse = dict(min=0.0, max=0.0) if coarse_offset is None else dict(min=float(np.min(coarse_offset) * 1e3), max=float(np.max(coarse_offset) * 1e3))
    return SyncFit(knots, values, int(x.size), float(slope), _summary(linear_residual), _summary(residual), coarse)


def probe_sync(probe_edges, probe_rate, probe_first_sample, sync_edges, sync_rate, sync_first_sample, knot_spacing=KNOT_SPACING_SECONDS):
    """Pair a probe's sync edges with the sync stream's and fit the map, in one call."""
    probe_index, sync_index, offset = pair_sync_edges(probe_edges, probe_rate, probe_first_sample, sync_edges, sync_rate, sync_first_sample)
    return fit_sync(edge_seconds(np.asarray(probe_edges)[probe_index], probe_rate), edge_seconds(np.asarray(sync_edges)[sync_index], sync_rate), offset, knot_spacing)


def transitions(line):
    """First sample of every change of a 0/1 line, either direction."""
    line = np.asarray(line)
    return np.flatnonzero(line[1:] != line[:-1]) + 1


def iteration_edges(iteration_idx_vector, trial_line, iteration_line, mode, rate):
    """Per trial, the edge each pipeline iteration index was found from, or -1 where the index isn't at an edge.

    The pipeline stores indices a fixed few samples before their edges (`pipeline_index_lag`); this finds the edge. An
    index the pipeline's fix path borrowed from ViRMEn's logged times has no edge there and gets -1, so it can't pass for
    a measurement.
    """
    first_lag, later_lag = pipeline_index_lag(mode, rate)
    trial_line = np.asarray(trial_line)
    trial_edges, iteration_changes = np.flatnonzero(trial_line[1:] > trial_line[:-1]) + 1, transitions(iteration_line)
    out = []
    for indices in iteration_idx_vector:
        indices = np.asarray(indices, dtype=np.int64).ravel()
        lags = np.full(indices.size, later_lag)
        if indices.size:
            lags[0] = first_lag
        candidates = indices + lags
        found = np.isin(candidates, iteration_changes)
        if indices.size:
            found[0] = np.isin(candidates[0], trial_edges)
        out.append(np.where(found, candidates, -1))
    return out


def iteration_times(iteration_idx_vector, edges, rate, iteration_time=ITERATION_TIME):
    """Per trial, each iteration's time in sync-stream seconds.

    "pipeline_index": the pipeline's stored index over the rate, as BehaviorSync's users compute it. "edge": the edge the
    index was found from; an index with no edge there (borrowed from ViRMEn's times by the fix path) keeps the index
    plus the same lag, so it moves with its neighbours.
    """
    if iteration_time not in ITERATION_TIMES:
        raise ValueError(f"iteration_time must be one of {ITERATION_TIMES}, not {iteration_time!r}")
    out = []
    for indices, found in zip(iteration_idx_vector, edges):
        indices = np.asarray(indices, dtype=np.int64).ravel()
        if iteration_time == "pipeline_index":
            out.append(indices / rate)
        else:
            found = np.asarray(found, dtype=np.int64)
            out.append(np.where(found >= 0, edge_seconds(found, rate), edge_seconds(indices + _lag(indices, found), rate)))
    return out


def _lag(indices, found):
    """The pipeline's lag in this trial, from its own measured edges, for the indices that have none."""
    measured = found >= 0
    lags = np.zeros(indices.size, dtype=np.int64)
    if measured[1:].any():
        lags[1:] = int(np.median(found[1:][measured[1:]] - indices[1:][measured[1:]]))
    if measured[0]:
        lags[0] = found[0] - indices[0]
    return lags


def behavior_timestamps(per_trial_times, logged_iterations):
    """Every logged iteration's time, trial after trial, for VirmenDataInterface.set_aligned_timestamps.

    `logged_iterations` is each saved trial's number of logged iterations (its `trial.time`). The pipeline gives a time
    to exactly those; where it gives none (a trial past the last trial pulse, or a trial with a different count), this
    raises rather than guessing.
    """
    if len(per_trial_times) < len(logged_iterations):
        raise ValueError(f"the pipeline synced {len(per_trial_times)} trials, the log has {len(logged_iterations)}: trial {len(per_trial_times) + 1} on has no pulse times")
    for trial, (times, n) in enumerate(zip(per_trial_times, logged_iterations), start=1):
        if np.size(times) != n:
            difference = n - np.size(times)
            raise ValueError(f"trial {trial}: {abs(difference)} logged iteration{'s' if abs(difference) != 1 else ''} {'without' if difference > 0 else 'more than the'} pulse times ({n} logged, {np.size(times)} synced)")
    flat = np.concatenate([np.asarray(t, dtype="float64") for t in per_trial_times[: len(logged_iterations)]]) if logged_iterations else np.zeros(0)
    if np.any(np.diff(flat) < 0):
        raise ValueError("iteration times run backwards; is this the wrong log?")
    return flat


@dataclass(frozen=True)
class EphysClock:
    """Seconds on the recording's clock for any sample of a session's streams. t = 0 is the sync stream's first sample."""

    sync_kind: str  # "nidq" or "obx"
    sync_rate: float
    probe_clock: str
    probes: dict = field(default_factory=dict)  # probe name -> SyncFit (always fitted, so its size can be reported)
    probe_rates: dict = field(default_factory=dict)  # probe name -> imSampRate from its meta
    shift: float = 0.0  # seconds added to everything, if needed so that nothing is negative

    def __post_init__(self):
        if self.probe_clock not in PROBE_CLOCKS:
            raise ValueError(f"probe_clock must be one of {PROBE_CLOCKS}, not {self.probe_clock!r}")

    def sync_samples_to_session(self, samples):
        """Sync-stream (NIDQ or OneBox) sample indices to seconds in the file."""
        return np.asarray(samples, dtype="float64") / self.sync_rate + self.shift

    def probe_samples_to_session(self, probe, samples, rate=None):
        """Sample indices into a probe's AP file (or, with its own `rate`, its LF file) to seconds in the file.

        `rate` defaults to the AP meta's imSampRate; pass the LF rate for LF samples, or params.py's rate under
        probe_clock "params_rate". Under "sync_edges" the probe's seconds then go through its sync map.
        """
        if probe not in self.probe_rates:
            raise KeyError(f"no clock for {probe}; the clock has {sorted(self.probe_rates)}")
        rate = self.probe_rates[probe] if rate is None else rate
        seconds = np.asarray(samples, dtype="float64") / rate
        if self.probe_clock == "sync_edges":
            seconds = self.probes[probe](seconds)
        return seconds + self.shift

    def shifted(self, seconds):
        return EphysClock(self.sync_kind, self.sync_rate, self.probe_clock, dict(self.probes), dict(self.probe_rates), self.shift + seconds)
