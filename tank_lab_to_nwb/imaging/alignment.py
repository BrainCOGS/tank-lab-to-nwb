"""Put imaging frames on the behavior clock using the I2C packets ViRMEn sends. See docs/imaging-timing.md."""

import json
from dataclasses import dataclass, replace
from datetime import timedelta

import numpy as np

from ..virmen.loader import records
from ..virmen.session import Series, encode_setting

MIN_PACKETS = 100
# Rejection follows the pipeline's rule: more than max(5 MADs, 5 ms) from the line, until the kept set stops changing.
_REJECT_MADS, _REJECT_FLOOR_SECONDS = 5.0, 0.005
# Beyond these the two clocks aren't related by a line, so the session needs a person to look at it.
MAX_RATE_DIFFERENCE_PPM = 100.0
MAX_REJECTED_FRACTION = 0.05
MAX_RESIDUAL_SECONDS = 0.010

_IN_PROGRESS_FIELDS = {
    "time": ("seconds", "Seconds since this trial started."),
    "position": ("n.a.", "x (cm), y (cm) and view angle (radians, unwrapped) in the virtual maze."),
    "velocity": ("n.a.", "x (cm/s), y (cm/s) and angular (radians/s) velocity."),
    "collision": ("n.a.", "Whether the animal was colliding with a wall."),
    "sensorDots": ("dots", "Raw motion sensor readings."),
}


class AlignmentError(ValueError):
    """The packets don't tie these frames to this log."""


@dataclass(frozen=True)
class ClockFit:
    """scanimage_time = slope * elapsed + intercept, from packets matched to logged iterations."""

    slope: float
    intercept: float
    used: np.ndarray  # per packet, whether it was fit
    n_packets: int
    n_first_iteration: int
    n_rejected: int
    residual_ms: dict  # over all packets, including rejected ones
    kept_std_ms: float
    within_5ms_fraction: float

    def to_elapsed(self, scanimage_time):
        return (np.asarray(scanimage_time, dtype="float64") - self.intercept) / self.slope


def fit_clock(times, elapsed, iterations):
    """Fit the line, rejecting late packets. A trial's first iteration isn't fit, because it's always sent late."""
    times, elapsed, iterations = (np.asarray(v) for v in (times, elapsed, iterations))
    if times.size < MIN_PACKETS:
        raise AlignmentError(f"only {times.size} I2C packets match the log, too few to fit the clocks; {MIN_PACKETS} are needed")
    center = elapsed.mean()
    eligible = iterations > 1
    if eligible.sum() < MIN_PACKETS:
        raise AlignmentError(f"only {eligible.sum()} usable I2C packets match the log; {MIN_PACKETS} are needed")
    keep = eligible.copy()
    slope, intercept = np.polyfit(elapsed[keep] - center, times[keep], 1)
    for _ in range(50):
        residual = times - (slope * (elapsed - center) + intercept)
        middle = np.median(residual[keep])
        mad = 1.4826 * np.median(np.abs(residual[keep] - middle))
        new_keep = eligible & (np.abs(residual - middle) <= max(_REJECT_MADS * mad, _REJECT_FLOOR_SECONDS))
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
        slope, intercept = np.polyfit(elapsed[keep] - center, times[keep], 1)
    intercept = intercept - slope * center
    residual = times - (slope * elapsed + intercept)
    ppm = abs(slope - 1) * 1e6
    if ppm > MAX_RATE_DIFFERENCE_PPM:
        raise AlignmentError(f"the clocks differ in rate by {ppm:.0f} ppm, more than the {MAX_RATE_DIFFERENCE_PPM:.0f} ppm allowed")
    n_rejected = int(eligible.sum() - keep.sum())
    if n_rejected > MAX_REJECTED_FRACTION * eligible.sum():
        raise AlignmentError(f"{n_rejected} of {eligible.sum()} packets are off the line, so one line doesn't fit this session")
    low, high = np.percentile(residual, [1, 99])
    if max(abs(low), abs(high)) > MAX_RESIDUAL_SECONDS:
        raise AlignmentError(f"packets are {low * 1e3:.1f} to {high * 1e3:.1f} ms from the line, so one line doesn't fit this session")
    p1, p50, p99 = np.percentile(residual, [1, 50, 99]) * 1e3
    return ClockFit(
        slope=float(slope),
        intercept=float(intercept),
        used=keep,
        n_packets=int(times.size),
        n_first_iteration=int((~eligible).sum()),
        n_rejected=n_rejected,
        residual_ms=dict(min=float(residual.min() * 1e3), p1=float(p1), p50=float(p50), p99=float(p99), max=float(residual.max() * 1e3)),
        kept_std_ms=float(residual[keep].std() * 1e3),
        within_5ms_fraction=float((np.abs(residual) < 0.005).mean()),
    )


def _saved_iterations(log):
    """How many iterations of the trial in progress were saved, and its fields, or (0, None) if there's no such trial."""
    trial = log.get("currentTrial")
    count = log.get("currentIt")
    if not isinstance(trial, dict) or count is None or np.size(count) != 1:
        return 0, None
    count = int(np.asarray(count).item())
    return (count, trial) if count > 0 else (0, None)


def iteration_clock(log):
    """Each trial's elapsed-clock start and per-iteration times, including the trial that was running when the log was saved."""
    blocks = records(log["block"])
    clock = {}
    for block_number, block in enumerate(blocks, start=1):
        for trial_number, trial in enumerate(records(block["trial"]), start=1):
            clock[(block_number, trial_number)] = (float(np.asarray(trial["start"]).item()), np.asarray(trial["time"], dtype="float64").ravel())
    count, current = _saved_iterations(log)
    if count and blocks:
        key = (len(blocks), len(records(blocks[-1]["trial"])) + 1)
        clock[key] = (float(np.asarray(current["start"]).item()), np.asarray(current["time"], dtype="float64").ravel()[:count])
    return clock


def match_packets(packets, clock):
    """The elapsed time of each packet's iteration. Every packet must be a logged iteration, in order."""
    blocks, trials, iterations = packets["block"], packets["trial"], packets["iteration"]
    elapsed = np.empty(blocks.size, dtype="float64")
    for position, (block, trial, iteration) in enumerate(zip(blocks.tolist(), trials.tolist(), iterations.tolist())):
        start, times = clock.get((block, trial), (None, ()))
        if start is None or not 1 <= iteration <= len(times):
            raise AlignmentError(f"no logged iteration for I2C packet {position}: block {block}, trial {trial}, iteration {iteration}. Is this the wrong log for these TIFFs?")
        elapsed[position] = start + times[iteration - 1]
    order = np.stack((blocks, trials, iterations), axis=1)
    if len(order) > 1:
        differences = order[1:] - order[:-1]
        first_change = np.argmax(differences != 0, axis=1)
        step = differences[np.arange(len(differences)), first_change]
        if np.any(step <= 0):
            raise AlignmentError(f"I2C packets are out of order at packet {int(np.argmax(step <= 0)) + 1}")
    return elapsed


def frame_labels(acquisition, trials_per_block):
    """Per frame, the behavior block, trial and iteration as SyncImagingBehavior stores them in the pipeline (0 = none).

    This follows the canonical MATLAB: the first packet of each frame, interior frames with no packet repeating the last
    frame that had one, nothing before the first or after the last packet, and frames of a trial absent from the log (the
    one running when ViRMEn stopped) cleared. These labels mark which header received a packet, so they run up to a frame
    late at trial onsets. Use the frame timestamps to decide which frames fall in a trial.
    """
    n_frames = acquisition.frames["frame_number"].size
    labels = np.zeros((3, n_frames), dtype="int64")
    packets = acquisition.packets
    has_packet = np.zeros(n_frames, dtype=bool)
    frame_index = packets["frame_index"]
    first = np.ones(frame_index.size, dtype=bool)
    first[1:] = frame_index[1:] != frame_index[:-1]
    labels[0, frame_index[first]] = packets["block"][first]
    labels[1, frame_index[first]] = packets["trial"][first]
    labels[2, frame_index[first]] = packets["iteration"][first]
    has_packet[frame_index[first]] = True
    if has_packet.any():
        with_packet = np.flatnonzero(has_packet)
        interior = np.arange(with_packet[0], with_packet[-1] + 1)
        latest = with_packet[np.searchsorted(with_packet, interior, side="right") - 1]
        labels[:, interior] = labels[:, latest]
    for block in np.unique(labels[0][labels[0] > 0]).tolist():
        in_block = labels[0] == block
        n_trials = trials_per_block[block - 1] if block <= len(trials_per_block) else 0
        last_trial = labels[1][in_block].max()
        if last_trial == n_trials + 1:
            labels[:, in_block & (labels[1] == last_trial)] = 0
        elif last_trial > n_trials:
            raise AlignmentError(f"frames carry trial {last_trial} of block {block}, which has only {n_trials} trials")
    return dict(block=labels[0], trial=labels[1], iteration=labels[2])


def session_start_shift(frame_times):
    """Whole microseconds to move session_start_time back so that no frame has a negative time. 0 if none would."""
    if not np.size(frame_times) or np.min(frame_times) >= 0:
        return 0
    return int(np.ceil(-float(np.min(frame_times)) * 1e6))


def _moved(columns, names, seconds):
    return [replace(column, data=column.data + seconds) if column.name in names else column for column in columns]


def shifted_session(session, shift_microseconds):
    """A copy of the session with session_start_time earlier by the shift and every stored time later by it."""
    seconds = shift_microseconds / 1e6
    settings, shifted_rows = [], []
    for key, value, meaning in session.settings:
        if key == "derived.clock_offset":
            value = json.dumps([offset + seconds for offset in json.loads(value)])
        settings.append((key, value, meaning))
    shifted_rows.append(("derived.session_start_shift_seconds", json.dumps(seconds), "Seconds session_start_time was moved back from ViRMEn's session.start, so that the first imaging frame has no negative time. derived.clock_offset includes this shift."))
    shifted_rows.append(("derived.virmen_session_start", session.session_start_time.isoformat(), "ViRMEn's session.start, before the shift."))
    return replace(
        session,
        session_start_time=session.session_start_time - timedelta(microseconds=shift_microseconds),
        trials=_moved(session.trials, ("start_time", "stop_time"), seconds),
        epochs=_moved(session.epochs, ("start_time", "stop_time", "wall_clock_start"), seconds),
        series=[replace(series, timestamps=series.timestamps + seconds) for series in session.series],
        settings=settings + shifted_rows,
    )


def _is_written(tail):
    return np.any(tail != 0) if tail.dtype.kind != "f" else np.any((tail != 0) & ~np.isnan(tail))


def in_progress_series(log, offset):
    """The trial running when ViRMEn saved the log, at full precision, and what else was recorded about it.

    Later iterations of its preallocated arrays must still be zero; `offset` moves its elapsed times to the session clock.
    """
    count, trial = _saved_iterations(log)
    if not count:
        return [], None
    blocks = records(log["block"])
    time = np.asarray(trial["time"], dtype="float64").ravel()
    allocated = time.size
    start = float(np.asarray(trial["start"]).item())
    timestamps = start + offset + time[:count]
    series, other = [], {}
    for name, value in trial.items():
        if isinstance(value, np.ndarray) and value.dtype != object and value.ndim and value.shape[0] == allocated and allocated >= count and name in _IN_PROGRESS_FIELDS:
            if _is_written(value[count:]):
                raise AlignmentError(f"currentTrial.{name} has values past iteration {count}, the saved count")
            unit, description = _IN_PROGRESS_FIELDS[name]
            data = value[:count, 0] if value.ndim == 2 and value.shape[1] == 1 else value[:count]
            series.append(Series(name, data, timestamps, unit, description, "in_progress"))
        elif np.size(value) if not isinstance(value, str) else value:
            other[name] = value.item() if isinstance(value, np.ndarray) and value.size == 1 and value.dtype != object else value
    record = dict(other, block=len(blocks), trial=len(records(blocks[-1]["trial"])) + 1, saved_iterations=count)
    return series, record


def in_progress_settings(record):
    """Settings rows describing the trial in progress, one per field."""
    meaning = "The trial running when ViRMEn saved the log. It isn't in the trials table; its per-iteration data is in processing/in_progress_trial."
    return [(f"derived.in_progress_trial.{name}", encode_setting(value), meaning) for name, value in record.items()]
