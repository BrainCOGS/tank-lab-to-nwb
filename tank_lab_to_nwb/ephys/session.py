"""Read one ephys session's sync files and behavior log, and build its clock. No DB is needed. See docs/ephys-timing.md.

Libraries do the reading where they read correctly:

- The NIDQ or OneBox digital word comes from SpikeInterface's SpikeGLX reader (neo), which returns the saved int16 word
  as is. `read_sync_word` checks it against a direct read of the same channel on the first and last million samples.
- A probe's SY word comes from SpikeInterface's `<stream>-SYNC` stream, read in short windows.
- The geometry comes from neuroconv and probeinterface (`probe.py`).

The hand-written `spikeglx.py` is kept for the meta checks and for that cross-check, so a disagreement with the DB can't
come from the reader unnoticed.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..imaging.alignment import AlignmentError, session_start_shift, shifted_session
from ..virmen.loader import load_log, records
from ..virmen.session import normalize_log
from . import pipeline_sync as ps
from . import spikeglx as sg
from .clock import EphysClock, fit_behavior, identity_sync, iteration_edges, iteration_samples_for_fit, probe_sync

# Pipeline rule (BehaviorSync.make): before this date the iteration count is a digital word, which U19 no longer supports.
DIGITAL_WORD_SYNC_BEFORE = "2021-06-01"
CROSS_CHECK_SAMPLES = 1_000_000


class SessionSyncError(ValueError):
    """The session's sync can't be built, for a reason stated in the message."""


def saved_trials(log):
    """Each saved trial's (start, per-iteration times) on the elapsed clock, in session order, as the pipeline fetches them."""
    out = []
    for block in records(log["block"]):
        for trial in records(block["trial"]):
            out.append((float(np.asarray(trial["start"]).item()), np.asarray(trial["time"], dtype="float64").ravel()))
    return out


def behavior_time_vector(trials):
    """The trials' times as the pipeline's `behavior_time`: an object array of (iterations, 1) float arrays."""
    out = np.empty(len(trials), dtype=object)
    for i, (_, times) in enumerate(trials):
        out[i] = np.asarray(times, dtype=np.float32).reshape(-1, 1)
    return out


def _neo_stream_id(stream):
    return "nidq" if stream.kind == "nidq" else re.sub(r"^.*\.(obx\d+)\.obx\.bin$", r"\1", stream.bin_path.name)



def read_sync_word(stream, word=0):
    """The digital word of a NIDQ or OneBox file, through SpikeInterface, checked against a direct read."""
    import spikeinterface.extractors as se

    channel = sg.digital_channel(stream, word)
    try:
        recording = se.read_spikeglx(sg.isolated_stream_folder(stream.bin_path), stream_id=_neo_stream_id(stream))
    except Exception as error:  # noqa: BLE001
        raise sg.SpikeGLXError(f"{stream.bin_path}: SpikeInterface can't read the {stream.kind} stream: {error}") from error
    # By name: neo puts a OneBox's SY word in a stream of its own, so positions shift.
    name = f"{_neo_stream_id(stream)}#XD{word}"
    if name not in set(recording.get_channel_ids()) or recording.get_num_samples() != stream.n_samples:
        raise sg.SpikeGLXError(f"{stream.bin_path}: SpikeInterface has no channel {name} or sees {recording.get_num_samples()} samples; the meta says {stream.n_samples}")
    traces = _read_column(recording, name, stream.n_samples)
    if traces.dtype != np.int16:
        raise sg.SpikeGLXError(f"{stream.bin_path}: SpikeInterface returned the digital word as {traces.dtype}, not int16")
    for start in (0, max(0, stream.n_samples - CROSS_CHECK_SAMPLES)):
        direct = sg.read_channel(stream, channel, start, start + CROSS_CHECK_SAMPLES)
        if not np.array_equal(direct, traces[start : start + direct.size]):
            raise sg.SpikeGLXError(f"{stream.bin_path}: SpikeInterface's digital word differs from the file's bytes near sample {start}")
    return traces


def read_probe_sync_edges(ap_bin, windows):
    """Rising edges of bit 6 of a probe's SY word, through SpikeInterface's SYNC stream, in the given sample windows."""
    import spikeinterface.extractors as se

    name = Path(ap_bin).name
    probe = re.search(r"\.(imec\d+)\.ap\.bin$", name)[1]
    recording = se.read_spikeglx(sg.isolated_stream_folder(ap_bin), stream_id=f"{probe}.ap-SYNC")
    edges = []
    for start, stop in windows:
        word = recording.get_traces(start_frame=start, end_frame=stop, return_in_uV=False).ravel()
        edges.append(sg.rising_edges(sg.bit(word, sg.SY_SYNC_BIT)) + start)
    return np.unique(np.concatenate(edges)).astype(np.int64)


@dataclass(frozen=True)
class SessionSync:
    """A session's clock and everything that shows how it was found."""

    clock: EphysClock
    session: object  # the normalized behavior session, on the same clock (shifted if needed)
    sync_stream: sg.Stream
    probe_streams: dict  # name -> Stream of the raw AP file
    pipeline: ps.PipelineSync
    edges: list  # per trial, the edge sample of each iteration, -1 if not measured
    trials: list
    probe_edges: dict = field(default_factory=dict)  # name -> (probe edge samples, sync edge samples) paired
    shift_microseconds: int = 0
    lines: tuple = ()  # (trial line, iteration line) as the pipeline reads them

    @property
    def lag_seconds(self):
        first, later = ps.pipeline_index_lag(self.pipeline.mode, self.sync_stream.rate)
        return first / self.sync_stream.rate, later / self.sync_stream.rate


def build_session_sync(session_folder, log_path, probes=None, sync_bin=None, every_seconds=30.0, session_date=None, file=None):
    """Build the clock of one session from its raw folder and behavior log. `probes` limits the probes synced.

    `log_path` is a path or an already decoded log; `file` names the log in the NWB file when it's decoded.
    """
    session_folder = Path(session_folder)
    if isinstance(log_path, (str, Path)):
        file = file or Path(log_path)
        log = load_log(log_path)
    else:
        log, log_path = log_path, file or "log"
    session = normalize_log(log, file=file)
    if session_date is not None and str(session_date) < DIGITAL_WORD_SYNC_BEFORE:
        raise SessionSyncError(f"{session_folder}: recorded {session_date}, before {DIGITAL_WORD_SYNC_BEFORE}, when the sync used a digital-word frame counter. U19 refuses these sessions too ('Old sessions < 2022 not suported anymore'), and so does this converter.")
    offsets = {round(float(offset), 9) for offset in session.coverage.clock["offsets"]}
    if len(offsets) != 1:
        raise SessionSyncError(f"{log_path}: ViRMEn's clock restarted ({len(offsets)} clock runs), and one line can't span that")
    clock_offset = float(session.coverage.clock["offsets"][0])
    sync_stream = sg.open_stream(sync_bin or sg.find_sync_stream(session_folder))
    word = read_sync_word(sync_stream)
    trial_line, iteration_line = ps.pipeline_lines(word, sync_stream.kind, sync_stream.rate, sync_stream.meta["fileTimeSecs"])
    # A OneBox's sync wave is in its SY word, read later; the digital word isn't needed after this.
    sync_edges = _nidq_sync_edges(sync_stream, word) if sync_stream.kind == "nidq" else None
    del word
    trials = saved_trials(log)
    if not trials:
        raise SessionSyncError(f"{log_path}: no saved trials")
    try:
        result = ps.pipeline_sync(trial_line, iteration_line, sync_stream.rate, behavior_time_vector(trials))
    except (IndexError, ValueError) as error:
        raise SessionSyncError(f"{sync_stream.bin_path}: the pipeline's iteration detection failed on these lines ({type(error).__name__}: {error}); are the trial and iteration lines wired as the pipeline expects?") from error
    if not (result.regular_sync_status == 1 or result.fixed_sync_status == 1):
        raise SessionSyncError(f"{sync_stream.bin_path}: the pipeline's sync fails on this session (regular {result.regular_sync_status}, fixed {result.fixed_sync_status}): the iteration pulses don't match the log's iterations. Only the ViRMEn-borrowed vectors exist, which assume no drift and aren't a measurement, so this converter stops.")
    edges = iteration_edges(result.iteration_idx_vector, trial_line, iteration_line, result.mode, sync_stream.rate)
    if sync_edges is None:
        sync_edges = _obx_sync_edges(sync_stream)
    samples, elapsed, numbers = iteration_samples_for_fit(edges, trials)
    try:
        behavior = fit_behavior(samples, elapsed, numbers, sync_stream.rate)
    except AlignmentError as error:
        raise SessionSyncError(f"{sync_stream.bin_path}: {error}") from error
    probe_files = sg.find_ap_files(session_folder)
    if probes is not None:
        missing = sorted(set(probes) - {f"imec{n}" for n in probe_files})
        if missing:
            raise SessionSyncError(f"{session_folder}: no raw AP file for {', '.join(missing)}")
        probe_files = {n: p for n, p in probe_files.items() if f"imec{n}" in probes}
    fits, rates, streams, paired = {sync_stream.kind: identity_sync(sync_edges.size)}, {sync_stream.kind: sync_stream.rate}, {}, {}
    for number, ap_bin in probe_files.items():
        name = f"imec{number}"
        stream = sg.open_stream(ap_bin)
        probe_edges = read_probe_sync_edges(ap_bin, sg.sampling_windows(stream, every_seconds=every_seconds))
        fit = probe_sync(probe_edges, stream.rate, stream.first_sample, sync_edges, sync_stream.rate, sync_stream.first_sample)
        fits[name], rates[name], streams[name] = fit, stream.rate, stream
        paired[name] = probe_edges
    clock = EphysClock(sync_stream.kind, sync_stream.rate, behavior, clock_offset, fits, rates)
    # Nothing in the file may have a negative time: move session_start_time back to the earliest file's first sample.
    starts = [clock.sync_samples_to_session([0])[0]] + [clock.probe_samples_to_session(name, [0])[0] for name in streams]
    shift = session_start_shift(np.array(starts))
    if shift:
        session = shifted_session(session, shift)
        clock = clock.shifted(shift / 1e6)
    return SessionSync(clock, session, sync_stream, streams, result, edges, trials, paired, shift, (trial_line, iteration_line))


def _nidq_sync_edges(stream, word):
    """The NIDQ's sync edges, from the digital word already read, on the line its meta names."""
    channel, line = sg.sync_line_of_nidq(stream)
    if channel != sg.digital_channel(stream, 0):
        word = read_sync_word(stream, word=channel - sg.digital_channel(stream, 0))
    return sg.rising_edges(sg.bit(word, line)).astype(np.int64)


def _obx_sync_edges(stream):
    """A OneBox's own sync edges: bit 6 of its SY word, through SpikeInterface's SYNC stream, checked against a direct read."""
    import spikeinterface.extractors as se

    channel = sg.sync_channel(stream)
    recording = se.read_spikeglx(sg.isolated_stream_folder(stream.bin_path), stream_id=f"{_neo_stream_id(stream)}-SYNC")
    word = _read_column(recording, recording.get_channel_ids()[0], stream.n_samples)
    for start in (0, max(0, stream.n_samples - CROSS_CHECK_SAMPLES)):
        direct = sg.read_channel(stream, channel, start, start + CROSS_CHECK_SAMPLES)
        if not np.array_equal(direct, word[start : start + direct.size]):
            raise sg.SpikeGLXError(f"{stream.bin_path}: SpikeInterface's SY word differs from the file's bytes near sample {start}")
    return sg.rising_edges(sg.bit(word, sg.SY_SYNC_BIT)).astype(np.int64)


def _read_column(recording, channel_id, n_samples, chunk=2_000_000):
    """One channel of a recording, a chunk at a time: neo reads every channel of a block before picking one."""
    out = None
    for start in range(0, n_samples, chunk):
        piece = recording.get_traces(start_frame=start, end_frame=min(start + chunk, n_samples), channel_ids=[channel_id], return_in_uV=False).ravel()
        if out is None:
            out = np.empty(n_samples, dtype=piece.dtype)
        out[start : start + piece.size] = piece
    return out


def compare_with_db(pipeline, row):
    """Exact-match counts between this converter's pipeline vectors and a BehaviorSync row's `sync_data`."""
    data = row["sync_data"]
    report = dict(regular_sync_status=(pipeline.regular_sync_status, int(row["regular_sync_status"])), fixed_sync_status=(pipeline.fixed_sync_status, int(row["fixed_sync_status"])))
    for name in ("trial_idx_vector", "trial_idx_vector_from_virmen"):
        ours, theirs = np.asarray(getattr(pipeline, name)), np.asarray(data[name])
        report[name] = dict(ours=int(ours.size), db=int(theirs.size), equal=bool(ours.shape == theirs.shape and np.array_equal(ours, theirs)))
    for name in ("iteration_idx_vector", "iteration_idx_vector_from_virmen"):
        ours, theirs = getattr(pipeline, name), data[name]
        n = min(len(ours), len(theirs))
        same_trials = sum(1 for i in range(n) if np.array_equal(np.asarray(ours[i]).ravel(), np.asarray(theirs[i]).ravel()))
        values = sum(int(min(np.size(ours[i]), np.size(theirs[i]))) for i in range(n))
        equal_values = sum(int(np.sum(np.asarray(ours[i]).ravel()[: min(np.size(ours[i]), np.size(theirs[i]))] == np.asarray(theirs[i]).ravel()[: min(np.size(ours[i]), np.size(theirs[i]))])) for i in range(n))
        report[name] = dict(trials_ours=len(ours), trials_db=len(theirs), trials_identical=same_trials, values_compared=values, values_equal=equal_values)
    return report

