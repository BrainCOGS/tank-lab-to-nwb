"""Write a ViRMEn session and its spike-sorted units to one NWB file, both on the ephys recording's clock. Nothing here uploads."""

import json
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.common import DynamicTable, DynamicTableRegion, VectorData, VectorIndex
from hdmf.data_utils import AbstractDataChunkIterator, DataChunk
from pynwb import NWBHDF5IO
from pynwb.misc import Units

from ..virmen.convert import KEYWORDS, PLACEHOLDER_METADATA, build_nwbfile
from . import pipeline_sync as ps
from .probe import add_probe_electrodes, raw_channels
from .sorting import read_sorting

# Allen's term for the whole brain, the most these files can honestly claim; the pipeline's probe locations aren't used yet.
LOCATION = "root"
COMPRESSION = dict(compression="gzip", compression_opts=4)
TIME_MEANING = (
    "Seconds on the recording's clock, from the sync stream's first sample (session_start_time): the spike's sample over the probe's rate, "
    "through the probe's sync map if ephys_sync_alignment's probe_clock is sync_edges. See ephys_sync_alignment."
)


class SampleOriginError(ValueError):
    """The sorted samples can't be shown to start where the raw AP file starts."""


@dataclass(frozen=True)
class ProbeUnits:
    name: str
    sorted: object  # SortedProbe
    electrodes: object  # ProbeElectrodes
    times_of: object  # () -> every unit's spike times in the file, unit after unit, made again from the sorter's files
    counts: np.ndarray  # spikes per unit
    position_check: tuple
    electrodes_rate: float  # the AP file's imSampRate: one sample is the units' resolution
    peak_channel: np.ndarray  # per unit, the saved AP channel of its peak
    departure: dict  # what the probe clock changed: spikes in behavior, and how many land in another iteration than sample / imSampRate puts them


class ProbeSpikeTimes(AbstractDataChunkIterator):
    """Every probe's spike times, unit after unit, made one probe at a time while HDF5 writes them.

    `times_of(i)` makes probe i's times; chunks never cross a probe boundary, so only one probe's times are held.
    """

    def __init__(self, counts, times_of, chunk=1_000_000):
        self._counts = [int(c) for c in counts]
        self._times_of = times_of
        self._chunk = int(chunk)
        self._total = sum(self._counts)
        self.peak_held = 0
        self._chunks = None

    def __iter__(self):
        self._chunks = self._generate()
        return self

    def __next__(self):
        if self._chunks is None:
            self._chunks = self._generate()
        return next(self._chunks)

    def _generate(self):
        start = 0
        for i, count in enumerate(self._counts):
            if not count:
                continue
            times = np.asarray(self._times_of(i), dtype="float64")
            if times.size != count:
                raise ValueError(f"probe {i}: {times.size} spike times for {count} spikes")
            self.peak_held = max(self.peak_held, times.size)
            for offset in range(0, count, self._chunk):
                piece = times[offset : offset + self._chunk]
                yield DataChunk(data=piece, selection=np.s_[start + offset : start + offset + piece.size])
            start += count
            del times

    def __len__(self):
        # hdmf's tables check a column's length against its index before anything is written.
        return self._total

    def recommended_chunk_shape(self):
        return (min(self._chunk, max(self._total, 1)),)

    def recommended_data_shape(self):
        return (self._total,)

    @property
    def chunk_shape(self):
        return self.recommended_chunk_shape()

    @property
    def dtype(self):
        return np.dtype("float64")

    @property
    def maxshape(self):
        return (self._total,)


def _compressed(data):
    return H5DataIO(data=data, **COMPRESSION) if np.size(data) else data


def _table(name, description, columns):
    built = [VectorData(name=column, description=text, data=_compressed(np.asarray(data))) for column, text, data in columns]
    return DynamicTable(name=name, description=description, columns=built, id=np.arange(len(columns[0][2])))


def _rows_table(name, description, rows):
    return _table(name, description, [
        ("key", "Name of the value.", np.array([row[0] for row in rows], dtype=object)),
        ("value", "The value, as text.", np.array([str(row[1]) for row in rows], dtype=object)),
        ("meaning", "What the value means.", np.array([row[2] for row in rows], dtype=object)),
    ])


def check_sample_origin(sorted_probe, stream):
    """Refuse a sorting whose sample 0 can't be shown to be sample 0 of the raw AP file. Returns how it was shown."""
    origin, bounds = sorted_probe.sample_origin, sorted_probe.n_sorted_samples
    if origin.startswith("unknown") and bounds is None:
        raise SampleOriginError(f"{sorted_probe.folder}: nothing says which binary was sorted or how many samples it had, so its spike samples can't be put on the raw file's clock")
    if bounds is not None and not bounds[0] <= stream.n_samples <= bounds[1]:
        told = f"{bounds[0]} samples" if bounds[0] == bounds[1] else f"{bounds[0]} to {bounds[1]} samples"
        raise SampleOriginError(f"{sorted_probe.folder}: the sorter read {told}, but {stream.bin_path.name} has {stream.n_samples}. A trimmed or concatenated input moves sample 0, and nothing here says by how much.")
    last = max((int(train[-1]) for train in sorted_probe.spike_samples if train.size), default=0)
    if last >= stream.n_samples:
        raise SampleOriginError(f"{sorted_probe.folder}: a spike at sample {last} is past the end of {stream.bin_path.name} ({stream.n_samples} samples)")
    if bounds is None:
        return f"{origin}; nothing records how many samples were sorted"
    exact = "exactly" if bounds[0] == bounds[1] else f"within one batch ({bounds[0]} to {bounds[1]})"
    return f"{origin}; the sorter read {exact} as many samples as {stream.bin_path.name} has ({stream.n_samples})"


def probe_spike_times(folder, to_seconds):
    """A sorter folder's spike times in the file, unit after unit in cluster-id order, each unit's in file order."""
    folder = Path(folder)
    spike_times = np.load(folder / "spike_times.npy").ravel()
    name = "spike_clusters.npy" if (folder / "spike_clusters.npy").is_file() else "spike_templates.npy"
    order = np.argsort(np.load(folder / name).ravel(), kind="stable")
    return to_seconds(spike_times[order].astype(np.int64))


def prepare_units(nwbfile, sync, sorter_folders):
    """Read each probe's sorting, add its electrodes, and put its spikes on the file's clock. `sorter_folders` maps probe name to folder."""
    prepared = []
    iterations = np.concatenate(sync.iteration_seconds)
    for name, folder in sorter_folders.items():
        if name not in sync.probe_streams:
            raise KeyError(f"{folder}: sorted probe {name} has no raw AP file in the session folder")
        stream = sync.probe_streams[name]
        sorted_probe = read_sorting(folder)
        how = check_sample_origin(sorted_probe, stream)
        electrodes = add_probe_electrodes(nwbfile, stream.bin_path, location=LOCATION)
        channels, channel_source, difference, offset = raw_channels(electrodes, sorted_probe.channel_map, sorted_probe.channel_positions)
        position_check = (difference, offset, channel_source)
        peak = channels[sorted_probe.peak_channel]
        counts = np.array([train.size for train in sorted_probe.spike_samples], dtype=np.int64)
        # Under "params_rate" a spike is its sample over params.py's sample_rate, as the DB's CuratedClustering stores it.
        rate = sorted_probe.sample_rate_params if sync.clock.probe_clock == "params_rate" else None
        changed = within = 0
        for train in sorted_probe.spike_samples:
            times = sync.clock.probe_samples_to_session(name, train, rate=rate)
            plain = train / stream.rate + sync.clock.shift
            inside = (plain >= iterations[0]) & (plain < iterations[-1])
            changed += int(np.count_nonzero(np.searchsorted(iterations, times[inside], side="right") != np.searchsorted(iterations, plain[inside], side="right")))
            within += int(inside.sum())
        # Neither samples nor times are kept: the writer makes each probe's times again when HDF5 asks for them.
        sorted_probe = replace(sorted_probe, spike_samples=[], n_spikes_read=int(counts.sum()))
        departure = dict(spikes_in_behavior=within, spikes_changing_iteration=changed, sample_rate_used=repr(rate or stream.rate))
        times_of = partial(probe_spike_times, sorted_probe.folder, lambda samples, name=name, rate=rate: sync.clock.probe_samples_to_session(name, samples, rate=rate))
        prepared.append((ProbeUnits(name, sorted_probe, electrodes, times_of, counts, position_check, stream.rate, peak, departure), how))
    return prepared


def _units_columns(prepared):
    """Columns of the units table across probes; a column one probe lacks is NaN or empty there."""
    names, descriptions, kinds = [], {}, {}
    for probe, _ in prepared:
        for column in probe.sorted.columns:
            if column.name not in descriptions:
                names.append(column.name)
                descriptions[column.name] = column.description
                kinds[column.name] = "f" if np.asarray(column.data).dtype.kind in "fiu" else "U"
    values = {name: [] for name in names}
    for probe, _ in prepared:
        own = {column.name: column.data for column in probe.sorted.columns}
        n = probe.sorted.unit_ids.size
        for name in names:
            if name in own:
                values[name].append(np.asarray(own[name], dtype="float64") if kinds[name] == "f" else np.asarray(own[name]).astype(str))
            else:
                values[name].append(np.full(n, np.nan) if kinds[name] == "f" else np.full(n, "", dtype=object).astype(str))
    return [(name, descriptions[name], np.concatenate(values[name]) if kinds[name] == "f" else np.concatenate(values[name]).astype(object)) for name in names]


def add_units(nwbfile, prepared):
    """One units table for every probe, built from whole columns so a long session's spikes stay numpy arrays."""
    resolution = 1 / min(probe.electrodes_rate for probe, _ in prepared) if prepared else -1.0
    counts = np.concatenate([probe.counts for probe, _ in prepared])
    flat = ProbeSpikeTimes([int(probe.counts.sum()) for probe, _ in prepared], lambda i: prepared[i][0].times_of())
    peak = np.concatenate([probe.peak_channel for probe, _ in prepared])
    rows = np.concatenate([probe.electrodes.rows[probe.peak_channel] for probe, _ in prepared])
    groups = [nwbfile.electrode_groups[probe.electrodes.group_names[int(channel)]] for probe, _ in prepared for channel in probe.peak_channel]
    spike_times = VectorData(name="spike_times", description=TIME_MEANING, data=H5DataIO(data=flat, **COMPRESSION))
    electrodes = DynamicTableRegion(name="electrodes", description="The electrode of the unit's peak channel.", data=rows, table=nwbfile.electrodes)
    # The index goes first: hdmf leaves a column written from an iterator out of its length check, and would then
    # not find the index's target.
    columns = [
        VectorIndex(name="spike_times_index", data=np.cumsum(counts), target=spike_times),
        spike_times,
        electrodes,
        VectorIndex(name="electrodes_index", data=np.arange(1, counts.size + 1), target=electrodes),
        VectorData(name="electrode_group", description="Electrode group (probe shank) of the peak channel.", data=groups),
        VectorData(name="probe", description="The probe this unit was sorted from: imec<N>, as in the SpikeGLX file names.", data=np.concatenate([np.full(probe.counts.size, probe.name, dtype=object) for probe, _ in prepared])),
        VectorData(name="cluster_id", description="The cluster's id in the sorter's spike_clusters.npy, unique within its probe.", data=np.concatenate([probe.sorted.unit_ids for probe, _ in prepared])),
        VectorData(name="peak_channel", description="Saved AP channel of the AP file (0-383) where the unit's most frequent template has its largest peak-to-peak. See sorting_<probe> channel_source for how sorter channels were matched to saved ones.", data=peak),
        *[VectorData(name=name, description=description, data=data) for name, description, data in _units_columns(prepared)],
    ]
    nwbfile.units = Units(
        name="units",
        resolution=resolution,
        description=f"Spike-sorted units of every probe, one row per cluster. spike_times: {TIME_MEANING}",
        id=np.arange(counts.size),
        columns=columns,
    )


RULE = (
    "The lab's rule (U19 BehaviorSync; SYNC_DATA_PLAN.md): the file's clock is the recording's, t = 0 at the sync stream's first sample. "
    "Each logged ViRMEn iteration takes the time of its own pulse, iteration_idx_vector / rate, one for one, and the behavior's frames are "
    "those times. Nothing is fitted to ViRMEn's clock. Spikes are their sample over the probe's rate."
)
PROBE_CLOCK_MEANING = {
    "sync_edges": "Departure from the rule: probe seconds (sample / imSampRate) go through a piecewise-linear map fitted to the 1 Hz sync edges, which removes the drift between the probe's and the sync stream's crystals that the calibrated rates leave. The rule itself is meta_rates.",
    "meta_rates": "The rule: sample / imSampRate, the calibrated rate in the AP meta file, with no drift correction.",
    "params_rate": "sample / params.py sample_rate (the nominal 30000), what the DB's CuratedClustering stores; neither calibrated nor drift-corrected.",
}
ITERATION_TIME_MEANING = {
    "pipeline_index": "The rule: the pipeline's stored index over the rate. In counter_bit0 mode that index is pipeline.index_lag_seconds before the line's edge.",
    "edge": "Departure from the rule: the line's edge itself, half a sample before its first high sample. An index with no edge there keeps the index plus the trial's lag.",
}


def alignment_rows(sync, prepared=()):
    clock, stream, pipeline = sync.clock, sync.sync_stream, sync.pipeline
    first_lag, later_lag = ps.pipeline_index_lag(pipeline.mode, stream.rate)
    n_edges = sum(int(np.sum(np.asarray(e) >= 0)) for e in sync.edges)
    n_iterations = sum(np.asarray(e).size for e in sync.edges)
    seconds = np.concatenate(sync.iteration_seconds)
    rows = [
        ("rule", "pulse per iteration, recording clock", RULE),
        ("probe_clock", clock.probe_clock, PROBE_CLOCK_MEANING[clock.probe_clock]),
        ("iteration_time", sync.iteration_time, ITERATION_TIME_MEANING[sync.iteration_time]),
        ("vectors", sync.vectors, {"regular": "The pipeline's regular pulse matching (BehaviorSync.regular_sync_status 1).", "fixed": "The pipeline's fix path (fixed_sync_status 1): a few missed pulses were filled in from ViRMEn's logged times; see n_not_edges.", "virmen_borrowed": "LOWER ACCURACY, converted only because it was asked for: the pipeline's pulse matching failed, so each iteration is its trial's start pulse plus ViRMEn's logged time (iteration_idx_vector_from_virmen), with no drift correction within a trial."}[sync.vectors]),
        ("sync.file", stream.bin_path.name, f"The {stream.kind} file whose digital lines carry ViRMEn's trial and iteration signals and the 1 Hz sync wave. Its first sample is t = 0."),
        ("sync.rate", repr(stream.rate), f"Its calibrated rate from the meta file ({'niSampRate' if stream.kind == 'nidq' else 'obSampRate'}), which divides the indices. The DB stores it rounded to a float, which the pipeline didn't compute with."),
        ("sync.file_create_time", stream.meta.get("fileCreateTime", ""), "The sync file's fileCreateTime on the rig's clock: session_start_time, to the second."),
        ("pipeline.mode", pipeline.mode, "How the pipeline read the iteration line: counter_bit0, bit 0 of ViRMEn's iteration counter, which changes every iteration; or pulse_signal, one pulse per iteration."),
        ("pipeline.regular_sync_status", str(pipeline.regular_sync_status), "As BehaviorSync.regular_sync_status, recomputed from the raw file."),
        ("pipeline.fixed_sync_status", str(pipeline.fixed_sync_status), "As BehaviorSync.fixed_sync_status, recomputed from the raw file."),
        ("pipeline.index_lag_seconds", repr(later_lag / stream.rate), f"The pipeline stores each later iteration's index {later_lag} samples before the line's edge (the first iteration's {first_lag} before), because it adds the trial's start rather than its search window's."),
        ("iterations", str(n_iterations), "Logged iterations given a time, every one of the saved trials."),
        ("iterations.with_edge", str(n_edges), "Of those, iterations whose index sits at an edge of the line."),
        ("iterations.first_time", repr(float(seconds[0])), "The first iteration's time in this file: when behavior began in the recording."),
        ("iterations.last_time", repr(float(seconds[-1])), "The last iteration's time."),
        ("between_iterations", "interpolated", "A time that isn't an iteration's, such as a trial's stop or a block's start, is placed linearly between the iterations around it, or keeps its distance from the first or last one outside them (VirmenDataInterface's rule). Trials the pipeline didn't sync, or with a different iteration count, stop the conversion."),
    ]
    for name, sync_fit in clock.probes.items():
        if name == clock.sync_kind:
            continue
        x = np.array([0.0, sync.probe_streams[name].n_samples / 2, sync.probe_streams[name].n_samples]) / clock.probe_rates[name]
        correction = (sync_fit(x) - x) * 1e3
        rows += [
            (f"{name}.rate", repr(clock.probe_rates[name]), "Calibrated imSampRate from the AP meta file."),
            (f"{name}.sync_edges", str(sync_fit.n_edges), "Paired 1 Hz sync edges, read from short windows of the AP file's SY channel."),
            (f"{name}.sync_correction_ms", f"start {correction[0]:.3f}, middle {correction[1]:.3f}, end {correction[2]:.3f}", "How far the sync map moves this probe's samples from sample / imSampRate, at the AP file's start, middle and end, in ms. Applied only if probe_clock is sync_edges."),
            (f"{name}.rate_difference_ppm", repr((sync_fit.linear_slope - 1) * 1e6), "Drift of this probe's clock against the sync stream left after the calibrated rates, in parts per million: 1 ppm is 3.6 ms per hour."),
            (f"{name}.line_residual_max_us", repr(sync_fit.linear_residual_us["max"]), "Largest edge residual of the best single line, in microseconds: the wander the piecewise map follows."),
            (f"{name}.map_residual_max_us", repr(sync_fit.residual_us["max"]), "Largest edge residual of the piecewise-linear map, in microseconds. Mostly the sync stream's sampling step."),
            (f"{name}.map_residual_std_us", repr(sync_fit.residual_us["std"]), "Standard deviation of those residuals, in microseconds; a 5 kHz NIDQ step alone gives 58."),
            (f"{name}.map_knots", str(sync_fit.knots.size), "Knots of the piecewise-linear map; see ephys_sync_maps."),
        ]
    for probe, _ in prepared:
        d = probe.departure
        fraction = d["spikes_changing_iteration"] / d["spikes_in_behavior"] if d["spikes_in_behavior"] else 0.0
        rows += [
            (f"{probe.name}.spikes_in_behavior", str(d["spikes_in_behavior"]), "Spikes between the first and last iteration, by sample / imSampRate."),
            (f"{probe.name}.spikes_changing_iteration", f"{d['spikes_changing_iteration']} ({fraction:.2%})", "Of those, spikes this file's probe_clock puts in another iteration than sample / imSampRate does: the size of the departure, 0 under meta_rates."),
        ]
    if sync.shift_microseconds:
        rows.append(("shift_seconds", repr(sync.shift_microseconds / 1e6), "Seconds added to every time so that nothing is negative: a probe mapped onto the sync stream started before it. session_start_time is earlier by the same amount."))
    return rows


def sorting_rows(probe, how):
    """(key, value, meaning) rows saying what sorted one probe, from which input, and which files were read or left out."""
    s = probe.sorted
    rows = [
        ("folder", str(s.folder), "The sorter output folder."),
        ("sorter", s.sorter, "The sorter, as its files say."),
        ("sorter_version", s.sorter_version, "Its version, or unknown when no file records it."),
        ("sorter_evidence", s.sorter_evidence, "Which files said so."),
        ("sorted_input", s.sorted_input, "The binary the sorter read, as its files name it."),
        ("sample_origin", s.sample_origin, "What sample 0 of spike_times.npy is."),
        ("sample_origin_check", how, "Why that sample 0 is taken as the raw AP file's."),
        ("params_sample_rate", repr(s.sample_rate_params), "params.py sample_rate, the nominal rate the sorter was told; the DB's CuratedClustering divides by it. Used for spike times only under probe_clock params_rate."),
        ("spike_time_rate", probe.departure["sample_rate_used"], "The rate this file divided spike samples by, before any sync map."),
        ("n_units", str(s.unit_ids.size), "Clusters written to the units table."),
        ("n_spikes", str(int(probe.counts.sum())), "Spikes written."),
        ("channel_position_offset_um", json.dumps([float(v) for v in probe.position_check[1]]), "Constant offset of channel_positions.npy from the meta file's geometry, x and y; the shapes agree within channel_position_max_difference_um."),
        ("channel_position_max_difference_um", repr(probe.position_check[0]), "Largest difference after that offset."),
        ("channel_source", probe.position_check[2], "How each sorter channel was matched to a saved AP channel."),
    ]
    rows += [(f"file.{name}", text, "A per-cluster table that was read, and its rows.") for name, text in s.files.items()]
    rows += [(f"note.{i}", note, "Something about the tables a reader should know.") for i, note in enumerate(s.notes, start=1)]
    rows += [(f"ignored.{i}", name, "A file in the sorter folder this converter didn't read.") for i, name in enumerate(s.ignored, start=1)]
    return rows


def _add_ecephys_tables(nwbfile, sync, prepared):
    module = nwbfile.create_processing_module("ecephys", "How behavior and spikes were put on the recording's clock, and what sorted the spikes.")
    module.add(_rows_table("ephys_sync_alignment", "How the behavior and the ephys streams were put on one clock, and how far to trust it.", alignment_rows(sync, prepared)))
    trial, iteration, edge, index = [], [], [], []
    for t, (edges, indices) in enumerate(zip(sync.edges, sync.iteration_idx_vector), start=1):
        edges, indices = np.asarray(edges), np.asarray(indices).ravel()
        trial.append(np.full(edges.size, t))
        iteration.append(np.arange(1, edges.size + 1))
        edge.append(edges)
        index.append(indices)
    trial, iteration, edge, index = (np.concatenate(v).astype(np.int64) for v in (trial, iteration, edge, index))
    edge_time = np.where(edge >= 0, (edge - 0.5) / sync.clock.sync_rate + sync.clock.shift, np.nan)
    module.add(_table("ephys_iteration_edges", "Each logged iteration, the pipeline's sample for it, the line's edge, and the time this file gives it.", [
        ("trial", "1-based trial in session order (the trials table's row + 1).", trial),
        ("iteration", "1-based iteration within the trial.", iteration),
        ("pipeline_index", "The sample index the pipeline stores for this iteration (BehaviorSync.sync_data iteration_idx_vector).", index),
        ("edge_sample", "Sample of the sync stream where the line changed for this iteration, or -1 if the pipeline borrowed this index from ViRMEn's times.", edge),
        ("edge_time", "The edge's time in this file (half a sample before its first high sample), NaN if there's no edge.", edge_time),
        ("time", "The iteration's time in this file, by ephys_sync_alignment's iteration_time: the behavior frames' timestamps.", np.concatenate(sync.iteration_seconds)),
    ]))
    knot_probe, knot_seconds, knot_values = [], [], []
    for name, fit in sync.clock.probes.items():
        if name == sync.clock.sync_kind:
            continue
        knot_probe += [name] * fit.knots.size
        knot_seconds.append(fit.knots)
        knot_values.append(fit.values)
    if knot_probe:
        module.add(_table("ephys_sync_maps", "Knots of each probe's piecewise-linear map from its seconds (sample / imSampRate) to sync-stream seconds; linear between knots, and along the end segments outside them.", [
            ("probe", "Probe.", np.array(knot_probe, dtype=object)),
            ("probe_seconds", "Knot on the probe's clock.", np.concatenate(knot_seconds)),
            ("sync_seconds", "Its value on the sync stream's clock.", np.concatenate(knot_values)),
        ]))
    for probe, how in prepared:
        module.add(_rows_table(f"sorting_{probe.name}", f"What sorted {probe.name}, from which input, and which files were read.", sorting_rows(probe, how)))


def build_ephys_nwbfile(sync, sorter_folders, metadata=None):
    """The NWB file for behavior with sorted units, ready to write."""
    placeholder = {
        **PLACEHOLDER_METADATA,
        "keywords": [*KEYWORDS, "extracellular electrophysiology", "Neuropixels", "spike sorting"],
        "experiment_description": "PLACEHOLDER: ViRMEn behavior with Neuropixels recordings sorted by Kilosort, on the recording clock. Experimenter, institution, lab and brain locations are not real yet.",
        "session_description": f"ViRMEn behavior of {sync.session.lab_meta.get('experiment_name', 'an unnamed experiment')} with spike-sorted Neuropixels units, both on the ephys recording's clock.",
    }
    nwbfile = build_nwbfile(sync.session, {**placeholder, **(metadata or {})})
    prepared = prepare_units(nwbfile, sync, sorter_folders)
    add_units(nwbfile, prepared)
    _add_ecephys_tables(nwbfile, sync, prepared)
    return nwbfile, prepared


def write_ephys_nwbfile(sync, sorter_folders, nwbfile_path, metadata=None):
    nwbfile, prepared = build_ephys_nwbfile(sync, sorter_folders, metadata)
    with NWBHDF5IO(str(nwbfile_path), "w") as io:
        io.write(nwbfile)
    return prepared


def conversion_report(sync, prepared):
    return dict(
        alignment={key: value for key, value, _ in alignment_rows(sync, prepared)},
        probes={probe.name: dict(sorter=probe.sorted.sorter, version=probe.sorted.sorter_version, units=int(probe.sorted.unit_ids.size), spikes=int(probe.counts.sum()), folder=str(probe.sorted.folder), sample_origin_check=how, notes=probe.sorted.notes) for probe, how in prepared},
        coverage=sync.session.coverage.as_dict(),
    )


def write_report(path, report):
    Path(path).write_text(json.dumps(report, indent=1, default=str))
