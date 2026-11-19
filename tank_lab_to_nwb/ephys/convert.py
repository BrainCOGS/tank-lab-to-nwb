"""Write a ViRMEn session and its spike-sorted units to one NWB file, on the behavior clock. Nothing here uploads."""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.common import DynamicTable, DynamicTableRegion, VectorData, VectorIndex
from pynwb import NWBHDF5IO
from pynwb.misc import Units

from ..virmen.convert import KEYWORDS, PLACEHOLDER_METADATA, build_nwbfile
from . import pipeline_sync as ps
from .probe import add_probe_electrodes, check_channel_positions
from .sorting import read_sorting

# Allen's term for the whole brain, the most these files can honestly claim; the pipeline's probe locations aren't used yet.
LOCATION = "root"
COMPRESSION = dict(compression="gzip", compression_opts=4)
TIME_MEANING = (
    "Seconds since session_start_time on the behavior clock: probe sample -> probe seconds (sample / imSampRate) -> sync-stream seconds "
    "through the piecewise-linear map fitted to the 1 Hz sync edges -> ViRMEn elapsed seconds through the line fitted to the iteration edges -> "
    "plus derived.clock_offset. See ephys_sync_alignment."
)


class SampleOriginError(ValueError):
    """The sorted samples can't be shown to start where the raw AP file starts."""


@dataclass(frozen=True)
class ProbeUnits:
    name: str
    sorted: object  # SortedProbe
    electrodes: object  # ProbeElectrodes
    times: np.ndarray  # every unit's spike times in the file, unit after unit
    counts: np.ndarray  # spikes per unit
    position_check: tuple
    electrodes_rate: float  # the AP file's imSampRate: one sample is the units' resolution


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


def prepare_units(nwbfile, sync, sorter_folders):
    """Read each probe's sorting, add its electrodes, and put its spikes on the file's clock. `sorter_folders` maps probe name to folder."""
    prepared = []
    for name, folder in sorter_folders.items():
        if name not in sync.probe_streams:
            raise KeyError(f"{folder}: sorted probe {name} has no raw AP file in the session folder")
        stream = sync.probe_streams[name]
        sorted_probe = read_sorting(folder)
        how = check_sample_origin(sorted_probe, stream)
        electrodes = add_probe_electrodes(nwbfile, stream.bin_path, location=LOCATION)
        position_check = check_channel_positions(electrodes, sorted_probe.channel_map, sorted_probe.channel_positions)
        counts = np.array([train.size for train in sorted_probe.spike_samples], dtype=np.int64)
        times = np.empty(int(counts.sum()), dtype="float64")
        stop = 0
        for train in sorted_probe.spike_samples:
            times[stop : stop + train.size] = sync.clock.probe_samples_to_session(name, train)
            stop += train.size
        # The samples aren't needed once mapped; dropping them halves what a four-probe session holds.
        sorted_probe = replace(sorted_probe, spike_samples=[], n_spikes_read=int(counts.sum()))
        prepared.append((ProbeUnits(name, sorted_probe, electrodes, times, counts, position_check, stream.rate), how))
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
    flat = np.concatenate([probe.times for probe, _ in prepared])
    peak = np.concatenate([probe.sorted.peak_channel for probe, _ in prepared])
    rows = np.concatenate([probe.electrodes.rows[probe.sorted.peak_channel] for probe, _ in prepared])
    groups = [nwbfile.electrode_groups[probe.electrodes.group_names[int(channel)]] for probe, _ in prepared for channel in probe.sorted.peak_channel]
    spike_times = VectorData(name="spike_times", description=TIME_MEANING, data=_compressed(flat))
    electrodes = DynamicTableRegion(name="electrodes", description="The electrode of the unit's peak channel.", data=rows, table=nwbfile.electrodes)
    columns = [
        spike_times,
        VectorIndex(name="spike_times_index", data=np.cumsum(counts), target=spike_times),
        electrodes,
        VectorIndex(name="electrodes_index", data=np.arange(1, counts.size + 1), target=electrodes),
        VectorData(name="electrode_group", description="Electrode group (probe shank) of the peak channel.", data=groups),
        VectorData(name="probe", description="The probe this unit was sorted from: imec<N>, as in the SpikeGLX file names.", data=np.concatenate([np.full(probe.counts.size, probe.name, dtype=object) for probe, _ in prepared])),
        VectorData(name="cluster_id", description="The cluster's id in the sorter's spike_clusters.npy, unique within its probe.", data=np.concatenate([probe.sorted.unit_ids for probe, _ in prepared])),
        VectorData(name="peak_channel", description="Saved AP channel (row of the sorter's channel_map.npy) where the unit's most frequent template has its largest peak-to-peak.", data=peak),
        *[VectorData(name=name, description=description, data=data) for name, description, data in _units_columns(prepared)],
    ]
    nwbfile.units = Units(
        name="units",
        resolution=resolution,
        description=f"Spike-sorted units of every probe, one row per cluster. spike_times: {TIME_MEANING}",
        id=np.arange(counts.size),
        columns=columns,
    )


def alignment_rows(sync):
    clock, fit, stream, pipeline = sync.clock, sync.clock.behavior, sync.sync_stream, sync.pipeline
    first_lag, later_lag = ps.pipeline_index_lag(pipeline.mode, stream.rate)
    n_edges = sum(int(np.sum(np.asarray(e) >= 0)) for e in sync.edges)
    n_iterations = sum(np.asarray(e).size for e in sync.edges)
    rows = [
        ("sync.file", stream.bin_path.name, f"The {stream.kind} file whose digital lines carry ViRMEn's trial and iteration signals and the 1 Hz sync wave."),
        ("sync.rate", repr(stream.rate), f"Its calibrated sampling rate from the meta file, samples per second ({'niSampRate' if stream.kind == 'nidq' else 'obSampRate'})."),
        ("sync.first_sample", str(stream.first_sample), "Its firstSample: the run's sample count when the file began, used only to pair sync edges."),
        ("pipeline.mode", pipeline.mode, "How the pipeline's detection read the iteration line: counter_bit0, bit 0 of ViRMEn's iteration counter, which changes every iteration; or pulse_signal, one pulse per iteration."),
        ("pipeline.regular_sync_status", str(pipeline.regular_sync_status), "1 if the pipeline's regular detection matched every trial's iterations, as BehaviorSync.regular_sync_status."),
        ("pipeline.fixed_sync_status", str(pipeline.fixed_sync_status), "1 if the pipeline's fix path was needed and succeeded, as BehaviorSync.fixed_sync_status. Indices it borrowed from ViRMEn's logged times aren't edges and weren't fit."),
        ("pipeline.index_lag_seconds", repr(later_lag / stream.rate), f"The pipeline stores each later iteration's index {later_lag} samples before the edge it found (the first iteration's {first_lag} sample before), because it adds the trial's start rather than its search window's. This file uses the edges; BehaviorSync's vectors are early by this much."),
        ("fit.method", "imaging.alignment.fit_clock on iteration edges", "A straight line from each matched iteration's logged elapsed time to its edge on the sync stream, each trial's first iteration excluded, rejecting edges further than max(5 x MAD, 5 ms) from the median residual until the kept set stops changing: the rule imaging uses for I2C packets."),
        ("fit.slope", repr(fit.slope), "Sync-stream seconds per ViRMEn elapsed second."),
        ("fit.intercept", repr(fit.intercept), "Sync-stream seconds when ViRMEn's elapsed clock read zero."),
        ("fit.clock_offset", repr(clock.clock_offset), "Seconds added to elapsed time to get this file's time, including derived.session_start_shift_seconds if any."),
        ("fit.n_edges", str(n_edges), "Iteration edges found on the line at the pipeline's matched indices."),
        ("fit.n_not_edges", str(n_iterations - n_edges), "Matched iterations with no edge at their index: borrowed from ViRMEn's times by the pipeline's fix path. Not fit."),
        ("fit.n_first_iteration", str(fit.n_first_iteration), "Edges of a trial's first iteration, not fit."),
        ("fit.n_rejected", str(fit.n_rejected), "Edges rejected as off the line."),
        ("fit.residual_p1_ms", repr(fit.residual_ms["p1"]), "1st percentile of edge residuals."),
        ("fit.residual_p50_ms", repr(fit.residual_ms["p50"]), "Median edge residual."),
        ("fit.residual_p99_ms", repr(fit.residual_ms["p99"]), "99th percentile of edge residuals."),
        ("fit.kept_std_ms", repr(fit.kept_std_ms), "Standard deviation of the fit edges' residuals: how far ViRMEn's output sits from its logged times, not spike-time error."),
        ("fit.send_latency", "not measured", "A constant, not corrected: the delay from ViRMEn logging an iteration's time to its digital output changing. Behavior events run early relative to spikes by it."),
    ]
    for name, sync_fit in clock.probes.items():
        if name == clock.sync_kind:
            continue
        rows += [
            (f"{name}.rate", repr(clock.probe_rates[name]), "Calibrated imSampRate from the AP meta file."),
            (f"{name}.sync_edges", str(sync_fit.n_edges), "Paired 1 Hz sync edges, read from short windows of the AP file's SY channel."),
            (f"{name}.rate_difference_ppm", repr((sync_fit.linear_slope - 1) * 1e6), "Drift of this probe's clock against the sync stream left after the calibrated rates, in parts per million: 1 ppm is 3.6 ms per hour."),
            (f"{name}.line_residual_max_us", repr(sync_fit.linear_residual_us["max"]), "Largest edge residual of the best single line, in microseconds: the wander the piecewise map follows."),
            (f"{name}.map_residual_max_us", repr(sync_fit.residual_us["max"]), "Largest edge residual of the piecewise-linear map, in microseconds. Mostly the sync stream's sampling step."),
            (f"{name}.map_residual_std_us", repr(sync_fit.residual_us["std"]), "Standard deviation of those residuals, in microseconds; a 5 kHz NIDQ step alone gives 58."),
            (f"{name}.map_knots", str(sync_fit.knots.size), "Knots of the piecewise-linear map; see ephys_sync_maps."),
            (f"{name}.coarse_offset_ms", f"{sync_fit.coarse_offset_ms['min']:.3f}..{sync_fit.coarse_offset_ms['max']:.3f}", "Offset between paired edges on the firstSample timeline, before fitting, in ms. Its spread is the drift."),
        ]
    if sync.shift_microseconds:
        rows.append(("derived.session_start_shift_seconds", repr(sync.shift_microseconds / 1e6), "Seconds session_start_time was moved back from ViRMEn's session.start, so that no ephys sample has a negative time."))
    return rows


def _add_ecephys_tables(nwbfile, sync, prepared):
    module = nwbfile.create_processing_module("ecephys", "How spikes were put on the behavior clock, and what sorted them.")
    module.add(_rows_table("ephys_sync_alignment", "How the ephys streams were aligned to behavior, and how far to trust it.", alignment_rows(sync)))
    trial, iteration, edge, index = [], [], [], []
    for t, (edges, indices) in enumerate(zip(sync.edges, sync.pipeline.iteration_idx_vector), start=1):
        edges, indices = np.asarray(edges), np.asarray(indices).ravel()
        trial.append(np.full(edges.size, t))
        iteration.append(np.arange(1, edges.size + 1))
        edge.append(edges)
        index.append(indices)
    trial, iteration, edge, index = (np.concatenate(v).astype(np.int64) for v in (trial, iteration, edge, index))
    edge_time = np.where(edge >= 0, sync.clock.sync_samples_to_session(np.maximum(edge, 0)), np.nan)
    module.add(_table("ephys_iteration_edges", "Each logged iteration the pipeline matched to the sync stream's iteration line.", [
        ("trial", "1-based trial in session order (the trials table's row + 1).", trial),
        ("iteration", "1-based iteration within the trial.", iteration),
        ("pipeline_index", "The sample index the pipeline stores for this iteration (BehaviorSync.sync_data iteration_idx_vector).", index),
        ("edge_sample", "Sample of the sync stream where the line changed for this iteration, or -1 if the pipeline borrowed this index from ViRMEn's times.", edge),
        ("edge_time", "The edge's time in this file, NaN if there's no edge.", edge_time),
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
        s = probe.sorted
        rows = [
            ("folder", str(s.folder), "The sorter output folder."),
            ("sorter", s.sorter, "The sorter, as its files say."),
            ("sorter_version", s.sorter_version, "Its version, or unknown when no file records it."),
            ("sorter_evidence", s.sorter_evidence, "Which files said so."),
            ("sorted_input", s.sorted_input, "The binary the sorter read, as its files name it."),
            ("sample_origin", s.sample_origin, "What sample 0 of spike_times.npy is."),
            ("sample_origin_check", how, "Why that sample 0 is taken as the raw AP file's."),
            ("params_sample_rate", repr(s.sample_rate_params), "params.py sample_rate. Not used: spike times use the meta file's calibrated imSampRate and the sync map."),
            ("n_units", str(s.unit_ids.size), "Clusters written to the units table."),
            ("n_spikes", str(int(probe.counts.sum())), "Spikes written."),
            ("channel_position_offset_um", json.dumps([float(v) for v in probe.position_check[1]]), "Constant offset of channel_positions.npy from the meta file's geometry, x and y; the shapes agree within channel_position_max_difference_um."),
            ("channel_position_max_difference_um", repr(probe.position_check[0]), "Largest difference after that offset."),
        ]
        rows += [(f"file.{name}", text, "A per-cluster table that was read, and its rows.") for name, text in s.files.items()]
        rows += [(f"note.{i}", note, "Something about the tables a reader should know.") for i, note in enumerate(s.notes, start=1)]
        module.add(_rows_table(f"sorting_{probe.name}", f"What sorted {probe.name}, from which input, and which files were read.", rows))


def build_ephys_nwbfile(sync, sorter_folders, metadata=None):
    """The NWB file for behavior with sorted units, ready to write."""
    placeholder = {
        **PLACEHOLDER_METADATA,
        "keywords": [*KEYWORDS, "extracellular electrophysiology", "Neuropixels", "spike sorting"],
        "experiment_description": "PLACEHOLDER: ViRMEn behavior with Neuropixels recordings sorted by Kilosort. Experimenter, institution, lab and brain locations are not real yet.",
        "session_description": f"ViRMEn behavior of {sync.session.lab_meta.get('experiment_name', 'an unnamed experiment')} with spike-sorted Neuropixels units on the behavior clock.",
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
        alignment={key: value for key, value, _ in alignment_rows(sync)},
        probes={probe.name: dict(sorter=probe.sorted.sorter, version=probe.sorted.sorter_version, units=int(probe.sorted.unit_ids.size), spikes=int(probe.counts.sum()), folder=str(probe.sorted.folder), sample_origin_check=how, notes=probe.sorted.notes) for probe, how in prepared},
        coverage=sync.session.coverage.as_dict(),
    )


def write_report(path, report):
    Path(path).write_text(json.dumps(report, indent=1, default=str))
