"""Check an ephys NWB file against its sources, reading the raw HDF5 and rereading the sorter files by a separate path.

The writer splits spike trains with an argsort and maps them per unit. This rereads `spike_times.npy` and
`spike_clusters.npy`, selects each cluster with a mask, maps the samples through the clock again, and compares the
stored times exactly. Electrode rows, unit labels and the behavior session are checked too.
"""

from pathlib import Path

import h5py
import numpy as np

from ..virmen.verify import RoundTripError, verify_round_trip


def _strings(values):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values])


def _check_iterations(f, sync):
    """Every behavior frame's timestamp must be its iteration's time, recomputed here from the vectors and the rate."""
    expected = np.concatenate([np.asarray(v, dtype=np.int64).ravel() / sync.sync_stream.rate for v in sync.iteration_idx_vector]) + sync.clock.shift
    if sync.iteration_time == "edge":
        expected = np.concatenate(sync.iteration_seconds)
    stored = f["processing/ecephys/ephys_iteration_edges/time"][:]
    if not np.array_equal(stored, expected):
        raise RoundTripError("ephys_iteration_edges time differs from the pipeline vectors over the rate")
    frames = None
    for item in f["processing/behavior"].values():
        if isinstance(item, h5py.Group) and "timestamps" in item and item["timestamps"].shape == expected.shape:
            frames = item["timestamps"][:]
            break
    if frames is None or not np.array_equal(frames, expected):
        raise RoundTripError("the behavior frames' timestamps aren't the iteration times")


def verify_ephys(sync, prepared, nwbfile_path):
    """Raise RoundTripError at the first stored value that differs from the sources. Returns counts checked."""
    verify_round_trip(sync.session, nwbfile_path)
    checked = dict(units=0, spikes=0)
    with h5py.File(nwbfile_path, "r") as f:
        _check_iterations(f, sync)
        checked["iterations"] = int(sum(np.size(t) for t in sync.iteration_seconds))
        units = f["units"]
        index = units["spike_times_index"][:]
        times = units["spike_times"]  # read a probe's slice at a time
        probes = _strings(units["probe"][:])
        clusters = units["cluster_id"][:]
        electrodes = units["electrodes"][:]
        starts = np.concatenate(([0], index[:-1]))
        row = 0
        for probe, _ in prepared:
            folder = Path(probe.sorted.folder)
            spike_times = np.load(folder / "spike_times.npy").ravel().astype(np.int64)
            name = "spike_clusters.npy" if (folder / "spike_clusters.npy").is_file() else "spike_templates.npy"
            spike_clusters = np.load(folder / name).ravel().astype(np.int64)
            ids = np.unique(spike_clusters)
            first, last = (starts[row], index[row + ids.size - 1]) if ids.size else (0, 0)
            stored_probe = times[first:last]
            if not np.array_equal(clusters[row : row + ids.size], ids) or set(probes[row : row + ids.size]) != {probe.name}:
                raise RoundTripError(f"units of {probe.name}: rows {row}..{row + ids.size} don't hold its clusters {ids[:5]}...")
            rate = probe.sorted.sample_rate_params if sync.clock.probe_clock == "params_rate" else None
            for i, cluster in enumerate(ids):
                expected = sync.clock.probe_samples_to_session(probe.name, spike_times[spike_clusters == cluster], rate=rate)
                stored = stored_probe[starts[row + i] - first : index[row + i] - first]
                if stored.shape != expected.shape or not np.array_equal(stored, expected):
                    raise RoundTripError(f"{probe.name} cluster {cluster}: stored spike times differ from the sorter's samples on the clock")
                channel = int(probe.peak_channel[i])
                if int(electrodes[row + i]) != int(probe.electrodes.rows[channel]):
                    raise RoundTripError(f"{probe.name} cluster {cluster}: electrode row {electrodes[row + i]}, expected {probe.electrodes.rows[channel]}")
                checked["spikes"] += int(stored.size)
            for column in probe.sorted.columns:
                stored = units[column.name][row : row + ids.size]
                expected = np.asarray(column.data)
                if expected.dtype.kind in "fiu":
                    same = np.array_equal(stored.astype("float64"), expected.astype("float64"), equal_nan=True)
                else:
                    same = np.array_equal(_strings(stored), expected.astype(str))
                if not same:
                    raise RoundTripError(f"{probe.name}: units column {column.name} differs from the sorter's files")
            row += ids.size
            checked["units"] += int(ids.size)
        if row != index.size:
            raise RoundTripError(f"the units table has {index.size} rows, the sorter folders {row} clusters")
        n_electrodes = sum(probe.electrodes.rows.size for probe, _ in prepared)
        if f["general/extracellular_ephys/electrodes/id"].shape[0] < n_electrodes:
            raise RoundTripError("fewer electrode rows than saved AP channels")
    return checked
