"""Units alone, for a sorting that nothing ties to a recording or to behavior. Asked for by name; never a fallback.

A spike's time is its sample over the rate of the binary the sorter read (its meta file's imSampRate), seconds since that
binary's first sample. Electrodes come from the meta file beside that binary. No behavior, no sync file and no other
stream is involved, and the file says so. Use it for sortings of clips whose place in a recording isn't known (S7a/b),
not for a session whose sync file and log exist: those go through `convert_session.convert`.
"""

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np
from pynwb import NWBHDF5IO, NWBFile
from pynwb.file import Subject

from ..virmen.timebase import EASTERN
from ..virmen.verify import RoundTripError
from . import spikeglx as sg
from .convert import (
    LOCATION,
    ProbeUnits,
    _rows_table,
    add_units,
    check_sample_origin,
    probe_spike_times,
    sorting_rows,
)
from .probe import add_probe_electrodes, probe_name, raw_channels
from .sorting import read_sorting

CLOCK = "own: seconds since the first sample of the sorted binary, sample / that binary's imSampRate"
TIED_TO = (
    "Nothing ties these times to a recording or to behavior: no sync file or behavior log was read, and the sorted binary's "
    "place in any longer recording isn't known. Times here can't be compared with any other file's."
)


class UnitsOnlyError(ValueError):
    """A units-only conversion can't find the binary the sorter read, or the caller didn't say what the file is."""


def find_sorted_binary(sorted_probe, given=None):
    """The AP .bin the sorter read: `given`, or the one its own files name (a file, or a folder holding one)."""
    candidate = Path(given) if given is not None else Path(sorted_probe.sorted_input) if sorted_probe.sorted_input != "unknown" else None
    if candidate is not None and candidate.is_dir():
        found = sorted(candidate.glob("*.ap.bin"))
        candidate = found[0] if len(found) == 1 else None
    if candidate is None or not candidate.is_file():
        raise UnitsOnlyError(f"{sorted_probe.folder}: can't find the binary the sorter read ({sorted_probe.sorted_input}); name it with binaries={{probe: path}} (--binary imec0=PATH)")
    return candidate


def build_units_only_nwbfile(sorter_folders, subject_id, session_id, binaries=None, metadata=None):
    """The NWB file of units only, and what was prepared for it."""
    if not subject_id or not session_id:
        raise UnitsOnlyError("a units-only file needs subject_id and session_id from the caller: a sorter folder doesn't say whose or which session it is")
    binaries = binaries or {}
    metadata = metadata or {}
    prepared, clock_rows, starts = [], [], []
    for name, folder in sorter_folders.items():
        sorted_probe = read_sorting(folder)
        binary = find_sorted_binary(sorted_probe, binaries.get(name))
        stream = sg.open_stream(binary)
        if probe_name(binary) != name:
            raise UnitsOnlyError(f"{binary}: is {probe_name(binary)}'s AP file, not {name}'s")
        how = check_sample_origin(sorted_probe, stream)
        starts.append(datetime.fromisoformat(stream.meta["fileCreateTime"]).replace(tzinfo=EASTERN))
        prepared.append((name, sorted_probe, stream, how))
        clock_rows += [
            (f"{name}.sorted_binary", str(binary), "The binary the sorter read; sample 0 of spike_times.npy is its first sample."),
            (f"{name}.rate", repr(stream.rate), "Its meta file's imSampRate, which divides the spike samples."),
            (f"{name}.n_samples", str(stream.n_samples), "Its samples."),
            (f"{name}.meta_first_sample", str(stream.first_sample), "firstSample in its meta file. A clip made by copying the meta keeps the original's, so this doesn't say where the clip starts."),
        ]
    nwbfile = NWBFile(
        session_description=metadata.get("session_description", f"Spike-sorted units only, of {', '.join(sorter_folders)}: no behavior and no recording clock. {TIED_TO}"),
        identifier=str(uuid4()),
        session_start_time=min(starts),
        session_id=session_id,
        experimenter=metadata.get("experimenter", ["Placeholder, Experimenter"]),
        institution=metadata.get("institution", "Placeholder Institution"),
        lab=metadata.get("lab", "Placeholder Lab"),
        experiment_description=metadata.get("experiment_description", "PLACEHOLDER: spike-sorted Neuropixels units without behavior. Experimenter, institution and lab are not real yet."),
        keywords=["extracellular electrophysiology", "Neuropixels", "spike sorting"],
        subject=Subject(subject_id=subject_id, species="Mus musculus", sex="U", age="P90D", description=metadata.get("subject_description", "PLACEHOLDER: sex and age are not real values.")),
    )
    units = []
    for name, sorted_probe, stream, how in prepared:
        electrodes = add_probe_electrodes(nwbfile, stream.bin_path, location=LOCATION)
        channels, channel_source, difference, offset = raw_channels(electrodes, sorted_probe.channel_map, sorted_probe.channel_positions)
        counts = np.array([train.size for train in sorted_probe.spike_samples], dtype=np.int64)
        trimmed = replace(sorted_probe, spike_samples=[], n_spikes_read=int(counts.sum()))
        times_of = lambda folder=sorted_probe.folder, rate=stream.rate: probe_spike_times(folder, lambda samples: samples / rate)  # noqa: E731
        departure = dict(spikes_in_behavior=0, spikes_changing_iteration=0, sample_rate_used=repr(stream.rate))
        units.append((ProbeUnits(name, trimmed, electrodes, times_of, counts, (difference, offset, channel_source), stream.rate, channels[sorted_probe.peak_channel], departure), how))
    add_units(nwbfile, units)
    module = nwbfile.create_processing_module("ecephys", "What sorted these units, and the clock their times are on.")
    session_start = (
        "session_start_time is the earliest fileCreateTime in the sorted binaries' meta files. A clip made by copying the "
        "meta carries the original recording's time, which is not when the clip's first sample was taken."
    )
    module.add(_rows_table("units_only_clock", "The clock of every time in this file: the sorted binary's own.", [
        ("clock", CLOCK, "How spike samples became times."),
        ("tied_to", TIED_TO, "What these times can be compared with."),
        ("session_start_time", min(starts).isoformat(), session_start),
        *clock_rows,
    ]))
    for probe, how in units:
        module.add(_rows_table(f"sorting_{probe.name}", f"What sorted {probe.name}, from which input, and which files were read.", sorting_rows(probe, how)))
    return nwbfile, units


def verify_units_only(units, nwbfile_path):
    """Every stored spike time must be its sample over the sorted binary's rate, recomputed here by a separate path."""
    checked = dict(units=0, spikes=0)
    with h5py.File(nwbfile_path, "r") as f:
        index = f["units/spike_times_index"][:]
        starts = np.concatenate(([0], index[:-1]))
        clusters = f["units/cluster_id"][:]
        row = 0
        for probe, _ in units:
            folder = Path(probe.sorted.folder)
            spike_times = np.load(folder / "spike_times.npy").ravel().astype(np.int64)
            spike_clusters = np.load(folder / "spike_clusters.npy").ravel()
            ids = np.unique(spike_clusters)
            stored_probe = f["units/spike_times"][starts[row] : index[row + ids.size - 1]]
            for i, cluster in enumerate(ids):
                if clusters[row + i] != cluster:
                    raise RoundTripError(f"{probe.name}: row {row + i} holds cluster {clusters[row + i]}, expected {cluster}")
                expected = spike_times[spike_clusters == cluster] / probe.electrodes_rate
                stored = stored_probe[starts[row + i] - starts[row] : index[row + i] - starts[row]]
                if not np.array_equal(stored, expected):
                    raise RoundTripError(f"{probe.name} cluster {cluster}: stored spike times aren't sample / rate")
                checked["spikes"] += int(stored.size)
            row += ids.size
            checked["units"] += int(ids.size)
    return checked


def write_units_only(sorter_folders, nwbfile_path, subject_id, session_id, binaries=None, metadata=None):
    nwbfile, units = build_units_only_nwbfile(sorter_folders, subject_id, session_id, binaries, metadata)
    with NWBHDF5IO(str(nwbfile_path), "w") as io:
        io.write(nwbfile)
    return units, verify_units_only(units, nwbfile_path)
