"""Check an NWB file against the normalized session, bit for bit, reading the raw HDF5 rather than through pynwb."""

from datetime import datetime

import h5py
import numpy as np


class RoundTripError(ValueError):
    """What's on disk differs from what was normalized."""


def _same_numeric(stored, expected):
    return stored.dtype == expected.dtype and stored.shape == expected.shape and stored.tobytes() == expected.tobytes()


def _check_dataset(dataset, expected, label):
    if expected.dtype == object:
        stored = dataset.asstr()[()] if dataset.size else np.empty(0, dtype=object)
        if list(np.asarray(stored).ravel()) != list(expected.ravel()):
            raise RoundTripError(f"{label}: text on disk differs")
        return
    stored = dataset[()]
    if not _same_numeric(np.asarray(stored), expected):
        raise RoundTripError(f"{label}: dtype {stored.dtype} {stored.shape} on disk, expected {expected.dtype} {expected.shape}, or the bits differ")


def _check_table(group, columns, table_name):
    for column in columns:
        label = f"{table_name}.{column.name}"
        if column.name not in group:
            raise RoundTripError(f"{label}: missing from the file")
        if column.region:
            if not np.array_equal(group[column.name][()], column.data):
                raise RoundTripError(f"{label}: region differs")
            continue
        _check_dataset(group[column.name], column.data, label)
        # hdmf stores an index in the smallest unsigned type that fits, so compare the values.
        for suffix, index in (("_index", column.index), ("_index_index", column.index_of_index)):
            if index is not None and not np.array_equal(group[column.name + suffix][()], index):
                raise RoundTripError(f"{label}: {suffix} on disk differs")


def _check_attributes(group, attributes, label):
    for name, expected in attributes.items():
        stored = group.attrs.get(name)
        full = f"{label}.{name}"
        if stored is None:
            raise RoundTripError(f"{full}: missing from the file")
        if isinstance(expected, datetime):
            if datetime.fromisoformat(str(stored)) != expected or datetime.fromisoformat(str(stored)).utcoffset() != expected.utcoffset():
                raise RoundTripError(f"{full}: {stored} on disk, expected {expected.isoformat()}")
        elif isinstance(expected, str):
            if str(stored) != expected:
                raise RoundTripError(f"{full}: {stored!r} on disk, expected {expected!r}")
        elif isinstance(expected, bool):
            if bool(stored) != expected:
                raise RoundTripError(f"{full}: {stored!r} on disk, expected {expected!r}")
        elif not _same_numeric(np.asarray(stored), np.asarray(expected)):
            raise RoundTripError(f"{full}: {stored!r} on disk, expected {expected!r}")


def _series_group(behavior, series):
    return behavior[series.container][series.name] if series.container else behavior[series.name]


def _check_series(behavior, series):
    group = _series_group(behavior, series)
    _check_dataset(group["data"], series.data, f"series {series.name}")
    _check_dataset(group["timestamps"], series.timestamps, f"series {series.name} timestamps")
    if group["data"].attrs["unit"] != series.unit:
        raise RoundTripError(f"series {series.name}: unit {group['data'].attrs['unit']} on disk, expected {series.unit}")


def verify_round_trip(session, nwbfile_path):
    """Raise RoundTripError unless the file holds exactly the session's normalized values."""
    with h5py.File(nwbfile_path, "r") as file:
        _check_table(file["intervals/trials"], session.trials, "trials")
        _check_table(file["intervals/epochs"], session.epochs, "epochs")
        lab = file["general/LabMetaData"]
        _check_attributes(lab, session.lab_meta, "lab metadata")
        _check_attributes(lab["rig"], session.rig, "rig")
        if session.animal:
            _check_attributes(lab["animal"], session.animal, "animal")
        _check_table(lab["mazes"], session.mazes, "mazes")
        if session.stimulus_pairs:
            _check_table(lab["stimulus_pairs"], session.stimulus_pairs, "stimulus pairs")
        rows = list(zip(*(lab["settings"][name].asstr()[()] for name in ("key", "value", "meaning")))) if session.settings else []
        if [tuple(map(str, row)) for row in rows] != [tuple(row) for row in session.settings]:
            raise RoundTripError("settings: rows on disk differ")
        for series in session.series:
            _check_series(file["processing/behavior"], series)
        if file["general/subject/subject_id"][()].decode() != session.subject_id:
            raise RoundTripError("subject_id differs")
        if file["general/session_id"][()].decode() != session.session_id:
            raise RoundTripError("session_id differs")
        stored_start = datetime.fromisoformat(file["session_start_time"][()].decode())
        if stored_start != session.session_start_time or stored_start.utcoffset() != session.session_start_time.utcoffset():
            raise RoundTripError("session_start_time differs")
