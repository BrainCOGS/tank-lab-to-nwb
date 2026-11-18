"""Write a normalized session to NWB, then check what's on disk."""

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.common import DynamicTableRegion, VectorData, VectorIndex
from ndx_tank_metadata import (
    AnimalSettings,
    LabMetaDataExtension,
    MazeExtension,
    RigExtension,
    StimulusPairs,
    VirmenEpochs,
    VirmenSettings,
    VirmenTrials,
)
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import Position, SpatialSeries
from pynwb.file import Subject

from .loader import load_log
from .session import normalize_log
from .verify import verify_round_trip

REFERENCE_FRAME = "Virtual maze coordinates: y runs along the track and x across it."
# Clearly fake until these come from DataJoint inside the pipeline.
KEYWORDS = ["ViRMEn", "virtual reality", "mouse behavior"]

PLACEHOLDER_METADATA = dict(
    keywords=KEYWORDS,
    experimenter=["Placeholder, Experimenter"],
    institution="Placeholder Institution",
    lab="Placeholder Lab",
    experiment_description="PLACEHOLDER: ViRMEn behavior session. Experimenter, institution and lab are not real yet.",
    subject=dict(species="Mus musculus", sex="U", age="P90D", description="PLACEHOLDER: sex and age are not real values."),
)


def _table_columns(columns, target_table=None):
    built = []
    for column in columns:
        if column.region:
            built.append(DynamicTableRegion(name=column.name, description=column.description, data=column.data, table=target_table))
            continue
        flat = VectorData(name=column.name, description=column.description, data=column.data)
        built.append(flat)
        if column.index is not None:
            inner = VectorIndex(name=f"{column.name}_index", data=column.index, target=flat)
            built.append(inner)
            if column.index_of_index is not None:
                built.append(VectorIndex(name=f"{column.name}_index_index", data=column.index_of_index, target=inner))
    return built


def _settings_table(session):
    rows = session.settings
    columns = [
        VectorData(name=name, description=description, data=np.array([row[position] for row in rows], dtype=object))
        for position, (name, description) in enumerate(
            [("key", "ViRMEn log path of the value."), ("value", "The recorded value, as text."), ("meaning", "What the value means.")]
        )
    ]
    return VirmenSettings(name="settings", description="ViRMEn settings of unknown or experiment-defined meaning.", columns=columns)


def _compressed(data):
    return H5DataIO(data=data, compression="gzip") if np.size(data) else data


def _add_series(nwbfile, session):
    behavior = nwbfile.create_processing_module("behavior", "ViRMEn behavior, per frame.")
    clocks = {}
    for item in session.series:
        timing = {"timestamps": clocks[item.clock]} if item.clock in clocks else {"timestamps": _compressed(item.timestamps)}
        common = dict(name=item.name, data=_compressed(item.data), unit=item.unit, description=item.description, **timing)
        if item.spatial:
            built = SpatialSeries(reference_frame=REFERENCE_FRAME, **common)
            behavior.add(Position(name=item.container, spatial_series=built))
        else:
            built = TimeSeries(**common)
            behavior.add(built)
        clocks.setdefault(item.clock, built)


def build_nwbfile(session, metadata=None):
    metadata = {**PLACEHOLDER_METADATA, **(metadata or {})}
    subject = Subject(subject_id=session.subject_id, **metadata["subject"])
    nwbfile = NWBFile(
        session_description=metadata.get("session_description", f"ViRMEn behavior session of {session.lab_meta.get('experiment_name', 'an unnamed experiment')}."),
        identifier=str(uuid4()),
        session_start_time=session.session_start_time,
        session_id=session.session_id,
        experimenter=metadata["experimenter"],
        institution=metadata["institution"],
        lab=metadata["lab"],
        experiment_description=metadata["experiment_description"],
        keywords=metadata["keywords"],
        subject=subject,
    )
    add_session_to_nwbfile(nwbfile, session)
    return nwbfile


def add_session_to_nwbfile(nwbfile, session):
    """Add the lab metadata, trials, epochs and behavior series of a normalized session."""
    optional = {}
    if session.animal:
        optional["animal"] = AnimalSettings(name="animal", **session.animal)
    pairs = None
    if session.stimulus_pairs:
        pairs = StimulusPairs(name="stimulus_pairs", description="Stimulus pairs per block.", columns=_table_columns(session.stimulus_pairs))
        optional["stimulus_pairs"] = pairs
    mazes = MazeExtension(name="mazes", description="Mazes configured for the session.", columns=_table_columns(session.mazes))
    lab_meta = LabMetaDataExtension(
        name="LabMetaData", rig=RigExtension(name="rig", **session.rig), mazes=mazes, settings=_settings_table(session), **optional, **session.lab_meta
    )
    # The stimulus pairs must be attached before the trials table that references them.
    nwbfile.add_lab_meta_data(lab_meta)
    nwbfile.trials = VirmenTrials(name="trials", description="ViRMEn trials.", columns=_table_columns(session.trials, pairs))
    nwbfile.epochs = VirmenEpochs(name="epochs", description="ViRMEn blocks.", columns=_table_columns(session.epochs))
    _add_series(nwbfile, session)


def write_session(session, nwbfile_path, metadata=None):
    with NWBHDF5IO(str(nwbfile_path), "w") as io:
        io.write(build_nwbfile(session, metadata))
    return Path(nwbfile_path)


def convert_log(log_path, nwbfile_path, metadata=None):
    """Convert one log, verify the file against the normalized values, and write its coverage report beside it."""
    return write_and_verify(normalize_log(load_log(log_path), file=log_path), nwbfile_path, metadata)


def write_and_verify(session, nwbfile_path, metadata=None):
    """Write a normalized session, verify the file against it, and write its coverage report beside it."""
    write_session(session, nwbfile_path, metadata)
    verify_round_trip(session, nwbfile_path)
    report = Path(nwbfile_path).with_suffix(".coverage.json")
    report.write_text(json.dumps(session.coverage.as_dict(), indent=1))
    return session
