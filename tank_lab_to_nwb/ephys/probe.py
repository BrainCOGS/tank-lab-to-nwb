"""Per-probe devices, electrode groups and electrodes, written by neuroconv from the SpikeGLX meta file.

Everything here goes through neuroconv's `SpikeGLXRecordingInterface` and `add_recording_metadata_to_nwbfile`, which take
the geometry from probeinterface's reading of the imro table. So a raw AP or LF series that neuroconv adds later (the
same probe, the same interface) finds these rows and reuses them instead of writing a second set. Nothing reads samples:
the interface maps the binary lazily.

neuroconv makes one electrode group per probe shank (`NeuropixelsImec0` for one shank, `NeuropixelsImec0Shank<k>` for a
multi-shank probe), each linked to one Neuropixels device per probe that carries the part and serial numbers.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class ProbeError(ValueError):
    """The probe's meta file and its sorting don't describe the same channels."""


@dataclass(frozen=True)
class ProbeElectrodes:
    """Where one probe's saved AP channels landed in the electrodes table."""

    name: str  # "imec0"
    device_name: str
    serial_number: str
    part_number: str
    rows: np.ndarray  # electrodes table row of each saved AP channel, in saved order
    group_names: np.ndarray  # electrode group of each saved AP channel
    positions: np.ndarray  # (channels, 2) x, y in micrometers, as probeinterface reads the imro table
    shank_ids: np.ndarray


def probe_name(ap_bin):
    match = re.search(r"\.(imec\d+)\.ap\.bin$", Path(ap_bin).name)
    if not match:
        raise ProbeError(f"{ap_bin}: not a SpikeGLX imec AP file name")
    return match[1]


def probe_interface(ap_bin):
    """neuroconv's interface to a probe's AP stream. It memory-maps the binary; nothing is read until data is written."""
    from neuroconv.datainterfaces import SpikeGLXRecordingInterface

    from .spikeglx import isolated_stream_folder

    ap_bin = Path(ap_bin)
    return SpikeGLXRecordingInterface(folder_path=isolated_stream_folder(ap_bin), stream_id=f"{probe_name(ap_bin)}.ap")


def add_probe_electrodes(nwbfile, ap_bin, location="unknown", interface=None):
    """Create, or find, the device, electrode groups and electrodes of one probe, and say which rows they are.

    Pass `interface` to reuse one already made (for the raw AP series). Calling this twice for one probe adds nothing.
    """
    from neuroconv.tools.spikeinterface import add_recording_metadata_to_nwbfile
    from neuroconv.tools.spikeinterface.spikeinterface import _build_channel_id_to_electrodes_table_map

    interface = interface or probe_interface(ap_bin)
    recording = interface.recording_extractor
    metadata = interface.get_metadata()
    for group in metadata["Ecephys"]["ElectrodeGroups"].values():
        group["location"] = location
    # neuroconv fills the electrodes' own location column from this property.
    recording.set_property("brain_area", np.full(recording.get_num_channels(), location))
    add_recording_metadata_to_nwbfile(recording, nwbfile, metadata)
    mapping = _build_channel_id_to_electrodes_table_map(recording, nwbfile)
    channel_ids = recording.get_channel_ids()
    rows = np.array([mapping[channel] for channel in channel_ids], dtype=np.int64)
    probe = recording.get_probe()
    device = next(iter(metadata["Devices"].values()))
    return ProbeElectrodes(
        name=probe_name(ap_bin),
        device_name=device["name"],
        serial_number=str(probe.serial_number or ""),
        part_number=str(probe.model_name or ""),
        rows=rows,
        group_names=np.asarray(recording.get_property("group_name")).astype(str),
        positions=np.asarray(recording.get_channel_locations(), dtype="float64"),
        shank_ids=np.asarray(probe.shank_ids).astype(str),
    )


def check_channel_positions(electrodes, channel_map, channel_positions, tolerance_um=0.5):
    """Compare the sorter's channel map with the meta file's geometry. Returns (max difference, x and y offsets).

    Kilosort's channel_map indexes the saved channels of the binary it read. Its positions come from the chanmap the
    pipeline wrote, which may place the probe differently, by a constant offset per axis; anything else means the
    sorting doesn't belong to this probe or this imro table.
    """
    channel_map = np.asarray(channel_map, dtype=np.int64)
    if channel_map.size and (channel_map.min() < 0 or channel_map.max() >= electrodes.rows.size):
        raise ProbeError(f"{electrodes.name}: channel_map.npy names channels up to {channel_map.max()}, but the meta file saves {electrodes.rows.size} AP channels")
    expected = electrodes.positions[channel_map]
    offset = np.median(np.asarray(channel_positions) - expected, axis=0)
    difference = np.max(np.abs(np.asarray(channel_positions) - expected - offset)) if channel_map.size else 0.0
    if difference > tolerance_um:
        raise ProbeError(f"{electrodes.name}: channel_positions.npy differs from the meta file's geometry by up to {difference:.1f} um beyond a constant offset; is this sorting from another probe or imro table?")
    return float(difference), offset
