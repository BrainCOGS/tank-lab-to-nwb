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


def raw_channels(electrodes, channel_map, channel_positions, tolerance_um=0.5):
    """(saved AP channel of each sorter channel, how it was found, max difference, x and y offset).

    Kilosort 4 writes `channel_map.npy` as saved channel numbers of the file it read. MATLAB Kilosort 2/3 (rezToPhy,
    rezToPhy2) writes it as 0..N-1, indices into its own whitened `temp_wh.dat`, which holds only the chanmap's connected
    channels: on S2 the reference channel 127 is left out, so from index 127 on the indices are one channel off.
    `channel_positions.npy` is right in both. So the channel map is used when its positions match the meta file's
    geometry up to one constant offset per axis (the lab's chanmaps place the probe 27 um over in x); otherwise, when channel_map.npy is exactly 0..N-1 (MATLAB's), every
    sorter channel is placed on the one contact its position names, under one offset that places all of them.
    """
    channel_map = np.asarray(channel_map, dtype=np.int64)
    positions = np.asarray(channel_positions, dtype="float64")
    geometry = electrodes.positions
    if channel_map.size and channel_map.min() >= 0 and channel_map.max() < geometry.shape[0]:
        offset = np.median(positions - geometry[channel_map], axis=0)
        difference = np.max(np.abs(positions - geometry[channel_map] - offset)) if channel_map.size else 0.0
        if difference <= tolerance_um:
            return channel_map, "channel_map.npy", float(difference), offset
    if not np.array_equal(channel_map, np.arange(channel_map.size)):
        raise ProbeError(f"{electrodes.name}: channel_positions.npy differs from the meta file's geometry beyond a constant offset; is this sorting from another probe or imro table?")
    for candidate in np.unique(np.round(positions[0] - geometry, 3), axis=0):
        shifted = positions - candidate
        distance = np.abs(shifted[:, None, :] - geometry[None, :, :]).max(axis=2)
        nearest = distance.argmin(axis=1)
        if distance[np.arange(nearest.size), nearest].max() <= tolerance_um and np.unique(nearest).size == nearest.size:
            difference = float(distance[np.arange(nearest.size), nearest].max())
            return nearest.astype(np.int64), "channel_positions.npy: channel_map.npy indexes the sorter's own whitened file, not the saved channels", difference, candidate
    raise ProbeError(f"{electrodes.name}: channel_positions.npy doesn't match the meta file's geometry under any one offset; is this sorting from another probe or imro table?")


def check_channel_positions(electrodes, channel_map, channel_positions, tolerance_um=0.5):
    """(max difference, x and y offsets) of the sorter's positions from the meta geometry; ProbeError if they don't match."""
    channel_map = np.asarray(channel_map, dtype=np.int64)
    if channel_map.size and (channel_map.min() < 0 or channel_map.max() >= electrodes.rows.size):
        raise ProbeError(f"{electrodes.name}: channel_map.npy names channels up to {channel_map.max()}, but the meta file saves {electrodes.rows.size} AP channels")
    _, _, difference, offset = raw_channels(electrodes, channel_map, channel_positions, tolerance_um)
    return difference, offset
