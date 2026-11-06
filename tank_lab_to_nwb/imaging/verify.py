"""Check an imaging NWB file against its sources, reading the raw HDF5 and rereading the TIFFs by a separate path.

The writer reads pixels straight from file offsets using its own header parser. This uses tifffile's page reader and its
own small parsing, so a bug in one shouldn't hide in the other.
"""

import re
from pathlib import Path

import h5py
import numpy as np
import tifffile

from ..virmen.verify import RoundTripError, _check_series, verify_round_trip

_CACHE_BYTES = 256_000_000
_FIELD = r"^{} = (.*)$"
_PACKET = re.compile(r"\{\s*([-+0-9.eE]+)\s*,\s*\[([^\]]*)\]\s*\}")


def _field(text, name):
    return re.search(_FIELD.format(name), text, re.MULTILINE).group(1).strip()


def _equal(stored, expected, label):
    stored, expected = np.asarray(stored), np.asarray(expected)
    if stored.shape != expected.shape or not np.array_equal(stored, expected):
        difference = "" if stored.shape != expected.shape else f", first at row {int(np.argmax(stored.ravel() != expected.ravel()))}"
        raise RoundTripError(f"{label}: on disk differs from the source{difference}")


def _read_sources(tiff_paths):
    """Frame numbers, times and packets, parsed here from the raw descriptions."""
    numbers, times, packets = [], [], []
    for path in sorted(map(Path, tiff_paths), key=lambda p: p.name):
        with tifffile.TiffFile(path) as tif:
            tif.pages.cache = False
            tif.pages.useframes = False
            for page in tif.pages:
                text = page.description
                numbers.append(int(_field(text, "frameNumbers")))
                times.append(float(_field(text, "frameTimestamps_sec")))
                for stamp, data in _PACKET.findall(_field(text, "I2CData")):
                    raw = [int(byte) for byte in re.split(r"[\s,]+", data.strip())]
                    packets.append((numbers[-1] - 1, float(stamp), raw[0] + 256 * raw[1], raw[2] + 256 * raw[3], raw[4] + 256 * raw[5]))
    return np.array(numbers), np.array(times), packets


def _check_packets(table, packets):
    stored = np.column_stack([table[name][()] for name in ("frame_index", "timestamp", "block", "trial", "iteration")]) if packets else np.zeros((0, 5))
    expected = np.array(packets, dtype="float64") if packets else np.zeros((0, 5))
    _equal(stored, expected, "I2C packets")


def _alignment_values(table):
    return dict(zip(table["key"].asstr()[()], table["value"].asstr()[()]))


def _check_pixels(file, tiff_paths, numbers, n_planes, limit):
    """Every frame of every plane, against the page tifffile reads for that frame number, in order."""
    positions = np.zeros(n_planes, dtype=int)
    sets = [file[f"acquisition/TwoPhotonSeriesPlane{plane}/data"] for plane in range(n_planes)]
    for plane, data in enumerate(sets):
        expected = int(np.sum((numbers - 1) % n_planes == plane))
        expected = expected if limit is None else min(expected, limit)
        if data.shape[0] != expected:
            raise RoundTripError(f"plane {plane}: {data.shape[0]} frames on disk, expected {expected}")
        if data.dtype != np.int16:
            raise RoundTripError(f"plane {plane}: dtype {data.dtype} on disk")
    checked = 0
    index = 0
    for path in sorted(map(Path, tiff_paths), key=lambda p: p.name):
        with tifffile.TiffFile(path) as tif:
            tif.pages.cache = False
            for page in tif.pages:
                plane = int((numbers[index] - 1) % n_planes)
                index += 1
                position = positions[plane]
                positions[plane] += 1
                if position >= sets[plane].shape[0]:
                    continue
                if not np.array_equal(sets[plane][position], page.asarray().T):
                    raise RoundTripError(f"plane {plane}, frame {position}: pixels on disk differ from {path.name} page {page.index}")
                checked += 1
    return checked


def verify_imaging(prepared, nwbfile_path, tiff_paths, max_frames_per_plane=None):
    """Raise RoundTripError unless the file holds exactly what the TIFFs and the log hold. Returns what was checked."""
    verify_round_trip(prepared.session, nwbfile_path)
    numbers, times, packets = _read_sources(tiff_paths)
    if not np.array_equal(numbers, np.arange(1, numbers.size + 1)):
        raise RoundTripError("frame numbers in the TIFFs aren't 1, 2, 3...")
    n_planes = prepared.acquisition.layout.n_planes
    with h5py.File(nwbfile_path, "r", rdcc_nbytes=_CACHE_BYTES) as file:
        scanimage = file["processing/scanimage"]
        frames = scanimage["frames"]
        _equal(frames["frame_number"][()], numbers, "frame numbers")
        _equal(frames["timestamp"][()], times, "frame timestamps")
        _equal(frames["plane"][()], (numbers - 1) % n_planes, "frame planes")
        _equal(frames["session_time"][()], prepared.frame_times, "frame session times")
        _equal(frames["behavior_block"][()], prepared.labels["block"], "behavior_block")
        _equal(frames["behavior_trial"][()], prepared.labels["trial"], "behavior_trial")
        _equal(frames["behavior_iteration"][()], prepared.labels["iteration"], "behavior_iteration")
        _check_packets(scanimage["i2c_packets"], packets)
        _equal(scanimage["i2c_packets/elapsed"][()], prepared.packet_elapsed, "packet elapsed times")
        _equal(scanimage["i2c_packets/session_time"][()], prepared.packet_times, "packet session times")

        # The timestamps again, from the line as the file itself records it.
        values = _alignment_values(scanimage["alignment"])
        slope, intercept, offset = (float(values[f"fit.{name}"]) for name in ("slope", "intercept", "clock_offset"))
        recomputed = (times - intercept) / slope + offset
        for plane in range(n_planes):
            stored = file[f"acquisition/TwoPhotonSeriesPlane{plane}/timestamps"][()]
            expected = recomputed[(numbers - 1) % n_planes == plane]
            if max_frames_per_plane is not None:
                expected = expected[:max_frames_per_plane]
            _equal(stored, expected, f"plane {plane} timestamps")
            if stored.size > 1 and np.any(np.diff(stored) <= 0):
                raise RoundTripError(f"plane {plane}: timestamps don't always increase")
        for series in prepared.in_progress:
            _check_series(file["processing/in_progress_trial"], series)
        checked = _check_pixels(file, tiff_paths, numbers, n_planes, max_frames_per_plane)
    return dict(frames_checked=checked, frames_in_tiffs=int(numbers.size), packets_checked=len(packets))
