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
from .alignment import alignment_rows
from .scanimage import SPLIT_SOURCE

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


_INTEGER_COLUMNS = {
    "acquisition_number": "acquisitionNumbers",
    "frame_number_acquisition": "frameNumberAcquisition",
    "end_of_acquisition": "endOfAcquisition",
    "end_of_acquisition_mode": "endOfAcquisitionMode",
    "dc_over_voltage": "dcOverVoltage",
}
_FLOAT_COLUMNS = {
    "acq_trigger_timestamp": "acqTriggerTimestamps_sec",
    "next_file_marker_timestamp": "nextFileMarkerTimestamps_sec",
}


_SPLIT = re.compile(r"_ROI(\d+)_z(\d+)_(\d{5})\.tif$")


def _file_order(path):
    """(depth, ROI, number) for a split file, so files sort as the writer lists them. Plain files sort by name."""
    match = _SPLIT.search(path.name)
    return (int(match[2]), int(match[1]), int(match[3]), path.name) if match else (0, 0, 0, path.name)


def _read_headers(tiff_paths):
    """Every header value the file should hold, parsed here from the raw TIFFs. A split acquisition's frames are those of each depth's lowest ROI."""
    paths = sorted(map(Path, tiff_paths), key=_file_order)
    first_at_depth = {}
    for path in paths:
        depth, roi, _, _ = _file_order(path)
        first_at_depth[depth] = min(first_at_depth.get(depth, roi), roi)
    columns = {name: [] for name in ("file_index", "page_index", "frame_number", "timestamp", *_INTEGER_COLUMNS, *_FLOAT_COLUMNS)}
    packets, epochs, files, file_times = [], set(), [], {}
    for file_index, path in enumerate(paths):
        depth, roi, _, _ = _file_order(path)
        speaks = roi == first_at_depth[depth]
        with tifffile.TiffFile(path) as tif:
            tif.pages.cache = False
            tif.pages.useframes = False
            times = []
            for page_index, page in enumerate(tif.pages):
                text = page.description
                if page_index == 0:
                    tags = page.tags
                    files.append(dict(file_name=path.name, software=tags["Software"].value if "Software" in tags else "", artist=tags["Artist"].value if "Artist" in tags else "", n_frames=0))
                files[-1]["n_frames"] += 1
                times.append(float(_field(text, "frameTimestamps_sec")))
                epochs.add(_field(text, "epoch"))
                if not speaks:
                    continue
                columns["file_index"].append(file_index)
                columns["page_index"].append(page_index)
                number = int(_field(text, "frameNumbers"))
                columns["frame_number"].append(number)
                columns["timestamp"].append(times[-1])
                for name, header in _INTEGER_COLUMNS.items():
                    columns[name].append(int(_field(text, header)))
                for name, header in _FLOAT_COLUMNS.items():
                    columns[name].append(float(_field(text, header)))
                for stamp, data in _PACKET.findall(_field(text, "I2CData")):
                    raw = [int(byte) for byte in re.split(r"[\s,]+", data.strip())]
                    packets.append((number - 1, float(stamp), raw[0] + 256 * raw[1], raw[2] + 256 * raw[3], raw[4] + 256 * raw[5]))
            file_times[file_index] = times
    order = np.argsort(columns["frame_number"], kind="stable")
    packets.sort(key=lambda packet: (packet[0], packet[1]))
    return dict(columns={name: np.array(values)[order] for name, values in columns.items()}, packets=packets, epochs=epochs, files=files, file_times=file_times, paths=paths)


def _check_packets(table, packets):
    stored = np.column_stack([table[name][()] for name in ("frame_index", "timestamp", "block", "trial", "iteration")]) if packets else np.zeros((0, 5))
    expected = np.array(packets, dtype="float64") if packets else np.zeros((0, 5))
    _equal(stored, expected, "I2C packets")


def _alignment_values(table):
    return dict(zip(table["key"].asstr()[()], table["value"].asstr()[()]))


def check_scanimage_tables(scanimage, prepared, headers, extra_alignment_rows=()):
    """Every ScanImage table in processing/ophys against the TIFF headers and the alignment, shared by every view of a session."""
    columns, packets = headers["columns"], headers["packets"]
    numbers = columns["frame_number"]
    frames = scanimage["scanimage_frames"]
    for name, values in columns.items():
        _equal(frames[name][()], values, f"frames.{name}")
    _equal(frames["plane"][()], (numbers - 1) % prepared.acquisition.layout.n_planes, "frame planes")
    _equal(frames["session_time"][()], prepared.frame_times, "frame session times")
    _equal(frames["behavior_block"][()], prepared.labels["block"], "behavior_block")
    _equal(frames["behavior_trial"][()], prepared.labels["trial"], "behavior_trial")
    _equal(frames["behavior_iteration"][()], prepared.labels["iteration"], "behavior_iteration")
    table = scanimage["scanimage_i2c_packets"]
    _check_packets(table, packets)
    _equal(table["elapsed"][()], prepared.packet_elapsed, "packet elapsed times")
    _equal(table["session_time"][()], prepared.packet_times, "packet session times")
    _equal(table["used_in_fit"][()], prepared.fit.used, "used_in_fit")
    files = scanimage["scanimage_files"]
    for name in ("file_name", "software", "artist"):
        _equal(files[name].asstr()[()], np.array([file[name] for file in headers["files"]], dtype=object), f"files.{name}")
    _equal(files["n_frames"][()], [file["n_frames"] for file in headers["files"]], "files.n_frames")
    alignment = scanimage["scanimage_alignment"]
    rows = [*alignment_rows(prepared), *extra_alignment_rows]
    for position, name in enumerate(("key", "value", "meaning")):
        _equal(alignment[name].asstr()[()], np.array([row[position] for row in rows], dtype=object), f"alignment.{name}")
    if headers["epochs"] != {_alignment_values(alignment)["scanimage.epoch"]}:
        raise RoundTripError(f"scanimage.epoch: the TIFFs say {sorted(headers['epochs'])}")


def _check_pixels(file, acquisition, headers, limit):
    """Every frame of every field, and the flyback rows, against the page tifffile reads for it. The rows must use up the page."""
    fields = acquisition.fields
    sets = {field.label: file[f"acquisition/TwoPhotonSeries{field.label}/data"] for field in fields}
    native = acquisition.dtype.newbyteorder("=")
    for field in fields:
        expected = field.frame_indices.size if limit is None else min(field.frame_indices.size, limit)
        if sets[field.label].shape[0] != expected:
            raise RoundTripError(f"{field.label}: {sets[field.label].shape[0]} frames on disk, expected {expected}")
        if sets[field.label].dtype != native:
            raise RoundTripError(f"{field.label}: dtype {sets[field.label].dtype} on disk")
    positions = {label: {} for label in sets}
    for field in fields:
        positions[field.label] = {(int(file_index), int(offset)): position for position, (file_index, offset) in enumerate(zip(field.file_index, field.data_offset))}
    checked = 0
    for file_index, path in enumerate(headers["paths"]):
        here = [field for field in fields if file_index in set(field.file_index.tolist())]
        with tifffile.TiffFile(path) as tif:
            tif.pages.cache = False
            for page in tif.pages:
                pixels = page.asarray()
                key = (file_index, int(page.dataoffsets[0]))
                used = []
                on_page = [field for field in here if key in positions[field.label]]
                for field in on_page:
                    position = positions[field.label][key]
                    used.append(field.rows)
                    if position < sets[field.label].shape[0]:
                        if not np.array_equal(sets[field.label][position], pixels[field.rows[0] : field.rows[1]].T):
                            raise RoundTripError(f"{field.label}, frame {position}: pixels on disk differ from {path.name} page {page.index}")
                        checked += 1
                if on_page:
                    plane = on_page[0].plane
                    gaps = acquisition.layout.flyback_rows.get(plane, ())
                    position = positions[on_page[0].label][key]
                    if gaps and position < sets[on_page[0].label].shape[0]:
                        rows = np.concatenate([pixels[start:stop] for start, stop in gaps], axis=0)
                        if not np.array_equal(file[f"acquisition/FlybackLinesPlane{plane}/data"][position], rows.T):
                            raise RoundTripError(f"plane {plane}, frame {position}: flyback rows on disk differ from {path.name} page {page.index}")
                    covered = sorted([*used, *gaps])
                    if covered[0][0] != 0 or covered[-1][1] != pixels.shape[0] or any(a[1] != b[0] for a, b in zip(covered, covered[1:])):
                        raise RoundTripError(f"{path.name} page {page.index}: the stored rows {covered} don't use up its {pixels.shape[0]} rows")
    return checked


def _expected_field_times(field, headers, acquisition):
    """A field's frame times on the ScanImage clock, from the TIFFs: its own files' for split copies, or from its first row in a stack."""
    if acquisition.source == SPLIT_SOURCE:
        ordered = {}
        for file_index in sorted(set(field.file_index.tolist())):
            ordered[file_index] = headers["file_times"][file_index]
        return np.concatenate([np.array(times) for times in ordered.values()])
    line_period = float(_field(headers["files"][0]["software"], r"SI\.hRoiManager\.linePeriod"))
    return headers["columns"]["timestamp"][field.frame_indices] + field.rows[0] * line_period


def verify_imaging(prepared, nwbfile_path, tiff_paths, max_frames_per_plane=None, extra_alignment_rows=()):
    """Raise RoundTripError unless the file holds exactly what the TIFFs and the log hold. Returns what was checked."""
    verify_round_trip(prepared.session, nwbfile_path)
    headers = _read_headers(tiff_paths)
    numbers, times = headers["columns"]["frame_number"], headers["columns"]["timestamp"]
    if not np.array_equal(numbers, np.arange(1, numbers.size + 1)):
        raise RoundTripError("frame numbers in the TIFFs aren't 1, 2, 3...")
    acquisition = prepared.acquisition
    with h5py.File(nwbfile_path, "r", rdcc_nbytes=_CACHE_BYTES) as file:
        scanimage = file["processing/ophys"]
        check_scanimage_tables(scanimage, prepared, headers, extra_alignment_rows)

        # The timestamps again, from the line as the file itself records it.
        values = _alignment_values(scanimage["scanimage_alignment"])
        slope, intercept, offset = (float(values[f"fit.{name}"]) for name in ("slope", "intercept", "clock_offset"))
        for field in acquisition.fields:
            stored = file[f"acquisition/TwoPhotonSeries{field.label}/timestamps"][()]
            expected = (_expected_field_times(field, headers, acquisition) - intercept) / slope + offset
            if max_frames_per_plane is not None:
                expected = expected[:max_frames_per_plane]
            _equal(stored, expected, f"{field.label} timestamps")
            if stored.size > 1 and np.any(np.diff(stored) <= 0):
                raise RoundTripError(f"{field.label}: timestamps don't always increase")
        for plane in range(acquisition.layout.n_planes):
            if acquisition.layout.flyback_rows.get(plane):
                expected = (times[acquisition.plane_frames(plane)] - intercept) / slope + offset
                if max_frames_per_plane is not None:
                    expected = expected[:max_frames_per_plane]
                _equal(file[f"acquisition/FlybackLinesPlane{plane}/timestamps"][()], expected, f"plane {plane} flyback timestamps")
        for series in prepared.in_progress:
            _check_series(file["processing/behavior"], series)
        checked = _check_pixels(file, acquisition, headers, max_frames_per_plane)
    return dict(frames_checked=checked, frames_in_tiffs=int(numbers.size), packets_checked=len(headers["packets"]))
