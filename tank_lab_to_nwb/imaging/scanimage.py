"""Read ScanImage TIFFs: every per-frame header field, the I2C packets from ViRMEn, and how pages map to planes.

Anything this can't account for is an error rather than a silent drop: an unknown header field, a used aux trigger,
a missing frame, or a layout it can't map to planes.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

_FILE_NAME = re.compile(r"^(?P<stem>.*)_(?P<number>\d{5})\.tif$")
_PACKET = re.compile(r"\{([0-9.eE+-]+)\s*,\s*\[([^\]]*)\]\}")
_PACKET_BYTES = 6  # three little-endian uint16: block, trial, iteration
_INTEGER_FIELDS = {
    "frameNumbers": "frame_number",
    "acquisitionNumbers": "acquisition_number",
    "frameNumberAcquisition": "frame_number_acquisition",
    "endOfAcquisition": "end_of_acquisition",
    "endOfAcquisitionMode": "end_of_acquisition_mode",
    "dcOverVoltage": "dc_over_voltage",
}
_FLOAT_FIELDS = {
    "frameTimestamps_sec": "timestamp",
    "acqTriggerTimestamps_sec": "acq_trigger_timestamp",
    "nextFileMarkerTimestamps_sec": "next_file_marker_timestamp",
}
_AUX_FIELDS = ("auxTrigger0", "auxTrigger1", "auxTrigger2", "auxTrigger3")
_KNOWN_FIELDS = {*_INTEGER_FIELDS, *_FLOAT_FIELDS, *_AUX_FIELDS, "epoch", "I2CData"}


class ScanImageError(ValueError):
    """These TIFFs can't be read without losing or misplacing something."""


@dataclass(frozen=True)
class SourceFile:
    path: Path
    n_frames: int
    software: str
    artist: str

    @property
    def name(self):
        return self.path.name


@dataclass(frozen=True)
class Layout:
    """How an acquisition's pages are laid out. frame_shape is (rows, columns); the sizes are (x, y)."""

    n_planes: int
    zs_um: tuple
    frame_shape: tuple
    frame_rate: float
    frame_period: float
    volume_rate: float
    line_period: float
    field_of_view_um: tuple
    pixel_size_um: tuple
    version: str
    frames_per_file: float  # logFramesPerFile, inf when the acquisition isn't split
    n_volumes: float  # the volumes requested, which a stopped acquisition falls short of


@dataclass(frozen=True)
class Acquisition:
    files: list
    frames: dict  # column name -> array with one entry per page
    packets: dict  # column name -> array with one entry per I2C packet
    epoch: str
    layout: Layout
    dtype: np.dtype

    def plane_frames(self, plane):
        """The frame indices of one plane, in time order. A partial last volume gives planes of different lengths."""
        if not 0 <= plane < self.layout.n_planes:
            raise IndexError(f"plane {plane} is outside the {self.layout.n_planes}-plane stack")
        return np.arange(self.frames["frame_number"].size)[plane :: self.layout.n_planes]

    @property
    def planes(self):
        return (self.frames["frame_number"] - 1) % self.layout.n_planes


def _header_values(software):
    values = {}
    for line in software.splitlines():
        key, separator, value = line.partition(" = ")
        if separator:
            values[key.strip()] = value.strip()
    return values


def _required(values, name, where):
    if f"SI.{name}" not in values:
        raise ScanImageError(f"{where}: no {name} in the ScanImage header, so the planes can't be mapped")
    return values[f"SI.{name}"]


def _number(values, name, where):
    text = _required(values, name, where)
    try:
        return float(text)
    except ValueError as error:
        raise ScanImageError(f"{where}: {name} is {text!r}, not a number") from error


def _rois(artist, where):
    try:
        group = json.loads(artist)["RoiGroups"]["imagingRoiGroup"]["rois"]
    except (ValueError, KeyError, TypeError) as error:
        raise ScanImageError(f"{where}: can't read the ROI group from the Artist tag") from error
    return group if isinstance(group, list) else [group]


def _layout(software, artist, where):
    """Read the layout, refusing anything other than whole frames of one channel, one per plane, with one ROI."""
    values = _header_values(software)
    channels = _required(values, "hChannels.channelSave", where)
    if channels != "1":
        raise ScanImageError(f"{where}: channelSave is {channels}; only one saved channel, channel 1, is supported")
    for name in ("hScan2D.logAverageFactor", "hStackManager.framesPerSlice"):
        if _number(values, name, where) != 1:
            raise ScanImageError(f"{where}: {name} is {_required(values, name, where)}; only 1 is supported")
    n_planes = int(_number(values, "hStackManager.actualNumSlices", where))
    if int(_number(values, "hStackManager.numFramesPerVolumeWithFlyback", where)) != n_planes:
        raise ScanImageError(f"{where}: flyback frames are saved in each volume, and they aren't handled yet")
    if _required(values, "hFastZ.discardFlybackFrames", where) not in ("false", "0") or _number(values, "hFastZ.numDiscardFlybackFrames", where) != 0:
        raise ScanImageError(f"{where}: flyback frames are discarded, and that isn't handled yet")
    zs = tuple(float(z) for z in _required(values, "hStackManager.zs", where).strip("[]").replace(";", " ").split())
    if len(zs) != n_planes:
        raise ScanImageError(f"{where}: hStackManager.zs lists {len(zs)} depths for {n_planes} planes")
    rois = _rois(artist, where)
    if len(rois) != 1:
        raise ScanImageError(f"{where}: {len(rois)} ROIs in one page; multi-ROI data isn't supported yet")
    scanfield = rois[0]["scanfields"]
    scanfield = scanfield[0] if isinstance(scanfield, list) else scanfield
    columns, rows = (int(v) for v in scanfield["pixelResolutionXY"])
    if rows != int(_number(values, "hRoiManager.linesPerFrame", where)):
        raise ScanImageError(f"{where}: hRoiManager.linesPerFrame disagrees with the ROI's {rows} lines")
    if columns != int(_number(values, "hRoiManager.pixelsPerLine", where)):
        raise ScanImageError(f"{where}: hRoiManager.pixelsPerLine disagrees with the ROI's {columns} pixels")
    degrees_to_um = _number(values, "objectiveResolution", where)
    size_x, size_y = (float(v) * degrees_to_um for v in scanfield["sizeXY"])
    return Layout(
        n_planes=n_planes,
        zs_um=zs,
        frame_shape=(rows, columns),
        frame_rate=_number(values, "hRoiManager.scanFrameRate", where),
        frame_period=_number(values, "hRoiManager.scanFramePeriod", where),
        volume_rate=_number(values, "hRoiManager.scanVolumeRate", where),
        line_period=_number(values, "hRoiManager.linePeriod", where),
        field_of_view_um=(size_x, size_y),
        pixel_size_um=(size_x / columns, size_y / rows),
        version=_required(values, "VERSION_MAJOR", where),
        frames_per_file=_number(values, "hScan2D.logFramesPerFile", where),
        n_volumes=_number(values, "hStackManager.numVolumes", where),
    )


def _parse_frame_header(text, where):
    """One page's ImageDescription as a dict of raw text, refusing fields it doesn't know."""
    fields = {}
    for line in text.splitlines():
        key, separator, value = line.partition(" = ")
        if not separator:
            if line.strip():
                raise ScanImageError(f"{where}: can't read header line {line!r}")
            continue
        if key not in _KNOWN_FIELDS:
            raise ScanImageError(f"{where}: unknown header field {key}, which would be lost")
        fields[key] = value.strip()
    missing = _KNOWN_FIELDS - set(fields)
    if missing:
        raise ScanImageError(f"{where}: header lacks {sorted(missing)}")
    return fields


def _parse_packets(text, where):
    """The (time, block, trial, iteration) of each packet in one frame's I2CData text."""
    found = _PACKET.findall(text)
    if len(found) != text.count("["):
        raise ScanImageError(f"{where}: can't read I2CData {text!r}")
    packets = []
    for time, data in found:
        raw = bytes(int(byte) for byte in re.split(r"[\s,]+", data.strip()) if byte)
        if len(raw) != _PACKET_BYTES:
            raise ScanImageError(f"{where}: an I2C packet has {len(raw)} bytes, expected {_PACKET_BYTES}: {text!r}")
        packets.append((float(time), *np.frombuffer(raw, dtype="<u2").tolist()))
    return packets


def _read_file(path):
    """The header fields and packets of every page in one file."""
    rows, packets = [], []
    with tifffile.TiffFile(path) as tif:
        tif.pages.cache = False
        tif.pages.useframes = False
        file_size = path.stat().st_size
        first = tif.pages[0]
        tags = first.tags
        if "Software" not in tags or "Artist" not in tags:
            raise ScanImageError(f"{path.name}: no ScanImage metadata, so this isn't an original acquisition file")
        software, artist = tags["Software"].value, tags["Artist"].value
        shape, dtype = first.shape, np.dtype(first.dtype).newbyteorder(tif.byteorder)
        for page_index, page in enumerate(tif.pages):
            where = f"{path.name} page {page_index}"
            if page.compression != 1 or len(page.dataoffsets) != 1 or page.shape != shape:
                raise ScanImageError(f"{where}: expected one uncompressed strip of shape {shape}")
            if page.dataoffsets[0] + page.databytecounts[0] > file_size:
                raise ScanImageError(f"{where}: the file is cut off inside this frame")
            fields = _parse_frame_header(page.description, where)
            for name in _AUX_FIELDS:
                if fields[name] != "[]":
                    raise ScanImageError(f"{where}: {name} holds {fields[name]}, which isn't stored yet")
            row = {column: int(fields[name]) for name, column in _INTEGER_FIELDS.items()}
            row.update({column: float(fields[name]) for name, column in _FLOAT_FIELDS.items()})
            row.update(page_index=page_index, data_offset=int(page.dataoffsets[0]), epoch=fields["epoch"])
            rows.append(row)
            packets.extend((page_index, *packet) for packet in _parse_packets(fields["I2CData"], where))
    return rows, packets, software, artist, shape, dtype


def _sorted_files(paths):
    if not paths:
        raise ScanImageError("no TIFF files were given")
    numbered = []
    for path in map(Path, paths):
        match = _FILE_NAME.match(path.name)
        if not match:
            raise ScanImageError(f"{path.name}: expected a name ending in _NNNNN.tif")
        numbered.append((int(match["number"]), match["stem"], path))
    numbered.sort(key=lambda item: item[0])
    numbers = [number for number, _, _ in numbered]
    if numbers != list(range(1, len(numbers) + 1)):
        raise ScanImageError(f"expected consecutive files from 00001, got file {', '.join(f'{n:05d}' for n in numbers)}")
    stems = {stem for _, stem, _ in numbered}
    if len(stems) != 1:
        raise ScanImageError(f"these files belong to different acquisitions: {sorted(stems)}")
    stem = stems.pop()
    given = {path.resolve() for _, _, path in numbered}
    for folder in {path.parent for _, _, path in numbered}:
        for neighbor in sorted(folder.iterdir()):
            match = _FILE_NAME.match(neighbor.name)
            if match and match["stem"] == stem and neighbor.resolve() not in given:
                raise ScanImageError(f"{neighbor.name} is part of this acquisition but wasn't given, so it would be left out")
    return [path for _, _, path in numbered]


def _check_complete(files, layout):
    """A full last file may have had more after it, unless every requested volume is already here.

    ScanImage's end-of-acquisition field reads 0 on the frames of the data seen so far, so it can't settle this.
    """
    if files[-1].n_frames < layout.frames_per_file:
        return
    if sum(file.n_frames for file in files) == layout.n_volumes * layout.n_planes:
        return
    raise ScanImageError(f"{files[-1].name} is full, with {files[-1].n_frames} frames, and fewer volumes than requested are here, so later files may be missing")


def read_acquisition(paths):
    """Read every page header of one acquisition, split across files. Pixels are left on disk."""
    paths = _sorted_files(paths)
    files, frame_rows, packet_rows, layout, shape, dtype, epoch = [], [], [], None, None, None, None
    previous = 0
    for file_index, path in enumerate(paths):
        try:
            rows, packets, software, artist, file_shape, file_dtype = _read_file(path)
        except tifffile.TiffFileError as error:
            raise ScanImageError(f"{path.name}: {error}") from error
        file_layout = _layout(software, artist, path.name)
        if layout is None:
            layout, shape, dtype = file_layout, file_shape, file_dtype
        elif file_layout != layout or file_shape != shape or file_dtype != dtype:
            raise ScanImageError(f"{path.name}: its layout differs from the first file's")
        if file_shape != layout.frame_shape:
            raise ScanImageError(f"{path.name}: pages are {file_shape}, but the header says {layout.frame_shape}")
        for row in rows:
            if row["frame_number"] != previous + 1 or row["frame_number_acquisition"] != previous + 1:
                raise ScanImageError(f"{path.name}: frame {row['frame_number']} follows frame {previous}")
            previous = row["frame_number"]
            if epoch is None:
                epoch = row["epoch"]
            elif row["epoch"] != epoch:
                raise ScanImageError(f"{path.name}: epoch {row['epoch']} differs from {epoch}, so this isn't one acquisition")
            row["file_index"] = file_index
        offset = len(frame_rows)
        packet_rows.extend((frame + offset, *rest) for frame, *rest in packets)
        frame_rows.extend(rows)
        files.append(SourceFile(path=path, n_frames=len(rows), software=software, artist=artist))
    _check_complete(files, layout)
    frames = {name: np.array([row[name] for row in frame_rows], dtype="float64" if name in _FLOAT_FIELDS.values() else "int64") for name in (
        "file_index", "page_index", "data_offset", *_INTEGER_FIELDS.values(), *_FLOAT_FIELDS.values())}
    packet_columns = list(zip(*packet_rows)) or [(), (), (), (), ()]
    packets = {
        name: np.array(column, dtype="float64" if name == "timestamp" else "int64")
        for name, column in zip(("frame_index", "timestamp", "block", "trial", "iteration"), packet_columns)
    }
    return Acquisition(files=files, frames=frames, packets=packets, epoch=epoch, layout=layout, dtype=np.dtype(dtype))


def read_frames(acquisition, indices):
    """The pixels of the given frames, in order, as (frames, rows, columns), read straight from their file offsets."""
    rows, columns = acquisition.layout.frame_shape
    native = acquisition.dtype.newbyteorder("=")
    frames = np.empty((len(indices), rows, columns), dtype=native)
    opened = {}
    try:
        for position, index in enumerate(indices):
            file_index = int(acquisition.frames["file_index"][index])
            if file_index not in opened:
                opened[file_index] = open(acquisition.files[file_index].path, "rb")
            handle = opened[file_index]
            handle.seek(int(acquisition.frames["data_offset"][index]))
            raw = np.frombuffer(handle.read(rows * columns * acquisition.dtype.itemsize), dtype=acquisition.dtype)
            frames[position] = raw.reshape(rows, columns)
    finally:
        for handle in opened.values():
            handle.close()
    return frames
