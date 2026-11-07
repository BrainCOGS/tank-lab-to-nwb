"""Read ScanImage TIFFs: every per-frame header field, the I2C packets from ViRMEn, and how pages map to planes.

Anything this can't account for is an error rather than a silent drop: an unknown header field, a used aux trigger,
a missing frame, or a layout it can't map to planes.

An acquisition comes as ScanImage saved it, each page stacking one plane's ROIs with flyback rows between them, or as the
lab's split copies, one folder per ROI and depth. Either way it becomes a list of fields, one per plane and ROI, which say
where their pixels are and when each frame was scanned.
"""

import json
import re
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path

import numpy as np
import tifffile

_FILE_NAME = re.compile(r"^(?P<stem>.*)_(?P<number>\d{5})\.tif$")
_SPLIT_NAME = re.compile(r"^(?P<stem>.*)_ROI(?P<roi>\d+)_z(?P<depth>\d+)_(?P<number>\d{5})\.tif$")
_SPLIT_FOLDER_GLOB = "ROI*_z*"
# The splitter writes times with num2str, to four decimals. Each stamp is off by up to half of this, so two ROIs' difference
# can swing over twice this while their lag is fixed.
_SPLIT_ROUNDING_SECONDS = 1e-4
STACK_SOURCE = "ScanImage TIFFs, as saved"
SPLIT_SOURCE = "split per-ROI TIFFs, not the original stack"
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
class Roi:
    """One ROI's place in an acquisition. Geometry is None when the files don't carry it."""

    number: int  # 1-based, as in the ROI order of the header and the split folders
    name: str
    plane: int
    rows: tuple  # (start, stop) within a stored page
    size_um: tuple | None = None
    pixel_size_um: tuple | None = None
    origin_um: tuple | None = None  # the top-left corner (x, y) and the plane's z


@dataclass(frozen=True)
class Field:
    """One plane and ROI through time: which frames it has, where its pixels are, and when each was scanned."""

    label: str  # names its NWB objects: Plane0, or Plane0Roi2 when there are several ROIs
    number: int
    name: str
    plane: int
    rows: tuple  # (start, stop) within a stored page
    frame_indices: np.ndarray  # rows of acquisition.frames
    file_index: np.ndarray  # per frame, the row of acquisition.files holding its pixels
    data_offset: np.ndarray  # per frame, where they start in that file
    timestamps: np.ndarray  # per frame, ScanImage seconds at the field's first row
    time_offset: float  # seconds from the frame's start to the field's first row, as far as the files say
    page_shape: tuple
    size_um: tuple | None = None
    pixel_size_um: tuple | None = None
    origin_um: tuple | None = None

    @property
    def shape(self):
        """(rows, columns) of one frame of this field."""
        return (self.rows[1] - self.rows[0], self.page_shape[1])


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
    frames_per_file: float  # logFramesPerFile, inf when the acquisition isn't split into files
    n_volumes: float  # the volumes requested, which a stopped acquisition falls short of
    rois: tuple = ()  # the Rois of the acquisition
    flyback_rows: dict = dataclass_field(default_factory=dict)  # plane -> (start, stop) rows between its ROIs


@dataclass(frozen=True)
class Acquisition:
    files: list
    frames: dict  # column name -> array with one entry per page
    packets: dict  # column name -> array with one entry per I2C packet
    epoch: str
    layout: Layout
    dtype: np.dtype
    fields: tuple = ()
    source: str = STACK_SOURCE
    completeness: str = "complete"
    split_packet_changes: int = 0  # I2C packets the splitter changed, because they shared their frame's time

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


def _common_settings(values, where):
    """The settings every page layout shares, refusing anything but whole frames of one channel, one per plane."""
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
    return n_planes, zs


def _scanfield(roi, where):
    """A ROI's one scanfield. Several that differ mean its size or place changes with depth, which isn't handled yet."""
    scanfields = roi["scanfields"]
    if not isinstance(scanfields, list):
        return scanfields
    shared = ("pixelResolutionXY", "sizeXY", "centerXY", "rotationDegrees")
    if any(any(other.get(key) != scanfields[0].get(key) for key in shared) for other in scanfields[1:]):
        raise ScanImageError(f"{where}: ROI {roi.get('name')} has scanfields that differ by depth, which isn't handled yet")
    return scanfields[0]


def _geometry(scanfield, z, degrees_to_um, where):
    """(size, pixel size, origin) of a scanfield, in micrometers. The origin is the outer corner of its first pixel, with the plane's z."""
    if scanfield.get("rotationDegrees", 0):
        raise ScanImageError(f"{where}: a scanfield is rotated by {scanfield['rotationDegrees']} degrees, which the stored geometry can't express")
    columns, rows = (int(v) for v in scanfield["pixelResolutionXY"])
    size_x, size_y = (float(v) * degrees_to_um for v in scanfield["sizeXY"])
    center_x, center_y = (float(v) * degrees_to_um for v in scanfield.get("centerXY", (0.0, 0.0)))
    return (size_x, size_y), (size_x / columns, size_y / rows), (center_x - size_x / 2, center_y - size_y / 2, z)


def _roi_zs(roi):
    return [float(z) for z in np.atleast_1d(roi.get("zs", []))]


def _stack_rois(raw_rois, zs, page_shape, degrees_to_um, where):
    """Each plane's ROIs placed in the page. ScanImage stacks them top to bottom with an even gap of flyback rows."""
    page_rows, page_columns = page_shape
    placed, flyback = [], {}
    used = set()
    for plane, z in enumerate(zs):
        here = [(number, roi) for number, roi in enumerate(raw_rois, start=1) if z in _roi_zs(roi)]
        if not here:
            raise ScanImageError(f"{where}: plane {plane} at z = {z:g} has no ROI")
        used.update(number for number, _ in here)
        heights = [int(_scanfield(roi, where)["pixelResolutionXY"][1]) for _, roi in here]
        spare = page_rows - sum(heights)
        gaps = len(here) - 1
        if spare < 0 or (gaps and spare % gaps) or (not gaps and spare):
            raise ScanImageError(f"{where}: plane {plane}'s ROIs total {sum(heights)} rows on a {page_rows}-row page, which doesn't leave even flyback gaps")
        gap = spare // gaps if gaps else 0
        row, plane_flyback = 0, []
        for (number, roi), height in zip(here, heights):
            scanfield = _scanfield(roi, where)
            if int(scanfield["pixelResolutionXY"][0]) != page_columns:
                raise ScanImageError(f"{where}: ROI {number} is {scanfield['pixelResolutionXY'][0]} pixels wide on a {page_columns}-column page")
            size, pixel, origin = _geometry(scanfield, z, degrees_to_um, where)
            placed.append(Roi(number, roi.get("name") or f"ROI{number:02d}", plane, (row, row + height), size, pixel, origin))
            row += height
            if len(plane_flyback) < gaps:
                plane_flyback.append((row, row + gap))
                row += gap
        flyback[plane] = tuple(plane_flyback)
    unused = sorted(set(range(1, len(raw_rois) + 1)) - used)
    if unused:
        raise ScanImageError(f"{where}: ROI {unused[0]} is on no plane of {list(zs)}")
    return tuple(placed), flyback


def _layout(software, artist, where, page_shape=None):
    """Read the layout of a stack's pages, refusing anything other than whole frames of one channel, one per plane."""
    values = _header_values(software)
    n_planes, zs = _common_settings(values, where)
    raw_rois = _rois(artist, where)
    mroi = _required(values, "hRoiManager.mroiEnable", where) in ("1", "true")
    if len(raw_rois) != 1 and not mroi:
        raise ScanImageError(f"{where}: {len(raw_rois)} ROIs in one page, but mroiEnable is off, so the rows can't be placed")
    degrees_to_um = _number(values, "objectiveResolution", where)
    if len(raw_rois) == 1:
        scanfield = _scanfield(raw_rois[0], where)
        columns, rows = (int(v) for v in scanfield["pixelResolutionXY"])
        if rows != int(_number(values, "hRoiManager.linesPerFrame", where)):
            raise ScanImageError(f"{where}: hRoiManager.linesPerFrame disagrees with the ROI's {rows} lines")
        if columns != int(_number(values, "hRoiManager.pixelsPerLine", where)):
            raise ScanImageError(f"{where}: hRoiManager.pixelsPerLine disagrees with the ROI's {columns} pixels")
        rois = tuple(
            Roi(1, raw_rois[0].get("name") or "ROI01", plane, (0, rows), *_geometry(scanfield, z, degrees_to_um, where)) for plane, z in enumerate(zs)
        )
        flyback = {plane: () for plane in range(n_planes)}
        frame_shape = (rows, columns)
    else:
        rois, flyback = _stack_rois(raw_rois, zs, page_shape, degrees_to_um, where)
        frame_shape = tuple(page_shape)
    return Layout(
        n_planes=n_planes,
        zs_um=zs,
        frame_shape=frame_shape,
        frame_rate=_number(values, "hRoiManager.scanFrameRate", where),
        frame_period=_number(values, "hRoiManager.scanFramePeriod", where),
        volume_rate=_number(values, "hRoiManager.scanVolumeRate", where),
        line_period=_number(values, "hRoiManager.linePeriod", where),
        field_of_view_um=rois[0].size_um,
        pixel_size_um=rois[0].pixel_size_um,
        version=_required(values, "VERSION_MAJOR", where),
        frames_per_file=_number(values, "hScan2D.logFramesPerFile", where),
        n_volumes=_number(values, "hStackManager.numVolumes", where),
        rois=rois,
        flyback_rows=flyback,
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


def _read_file(path, need_metadata=True):
    """The header fields and packets of every page in one file."""
    rows, packets = [], []
    with tifffile.TiffFile(path) as tif:
        tif.pages.cache = False
        tif.pages.useframes = False
        file_size = path.stat().st_size
        if not len(tif.pages):
            raise ScanImageError(f"{path.name}: no pages")
        first = tif.pages[0]
        tags = first.tags
        if need_metadata and ("Software" not in tags or "Artist" not in tags):
            raise ScanImageError(f"{path.name}: no ScanImage metadata, so this isn't an original acquisition file")
        software = tags["Software"].value if "Software" in tags else ""
        artist = tags["Artist"].value if "Artist" in tags else ""
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


def _completeness(total, last_name, last_is_short, layout, allow_incomplete):
    """What these files can say about being the whole acquisition. ScanImage's end-of-acquisition field reads 0 on the last
    frame of the data seen so far, so nothing marks the end: a short last file fits an acquisition stopped early, but it would
    also fit one cut off on a page boundary or with later files removed. A full one may have had more after it."""
    requested = layout.n_volumes * layout.n_planes
    if total > requested:
        raise ScanImageError(f"{total} frames, but the header requested {requested:g}, so these aren't one acquisition's files")
    if total == requested:
        return "complete: every requested volume is here"
    if last_is_short:
        return "no gap found: frames run in order and the last file is short, as when an acquisition is stopped early. That's not proof, since nothing in the files marks the end."
    reason = f"{last_name} is full, and fewer volumes than requested are here, so later files may be missing"
    if not allow_incomplete:
        raise ScanImageError(f"{reason}. Pass allow_incomplete to convert what's here anyway.")
    return f"not confirmed: {reason}"


def _stack_fields(layout, frames):
    several = len({roi.number for roi in layout.rois}) > 1
    fields = []
    for roi in layout.rois:
        indices = np.arange(frames["frame_number"].size)[roi.plane :: layout.n_planes]
        offset = roi.rows[0] * layout.line_period
        fields.append(Field(
            label=f"Plane{roi.plane}Roi{roi.number}" if several else f"Plane{roi.plane}",
            number=roi.number,
            name=roi.name,
            plane=roi.plane,
            rows=roi.rows,
            frame_indices=indices,
            file_index=frames["file_index"][indices],
            data_offset=frames["data_offset"][indices],
            timestamps=frames["timestamp"][indices] + offset,
            time_offset=offset,
            page_shape=layout.frame_shape,
            size_um=roi.size_um,
            pixel_size_um=roi.pixel_size_um,
            origin_um=roi.origin_um,
        ))
    return tuple(fields)


def _frame_table(frame_rows):
    return {name: np.array([row[name] for row in frame_rows], dtype="float64" if name in _FLOAT_FIELDS.values() else "int64") for name in (
        "file_index", "page_index", "data_offset", *_INTEGER_FIELDS.values(), *_FLOAT_FIELDS.values())}


def _packet_table(packet_rows):
    columns = list(zip(*packet_rows)) or [(), (), (), (), ()]
    return {
        name: np.array(column, dtype="float64" if name == "timestamp" else "int64")
        for name, column in zip(("frame_index", "timestamp", "block", "trial", "iteration"), columns)
    }


def _read_stack(paths, allow_incomplete):
    paths = _sorted_files(paths)
    files, frame_rows, packet_rows, layout, shape, dtype, epoch = [], [], [], None, None, None, None
    previous = 0
    for file_index, path in enumerate(paths):
        try:
            rows, packets, software, artist, file_shape, file_dtype = _read_file(path)
        except tifffile.TiffFileError as error:
            raise ScanImageError(f"{path.name}: {error}") from error
        file_layout = _layout(software, artist, path.name, file_shape)
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
    completeness = _completeness(len(frame_rows), files[-1].name, files[-1].n_frames < layout.frames_per_file, layout, allow_incomplete)
    frames = _frame_table(frame_rows)
    return Acquisition(
        files=files, frames=frames, packets=_packet_table(packet_rows), epoch=epoch, layout=layout, dtype=np.dtype(dtype),
        fields=_stack_fields(layout, frames), source=STACK_SOURCE, completeness=completeness,
    )


@dataclass
class _SplitGroup:
    """One folder of split files: every page of one ROI at one depth."""

    roi: int
    depth: int
    files: list = dataclass_field(default_factory=list)
    rows: list = dataclass_field(default_factory=list)
    packets: list = dataclass_field(default_factory=list)  # (position in this group, time, block, trial, iteration)
    shape: tuple = None
    dtype: np.dtype = None

    @property
    def label(self):
        return f"ROI{self.roi:02d}_z{self.depth}"


def _group_split_paths(paths):
    """Paths by (depth, ROI), refusing other files of this acquisition that weren't given."""
    groups, stems = {}, set()
    for path in map(Path, paths):
        match = _SPLIT_NAME.match(path.name)
        stems.add(match["stem"])
        groups.setdefault((int(match["depth"]), int(match["roi"])), []).append((int(match["number"]), path))
    if len(stems) != 1:
        raise ScanImageError(f"these files belong to different acquisitions: {sorted(stems)}")
    stem = stems.pop()
    given = {Path(path).resolve() for path in paths}
    roots = {Path(path).parent for path in paths} | {Path(path).parent.parent for path in paths}
    for root in sorted(roots):
        for neighbor in sorted([*root.glob("*.tif"), *root.glob(f"{_SPLIT_FOLDER_GLOB}/*.tif")]):
            match = _SPLIT_NAME.match(neighbor.name)
            if match and match["stem"] == stem and neighbor.resolve() not in given:
                raise ScanImageError(f"{neighbor.name} is part of this acquisition but wasn't given, so it would be left out")
    ordered = {}
    for key in sorted(groups):
        numbered = sorted(groups[key])
        if [number for number, _ in numbered] != list(range(1, len(numbered) + 1)):
            raise ScanImageError(f"ROI{key[1]:02d}_z{key[0]}: expected consecutive files from 00001, got {', '.join(f'{number:05d}' for number, _ in numbered)}")
        ordered[key] = [path for _, path in numbered]
    return ordered


def _read_split_group(key, paths, files):
    group = _SplitGroup(roi=key[1], depth=key[0])
    software, artist = "", ""
    for path in paths:
        try:
            rows, packets, file_software, file_artist, shape, dtype = _read_file(path, need_metadata=False)
        except tifffile.TiffFileError as error:
            raise ScanImageError(f"{path.name}: {error}") from error
        if group.shape not in (None, shape) or group.dtype not in (None, dtype):
            raise ScanImageError(f"{path.name}: its pages differ from the rest of {group.label}")
        group.shape, group.dtype = shape, dtype
        software, artist = software or file_software, artist or file_artist
        offset = len(group.rows)
        for row in rows:
            row["file_index"] = len(files)
        group.packets.extend((page + offset, *rest) for page, *rest in packets)
        group.rows.extend(rows)
        files.append(SourceFile(path=path, n_frames=len(rows), software=file_software, artist=file_artist))
    return group, software, artist


_SHARED_COLUMNS = ("frame_number", "acquisition_number", "frame_number_acquisition", "acq_trigger_timestamp", "next_file_marker_timestamp", "end_of_acquisition", "end_of_acquisition_mode", "dc_over_voltage", "epoch")


def _check_same_frames(first, other):
    """Every ROI at a depth must hold the same frames, the same I2C data and the same clock, apart from the splitter's lag."""
    if len(other.rows) != len(first.rows) or [file_size for file_size in other.page_counts] != [file_size for file_size in first.page_counts]:
        raise ScanImageError(f"{other.label} doesn't hold the same files and pages as {first.label}")
    for name in _SHARED_COLUMNS:
        if [row[name] for row in other.rows] != [row[name] for row in first.rows]:
            raise ScanImageError(f"{other.label} and {first.label} disagree about {name}")
    differences = np.array([b["timestamp"] - a["timestamp"] for a, b in zip(first.rows, other.rows)])
    if differences.size and differences.max() - differences.min() > 2 * _SPLIT_ROUNDING_SECONDS + 1e-9:
        raise ScanImageError(f"{other.label}'s frame times don't follow {first.label}'s by a fixed lag")
    return _changed_packets(first, other)


def _changed_packets(first, other):
    """How many I2C packets differ in time only because the splitter's replace also caught them.

    It replaces the frame's old time anywhere in the header, so a packet stamped with exactly that time takes each ROI's new
    frame time. Anything else that differs is real disagreement.
    """
    if len(other.packets) != len(first.packets):
        raise ScanImageError(f"{other.label} and {first.label} disagree about the I2C data: {len(other.packets)} and {len(first.packets)} packets")
    changed = 0
    for mine, theirs in zip(first.packets, other.packets):
        same_packet = mine[0] == theirs[0] and mine[2:] == theirs[2:]
        if same_packet and mine[1] == theirs[1]:
            continue
        caught = same_packet and mine[1] == first.rows[mine[0]]["timestamp"] and theirs[1] == other.rows[theirs[0]]["timestamp"]
        if not caught:
            raise ScanImageError(f"{other.label} and {first.label} disagree about the I2C data at frame {first.rows[mine[0]]['frame_number']}")
        changed += 1
    return changed


def _split_rois(groups, software, artist, where):
    """Each folder's ROI, from the ScanImage settings when the split files kept them, and from its folder name otherwise."""
    if not software or not artist:
        return {key: Roi(key[1], f"ROI{key[1]:02d}", key[0] - 1, (0, group.shape[0])) for key, group in groups.items()}, None
    values = _header_values(software)
    _, zs = _common_settings(values, where)
    raw_rois = _rois(artist, where)
    degrees_to_um = _number(values, "objectiveResolution", where)
    # The splitter numbers its depth folders in ascending z, while frames cycle through the planes in the header's order.
    depths = sorted({z for roi in raw_rois for z in _roi_zs(roi)})
    if tuple(depths) != tuple(zs):
        raise ScanImageError(f"{where}: the depth folders run in ascending z, {depths}, but the header's planes are {list(zs)}, so they can't be matched")
    for depth, z in enumerate(depths, start=1):
        for number, raw in enumerate(raw_rois, start=1):
            if z in _roi_zs(raw) and (depth, number) not in groups:
                raise ScanImageError(f"{where}: the ScanImage settings have ROI {number} at depth {depth}, but no files were given for it")
    rois = {}
    for key, group in groups.items():
        depth, number = key
        if not 1 <= number <= len(raw_rois) or depth > len(depths):
            raise ScanImageError(f"{group.label}: the ScanImage settings have {len(raw_rois)} ROIs at {len(depths)} depths")
        raw = raw_rois[number - 1]
        z = depths[depth - 1]
        if z not in _roi_zs(raw):
            raise ScanImageError(f"{group.label}: ROI {number} isn't at depth {depth}, z = {z:g}")
        scanfield = _scanfield(raw, group.label)
        columns, rows = (int(v) for v in scanfield["pixelResolutionXY"])
        if (rows, columns) != tuple(group.shape):
            raise ScanImageError(f"{group.label}: pages are {group.shape}, but the ROI is {rows} rows by {columns} pixels wide, so it may have been summed when split")
        rois[key] = Roi(number, raw.get("name") or f"ROI{number:02d}", depth - 1, (0, rows), *_geometry(scanfield, z, degrees_to_um, group.label))
    return rois, values


def _split_layout(groups, rois, values, where, first_times):
    n_planes = len({depth for depth, _ in groups})
    shape = next(iter(groups.values())).shape
    if values is None:
        # The split files kept no settings, so the only rate on hand is the frames' own.
        period = float(np.median(np.diff(first_times))) if first_times.size > 1 else float("nan")
        return Layout(
            n_planes=n_planes, zs_um=(float("nan"),) * n_planes, frame_shape=tuple(shape), frame_rate=n_planes / period, frame_period=period / n_planes, volume_rate=1 / period,
            line_period=None, field_of_view_um=None, pixel_size_um=None, version="unknown", frames_per_file=float("inf"), n_volumes=float("inf"),
            rois=tuple(rois.values()), flyback_rows={plane: () for plane in range(n_planes)},
        )
    settings_planes, zs = _common_settings(values, where)
    if settings_planes != n_planes:
        raise ScanImageError(f"{where}: the ScanImage settings have {settings_planes} planes, but there are {n_planes} depth folders")
    first = next(iter(rois.values()))
    return Layout(
        n_planes=n_planes, zs_um=zs, frame_shape=tuple(shape),
        frame_rate=_number(values, "hRoiManager.scanFrameRate", where), frame_period=_number(values, "hRoiManager.scanFramePeriod", where), volume_rate=_number(values, "hRoiManager.scanVolumeRate", where),
        line_period=_number(values, "hRoiManager.linePeriod", where), field_of_view_um=first.size_um, pixel_size_um=first.pixel_size_um, version=_required(values, "VERSION_MAJOR", where),
        frames_per_file=_number(values, "hScan2D.logFramesPerFile", where), n_volumes=_number(values, "hStackManager.numVolumes", where),
        rois=tuple(rois.values()), flyback_rows={plane: () for plane in range(n_planes)},
    )


def _read_split(paths, allow_incomplete):
    grouped = _group_split_paths(paths)
    files, groups, software, artist = [], {}, "", ""
    for key, group_paths in grouped.items():
        groups[key], file_software, file_artist = _read_split_group(key, group_paths, files)
        software, artist = software or file_software, artist or file_artist
    dtypes = {group.dtype for group in groups.values()}
    if len(dtypes) != 1:
        raise ScanImageError(f"the split files have different pixel types: {sorted(map(str, dtypes))}")
    for group in groups.values():
        group.page_counts = [sum(1 for row in group.rows if row["file_index"] == index) for index in sorted({row["file_index"] for row in group.rows})]
    # Each depth's lowest ROI has no lag, and its files speak for the frames of that depth.
    first_at_depth = {}
    for (depth, _), group in groups.items():
        first_at_depth.setdefault(depth, group)
    n_depths = len(first_at_depth)
    changed_packets = 0
    for (depth, _), group in groups.items():
        for row in group.rows:
            if (row["frame_number"] - 1) % n_depths != depth - 1:
                raise ScanImageError(f"{group.label}: frame {row['frame_number']} belongs to depth {(row['frame_number'] - 1) % n_depths + 1} of {n_depths}, so this folder's depth is wrong")
        if group is not first_at_depth[depth]:
            changed_packets = max(changed_packets, _check_same_frames(first_at_depth[depth], group))
    frame_rows, packet_rows = [], []
    for group in first_at_depth.values():
        frame_rows.extend(group.rows)
        packet_rows.extend((group.rows[position]["frame_number"] - 1, *rest) for position, *rest in group.packets)
    frame_rows.sort(key=lambda row: row["frame_number"])
    packet_rows.sort(key=lambda row: (row[0], row[1]))
    for position, row in enumerate(frame_rows, start=1):
        if row["frame_number"] != position or row["frame_number_acquisition"] != position:
            raise ScanImageError(f"{files[row['file_index']].name}: frame {row['frame_number']} where frame {position} should be")
    epochs = {row["epoch"] for row in frame_rows}
    if len(epochs) != 1:
        raise ScanImageError(f"the split files hold several epochs, so this isn't one acquisition: {sorted(epochs)}")
    where = files[0].name
    rois, values = _split_rois(groups, software, artist, where)
    first_times = np.array([row["timestamp"] for row in next(iter(first_at_depth.values())).rows])
    layout = _split_layout(groups, rois, values, where, first_times)
    frames = _frame_table(frame_rows)
    if values is None:
        reason = "these split files kept no ScanImage settings, so nothing says how many files there should be or which ROIs there were"
        if not allow_incomplete:
            raise ScanImageError(f"{reason}. Pass allow_incomplete to convert what's here anyway.")
        completeness = f"not confirmed: {reason}"
    else:
        last_group = next(iter(first_at_depth.values()))
        last_file = sorted({row["file_index"] for row in last_group.rows})[-1]
        completeness = _completeness(frames["frame_number"].size, files[last_file].name, last_group.page_counts[-1] * layout.n_planes < layout.frames_per_file, layout, allow_incomplete)
    several = len({number for _, number in groups}) > 1
    fields = []
    for key, group in groups.items():
        roi, lags = rois[key], first_at_depth[key[0]]
        times = np.array([row["timestamp"] for row in group.rows])
        lag = float(np.median(times - np.array([row["timestamp"] for row in lags.rows])))
        fields.append(Field(
            label=f"Plane{roi.plane}Roi{roi.number}" if several else f"Plane{roi.plane}",
            number=roi.number, name=roi.name, plane=roi.plane, rows=roi.rows,
            frame_indices=np.array([row["frame_number"] - 1 for row in group.rows]),
            file_index=np.array([row["file_index"] for row in group.rows]),
            data_offset=np.array([row["data_offset"] for row in group.rows]),
            timestamps=times, time_offset=lag, page_shape=tuple(group.shape),
            size_um=roi.size_um, pixel_size_um=roi.pixel_size_um, origin_um=roi.origin_um,
        ))
    return Acquisition(
        files=files, frames=frames, packets=_packet_table(packet_rows), epoch=epochs.pop(), layout=layout, dtype=np.dtype(dtypes.pop()),
        fields=tuple(fields), source=SPLIT_SOURCE, completeness=completeness, split_packet_changes=changed_packets,
    )


def read_acquisition(paths, allow_incomplete=False):
    """Read every page header of one acquisition, from a stack split across files or from the lab's per-ROI folders.

    Pixels are left on disk. `allow_incomplete` accepts files that can't be shown to be the whole acquisition, and records that.
    """
    if not paths:
        raise ScanImageError("no TIFF files were given")
    split = [_SPLIT_NAME.match(Path(path).name) is not None for path in paths]
    if any(split) and not all(split):
        raise ScanImageError("these are original TIFFs and split ROI TIFFs together; give either the originals or the split files")
    return _read_split(paths, allow_incomplete) if all(split) else _read_stack(paths, allow_incomplete)


def find_tiffs(folder):
    """One session's TIFFs in a folder: the original stack when it's there, otherwise the split ROI folders."""
    folder = Path(folder)
    for found in (folder.glob("originalStacks/*.tif"), folder.glob("*.tif"), folder.glob(f"{_SPLIT_FOLDER_GLOB}/*.tif")):
        found = sorted(found)
        if found:
            return found
    raise ScanImageError(f"{folder}: no TIFF files in it, in originalStacks, or in ROI folders")


def _read_pages(acquisition, file_index, data_offset, shape):
    """Pages by file and offset, in order, as (frames, rows, columns), read straight from the files."""
    rows, columns = shape
    native = acquisition.dtype.newbyteorder("=")
    frames = np.empty((len(file_index), rows, columns), dtype=native)
    opened = {}
    try:
        for position, (file, offset) in enumerate(zip(file_index.tolist(), data_offset.tolist())):
            if file not in opened:
                opened[file] = open(acquisition.files[file].path, "rb")
            handle = opened[file]
            handle.seek(offset)
            raw = np.frombuffer(handle.read(rows * columns * acquisition.dtype.itemsize), dtype=acquisition.dtype)
            frames[position] = raw.reshape(rows, columns)
    finally:
        for handle in opened.values():
            handle.close()
    return frames


def read_frames(acquisition, indices):
    """The pixels of the given frames, in order, as (frames, rows, columns), read straight from their file offsets."""
    indices = np.asarray(indices, dtype=int)
    return _read_pages(acquisition, acquisition.frames["file_index"][indices], acquisition.frames["data_offset"][indices], acquisition.layout.frame_shape)


def read_field(acquisition, field, positions):
    """The pixels of a field's frames at `positions`, as (frames, rows, columns)."""
    positions = np.asarray(positions, dtype=int)
    pages = _read_pages(acquisition, field.file_index[positions], field.data_offset[positions], field.page_shape)
    return pages[:, field.rows[0] : field.rows[1]]


def read_flyback(acquisition, plane, positions):
    """The rows between a plane's ROIs at `positions` of that plane, joined top to bottom."""
    indices = acquisition.plane_frames(plane)[np.asarray(positions, dtype=int)]
    pages = read_frames(acquisition, indices)
    return np.concatenate([pages[:, start:stop] for start, stop in acquisition.layout.flyback_rows[plane]], axis=1)
