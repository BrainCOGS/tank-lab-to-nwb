"""Read per-plane Suite2p results without importing Suite2p or torch."""

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import numpy as np

from ..virmen.coerce import coerce
from .scanimage import _FILE_NAME, _SPLIT_NAME, SPLIT_SOURCE


REGISTRATION_SERIES = ("xoff", "yoff", "corrXY", "xoff1", "yoff1", "corrXY1", "badframes", "badframes0")
TRACE_SERIES = (("Fluorescence", "F"), ("Neuropil", "Fneu"), ("Deconvolved", "spks"))
SUMMARY_IMAGES = ("meanImg", "meanImgE", "max_proj", "Vcorr", "refImg", "meanImg_crop")
# What a pre-1.0 ops.npy must hold to place a plane folder in a field and check it against the TIFFs.
PRE_1_0_REQUIRED = ("filelist", "nframes", "Ly", "Lx", "nplanes", "iplane", "frames_per_file", "xoff", "yoff")
# Pre-1.0 ops arrays larger than this, other than registration series and images, stay only in the archived ops.npy.
SETTINGS_ARRAY_LIMIT = 1000


def _npy(folder, name):
    value = np.load(folder / f"{name}.npy", allow_pickle=True)
    return value.item() if value.shape == () else value


def read_plane(folder):
    """A Suite2p plane folder's metadata in one shape, whichever Suite2p wrote it.

    Suite2p 1.x writes db.npy, settings.npy, reg_outputs.npy and detect_outputs.npy. Suite2p before 1.0 kept all of it in
    one flat ops.npy, with the file list as `filelist`. Returns layout, db (file_list, nframes, Ly, Lx, nplanes, iplane,
    nrois, iroi, lines, frames_per_file), settings (as recorded), registration (the settings that rebuild registered frames),
    allow_overlap, reg_outputs, detect_outputs and version. Anything it can't read that way is refused, saying why.
    """
    folder = Path(folder)
    if (folder / "db.npy").exists():
        missing = [name for name in ("settings", "reg_outputs") if not (folder / f"{name}.npy").exists()]
        if missing:
            raise ValueError(f"{folder}: a Suite2p 1.x plane folder (it has db.npy) without {', '.join(f'{m}.npy' for m in missing)}")
        settings = _npy(folder, "settings")
        return dict(
            layout="1.x",
            db=_npy(folder, "db"),
            settings=settings,
            registration=settings.get("registration", {}),
            allow_overlap=settings.get("extraction", {}).get("allow_overlap", False),
            reg_outputs=_npy(folder, "reg_outputs"),
            detect_outputs=_npy(folder, "detect_outputs") if (folder / "detect_outputs.npy").exists() else {},
            version=settings.get("version"),
        )
    if (folder / "ops.npy").exists():
        ops = _npy(folder, "ops")
        missing = [key for key in PRE_1_0_REQUIRED if key not in ops]
        if missing:
            raise ValueError(f"{folder}: a pre-1.0 Suite2p plane folder (ops.npy, no db.npy) whose ops lack {', '.join(missing)}, so it can't be placed and checked")
        reg = {key: ops[key] for key in (*REGISTRATION_SERIES, "meanImg", "meanImgE", "refImg", "meanImg_crop", "xrange", "yrange") if key in ops}
        detect = {key: ops[key] for key in ("max_proj", "Vcorr") if key in ops}
        db = dict(
            file_list=list(ops["filelist"]),
            **{key: ops[key] for key in ("nframes", "Ly", "Lx", "nplanes", "iplane", "frames_per_file")},
            **{key: ops.get(key) for key in ("nrois", "iroi", "lines", "dy", "dx")},
        )
        settings = {
            key: value
            for key, value in ops.items()
            if key not in reg and key not in detect and not (isinstance(value, np.ndarray) and value.size > SETTINGS_ARRAY_LIMIT)
        }
        return dict(layout="pre-1.0", db=db, settings=settings, registration=ops, allow_overlap=ops.get("allow_overlap", False),
                    reg_outputs=reg, detect_outputs=detect, version=ops.get("suite2p_version"))
    raise ValueError(f"{folder}: neither db.npy (Suite2p 1.x) nor ops.npy (Suite2p before 1.0), so it isn't a Suite2p plane folder this can read")


SPLIT_COPY = "split copy of the given TIFFs: negatives set to 0, as uint16"
SAME_PIXELS = "the given TIFFs"
_PLANE_FOLDER = re.compile(r"^plane(\d+)$")


@dataclass(frozen=True)
class Suite2pPlane:
    """One Suite2p plane folder and the acquisition field it holds.

    `split_copy` means Suite2p read the lab's split copies of the given stack, which set negative pixels to 0 and store
    uint16, so its registered frames are rebuilt from the stack's rows that way.
    """

    folder: Path
    field: object
    db: dict
    split_copy: bool = False
    meta: dict = None  # read_plane's result

    @property
    def input(self):
        return SPLIT_COPY if self.split_copy else SAME_PIXELS

    @property
    def suffix(self):
        """Names this field's NWB objects: 0 for a plane with one ROI, 0Roi2 for the second ROI on plane 0."""
        return self.field.label.removeprefix("Plane")


def _roots(roots):
    roots = [Path(roots)] if isinstance(roots, (str, Path)) else [Path(root) for root in roots]
    if not roots:
        raise ValueError("no Suite2p output folder was given")
    resolved = [root.resolve() for root in roots]
    for index, root in enumerate(resolved):
        for other in resolved[:index]:
            if root == other:
                raise ValueError(f"{roots[index]} is given more than once")
            if root in other.parents or other in root.parents:
                raise ValueError(f"{roots[index]} and another given folder are nested, so plane folders would be read twice")
    return roots


def source_base(roots):
    """The folder that source paths are relative to: the one Suite2p folder, or what several have in common."""
    roots = _roots(roots)
    return roots[0] if len(roots) == 1 else Path(os.path.commonpath([root.resolve() for root in roots]))


def _plane_folders(root):
    """{number: folder} of a Suite2p output folder's plane folders, refusing anything else named like one."""
    folders = {}
    for path in sorted(root.iterdir()) if root.is_dir() else []:
        if path.is_dir() and path.name.startswith("plane"):
            match = _PLANE_FOLDER.match(path.name)
            if not match:
                raise ValueError(f"{path}: named like a Suite2p plane folder but not numbered, so it can't be placed")
            folders[int(match[1])] = path
    if not folders:
        raise ValueError(f"{root}: no Suite2p plane folders")
    return folders


def _split_names(names):
    """(stem, ROI, depth, file numbers) of split file names, None if none are split, or an error if they're mixed or several."""
    matches = [_SPLIT_NAME.match(name) for name in names]
    if not any(matches):
        return None
    if not all(matches):
        raise ValueError("it read original and split files together")
    keys = {(m["stem"], int(m["roi"]), int(m["depth"])) for m in matches}
    if len(keys) != 1:
        raise ValueError(f"it read split files of several ROIs or depths: {sorted(keys)}")
    stem, roi, depth = keys.pop()
    return stem, roi, depth, [int(m["number"]) for m in matches]


def _field_for(where, number, db, acquisition):
    """The field a plane folder holds, and whether Suite2p read split copies of the given stack. Refuses anything ambiguous."""
    # A run on a Windows rig records paths with backslashes, sometimes after a forward-slash share; take the name either way.
    names = [PureWindowsPath(name).name for name in db.get("file_list", [])]
    try:
        split = _split_names(names)
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from None
    lines = db.get("lines")
    lines = None if lines is None or len(lines) == 0 else [int(line) for line in lines]
    nrois = int(db.get("nrois") or 1)
    if split is not None or acquisition.source == SPLIT_SOURCE:
        # One run per split folder, as the lab's pipeline does it: one plane, the folder's own files.
        if split is None:
            raise ValueError(f"{where}: it read original TIFFs, but the given TIFFs are split copies, which lost the stack's page layout")
        if int(db.get("nplanes", 1)) != 1 or nrois != 1 or lines is not None or number != 0 or int(db.get("iplane", 0)) != 0:
            raise ValueError(f"{where}: a run over one split folder should have one plane and no lines, plane0")
        stem, roi, depth, numbers = split
        if acquisition.source == SPLIT_SOURCE:
            found = [f for f in acquisition.fields if [acquisition.files[i].name for i in sorted(set(f.file_index.tolist()))] == names]
            if len(found) != 1:
                raise ValueError(f"{where}: its file order differs from every field's TIFFs")
            return found[0], False
        stack_stems = {_FILE_NAME.match(file.name)["stem"] for file in acquisition.files}
        if stack_stems != {stem} or numbers != list(range(1, len(acquisition.files) + 1)):
            raise ValueError(f"{where}: its split files {names[0]}... aren't one copy per given TIFF of this acquisition, so its file order differs")
        zs = list(acquisition.layout.zs_um)
        if zs != sorted(zs) or len(set(zs)) != len(zs):
            raise ValueError(f"{where}: the splitter numbers depth folders in ascending z, but the stack's planes are at {zs}, so they can't be matched")
        found = [f for f in acquisition.fields if f.number == roi and f.plane == depth - 1]
        if len(found) != 1:
            raise ValueError(f"{where}: ROI {roi} at depth {depth} isn't a field of these TIFFs")
        return found[0], True
    if names != [file.name for file in acquisition.files]:
        raise ValueError(f"{where}: its file order differs from TIFFs")
    layout = acquisition.layout
    if int(db.get("nplanes", -1)) != layout.n_planes:
        raise ValueError(f"{where}: Suite2p's nplanes is {db.get('nplanes')}, the TIFFs have {layout.n_planes} planes")
    iplane = int(db.get("iplane", -1))
    iroi = int(db.get("iroi") or 0)
    if number != iplane * nrois + iroi or not 0 <= iroi < nrois:
        raise ValueError(f"{where}: its folder number doesn't match iplane {iplane} and iroi {iroi} of {nrois} ROIs")
    on_plane = [f for f in acquisition.fields if f.plane == iplane]
    if not on_plane:
        raise ValueError(f"{where}: iplane {iplane} is no plane of the TIFFs")
    if nrois > 1 or (lines is not None and len(on_plane) > 1):
        if lines is None:
            raise ValueError(f"{where}: {nrois} ROIs but no lines")
        rows = (lines[0], lines[-1] + 1)
        found = [f for f in on_plane if f.rows == rows]
        if lines != list(range(*rows)) or len(found) != 1:
            raise ValueError(f"{where}: lines {lines[0]}..{lines[-1]} aren't the rows of one ROI on plane {iplane}, {[f.rows for f in on_plane]}")
        return found[0], False
    if len(on_plane) != 1 or on_plane[0].rows != (0, layout.frame_shape[0]):
        raise ValueError(f"{where}: Suite2p read whole pages, but plane {iplane} holds {len(on_plane)} ROIs with flyback rows between them; it needs lines")
    if lines is not None and lines != list(range(layout.frame_shape[0])):
        raise ValueError(f"{where}: lines {lines[0]}..{lines[-1]} aren't the rows of the one ROI on plane {iplane}")
    return on_plane[0], False


def match_planes(roots, acquisition):
    """Each Suite2p plane folder with the field it holds, in the acquisition's field order.

    `roots` is one Suite2p output folder, or several, such as one per split ROI folder. Every field must be held by exactly
    one plane folder, and every plane folder Suite2p's db describes must be there. Anything else stops the conversion.
    """
    found = []
    for root in _roots(roots):
        folders = _plane_folders(root)
        metas = {number: read_plane(folder) for number, folder in folders.items()}
        dbs = {number: meta["db"] for number, meta in metas.items()}
        first = next(iter(dbs.values()))
        expected = int(first.get("nplanes", 1)) * int(first.get("nrois") or 1)
        if set(folders) != set(range(expected)):
            raise ValueError(f"{root}: Suite2p plane folders {sorted(folders)} differ from the {expected} its db describes")
        for number, folder in folders.items():
            field, split_copy = _field_for(f"Suite2p plane {number} in {root}", number, dbs[number], acquisition)
            found.append(Suite2pPlane(folder=folder, field=field, db=dbs[number], split_copy=split_copy, meta=metas[number]))
    by_field = {}
    for plane in found:
        if plane.field.label in by_field:
            raise ValueError(f"{plane.field.label} is held by both {by_field[plane.field.label].folder} and {plane.folder}")
        by_field[plane.field.label] = plane
    missing = [field.label for field in acquisition.fields if field.label not in by_field]
    if missing:
        raise ValueError(f"Suite2p plane folders differ from the TIFF planes: no folder holds {', '.join(missing)}")
    return [by_field[field.label] for field in acquisition.fields]


def _read_files(acquisition, plane):
    """Rows of acquisition.files that Suite2p read for this plane."""
    if acquisition.source == SPLIT_SOURCE:
        return sorted(set(plane.field.file_index.tolist()))
    return list(range(len(acquisition.files)))


def load_suite2p(roots, prepared):
    """Validate counts and geometry against TIFF headers; never use combined's padded traces.

    Returns one dict per field, in field order, holding its Suite2p arrays and `source`, its Suite2pPlane.
    NumPy object files must come from a trusted run: loading them executes pickle.
    """
    acquisition = prepared.acquisition
    planes = []
    for source in match_planes(roots, acquisition):
        field, folder = source.field, source.folder
        where = f"{field.label} (Suite2p plane {folder.name[5:]} in {folder.parent})"
        data = {name: _npy(folder, name) for name in ("stat", "iscell", "F", "Fneu", "spks")}
        data.update({key: source.meta[key] for key in ("db", "settings", "reg_outputs", "detect_outputs")})
        nframes = field.frame_indices.size
        if not nframes:
            raise ValueError(f"{where}: empty plane cannot be converted")
        db = data["db"]
        if db["nframes"] != nframes:
            raise ValueError(f"{where}: nframes differs from TIFF frame count")
        if (db["Ly"], db["Lx"]) != field.shape:
            raise ValueError(f"{where}: geometry differs from TIFF headers")
        counts = [int(np.sum(field.file_index == index)) for index in _read_files(acquisition, source)]
        if not np.array_equal(db["frames_per_file"], counts):
            raise ValueError(f"{where}: frames_per_file differs from TIFF pages")
        nroi = len(data["stat"])
        for key in ("F", "Fneu", "spks"):
            if data[key].shape != (nroi, nframes):
                raise ValueError(f"{where}: {key} ROI/frame dimensions differ")
            data[key] = coerce(data[key], "float32", file=folder, field=key)
        if data["iscell"].shape != (nroi, 2):
            raise ValueError(f"{where}: iscell shape differs")
        coerce(data["iscell"][:, 0], "bool", file=folder, field="iscell")
        if not np.all(np.isfinite(data["iscell"][:, 1]) & (data["iscell"][:, 1] >= 0) & (data["iscell"][:, 1] <= 1)):
            raise ValueError(f"{where}: invalid cell probability")
        for key in REGISTRATION_SERIES:
            if key in data["reg_outputs"] and data["reg_outputs"][key].shape[0] != nframes:
                raise ValueError(f"{where}: {key} frame count differs")
        for roi in data["stat"]:
            x = coerce(roi["xpix"], "int32", file=folder, field="xpix")
            y = coerce(roi["ypix"], "int32", file=folder, field="ypix")
            coerce(roi["lam"], "float32", file=folder, field="lam")
            if not len(x) or x.shape != y.shape or x.shape != np.shape(roi["lam"]):
                raise ValueError(f"{where}: empty or inconsistent ROI pixels")
            if np.any((x < 0) | (x >= db["Lx"]) | (y < 0) | (y >= db["Ly"])) or not np.all(np.isfinite(roi["lam"])):
                raise ValueError(f"{where}: invalid ROI pixels/weights")
        data["source"] = source
        planes.append(data)
    return planes


def _json_value(value):
    def numeric(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Unsupported source value: {type(item).__name__}")

    return json.dumps(value, default=numeric, allow_nan=True)


def _sources(roots):
    """All artifacts of one or several Suite2p folders, including combined and MATLAB duplicates; registered movies stay external."""
    return sorted(p for root in _roots(roots) for p in root.rglob("*") if p.is_file() and p.name != "data.bin")


def _relative(path, base):
    return str(Path(path).resolve().relative_to(base.resolve()))


def _artifact_bytes(path):
    """A file's bytes without reading it all into memory. A zero-byte file can't be memory-mapped."""
    return np.memmap(path, mode="r", dtype="uint8") if path.stat().st_size else np.zeros(0, dtype="uint8")


def _flatten_settings(prefix, value):
    """Every leaf of a nested settings dictionary, named by its path."""
    if isinstance(value, dict):
        for key, inner in value.items():
            yield from _flatten_settings(f"{prefix}.{key}", inner)
    else:
        yield prefix, value


def _companion_rows(companion):
    """Alignment rows naming the raw file a Suite2p file goes with."""
    return [
        ("companion.rawtiff.identifier", companion["identifier"], "That file's NWB identifier. Its DANDI name has desc-rawtiff."),
        ("companion.rawtiff.sha256", companion["sha256"], "SHA256 of that file when this one was written."),
        ("companion.rawtiff.relation", companion["relation"], "How the two files relate."),
    ] if companion else []


def _recorded_values(file):
    """What ScanImage and the session recorded that pins down one acquisition, as stored in an NWB file."""
    from datetime import datetime

    from .verify import _alignment_values

    ophys = file["processing/ophys"]
    return {
        "session_start_time": datetime.fromisoformat(file["session_start_time"].asstr()[()]),
        "scanimage.epoch": _alignment_values(ophys["scanimage_alignment"]).get("scanimage.epoch"),
        "file_name": list(ophys["scanimage_files/file_name"].asstr()[()]),
        "n_frames": ophys["scanimage_files/n_frames"][()].tolist(),
        "frame_number": ophys["scanimage_frames/frame_number"][()],
        "timestamp": ophys["scanimage_frames/timestamp"][()],
    }


def _expected_values(prepared):
    acquisition = prepared.acquisition
    return {
        "session_start_time": prepared.session.session_start_time,
        "scanimage.epoch": acquisition.epoch,
        "file_name": [file.name for file in acquisition.files],
        "n_frames": [file.n_frames for file in acquisition.files],
        "frame_number": acquisition.frames["frame_number"],
        "timestamp": acquisition.frames["timestamp"],
    }


def _same(stored, expected):
    if isinstance(expected, np.ndarray):
        return stored.shape == expected.shape and np.array_equal(stored, expected)
    return stored == expected


def read_companion(raw_nwb, prepared):
    """What ties a raw TIFF file to this one. Refuses a file from another acquisition or on another clock."""
    import h5py

    from .alignment import AlignmentError

    with h5py.File(raw_nwb, "r") as file:
        stored = _recorded_values(file)
        for name, expected in _expected_values(prepared).items():
            if not _same(stored[name], expected):
                raise AlignmentError(f"{raw_nwb}: its {name} differs, so it comes from another acquisition")
        if not np.array_equal(file["processing/ophys/scanimage_frames/session_time"][()], prepared.frame_times):
            raise AlignmentError(f"{raw_nwb}: its frame times differ, so the two files aren't on the same clock")
        for field in prepared.acquisition.fields:
            # A ROI's frames start at its first row, so its times differ from the frame table's. A stub keeps only the first frames.
            name = f"acquisition/TwoPhotonSeries{field.label}/timestamps"
            if name not in file:
                raise AlignmentError(f"{raw_nwb}: it has no {field.label}, so its fields differ from this acquisition's")
            stored_times = file[name][()]
            if not np.array_equal(stored_times, prepared.field_times(field)[: stored_times.size]):
                raise AlignmentError(f"{raw_nwb}: its {field.label} times differ, so the two files aren't on the same clock")
        identifier = file["identifier"].asstr()[()]
    digest = hashlib.sha256()
    with open(raw_nwb, "rb") as handle:
        while chunk := handle.read(64_000_000):
            digest.update(chunk)
    return dict(
        identifier=identifier,
        sha256=digest.hexdigest(),
        relation="Raw frames of the same acquisition, on the same clock. When this file was written, that file's session_start_time, "
        "scanimage.epoch, TIFF names and page counts, every frame's number, ScanImage time and session time, and each field's frame times matched this one's.",
    )


def _settings_prefix(folder, base):
    """A plane folder's settings key prefix: plane0 for one Suite2p folder, ROI01_z1/suite2p/plane0 for several."""
    return _relative(folder, base)


def _imaging_plane_text(field, layout):
    depth = layout.zs_um[field.plane]
    several = field.label != f"Plane{field.plane}"
    place = f"ROI {field.number} ({field.name}) on plane {field.plane}" if several else f"Plane {field.plane}"
    at = "at an unknown z, because the split files kept no ScanImage settings" if np.isnan(depth) else f"ScanImage z={depth:g} um"
    if several and field.origin_um is not None:
        at += "; origin_coords is the outer corner of the first pixel, in ScanImage's reference space, not a measured brain position"
    return f"{place}, {at}. PLACEHOLDER: indicator, location and wavelengths await metadata."


def build_suite2p_nwbfile(prepared, root, metadata=None, mapping=None, companion=None):
    """Build ophys interfaces plus a byte-exact source archive for fields without an NWB equivalent.

    `root` is one Suite2p output folder or several; each plane folder's field comes from `match_planes`.
    """
    from hdmf.backends.hdf5.h5_utils import H5DataIO
    from hdmf.common import ElementIdentifiers
    from pynwb import TimeSeries
    from pynwb.core import ScratchData
    from pynwb.base import Images
    from pynwb.device import Device
    from pynwb.image import GrayscaleImage
    from pynwb.ophys import Fluorescence, ImageSegmentation, OpticalChannel, RoiResponseSeries
    from ..virmen.convert import KEYWORDS, PLACEHOLDER_METADATA, build_nwbfile
    from .convert import PLACEHOLDER_IMAGING, _add_in_progress, _add_scanimage_tables, _table, COMPRESSION

    planes = load_suite2p(root, prepared)
    base = source_base(root)
    nwb = build_nwbfile(
        prepared.session,
        {
            **PLACEHOLDER_METADATA,
            "keywords": [*KEYWORDS, "two-photon calcium imaging", "Suite2p"],
            "session_description": "ViRMEn behavior and Suite2p two-photon results, aligned using TIFF headers and behavior I2C events.",
            **(metadata or {}),
        },
    )
    bidiphase = sorted({repr(data["source"].meta["registration"].get("do_bidiphase")) for data in planes})
    versions = sorted({str(data["source"].meta["version"]) for data in planes})
    several = any(data["source"].field.label != f"Plane{data['source'].field.plane}" for data in planes)
    nwb.notes = (
        "Companion to the raw TIFF NWB conversion of this acquisition: both use prepare_session, the same behavior, "
        "session-start shift and frame-time fit. Raw pixels are external and are not duplicated here. "
        "Per-plane Suite2p folders define lengths; combined contains padded display traces. "
        "Original Suite2p artifacts except data.bin are preserved byte-for-byte in scratch; object NPY files require a trusted pickle reader. "
        f"registration.do_bidiphase is {' or '.join(bidiphase)} in the Suite2p settings. Suite2p version: {', '.join(versions)}. "
        + (
            "Each field, one ROI on one plane, has its own ImagingPlane, PlaneSegmentation and response series, named Plane<p>Roi<r>; "
            "processing/ophys/suite2p_fields says which Suite2p folder holds which field, and its times are when the field's first row was scanned. "
            if several
            else ""
        )
        + "Files of this acquisition share session_start_time, scanimage.epoch, the scanimage_files names and page counts, and the scanimage_frames numbers and times; "
        "processing/ophys/scanimage_alignment names the raw file in its companion rows, when it was given."
    )
    _add_scanimage_tables(nwb, prepared, _companion_rows(companion))
    _add_in_progress(nwb, prepared)
    module = nwb.processing["ophys"]
    segmentation = ImageSegmentation()
    module.add(segmentation)
    containers = {key: Fluorescence(name=key) for key in ("Fluorescence", "Neuropil", "Deconvolved")}
    for container in containers.values():
        module.add(container)
    device = Device(
        name="Microscope", description="Two-photon microscope; acquisition metadata in processing/scanimage."
    )
    nwb.add_device(device)
    layout = prepared.acquisition.layout
    image_coordinates = []
    field_rows = []
    for data in planes:
        source = data["source"]
        field, suffix = source.field, source.suffix
        plane_options = {}
        if field.pixel_size_um is not None:
            plane_options.update(grid_spacing=list(field.pixel_size_um), grid_spacing_unit="micrometers")
        if field.origin_um is not None and field.label != f"Plane{field.plane}":
            plane_options.update(origin_coords=list(field.origin_um), origin_coords_unit="micrometers")
        imaging_plane = nwb.create_imaging_plane(
            name=f"ImagingPlane{suffix}",
            optical_channel=OpticalChannel(
                name=f"OpticalChannel{suffix}", description="Only saved channel.", emission_lambda=float("nan")
            ),
            description=_imaging_plane_text(field, layout),
            device=device,
            imaging_rate=layout.volume_rate,
            **plane_options,
            **{k: v for k, v in PLACEHOLDER_IMAGING.items() if k != "emission_lambda"},
        )
        # emission_lambda belongs to OpticalChannel, not ImagingPlane.
        ps = segmentation.create_plane_segmentation(
            name=f"PlaneSegmentation{suffix}",
            imaging_plane=imaging_plane,
            id=ElementIdentifiers(name="id", data=np.array([], dtype="int64")),
            description="Every Suite2p ROI, including rejected cells. Full-frame x=column, y=row. stat_* columns are JSON source values; native dtype is preserved in the source archive.",
        )
        ps.add_column(
            "iscell",
            "Suite2p classifier decision; no ROI filtering.",
            data=[] if len(data["stat"]) else np.array([], dtype=bool),
        )
        ps.add_column(
            "cell_probability",
            "Suite2p classifier probability.",
            data=[] if len(data["stat"]) else np.array([], dtype="float64"),
        )
        stat_keys = sorted(set().union(*(set(roi) for roi in data["stat"])))
        for key in stat_keys:
            ps.add_column(
                f"stat_{key}", f"Original stat.{key} as JSON; NaN remains NaN; empty text means this ROI has no such key; see native source archive for dtype."
            )
        for index, roi in enumerate(data["stat"]):
            # Explicit integer fields and float32 weights avoid HDMF silently casting tuple floats to uint32.
            mask = np.empty(len(roi["xpix"]), dtype=[("x", "uint32"), ("y", "uint32"), ("weight", "float32")])
            mask["x"], mask["y"], mask["weight"] = roi["xpix"], roi["ypix"], roi["lam"]
            ps.add_roi(
                pixel_mask=[(np.uint32(x), np.uint32(y), np.float32(w)) for x, y, w in mask],
                iscell=bool(data["iscell"][index, 0]),
                cell_probability=float(data["iscell"][index, 1]),
                **{f"stat_{k}": _json_value(roi[k]) if k in roi else "" for k in stat_keys},
            )
        times = prepared.field_times(field)
        if len(data["stat"]):
            region = ps.create_roi_table_region(
                region=list(range(len(data["stat"]))), description="All ROIs in source order."
            )
            for name, key in TRACE_SERIES:
                containers[name].add_roi_response_series(
                    RoiResponseSeries(
                        name=field.label,
                        rois=region,
                        data=H5DataIO(data[key].T, **COMPRESSION),
                        timestamps=H5DataIO(times, **COMPRESSION),
                        unit="a.u.",
                        description=f"Suite2p {key}, unchanged float32 values. spks are deconvolved activity estimates, not measured spike counts.",
                    )
                )
        images = Images(
            name=f"SummaryImages{suffix}",
            description="Images are (x,y). Cropped detection images start at (xrange[0],yrange[0]); masks use full-frame coordinates.",
        )
        for source_name in ("reg_outputs", "detect_outputs"):
            for key, value in data[source_name].items():
                if (
                    isinstance(value, np.ndarray)
                    and value.ndim == 2
                    and key in SUMMARY_IMAGES
                ):
                    rows, cols = field.shape
                    xr = data["reg_outputs"].get("xrange", [0, cols])
                    yr = data["reg_outputs"].get("yrange", [0, rows])
                    if value.shape == (rows, cols):
                        origin = (0, 0)
                    elif value.shape == (yr[1] - yr[0], xr[1] - xr[0]):
                        origin = (int(xr[0]), int(yr[0]))
                    else:
                        raise ValueError(f"{field.label}: {key} image shape does not match full frame or crop")
                    image_coordinates.append((f"SummaryImages{suffix}/{key}", *origin, value.shape[1], value.shape[0]))
                    images.add_image(
                        GrayscaleImage(
                            name=key,
                            data=H5DataIO(value.T, **COMPRESSION),
                            description=f"{source_name}.{key}, transposed from (row,column). Crop origin: {data['reg_outputs'].get('xrange')}, {data['reg_outputs'].get('yrange')}; full-size images have origin (0,0).",
                        )
                    )
        if images.images:
            module.add(images)
        for key in REGISTRATION_SERIES:
            if key in data["reg_outputs"]:
                module.add(
                    TimeSeries(
                        name=f"{field.label}_{key}",
                        data=H5DataIO(data["reg_outputs"][key], **COMPRESSION),
                        timestamps=H5DataIO(times, **COMPRESSION),
                        unit="pixels" if "off" in key else "n.a.",
                        description=f"Suite2p registration {key}, unchanged; bad frames remain in all traces and timestamps.",
                    )
                )
        field_rows.append((field, source, _relative(source.folder, base)))
    setting_rows = []

    def add_settings(prefix, values):
        for key, value in values.items():
            name = f"{prefix}.{key}"
            if isinstance(value, dict):
                add_settings(name, value)
            else:
                setting_rows.append((name, _json_value(value)))

    for data in planes:
        add_settings(_settings_prefix(data["source"].folder, base), data["settings"])
    module.add(
        _table(
            "suite2p_settings",
            "Suite2p settings and version as recorded, without changing any value. Keys start with the plane folder. fs is processing provenance, not the source of timestamps.",
            [
                ("key", "Plane folder and original setting path.", np.array([r[0] for r in setting_rows], dtype=object)),
                ("value", "Original value as JSON.", np.array([r[1] for r in setting_rows], dtype=object)),
            ],
        )
    )
    module.add(
        _table(
            "suite2p_fields",
            "Which Suite2p plane folder holds which field, one ROI on one plane, and what pixels Suite2p read for it.",
            [
                ("field", "The field's name, which names its ImagingPlane, PlaneSegmentation and response series.", np.array([f.label for f, _, _ in field_rows], dtype=object)),
                ("plane", "0-based plane of the acquisition.", np.array([f.plane for f, _, _ in field_rows], dtype="int32")),
                ("roi", "1-based ROI number, in the header's ROI order and the split folders' numbering.", np.array([f.number for f, _, _ in field_rows], dtype="int32")),
                ("roi_name", "ROI name in the header, or its folder's when the split files kept no header.", np.array([f.name for f, _, _ in field_rows], dtype=object)),
                ("rows", "Rows of a stored page holding the field, as start:stop.", np.array([f"{f.rows[0]}:{f.rows[1]}" for f, _, _ in field_rows], dtype=object)),
                ("time_offset_seconds", "Seconds from a frame's start to the field's first row, which its times include.", np.array([f.time_offset for f, _, _ in field_rows], dtype="float64")),
                ("folder", "The Suite2p plane folder, relative to the Suite2p source paths.", np.array([r for _, _, r in field_rows], dtype=object)),
                ("input", "The pixels Suite2p read for this field.", np.array([s.input for _, s, _ in field_rows], dtype=object)),
            ],
        )
    )
    if image_coordinates:
        module.add(
            _table(
                "summary_image_coordinates",
                "Image coordinates relative to full-frame pixel masks. Subtract x_origin and y_origin from mask coordinates to index a cropped image.",
                [
                    (
                        "image",
                        "Image path within processing/ophys.",
                        np.array([r[0] for r in image_coordinates], dtype=object),
                    ),
                    *[
                        (key, meaning, np.array([r[i] for r in image_coordinates], dtype="int32"))
                        for i, key, meaning in (
                            (1, "x_origin", "Full-frame x of first image column."),
                            (2, "y_origin", "Full-frame y of first image row."),
                            (3, "width", "Image x dimension."),
                            (4, "height", "Image y dimension."),
                        )
                    ],
                ],
            )
        )
    if mapping is not None:
        nwb.add_scratch(
            json.dumps(mapping),
            name="suite2p_frame_mapping",
            description="Measured per-frame raw-to-registered reconstruction RMSE and neighboring-page controls. Sampled first, middle and last frame in every TIFF for each field. See docs/imaging-suite2p.md for numerical and sampling limits.",
        )
    inventory = []
    for index, path in enumerate(_sources(root)):
        relative = _relative(path, base)
        blob = _artifact_bytes(path)
        name = f"suite2p_source_{index:04d}"
        nwb.add_scratch(
            ScratchData(
                data=H5DataIO(blob, **COMPRESSION) if blob.size else blob,
                name=name,
                description=f"Original artifact bytes: {relative}. SHA256 {hashlib.sha256(blob).hexdigest()}. No interpretation or dtype conversion.",
            )
        )
        inventory.append((relative, name, hashlib.sha256(blob).hexdigest()))
    module.add(
        _table(
            "source_inventory",
            "Byte-exact source artifacts in scratch. Only data.bin is excluded (registered movie remains external). Combined and Fall.mat are archived as provenance, never used as trace sources.",
            [
                ("path", "Relative source path.", np.array([x[0] for x in inventory], dtype=object)),
                (
                    "scratch_name",
                    "ScratchData holding the original bytes.",
                    np.array([x[1] for x in inventory], dtype=object),
                ),
                ("sha256", "Source SHA256.", np.array([x[2] for x in inventory], dtype=object)),
            ],
        )
    )
    return nwb


def convert_suite2p(log, tiffs, root, output, file=None, metadata=None, raw_nwb=None, allow_incomplete=False):
    """Write, then independently verify; no upload or automatic organization. `root` is one Suite2p folder or several."""
    from pynwb import NWBHDF5IO
    from .convert import prepare_session

    prepared = prepare_session(log, tiffs, file=file, allow_incomplete=allow_incomplete)
    load_suite2p(root, prepared)
    from .suite2p_mapping import prove_frame_mapping

    mapping = prove_frame_mapping(prepared.acquisition, root)
    companion = read_companion(raw_nwb, prepared) if raw_nwb is not None else None
    with NWBHDF5IO(str(output), "w") as io:
        io.write(build_suite2p_nwbfile(prepared, root, metadata, mapping, companion))
    verify_suite2p(prepared, output, root, tiffs, companion=companion, mapping=mapping)
    return prepared


def verify_suite2p(prepared, output, root, tiffs, companion=None, mapping=None):
    """Compare HDF5 against fresh NumPy files and independently parsed TIFF headers, exactly."""
    import h5py
    from ..virmen.verify import verify_round_trip, _check_series
    from .verify import RoundTripError, _alignment_values, _expected_field_times, _read_headers, check_scanimage_tables

    def equal(actual, expected, label):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape or not np.array_equal(
            actual, expected, equal_nan=actual.dtype.kind in "fc" and expected.dtype.kind in "fc"
        ):
            raise RoundTripError(f"{label}: differs from source")

    verify_round_trip(prepared.session, output)
    headers = _read_headers(tiffs)
    numbers, times = headers["columns"]["frame_number"], headers["columns"]["timestamp"]
    base = source_base(root)
    planes = match_planes(root, prepared.acquisition)
    with h5py.File(output) as f:
        scan = f["processing/ophys"]
        check_scanimage_tables(scan, prepared, headers, _companion_rows(companion))
        values = _alignment_values(scan["scanimage_alignment"])
        slope, intercept, offset = [float(values[f"fit.{k}"]) for k in ("slope", "intercept", "clock_offset")]
        session_times = (times - intercept) / slope + offset
        equal(scan["scanimage_frames/session_time"][()], session_times, "session times")
        if mapping is not None and json.loads(f["scratch/suite2p_frame_mapping"].asstr()[()]) != mapping:
            raise RoundTripError("the stored frame mapping evidence differs from what was measured")
        for series in prepared.in_progress:
            _check_series(f["processing/behavior"], series)
        ophys = f["processing/ophys"]
        fields = ophys["suite2p_fields"]
        equal(fields["field"].asstr()[()], np.array([p.field.label for p in planes], dtype=object), "suite2p_fields field")
        equal(fields["folder"].asstr()[()], np.array([_relative(p.folder, base) for p in planes], dtype=object), "suite2p_fields folder")
        equal(fields["input"].asstr()[()], np.array([p.input for p in planes], dtype=object), "suite2p_fields input")
        equal(fields["plane"][()], [p.field.plane for p in planes], "suite2p_fields plane")
        equal(fields["roi"][()], [p.field.number for p in planes], "suite2p_fields roi")
        for plane in planes:
            field, folder, suffix = plane.field, plane.folder, plane.suffix
            field_times = (_expected_field_times(field, headers, prepared.acquisition) - intercept) / slope + offset
            meta = read_plane(folder)
            stat = np.load(folder / "stat.npy", allow_pickle=True)
            cells = np.load(folder / "iscell.npy")
            segmentation = ophys[f"ImageSegmentation/PlaneSegmentation{suffix}"]
            link = getattr(segmentation.get("imaging_plane", getlink=True), "path", None)
            if link != f"/general/optophysiology/ImagingPlane{suffix}":
                raise RoundTripError(f"PlaneSegmentation{suffix}: its imaging plane is {link}")
            equal(segmentation["id"][()], np.arange(len(stat)), "ROI ids")
            db, reg = meta["db"], meta["reg_outputs"]
            equal(segmentation["iscell"][()], cells[:, 0], "iscell")
            equal(segmentation["cell_probability"][()], cells[:, 1], "probability")
            cursor = 0
            for i, roi in enumerate(stat):
                end = int(segmentation["pixel_mask_index"][i])
                mask = segmentation["pixel_mask"][cursor:end]
                equal(mask["x"], roi["xpix"], "mask x")
                equal(mask["y"], roi["ypix"], "mask y")
                equal(mask["weight"], roi["lam"], "mask weights")
                cursor = end
                for key, value in roi.items():
                    equal(json.loads(segmentation[f"stat_{key}"].asstr()[i]), value, f"stat.{key}")
                for column in segmentation:
                    if column.startswith("stat_") and column.removeprefix("stat_") not in roi and segmentation[column].asstr()[i] != "":
                        raise RoundTripError(f"{column}: ROI {i} has no such key, but the file holds a value")
            for name, key in TRACE_SERIES:
                source = np.load(folder / f"{key}.npy", mmap_mode="r")
                if len(stat):
                    series = ophys[f"{name}/{field.label}"]
                    if series["data"].dtype != np.float32:
                        raise RoundTripError(key + " dtype differs")
                    equal(series["data"][()], source.T, key)
                    table = f[series["rois"].attrs["table"]].name
                    if table != f"/processing/ophys/ImageSegmentation/PlaneSegmentation{suffix}":
                        raise RoundTripError(f"{name}/{field.label}: rois point at {table}")
                    equal(series["rois"][()], np.arange(len(stat)), f"{key} rois")
                    equal(series["timestamps"][()], field_times, f"{field.label} {key} times")
            for source in ("reg_outputs", "detect_outputs"):
                for key, value in meta[source].items():
                    if key in REGISTRATION_SERIES:
                        if f"{field.label}_{key}" not in ophys:
                            raise RoundTripError("missing registration " + key)
                        equal(ophys[f"{field.label}_{key}/data"][()], value, key)
                        equal(ophys[f"{field.label}_{key}/timestamps"][()], field_times, f"{field.label} {key} times")
                    if (
                        isinstance(value, np.ndarray)
                        and value.ndim == 2
                        and key in SUMMARY_IMAGES
                    ):
                        if f"SummaryImages{suffix}/{key}" not in ophys:
                            raise RoundTripError("missing summary image " + key)
                        equal(ophys[f"SummaryImages{suffix}/{key}"][()], value.T, key)
                        table = ophys["summary_image_coordinates"]
                        row = list(table["image"].asstr()[()]).index(f"SummaryImages{suffix}/{key}")
                        origin = (0, 0) if value.shape == (db["Ly"], db["Lx"]) else (reg["xrange"][0], reg["yrange"][0])
                        equal(
                            [table["x_origin"][row], table["y_origin"][row], table["width"][row], table["height"][row]],
                            [*origin, value.shape[1], value.shape[0]],
                            "summary image coordinates",
                        )
        table = ophys["suite2p_settings"]
        stored = dict(zip(table["key"].asstr()[()], table["value"].asstr()[()]))
        expected = {}
        for plane in planes:
            original = read_plane(plane.folder)["settings"]
            expected.update(_flatten_settings(_settings_prefix(plane.folder, base), original))
        if set(stored) != set(expected):
            raise RoundTripError(f"Suite2p settings differ: {sorted(set(stored) ^ set(expected))}")
        for key, value in expected.items():
            equal(json.loads(stored[key]), value, "Suite2p setting " + key)
        inventory = ophys["source_inventory"]
        sources = _sources(root)
        if list(inventory["path"].asstr()[()]) != [_relative(p, base) for p in sources]:
            raise RoundTripError("source archive inventory differs")
        for index, (path, name) in enumerate(zip(sources, inventory["scratch_name"].asstr()[()])):
            source = _artifact_bytes(path)
            if inventory["sha256"].asstr()[index] != hashlib.sha256(source).hexdigest():
                raise RoundTripError("source archive hash differs")
            dataset = f[f"scratch/{name}"]
            if dataset.shape != source.shape:
                raise RoundTripError(f"{path.name}: archive length differs")
            for start in range(0, len(source), 8_000_000):
                equal(dataset[start : start + 8_000_000], source[start : start + 8_000_000], _relative(path, base))
    return {
        "fields_checked": len(planes),
        "frames_checked": len(numbers),
        "artifacts_checked": len(sources),
    }
