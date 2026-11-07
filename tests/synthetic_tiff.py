"""ScanImage TIFFs built by hand, with no project data."""

import json

import numpy as np
import tifffile

FRAME_PERIOD = 0.02
EPOCH = "[2026  7 15 11 59 50.000]"


def software(**overrides):
    values = {
        "SI.VERSION_MAJOR": "2022",
        "SI.objectiveResolution": "15",
        "SI.hChannels.channelSave": "1",
        "SI.hFastZ.discardFlybackFrames": "false",
        "SI.hFastZ.numDiscardFlybackFrames": "0",
        "SI.hRoiManager.linePeriod": "6.3e-05",
        "SI.hRoiManager.linesPerFrame": "3",
        "SI.hRoiManager.mroiEnable": "false",
        "SI.hRoiManager.pixelsPerLine": "4",
        "SI.hRoiManager.scanFramePeriod": "0.02",
        "SI.hRoiManager.scanFrameRate": "50",
        "SI.hRoiManager.scanVolumeRate": "25",
        "SI.hScan2D.logAverageFactor": "1",
        "SI.hScan2D.logFramesPerFile": "Inf",
        "SI.hStackManager.actualNumSlices": "2",
        "SI.hStackManager.framesPerSlice": "1",
        "SI.hStackManager.numFramesPerVolumeWithFlyback": "2",
        "SI.hStackManager.numVolumes": "50000",
        "SI.hStackManager.zs": "[10 50]",
    }
    values.update(overrides)
    return "\n".join(f"{key} = {value}" for key, value in values.items() if value is not None) + "\n"


def artist(n_rois=1, pixels=(4, 3), size=(8.0, 8.0)):
    scanfields = {"pixelResolutionXY": list(pixels), "sizeXY": list(size), "centerXY": [0, 0]}
    rois = [{"name": f"ROI {i}", "zs": [10, 50], "scanfields": scanfields} for i in range(n_rois)]
    return json.dumps({"RoiGroups": {"imagingRoiGroup": {"rois": rois[0] if n_rois == 1 else rois}}})


def frame_pixels(frame_number, rows=3, cols=4):
    """Every pixel differs across frames and positions, and some are negative, as in real data."""
    return (np.arange(rows * cols, dtype="int64").reshape(rows, cols) * 7 + frame_number * 30 - 500).astype("int16")


def i2c(*packets):
    """The I2CData text for (time, (block, trial, iteration)) packets."""
    if not packets:
        return "{}"
    parts = []
    for time, values in packets:
        data = ",".join(str(byte) for byte in np.array(values, dtype="<u2").tobytes())
        parts.append(f"{{{time:.9f}, [{data}]}}")
    return "{" + " ".join(parts) + " }"


def description(frame_number, *, time=None, i2c_text="{}", epoch=EPOCH, aux="[]", extra=""):
    time = (frame_number - 1) * FRAME_PERIOD if time is None else time
    return (
        f"frameNumbers = {frame_number}\n"
        "acquisitionNumbers = 1\n"
        f"frameNumberAcquisition = {frame_number}\n"
        f"frameTimestamps_sec = {time:.9f}\n"
        f"acqTriggerTimestamps_sec = {-0.000075165 if frame_number == 1 else -1.0:.9f}\n"
        "nextFileMarkerTimestamps_sec = -1.000000000\n"
        "endOfAcquisition = 0\n"
        "endOfAcquisitionMode = 0\n"
        "dcOverVoltage = 0\n"
        f"epoch = {epoch}\n"
        f"auxTrigger0 = {aux}\nauxTrigger1 = []\nauxTrigger2 = []\nauxTrigger3 = []\n"
        f"I2CData = {i2c_text}\n{extra}"
    )


def write_acquisition(folder, n_frames, frames_per_file=4, *, packets=None, software_text=None, artist_text=None, rows=3, cols=4, stem="mouse_00001", skip_frames=(), raw_i2c=None, **description_options):
    """Write one acquisition split across files, and return the paths. `packets` maps a frame number to its packets."""
    packets = packets or {}
    numbers = [n for n in range(1, n_frames + 1) if n not in skip_frames]
    paths = []
    for file_index, start in enumerate(range(0, len(numbers), frames_per_file), start=1):
        path = folder / f"{stem}_{file_index:05d}.tif"
        with tifffile.TiffWriter(path, bigtiff=True) as writer:
            for position, number in enumerate(numbers[start : start + frames_per_file]):
                options = dict(
                    description=description(number, i2c_text=(raw_i2c or {}).get(number, i2c(*packets.get(number, ()))), **description_options),
                    software=(software_text or software()) if position == 0 else None,
                    extratags=[(315, "s", 0, artist_text or artist(pixels=(cols, rows)), True)] if position == 0 else [],
                    contiguous=False,
                    metadata=None,
                )
                writer.write(frame_pixels(number, rows, cols), **options)
        paths.append(path)
    return paths


FLYBACK_ROWS = 2


def mroi_artist(rois):
    """The ROI group for `rois`, a list of dicts with rows, and optionally name, z, center and cols."""
    built = []
    for number, roi in enumerate(rois, start=1):
        built.append({
            "name": roi.get("name", f"area{number}"),
            "zs": roi.get("z", 10),
            "scanfields": roi.get("scanfields") or {"pixelResolutionXY": [roi.get("cols", 4), roi["rows"]], "sizeXY": [8.0, 8.0], "centerXY": roi.get("center", [float(number), 0.0]), "rotationDegrees": roi.get("rotation", 0)},
        })
    return json.dumps({"RoiGroups": {"imagingRoiGroup": {"rois": built}}})


def mroi_software(zs=(10,), page_rows=8, **overrides):
    many = len(zs) > 1
    return software(**{
        "SI.hRoiManager.mroiEnable": "1",
        "SI.hRoiManager.linesPerFrame": "3",
        "SI.hStackManager.actualNumSlices": str(len(zs)),
        "SI.hStackManager.numFramesPerVolumeWithFlyback": str(len(zs)),
        "SI.hStackManager.zs": ("[" + " ".join(map(str, zs)) + "]") if many else str(zs[0]),
        **overrides,
    })


def write_stack(folder, n_frames, rois, zs=(10,), page_rows=None, frames_per_file=4, software_text=None, stem="mouse_00001", **options):
    """A multi-ROI acquisition as ScanImage saves it: each page stacks its plane's ROIs with flyback rows between them."""
    first_plane = [roi for roi in rois if roi.get("z", 10) == zs[0]]
    page_rows = page_rows or sum(roi["rows"] for roi in first_plane) + FLYBACK_ROWS * (len(first_plane) - 1)
    return write_acquisition(
        folder, n_frames, frames_per_file=frames_per_file, software_text=software_text or mroi_software(zs), artist_text=mroi_artist(rois), rows=page_rows, cols=rois[0].get("cols", 4), stem=stem, **options
    )


def _matlab_number(value):
    """num2str: four decimals at most, with trailing zeros gone."""
    return f"{value:.4f}".rstrip("0").rstrip(".")


def split_stack(paths, folder, rois, zs=(10,), lag=0.001, keep_tags=False, software_text=None, artist_text=None, file_local_depth=False):
    """What the lab's ScanInfo splitter does to a stack: one folder per ROI and depth, negatives clipped into uint16,
    each ROI's frame time moved by a fixed lag and rounded, and in older runs no Software or Artist tags.

    Like the MATLAB, it replaces the old time everywhere in the header, so an I2C packet stamped with exactly the frame's
    time changes too. `file_local_depth` restarts the depth cycle in each file, as the MATLAB's zIdx does.
    """
    written = []
    for depth, z in enumerate(zs, start=1):
        row, lag_steps = 0, 0
        plane_rois = [(number, roi) for number, roi in enumerate(rois, start=1) if roi.get("z", 10) == z]
        for number, roi in plane_rois:
            where = folder / f"ROI{number:02d}_z{depth}"
            where.mkdir(parents=True, exist_ok=True)
            for path in paths:
                stem, file_number = path.stem.rsplit("_", 1)
                out = where / f"{stem}_ROI{number:02d}_z{depth}_{file_number}.tif"
                with tifffile.TiffFile(path) as source, tifffile.TiffWriter(out, bigtiff=False) as writer:
                    if file_local_depth:
                        pages = [page for position, page in enumerate(source.pages) if position % len(zs) == depth - 1]
                    else:
                        pages = [page for page in source.pages if (int(page.description.split("\n")[0].split(" = ")[1]) - 1) % len(zs) == depth - 1]
                    for position, page in enumerate(pages):
                        text = page.description
                        old = text.split("frameTimestamps_sec = ")[1].split("\n")[0]
                        new = _matlab_number(float(old) + lag * lag_steps)
                        head, packets = text.split("I2CData = ")
                        # A whole number would also hit fields like nextFileMarker = -1.000000000, which real times don't.
                        text = head.replace(f"frameTimestamps_sec = {old}", f"frameTimestamps_sec = {new}") + "I2CData = " + packets.replace(old, new)
                        pixels = np.clip(page.asarray()[row : row + roi["rows"]], 0, None).astype("uint16")
                        options = {}
                        if keep_tags and position == 0:
                            options = dict(software=software_text, extratags=[(315, "s", 0, artist_text, True)])
                        writer.write(pixels, description=text, contiguous=False, metadata=None, **options)
                written.append(out)
            lag_steps += 1
            row += roi["rows"] + FLYBACK_ROWS
    return written
