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
        "SI.hStackManager.actualNumSlices": "2",
        "SI.hStackManager.framesPerSlice": "1",
        "SI.hStackManager.numFramesPerVolumeWithFlyback": "2",
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
