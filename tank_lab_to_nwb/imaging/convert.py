"""Write raw two-photon TIFFs and their ViRMEn log to one NWB file, on one clock. Nothing here uploads."""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.common import DynamicTable, VectorData
from hdmf.data_utils import GenericDataChunkIterator
from pynwb import NWBHDF5IO, TimeSeries
from pynwb.device import Device
from pynwb.image import ImageSeries
from pynwb.ophys import OpticalChannel, TwoPhotonSeries

from ..virmen.coerce import coerce
from ..virmen.convert import KEYWORDS, PLACEHOLDER_METADATA, build_nwbfile
from ..virmen.loader import load_log, records
from ..virmen.session import NormalizedSession, normalize_log
from .alignment import (
    AlignmentError,
    ClockFit,
    alignment_rows,
    fit_clock,
    frame_labels,
    in_progress_series,
    in_progress_settings,
    iteration_clock,
    match_packets,
    session_start_shift,
    shifted_session,
)
from .scanimage import SPLIT_SOURCE, Acquisition, read_acquisition, read_field, read_flyback
from .verify import verify_imaging

COMPRESSION = dict(compression="gzip", compression_opts=4, shuffle=True)
_CHUNK_BYTES = 10_000_000
_BUFFER_BYTES = 1_000_000_000

# Values in the imaging tables that stand for "none", beyond the ViRMEn field registry's own.
SENTINELS = {
    **{name: dict(value=0, meaning="This frame has no label: it's before the first packet, after the last, or in the trial running at save.") for name in ("behavior_block", "behavior_trial", "behavior_iteration")},
    **{name: dict(value="NaN", meaning="PLACEHOLDER: the wavelength isn't recorded in these sources.") for name in ("excitation_lambda", "emission_lambda")},
}

PLACEHOLDER_IMAGING = dict(
    indicator="PLACEHOLDER: indicator not recorded here.",
    location="root",  # Allen's term for the whole brain, the most this file can honestly claim
    excitation_lambda=float("nan"),
    emission_lambda=float("nan"),
)


@dataclass(frozen=True)
class PreparedSession:
    """A behavior session and an acquisition on one clock. Whatever gets written next, raw or Suite2p, shares this."""

    session: NormalizedSession
    acquisition: Acquisition
    fit: ClockFit
    clock_offset: float  # seconds added to the elapsed clock, including the shift
    shift_microseconds: int
    frame_times: np.ndarray  # per frame, on the file's clock
    packet_elapsed: np.ndarray
    packet_times: np.ndarray  # per packet, on the file's clock
    labels: dict
    in_progress: list

    def field_times(self, field):
        """A field's frame times on the file's clock."""
        return self.fit.to_elapsed(field.timestamps) + self.clock_offset


def prepare_session(log, tiff_paths, file=None, allow_incomplete=False):
    """Read and align without writing. `log` is a path or an already decoded log."""
    if isinstance(log, (str, Path)):
        file = file or log
        log = load_log(log)
    acquisition = read_acquisition(tiff_paths, allow_incomplete=allow_incomplete)
    packets = acquisition.packets
    elapsed = match_packets(packets, iteration_clock(log))
    if np.any(np.diff(packets["timestamp"]) <= 0):
        raise AlignmentError("I2C packet times don't always increase")
    fit = fit_clock(packets["timestamp"], elapsed, packets["iteration"])
    session = normalize_log(log, file=file)
    offsets = {round(float(offset), 9) for offset in session.coverage.clock["offsets"]}
    if len(offsets) != 1:
        raise AlignmentError(f"ViRMEn's clock restarted in this session ({len(offsets)} clock runs), and one line can't span that")
    offset = float(session.coverage.clock["offsets"][0])
    shift = session_start_shift(fit.to_elapsed(acquisition.frames["timestamp"]) + offset)
    session = shifted_session(session, shift)
    clock_offset = offset + shift / 1e6
    in_progress, record = in_progress_series(log, clock_offset)
    if record is not None:
        session = replace(session, settings=[*session.settings, *in_progress_settings(record)])
    trials_per_block = [len(records(block["trial"])) for block in records(log["block"])]
    return PreparedSession(
        session=session,
        acquisition=acquisition,
        fit=fit,
        clock_offset=clock_offset,
        shift_microseconds=shift,
        frame_times=fit.to_elapsed(acquisition.frames["timestamp"]) + clock_offset,
        packet_elapsed=elapsed,
        packet_times=fit.to_elapsed(packets["timestamp"]) + clock_offset,
        labels=frame_labels(acquisition, trials_per_block),
        in_progress=in_progress,
    )


class _FramePixels(GenericDataChunkIterator):
    """Frames as (time, x, y), read a buffer at a time, so a series never has to fit in memory. `read` gives (frames, rows, columns)."""

    def __init__(self, read, n_frames, shape, dtype):
        self._read = read
        self._n_frames = n_frames
        self._shape = shape
        self._dtype = dtype.newbyteorder("=")
        rows, columns = shape
        frame_bytes = rows * columns * dtype.itemsize
        chunk = max(1, min(n_frames, _CHUNK_BYTES // frame_bytes))
        buffer = max(1, _BUFFER_BYTES // (frame_bytes * chunk)) * chunk
        super().__init__(chunk_shape=(chunk, columns, rows), buffer_shape=(min(n_frames, buffer), columns, rows), display_progress=False)

    def _get_data(self, selection):
        positions = np.arange(self._n_frames)[selection[0]]
        return self._read(positions).transpose(0, 2, 1)[(slice(None), selection[1], selection[2])]

    def _get_maxshape(self):
        rows, columns = self._shape
        return (self._n_frames, columns, rows)

    def _get_dtype(self):
        return self._dtype


def _compressed(data):
    return H5DataIO(data=data, **COMPRESSION) if np.size(data) else data


def _table(name, description, columns):
    built = [VectorData(name=column, description=text, data=_compressed(data)) for column, text, data in columns]
    return DynamicTable(name=name, description=description, columns=built, id=np.arange(len(columns[0][2])))


def _int32(data, field):
    """A header column as int32, exactly, or a LossyConversionError naming the column."""
    return coerce(data, "int32", file="ScanImage headers", field=field)


def _add_scanimage_tables(nwbfile, prepared, extra_alignment_rows=()):
    acquisition, frames, packets = prepared.acquisition, prepared.acquisition.frames, prepared.acquisition.packets
    module = nwbfile.create_processing_module("ophys", "ScanImage's per-frame headers, the I2C packets that align frames to behavior, and any processed results.")
    module.add(_table("scanimage_frames", "One row per TIFF page, in acquisition order.", [
        ("file_index", "Row of the scanimage_files table this page is in.", _int32(frames["file_index"], "file_index")),
        ("page_index", "0-based page within its file.", _int32(frames["page_index"], "page_index")),
        ("frame_number", "frameNumbers.", _int32(frames["frame_number"], "frame_number")),
        ("acquisition_number", "acquisitionNumbers.", _int32(frames["acquisition_number"], "acquisition_number")),
        ("frame_number_acquisition", "frameNumberAcquisition.", _int32(frames["frame_number_acquisition"], "frame_number_acquisition")),
        ("plane", "0-based plane, which names the TwoPhotonSeries that holds this page.", _int32(acquisition.planes, "plane")),
        ("timestamp", "frameTimestamps_sec, seconds on the ScanImage clock.", frames["timestamp"]),
        ("session_time", "This frame's time in the file, in seconds since session_start_time.", prepared.frame_times),
        ("acq_trigger_timestamp", "acqTriggerTimestamps_sec.", frames["acq_trigger_timestamp"]),
        ("next_file_marker_timestamp", "nextFileMarkerTimestamps_sec.", frames["next_file_marker_timestamp"]),
        ("end_of_acquisition", "endOfAcquisition.", _int32(frames["end_of_acquisition"], "end_of_acquisition")),
        ("end_of_acquisition_mode", "endOfAcquisitionMode.", _int32(frames["end_of_acquisition_mode"], "end_of_acquisition_mode")),
        ("dc_over_voltage", "dcOverVoltage.", _int32(frames["dc_over_voltage"], "dc_over_voltage")),
        ("behavior_block", "Behavior block as the pipeline's SyncImagingBehavior labels this frame, 0 for none. These labels mark which header got a packet: at a trial's onset, the first labelled frame can have started just before the trial began, or can run up to a frame late. Use the timestamps to find a trial's frames.", _int32(prepared.labels["block"], "behavior_block")),
        ("behavior_trial", "Behavior trial as the pipeline labels this frame. 0 for none, which includes the frames of the trial running when the log was saved, because the pipeline clears that trial. Its packets are in the scanimage_i2c_packets table.", _int32(prepared.labels["trial"], "behavior_trial")),
        ("behavior_iteration", "Per-trial iteration as the pipeline labels this frame, 0 for none.", _int32(prepared.labels["iteration"], "behavior_iteration")),
    ]))
    module.add(_table("scanimage_i2c_packets", "Every I2C packet from ViRMEn, as stored in the frame headers.", [
        ("frame_index", "0-based row of the scanimage_frames table whose header holds this packet. It can be the frame after the one the packet's time falls in.", _int32(packets["frame_index"], "frame_index")),
        ("timestamp", "Arrival time in seconds on the ScanImage clock.", packets["timestamp"]),
        ("block", "1-based block.", _int32(packets["block"], "block")),
        ("trial", "1-based trial within the block.", _int32(packets["trial"], "trial")),
        ("iteration", "1-based iteration within the trial.", _int32(packets["iteration"], "iteration")),
        ("elapsed", "Logged time of that iteration on ViRMEn's elapsed clock, in seconds.", prepared.packet_elapsed),
        ("session_time", "The packet's arrival mapped to this file's time, in seconds since session_start_time.", prepared.packet_times),
        ("used_in_fit", "Whether this packet was used in the fit.", prepared.fit.used),
    ]))
    module.add(_table("scanimage_files", "The source TIFFs and their full ScanImage metadata.", [
        ("file_name", "File name.", np.array([file.name for file in acquisition.files], dtype=object)),
        ("n_frames", "Pages in this file.", _int32([file.n_frames for file in acquisition.files], "n_frames")),
        ("software", "The first page's Software tag: all ScanImage settings.", np.array([file.software for file in acquisition.files], dtype=object)),
        ("artist", "The first page's Artist tag: the ROI groups, as JSON.", np.array([file.artist for file in acquisition.files], dtype=object)),
    ]))
    rows = [*alignment_rows(prepared), *extra_alignment_rows]
    module.add(_table("scanimage_alignment", "How frames were aligned to behavior, and how far to trust it.", [
        ("key", "Name of the value.", np.array([row[0] for row in rows], dtype=object)),
        ("value", "The value, as text.", np.array([row[1] for row in rows], dtype=object)),
        ("meaning", "What the value means.", np.array([row[2] for row in rows], dtype=object)),
    ]))


def _add_in_progress(nwbfile, prepared):
    if not prepared.in_progress:
        return
    module = nwbfile.processing["behavior"]
    clock = None
    for series in prepared.in_progress:
        timing = {"timestamps": clock} if clock is not None else {"timestamps": _compressed(series.timestamps)}
        built = TimeSeries(name=series.name, data=_compressed(series.data), unit=series.unit, description=f"The trial running when ViRMEn saved the log, which isn't in the trials table. Unrounded, per iteration. {series.description}", **timing)
        module.add(built)
        clock = clock or built


def _limited(count, limit):
    return count if limit is None else min(count, limit)


def _add_field(nwbfile, prepared, field, channel, device, imaging, limit):
    acquisition, layout = prepared.acquisition, prepared.acquisition.layout
    n_frames = _limited(field.frame_indices.size, limit)
    columns = field.page_shape[1]
    depth = layout.zs_um[field.plane]
    place = f"ROI {field.name}, on plane {field.plane} of {layout.n_planes}" if field.label != f"Plane{field.plane}" else f"Plane {field.plane} of {layout.n_planes}"
    at = "at an unknown z, because the split files kept no ScanImage settings" if np.isnan(depth) else f"at ScanImage z = {depth:g} um"
    plane_options = {}
    if field.pixel_size_um is not None:
        plane_options.update(grid_spacing=list(field.pixel_size_um), grid_spacing_unit="micrometers")
    if field.origin_um is not None and field.label != f"Plane{field.plane}":
        plane_options.update(origin_coords=list(field.origin_um), origin_coords_unit="micrometers")
    imaging_plane = nwbfile.create_imaging_plane(
        name=f"ImagingPlane{field.label.removeprefix('Plane')}",
        optical_channel=channel,
        description=f"{place}, {at}. PLACEHOLDER: the location says only that this is in the brain, and the wavelengths aren't recorded here.",
        device=device,
        excitation_lambda=imaging["excitation_lambda"],
        indicator=imaging["indicator"],
        location=imaging["location"],
        imaging_rate=layout.volume_rate,
        **plane_options,
    )
    series_options = {}
    if field.size_um is not None:
        series_options["field_of_view"] = [size * 1e-6 for size in field.size_um]
    if layout.line_period is not None:
        series_options["scan_line_rate"] = 1 / layout.line_period
    source = (
        "Split copy of this ROI, clipped at 0 by the splitter, so it isn't the raw signal: (time, x, y), where x runs along a scan line."
        if acquisition.source == SPLIT_SOURCE
        else "Raw frames, exactly as saved by ScanImage: (time, x, y), where x runs along a scan line."
    )
    nwbfile.add_acquisition(TwoPhotonSeries(
        name=f"TwoPhotonSeries{field.label}",
        imaging_plane=imaging_plane,
        data=H5DataIO(data=_FramePixels(lambda positions, f=field: read_field(acquisition, f, positions), n_frames, field.shape, acquisition.dtype), **COMPRESSION),
        unit="n.a.",
        timestamps=_compressed(prepared.field_times(field)[:n_frames]),
        dimension=[columns, field.shape[0]],
        description=f"{source} {place}. See processing/ophys/scanimage_alignment for how far to trust the timestamps.",
        **series_options,
    ))


def _add_flyback(nwbfile, prepared, plane, limit):
    """The rows between a plane's ROIs, so that with the fields they rebuild every page of the stack exactly."""
    acquisition = prepared.acquisition
    gaps = acquisition.layout.flyback_rows.get(plane)
    if not gaps:
        return
    indices = acquisition.plane_frames(plane)
    n_frames = _limited(indices.size, limit)
    rows = sum(stop - start for start, stop in gaps)
    columns = acquisition.layout.frame_shape[1]
    nwbfile.add_acquisition(ImageSeries(
        name=f"FlybackLinesPlane{plane}",
        data=H5DataIO(data=_FramePixels(lambda positions, p=plane: read_flyback(acquisition, p, positions), n_frames, (rows, columns), acquisition.dtype), **COMPRESSION),
        unit="n.a.",
        timestamps=_compressed(prepared.frame_times[indices][:n_frames]),
        dimension=[columns, rows],
        description=f"The rows ScanImage saved while flying between ROIs on plane {plane}, joined top to bottom: rows {', '.join(f'{start}:{stop}' for start, stop in gaps)} of each page. Not imaging data. They're kept because nothing is dropped, and with the plane's fields they rebuild each page exactly.",
    ))


def _add_planes(nwbfile, prepared, imaging, max_frames_per_plane):
    layout = prepared.acquisition.layout
    version = "an unknown version" if layout.version == "unknown" else layout.version
    device = Device(name="Microscope", description=f"Two-photon microscope run by ScanImage {version}.")
    nwbfile.add_device(device)
    channel = OpticalChannel(name="OpticalChannel", description="Channel 1, the only saved channel.", emission_lambda=imaging["emission_lambda"])
    for field in prepared.acquisition.fields:
        _add_field(nwbfile, prepared, field, channel, device, imaging, max_frames_per_plane)
    for plane in range(layout.n_planes):
        _add_flyback(nwbfile, prepared, plane, max_frames_per_plane)


def build_imaging_nwbfile(prepared, metadata=None, imaging=None, max_frames_per_plane=None, extra_alignment_rows=()):
    """The NWB file for raw imaging with behavior, ready to write. Pixels stream from the TIFFs when it's written."""
    placeholder = {
        **PLACEHOLDER_METADATA,
        "keywords": [*KEYWORDS, "two-photon calcium imaging", "ScanImage"],
        "experiment_description": "PLACEHOLDER: ViRMEn behavior with raw two-photon imaging. Experimenter, institution, lab and imaging details are not real yet.",
        "session_description": f"ViRMEn behavior with raw two-photon imaging of {prepared.session.lab_meta.get('experiment_name', 'an unnamed experiment')}, aligned frame by frame.",
    }
    nwbfile = build_nwbfile(prepared.session, {**placeholder, **(metadata or {})})
    _add_planes(nwbfile, prepared, {**PLACEHOLDER_IMAGING, **(imaging or {})}, max_frames_per_plane)
    _add_scanimage_tables(nwbfile, prepared, extra_alignment_rows)
    _add_in_progress(nwbfile, prepared)
    return nwbfile


def convert_session(log, tiff_paths, nwbfile_path, file=None, metadata=None, imaging=None, max_frames_per_plane=None, verify=True, allow_incomplete=False):
    """Convert one session, check it against the sources unless told not to, and write its report beside it. Returns what was prepared."""
    if max_frames_per_plane is not None and max_frames_per_plane < 1:
        raise ValueError(f"max_frames_per_plane must be at least 1, got {max_frames_per_plane}")
    prepared = prepare_session(log, tiff_paths, file=file, allow_incomplete=allow_incomplete)
    with NWBHDF5IO(str(nwbfile_path), "w") as io:
        io.write(build_imaging_nwbfile(prepared, metadata, imaging, max_frames_per_plane))
    if verify:
        verify_imaging(prepared, nwbfile_path, tiff_paths, max_frames_per_plane=max_frames_per_plane)
    Path(nwbfile_path).with_suffix(".report.json").write_text(json.dumps(conversion_report(prepared), indent=1))
    return prepared


def alignment_summary(prepared):
    """The alignment as plain data, for a report beside the file."""
    return {key: value for key, value, _ in alignment_rows(prepared)} | dict(frames=int(prepared.frame_times.size), first_frame_time=float(prepared.frame_times[0]), last_frame_time=float(prepared.frame_times[-1]), shift_seconds=prepared.shift_microseconds / 1e6)


def conversion_report(prepared):
    """The alignment and what happened to every log field. The trial in progress counts as dropped unless it was written."""
    report = dict(alignment=alignment_summary(prepared), coverage=prepared.session.coverage.as_dict(), sentinels=SENTINELS)
    if prepared.in_progress:
        report["coverage"]["dropped"].pop("log.currentTrial", None)
        report["coverage"]["dropped"].pop("log.currentIt", None)
    return report
