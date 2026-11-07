"""Read per-plane Suite2p results without importing Suite2p or torch."""

from pathlib import Path

import numpy as np

from ..virmen.coerce import coerce


def load_suite2p(root, prepared):
    """Validate counts and geometry against TIFF headers; never use combined's padded traces.

    NumPy object files must come from a trusted run: loading them executes pickle.
    """
    root = Path(root)
    acquisition = prepared.acquisition
    expected_folders = {f"plane{p}" for p in range(acquisition.layout.n_planes)}
    if {p.name for p in root.glob("plane*") if p.is_dir()} != expected_folders:
        raise ValueError("Suite2p plane folders differ from the TIFF planes")
    planes = []
    for plane in range(acquisition.layout.n_planes):
        folder = root / f"plane{plane}"
        data = {}
        for path in folder.glob("*.npy"):
            value = np.load(path, allow_pickle=True)
            data[path.stem] = value.item() if value.shape == () else value
        indices = acquisition.plane_frames(plane)
        nframes = len(indices)
        if not nframes:
            raise ValueError(f"plane {plane}: empty plane cannot be converted")
        db = data["db"]
        if db["nframes"] != nframes:
            raise ValueError(f"plane {plane}: nframes differs from TIFF frame count")
        if (db["Ly"], db["Lx"]) != acquisition.layout.frame_shape or db["iplane"] != plane:
            raise ValueError(f"plane {plane}: geometry differs from TIFF headers")
        if [Path(p).name for p in db["file_list"]] != [f.name for f in acquisition.files]:
            raise ValueError(f"plane {plane}: file order differs from TIFFs")
        counts = np.bincount(acquisition.frames["file_index"][indices], minlength=len(acquisition.files))
        if not np.array_equal(db["frames_per_file"], counts):
            raise ValueError(f"plane {plane}: frames_per_file differs from TIFF pages")
        nroi = len(data["stat"])
        for key in ("F", "Fneu", "spks"):
            if data[key].shape != (nroi, nframes):
                raise ValueError(f"plane {plane}: {key} ROI/frame dimensions differ")
            data[key] = coerce(data[key], "float32", file=folder, field=key)
        if data["iscell"].shape != (nroi, 2):
            raise ValueError(f"plane {plane}: iscell shape differs")
        coerce(data["iscell"][:, 0], "bool", file=folder, field="iscell")
        if not np.all(np.isfinite(data["iscell"][:, 1]) & (data["iscell"][:, 1] >= 0) & (data["iscell"][:, 1] <= 1)):
            raise ValueError(f"plane {plane}: invalid cell probability")
        for key in ("xoff", "yoff", "corrXY", "xoff1", "yoff1", "corrXY1", "badframes", "badframes0"):
            if key in data["reg_outputs"] and data["reg_outputs"][key].shape[0] != nframes:
                raise ValueError(f"plane {plane}: {key} frame count differs")
        for roi in data["stat"]:
            x = coerce(roi["xpix"], "int32", file=folder, field="xpix")
            y = coerce(roi["ypix"], "int32", file=folder, field="ypix")
            coerce(roi["lam"], "float32", file=folder, field="lam")
            if not len(x) or x.shape != y.shape or x.shape != np.shape(roi["lam"]):
                raise ValueError(f"plane {plane}: empty or inconsistent ROI pixels")
            if np.any((x < 0) | (x >= db["Lx"]) | (y < 0) | (y >= db["Ly"])) or not np.all(np.isfinite(roi["lam"])):
                raise ValueError(f"plane {plane}: invalid ROI pixels/weights")
        planes.append(data)
    return planes


def _json_value(value):
    import json

    return json.dumps(
        value.tolist() if isinstance(value, np.ndarray) else value.item() if isinstance(value, np.generic) else value,
        allow_nan=True,
    )


def _sources(root):
    """All artifacts, including combined and MATLAB duplicates; registered movies stay external."""
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.name != "data.bin")


def build_suite2p_nwbfile(prepared, root, metadata=None, mapping=None):
    """Build ophys interfaces plus a byte-exact source archive for fields without an NWB equivalent."""
    import hashlib
    from hdmf.backends.hdf5.h5_utils import H5DataIO
    from hdmf.common import ElementIdentifiers
    from pynwb import TimeSeries
    from pynwb.core import ScratchData
    from pynwb.base import Images
    from pynwb.device import Device
    from pynwb.image import GrayscaleImage
    from pynwb.ophys import Fluorescence, ImageSegmentation, OpticalChannel, RoiResponseSeries
    from ..virmen.convert import PLACEHOLDER_METADATA, build_nwbfile
    from .convert import PLACEHOLDER_IMAGING, _add_in_progress, _add_scanimage_tables, _table, COMPRESSION

    planes = load_suite2p(root, prepared)
    nwb = build_nwbfile(
        prepared.session,
        {
            **PLACEHOLDER_METADATA,
            "session_description": "ViRMEn behavior and Suite2p two-photon results, aligned using TIFF headers and behavior I2C events.",
            **(metadata or {}),
        },
    )
    nwb.notes = (
        "Companion to the raw TIFF NWB conversion of this acquisition: both use prepare_session, the same behavior, "
        "session-start shift and frame-time fit. Raw pixels are external and are not duplicated here. "
        "Per-plane Suite2p folders define lengths; combined contains padded display traces. "
        "Original Suite2p artifacts except data.bin are preserved byte-for-byte in scratch; object NPY files require a trusted pickle reader. "
        "registration.do_bidiphase is preserved without modification; decide separately whether this bidirectional rig needs correction."
    )
    _add_scanimage_tables(nwb, prepared)
    _add_in_progress(nwb, prepared)
    module = nwb.create_processing_module(
        "ophys",
        "Suite2p segmentation, fluorescence, neuropil, deconvolution and registration; per-plane TIFF frame times.",
    )
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
    for plane, data in enumerate(planes):
        ip = nwb.create_imaging_plane(
            name=f"ImagingPlane{plane}",
            optical_channel=OpticalChannel(
                name=f"OpticalChannel{plane}", description="Only saved channel.", emission_lambda=float("nan")
            ),
            description=f"Plane {plane}, ScanImage z={layout.zs_um[plane]:g} um. PLACEHOLDER: indicator, location and wavelengths await metadata.",
            device=device,
            imaging_rate=layout.volume_rate,
            grid_spacing=list(layout.pixel_size_um),
            grid_spacing_unit="micrometers",
            **{k: v for k, v in PLACEHOLDER_IMAGING.items() if k != "emission_lambda"},
        )
        # emission_lambda belongs to OpticalChannel, not ImagingPlane.
        ps = segmentation.create_plane_segmentation(
            name=f"PlaneSegmentation{plane}",
            imaging_plane=ip,
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
                f"stat_{key}", f"Original stat.{key} as JSON; NaN remains NaN; see native source archive for dtype."
            )
        for index, roi in enumerate(data["stat"]):
            # Explicit integer fields and float32 weights avoid HDMF silently casting tuple floats to uint32.
            mask = np.empty(len(roi["xpix"]), dtype=[("x", "uint32"), ("y", "uint32"), ("weight", "float32")])
            mask["x"], mask["y"], mask["weight"] = roi["xpix"], roi["ypix"], roi["lam"]
            ps.add_roi(
                pixel_mask=[(np.uint32(x), np.uint32(y), np.float32(w)) for x, y, w in mask],
                iscell=bool(data["iscell"][index, 0]),
                cell_probability=float(data["iscell"][index, 1]),
                **{f"stat_{k}": _json_value(roi[k]) if k in roi else "null" for k in stat_keys},
            )
        times = prepared.frame_times[prepared.acquisition.plane_frames(plane)]
        if len(data["stat"]):
            region = ps.create_roi_table_region(
                region=list(range(len(data["stat"]))), description="All ROIs in source order."
            )
            for name, key in (("Fluorescence", "F"), ("Neuropil", "Fneu"), ("Deconvolved", "spks")):
                containers[name].add_roi_response_series(
                    RoiResponseSeries(
                        name=f"Plane{plane}",
                        rois=region,
                        data=H5DataIO(data[key].T, **COMPRESSION),
                        timestamps=H5DataIO(times, **COMPRESSION),
                        unit="a.u.",
                        description=f"Suite2p {key}, unchanged float32 values. spks are deconvolved activity estimates, not measured spike counts.",
                    )
                )
        images = Images(
            name=f"SummaryImages{plane}",
            description="Images are (x,y). Cropped detection images start at (xrange[0],yrange[0]); masks use full-frame coordinates.",
        )
        for source in ("reg_outputs", "detect_outputs"):
            for key, value in data[source].items():
                if (
                    isinstance(value, np.ndarray)
                    and value.ndim == 2
                    and key in ("meanImg", "meanImgE", "max_proj", "Vcorr", "refImg", "meanImg_crop")
                ):
                    rows, cols = prepared.acquisition.layout.frame_shape
                    xr = data["reg_outputs"].get("xrange", [0, cols])
                    yr = data["reg_outputs"].get("yrange", [0, rows])
                    if value.shape == (rows, cols):
                        origin = (0, 0)
                    elif value.shape == (yr[1] - yr[0], xr[1] - xr[0]):
                        origin = (int(xr[0]), int(yr[0]))
                    else:
                        raise ValueError(f"plane {plane}: {key} image shape does not match full frame or crop")
                    image_coordinates.append((f"SummaryImages{plane}/{key}", *origin, value.shape[1], value.shape[0]))
                    images.add_image(
                        GrayscaleImage(
                            name=key,
                            data=H5DataIO(value.T, **COMPRESSION),
                            description=f"{source}.{key}, transposed from (row,column). Crop origin: {data['reg_outputs'].get('xrange')}, {data['reg_outputs'].get('yrange')}; full-size images have origin (0,0).",
                        )
                    )
        if images.images:
            module.add(images)
        for key in ("xoff", "yoff", "corrXY", "xoff1", "yoff1", "corrXY1", "badframes", "badframes0"):
            if key in data["reg_outputs"]:
                module.add(
                    TimeSeries(
                        name=f"Plane{plane}_{key}",
                        data=H5DataIO(data["reg_outputs"][key], **COMPRESSION),
                        timestamps=H5DataIO(times, **COMPRESSION),
                        unit="pixels" if "off" in key else "n.a.",
                        description=f"Suite2p registration {key}, unchanged; bad frames remain in all traces and timestamps.",
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
        import json

        nwb.add_scratch(
            json.dumps(mapping),
            name="suite2p_frame_mapping",
            description="Measured per-frame raw-to-registered reconstruction RMSE and neighboring-page controls. Sampled first, middle and last frame in every TIFF for each plane. See docs/imaging-suite2p.md for numerical and sampling limits.",
        )
    inventory = []
    for index, path in enumerate(_sources(root)):
        relative = str(path.relative_to(root))
        blob = np.memmap(path, mode="r", dtype="uint8")
        name = f"suite2p_source_{index:04d}"
        nwb.add_scratch(
            ScratchData(
                data=H5DataIO(blob, **COMPRESSION),
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


def convert_suite2p(log, tiffs, root, output, file=None, metadata=None):
    """Write, then independently verify; no upload or automatic organization."""
    from pynwb import NWBHDF5IO
    from .convert import prepare_session

    prepared = prepare_session(log, tiffs, file=file)
    load_suite2p(root, prepared)
    from .suite2p_mapping import prove_frame_mapping

    mapping = prove_frame_mapping(prepared.acquisition, root)
    with NWBHDF5IO(str(output), "w") as io:
        io.write(build_suite2p_nwbfile(prepared, root, metadata, mapping))
    verify_suite2p(prepared, output, root, tiffs)
    return prepared


def verify_suite2p(prepared, output, root, tiffs):
    """Compare HDF5 against fresh NumPy files and independently parsed TIFF headers, exactly."""
    import h5py
    import hashlib
    from ..virmen.verify import verify_round_trip, _check_series
    from .verify import RoundTripError, _read_sources, _check_packets, _alignment_values

    def equal(actual, expected, label):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape or not np.array_equal(actual, expected, equal_nan=True):
            raise RoundTripError(f"{label}: differs from source")

    verify_round_trip(prepared.session, output)
    numbers, times, packets = _read_sources(tiffs)
    with h5py.File(output) as f:
        scan = f["processing/scanimage"]
        equal(scan["frames/frame_number"][()], numbers, "frame numbers")
        equal(scan["frames/timestamp"][()], times, "TIFF timestamps")
        _check_packets(scan["i2c_packets"], packets)
        values = _alignment_values(scan["alignment"])
        slope, intercept, offset = [float(values[f"fit.{k}"]) for k in ("slope", "intercept", "clock_offset")]
        equal(
            [slope, intercept, offset], [prepared.fit.slope, prepared.fit.intercept, prepared.clock_offset], "clock fit"
        )
        session_times = (times - intercept) / slope + offset
        equal(scan["frames/plane"][()], (numbers - 1) % prepared.acquisition.layout.n_planes, "plane indices")
        equal(scan["frames/session_time"][()], prepared.frame_times, "prepared frame times")
        for key, column in prepared.acquisition.frames.items():
            if key in scan["frames"]:
                equal(scan[f"frames/{key}"][()], column, "header " + key)
        for key, expected in [
            ("file_name", [x.name for x in prepared.acquisition.files]),
            ("software", [x.software for x in prepared.acquisition.files]),
            ("artist", [x.artist for x in prepared.acquisition.files]),
        ]:
            if list(scan[f"files/{key}"].asstr()[()]) != expected:
                raise RoundTripError("source TIFF " + key + " differs")
        equal(scan["frames/session_time"][()], session_times, "session times")
        for key in ("block", "trial", "iteration"):
            equal(scan[f"frames/behavior_{key}"][()], prepared.labels[key], key)
        for key, expected in [
            ("elapsed", prepared.packet_elapsed),
            ("session_time", prepared.packet_times),
            ("used_in_fit", prepared.fit.used),
        ]:
            equal(scan[f"i2c_packets/{key}"][()], expected, key)
        for series in prepared.in_progress:
            _check_series(f["processing/in_progress_trial"], series)
        o = f["processing/ophys"]
        for plane in range(prepared.acquisition.layout.n_planes):
            folder = Path(root) / f"plane{plane}"
            stat = np.load(folder / "stat.npy", allow_pickle=True)
            cells = np.load(folder / "iscell.npy")
            ps = o[f"ImageSegmentation/PlaneSegmentation{plane}"]
            equal(ps["id"][()], np.arange(len(stat)), "ROI ids")
            db = np.load(folder / "db.npy", allow_pickle=True).item()
            reg = np.load(folder / "reg_outputs.npy", allow_pickle=True).item()
            equal(ps["iscell"][()], cells[:, 0], "iscell")
            equal(ps["cell_probability"][()], cells[:, 1], "probability")
            cursor = 0
            for i, roi in enumerate(stat):
                end = int(ps["pixel_mask_index"][i])
                mask = ps["pixel_mask"][cursor:end]
                equal(mask["x"], roi["xpix"], "mask x")
                equal(mask["y"], roi["ypix"], "mask y")
                equal(mask["weight"], roi["lam"], "mask weights")
                cursor = end
                import json

                for key, value in roi.items():
                    equal(json.loads(ps[f"stat_{key}"].asstr()[i]), value, f"stat.{key}")
            for name, key in [("Fluorescence", "F"), ("Neuropil", "Fneu"), ("Deconvolved", "spks")]:
                source = np.load(folder / f"{key}.npy", mmap_mode="r")
                if len(stat):
                    series = o[f"{name}/Plane{plane}"]
                    if series["data"].dtype != np.float32:
                        raise RoundTripError(key + " dtype differs")
                    equal(series["data"][()], source.T, key)
                    equal(
                        series["timestamps"][()],
                        session_times[(numbers - 1) % prepared.acquisition.layout.n_planes == plane],
                        f"{key} times",
                    )
            for source in ("reg_outputs", "detect_outputs"):
                meta = np.load(folder / f"{source}.npy", allow_pickle=True).item()
                for key, value in meta.items():
                    if f"Plane{plane}_{key}" in o:
                        equal(o[f"Plane{plane}_{key}/data"][()], value, key)
                        equal(
                            o[f"Plane{plane}_{key}/timestamps"][()],
                            session_times[(numbers - 1) % prepared.acquisition.layout.n_planes == plane],
                            key + " times",
                        )
                    if f"SummaryImages{plane}/{key}" in o:
                        equal(o[f"SummaryImages{plane}/{key}"][()], value.T, key)
                        table = o["summary_image_coordinates"]
                        row = list(table["image"].asstr()[()]).index(f"SummaryImages{plane}/{key}")
                        origin = (0, 0) if value.shape == (db["Ly"], db["Lx"]) else (reg["xrange"][0], reg["yrange"][0])
                        equal(
                            [table["x_origin"][row], table["y_origin"][row], table["width"][row], table["height"][row]],
                            [*origin, value.shape[1], value.shape[0]],
                            "summary image coordinates",
                        )
        inventory = o["source_inventory"]
        sources = _sources(root)
        if list(inventory["path"].asstr()[()]) != [str(p.relative_to(root)) for p in sources]:
            raise RoundTripError("source archive inventory differs")
        for index, (path, name) in enumerate(zip(sources, inventory["scratch_name"].asstr()[()])):
            source = np.memmap(path, mode="r", dtype="uint8")
            if inventory["sha256"].asstr()[index] != hashlib.sha256(source).hexdigest():
                raise RoundTripError("source archive hash differs")
            dataset = f[f"scratch/{name}"]
            if dataset.shape != source.shape:
                raise RoundTripError(f"{path.name}: archive length differs")
            for start in range(0, len(source), 8_000_000):
                equal(
                    dataset[start : start + 8_000_000], source[start : start + 8_000_000], str(path.relative_to(root))
                )
    return {
        "planes_checked": prepared.acquisition.layout.n_planes,
        "frames_checked": len(numbers),
        "artifacts_checked": len(sources),
    }
