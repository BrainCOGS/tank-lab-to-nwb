"""Synthetic Suite2p output: all coordinates and counts are deliberately small."""

import hashlib
import json

import h5py
import numpy as np
import pytest
from pynwb import validate

from tests.synthetic_tiff import frame_pixels, write_acquisition, write_stack
from tests.test_imaging_nwb import log, tiffs  # noqa: F401
from tank_lab_to_nwb.imaging import convert_suite2p as cli
from tank_lab_to_nwb.imaging.convert_suite2p import verify_combined
from tank_lab_to_nwb.imaging.scanimage import read_acquisition
from tank_lab_to_nwb.imaging.suite2p_mapping import is_separated, predict_traces, prove_frame_mapping
from tank_lab_to_nwb.imaging.alignment import AlignmentError
from tank_lab_to_nwb.imaging.convert import convert_session, prepare_session
from tank_lab_to_nwb.imaging.suite2p import convert_suite2p, load_suite2p, read_companion, verify_suite2p
from tank_lab_to_nwb.imaging.verify import RoundTripError


def _pixel_trace(plane):
    """Suite2p's fluorescence for the fixture's one-pixel ROI at row 1, column 2, of this plane's registered movie."""
    return np.array([frame_pixels(i)[1, 2] for i in range(plane + 1, 802, 2)], dtype="float32")


@pytest.fixture
def run(tmp_path, tiffs):
    root = tmp_path / "suite2p"
    for plane, count in enumerate((401, 400)):
        folder = root / f"plane{plane}"
        folder.mkdir(parents=True)
        db = dict(
            nframes=count,
            Ly=3,
            Lx=4,
            iplane=plane,
            nplanes=2,
            file_list=[str(p) for p in tiffs],
            frames_per_file=[150, 150, count - 300],
        )
        stat = [
            dict(
                xpix=np.array([2]),
                ypix=np.array([1]),
                lam=np.array([0.5], dtype="float32"),
                med=np.array([1.0, 2.0]),
                skew=np.float32(np.nan),
            )
        ]
        image = np.arange(12, dtype="float32").reshape(3, 4)
        for key, value in dict(
            db=db,
            stat=stat,
            settings=dict(version="1.1.0", registration=dict(do_bidiphase=False, nonrigid=False)),
            reg_outputs=dict(
                meanImg=image,
                yrange=[0, 3],
                xrange=[0, 4],
                xoff=np.zeros(count, dtype="int64"),
                yoff=np.zeros(count, dtype="int64"),
                badframes=np.zeros(count, dtype=bool),
            ),
            detect_outputs=dict(max_proj=image),
            ops=dict(),
            zcorr=np.array([]),
            iscell=np.array([[0.0, 0.125]]),
            F=_pixel_trace(plane)[None, :],
            Fneu=np.ones((1, count), dtype="float32"),
            spks=np.zeros((1, count), dtype="float32"),
        ).items():
            np.save(folder / f"{key}.npy", value)
        np.stack([frame_pixels(i) for i in range(plane + 1, 802, 2)]).tofile(folder / "data.bin")
    return root


def test_partial_last_volume_and_one_frame_mismatch(log, tiffs, run):
    prepared = prepare_session(log, tiffs, file="session.mat")
    planes = load_suite2p(run, prepared)
    assert [p["F"].shape for p in planes] == [(1, 401), (1, 400)]
    np.save(run / "plane1/F.npy", np.zeros((1, 401), dtype="float32"))
    with pytest.raises(ValueError, match="frame"):
        load_suite2p(run, prepared)


def test_conversion_preserves_masks_traces_provenance_and_exact_times(tmp_path, log, tiffs, run):
    from tank_lab_to_nwb.imaging.suite2p import convert_suite2p, verify_suite2p

    path = tmp_path / "out.nwb"
    prepared = convert_suite2p(log, tiffs, run, path, file="session.mat")
    assert not validate(path=str(path))
    with h5py.File(path) as f:
        assert list(f["acquisition"]) == []
        o = f["processing/ophys"]
        seg = o["ImageSegmentation/PlaneSegmentation0"]
        pixel = seg["pixel_mask"][0]
        assert (pixel["x"], pixel["y"], pixel["weight"]) == (2, 1, 0.5)
        image = o["SummaryImages0/meanImg"][()]
        assert image[2, 1] == 6
        assert seg["iscell"][0] == 0 and seg["cell_probability"][0] == 0.125
        assert np.isnan(float(seg["stat_skew"].asstr()[0]))
        for plane, length in enumerate((401, 400)):
            ts = o[f"Fluorescence/Plane{plane}/timestamps"][()]
            np.testing.assert_array_equal(ts, prepared.frame_times[plane::2])
            assert o[f"Fluorescence/Plane{plane}/data"].shape == (length, 1)
    with h5py.File(path, "r+") as f:
        d = f["processing/ophys/Fluorescence/Plane1/data"]
        d[0, 0] = np.nextafter(d[0, 0], np.float32(1))
    with pytest.raises(ValueError, match="F"):
        verify_suite2p(prepared, path, run, tiffs)


def test_reconstruction_uses_rigid_sign_and_uint16_scaling():
    from tank_lab_to_nwb.imaging.suite2p_mapping import reconstruct_frame

    raw = np.arange(12, dtype="int16").reshape(3, 4) - 6
    reg = dict(xoff=np.array([1]), yoff=np.array([-1]))
    settings = dict(nonrigid=False, do_bidiphase=False, bidiphase=0, two_step_registration=False)
    expected = np.array([[3, 4, 5, 2], [-5, -4, -3, -6], [-1, 0, 1, -2]], dtype="int16")
    np.testing.assert_array_equal(reconstruct_frame(raw, reg, settings, 0), expected)
    np.testing.assert_array_equal(reconstruct_frame((raw + 6).astype("uint16") * 2, reg, settings, 0), expected + 6)


def test_zero_rois_keep_the_plane_and_source_arrays(tmp_path, log, tiffs, run):
    from tank_lab_to_nwb.imaging.suite2p import convert_suite2p

    folder = run / "plane1"
    for key in ("F", "Fneu", "spks"):
        np.save(folder / f"{key}.npy", np.empty((0, 400), dtype="float32"))
    np.save(folder / "iscell.npy", np.empty((0, 2)))
    np.save(folder / "stat.npy", np.array([], dtype=object))
    path = tmp_path / "empty.nwb"
    convert_suite2p(log, tiffs, run, path, file="session.mat")
    assert not validate(path=str(path))
    with h5py.File(path) as f:
        assert f["processing/ophys/ImageSegmentation/PlaneSegmentation1/id"].shape == (0,)
        assert "Plane1" not in f["processing/ophys/Fluorescence"]


def test_nan_traces_roundtrip_without_sentinel(tmp_path, log, tiffs, run):
    from tank_lab_to_nwb.imaging.suite2p import convert_suite2p

    trace = np.load(run / "plane0/F.npy")
    trace[0, 3] = np.nan
    np.save(run / "plane0/F.npy", trace)
    path = tmp_path / "nan.nwb"
    convert_suite2p(log, tiffs, run, path, file="session.mat")
    with h5py.File(path) as f:
        assert np.isnan(f["processing/ophys/Fluorescence/Plane0/data"][3, 0])


def test_float32_conversion_has_zero_tolerance(log, tiffs, run):
    from tank_lab_to_nwb.virmen.coerce import LossyConversionError

    trace = np.zeros((1, 401), dtype="float64")
    trace[0, 0] = 0.1
    np.save(run / "plane0/F.npy", trace)
    with pytest.raises(LossyConversionError):
        load_suite2p(run, prepare_session(log, tiffs, file="session.mat"))


def test_nonrigid_reconstruction_applies_half_pixel_shift():
    from tank_lab_to_nwb.imaging.suite2p_mapping import reconstruct_frame

    raw = np.tile(np.arange(9, dtype="int16") * 10, (9, 1))
    reg = dict(
        xoff=np.array([0]),
        yoff=np.array([0]),
        xoff1=np.full((1, 16), 0.5, dtype="float32"),
        yoff1=np.zeros((1, 16), dtype="float32"),
    )
    result = reconstruct_frame(raw, reg, dict(nonrigid=True, block_size=[4, 4]), 0)
    np.testing.assert_array_equal(result[3], np.array([5, 15, 25, 35, 45, 55, 65, 75, 80]))


def test_converter_refuses_registered_movie_with_shifted_frame_order(tmp_path, log, tiffs, run):
    from tank_lab_to_nwb.imaging.suite2p import convert_suite2p
    from tests.synthetic_tiff import frame_pixels

    path = run / "plane0/data.bin"
    np.stack([frame_pixels(i) for i in range(3, 804, 2)]).tofile(path)
    with pytest.raises(ValueError, match="mapping"):
        convert_suite2p(log, tiffs, run, tmp_path / "bad.nwb", file="session.mat")


@pytest.mark.parametrize("field", ["xpix", "ypix", "lam"])
def test_nan_mask_values_are_refused(log, tiffs, run, field):
    stat = np.load(run / "plane0/stat.npy", allow_pickle=True)
    stat[0][field] = np.array([np.nan])
    np.save(run / "plane0/stat.npy", stat)
    with pytest.raises(ValueError):
        load_suite2p(run, prepare_session(log, tiffs, file="session.mat"))


def test_registered_movie_one_frame_short_is_refused(tmp_path, log, tiffs, run):
    from tank_lab_to_nwb.imaging.suite2p import convert_suite2p

    with (run / "plane1/data.bin").open("r+b") as f:
        f.truncate(399 * 3 * 4 * 2)
    with pytest.raises(ValueError, match="frame|mapping|mmap"):
        convert_suite2p(log, tiffs, run, tmp_path / "out.nwb", file="session.mat")


def test_empty_plane_is_refused(log, tiffs, run):
    from dataclasses import replace

    prepared = prepare_session(log, tiffs, file="session.mat")
    empty = replace(prepared.acquisition, frames={k: v[:0] for k, v in prepared.acquisition.frames.items()})
    with pytest.raises(ValueError, match="empty plane"):
        load_suite2p(run, replace(prepared, acquisition=empty))


@pytest.mark.parametrize(
    "dataset",
    [
        "processing/ophys/Fluorescence/Plane0/timestamps",
        "processing/ophys/Plane0_xoff/data",
        "processing/ophys/SummaryImages0/meanImg",
        "scratch/suite2p_source_0000",
    ],
)
def test_independent_readback_detects_corruption(tmp_path, log, tiffs, run, dataset):
    from tank_lab_to_nwb.imaging.suite2p import convert_suite2p, verify_suite2p

    path = tmp_path / "corrupt.nwb"
    prepared = convert_suite2p(log, tiffs, run, path, file="session.mat")
    with h5py.File(path, "r+") as f:
        d = f[dataset]
        key = tuple(0 for _ in d.shape)
        d[key] = d[key] + 1
    with pytest.raises(ValueError, match="differs"):
        verify_suite2p(prepared, path, run, tiffs)


def test_badframes_are_retained_without_dropping_trace_samples(tmp_path, log, tiffs, run):
    from tank_lab_to_nwb.imaging.suite2p import convert_suite2p

    source = run / "plane1/reg_outputs.npy"
    meta = np.load(source, allow_pickle=True).item()
    meta["badframes"][13] = True
    np.save(source, meta)
    path = tmp_path / "badframes.nwb"
    convert_suite2p(log, tiffs, run, path, file="session.mat")
    with h5py.File(path) as f:
        assert f["processing/ophys/Plane1_badframes/data"][13]
        assert f["processing/ophys/Fluorescence/Plane1/data"].shape == (400, 1)


def test_cropped_summary_image_has_explicit_full_frame_origin(tmp_path, log, tiffs, run):
    from tank_lab_to_nwb.imaging.suite2p import convert_suite2p

    folder = run / "plane0"
    reg = np.load(folder / "reg_outputs.npy", allow_pickle=True).item()
    reg["xrange"] = [1, 3]
    reg["yrange"] = [1, 3]
    np.save(folder / "reg_outputs.npy", reg)
    np.save(folder / "detect_outputs.npy", dict(max_proj=np.array([[5.0, 6.0], [9.0, 10.0]], dtype="float32")))
    path = tmp_path / "cropped.nwb"
    convert_suite2p(log, tiffs, run, path, file="session.mat")
    with h5py.File(path) as f:
        o = f["processing/ophys"]
        t = o["summary_image_coordinates"]
        row = list(t["image"].asstr()[()]).index("SummaryImages0/max_proj")
        assert (t["x_origin"][row], t["y_origin"][row]) == (1, 1)
        # Full-frame ROI (2,1) is crop coordinate (1,0), the very same image pixel.
        assert o["SummaryImages0/max_proj"][1, 0] == o["SummaryImages0/meanImg"][2, 1] == 6


def _write_combined(run):
    combined = run / "combined"
    combined.mkdir()
    for key in ("F", "Fneu", "spks"):
        a = np.load(run / "plane0" / f"{key}.npy")
        b = np.load(run / "plane1" / f"{key}.npy")
        np.save(combined / f"{key}.npy", np.concatenate([a, np.pad(b, ((0, 0), (0, 1)))]))
    np.save(combined / "iscell.npy", np.concatenate([np.load(run / f"plane{p}/iscell.npy") for p in range(2)]))
    stat = []
    for plane in range(2):
        roi = np.load(run / f"plane{plane}/stat.npy", allow_pickle=True)[0]
        roi["xpix"] += 4 * plane
        roi["med"][1] += 4 * plane
        roi["iplane"] = plane
        stat.append(roi)
    np.save(combined / "stat.npy", stat)
    return combined


def test_combined_is_only_tiling_and_padding(run):
    from tank_lab_to_nwb.imaging.convert_suite2p import verify_combined

    combined = _write_combined(run)
    assert verify_combined(run)["roi_values_match"]
    trace = np.load(combined / "F.npy")
    trace[1, -1] = 1
    np.save(combined / "F.npy", trace)
    with pytest.raises(ValueError, match="padding"):
        verify_combined(run)


def test_inventory_includes_matlab_fields_and_excluded_movie_reason(run):
    from scipy.io import savemat
    from tank_lab_to_nwb.imaging.convert_suite2p import artifact_inventory

    savemat(run / "plane0/Fall.mat", dict(redcell=np.empty((0, 0)), F=np.array([[1.0]], dtype="float32")))
    inventory = artifact_inventory(run)
    assert inventory["plane0/Fall.mat"]["fields"]["redcell"]["shape"] == [0, 0]
    assert inventory["plane0/Fall.mat"]["fields"]["F"]["matlab_class"] == "single"
    assert inventory["plane0/data.bin"]["storage"].startswith("excluded:")


def test_bidiphase_decision_and_version_are_visible_without_unpickling(tmp_path, log, tiffs, run):
    from tank_lab_to_nwb.imaging.suite2p import convert_suite2p

    path = tmp_path / "settings.nwb"
    convert_suite2p(log, tiffs, run, path, file="session.mat")
    with h5py.File(path) as f:
        table = f["processing/ophys/suite2p_settings"]
        keys = list(table["key"].asstr()[()])
        values = list(table["value"].asstr()[()])
        assert values[keys.index("plane0.registration.do_bidiphase")] == "false"
        assert values[keys.index("plane0.version")] == '"1.1.0"'
        assert "open decision" not in table.attrs["description"]
        assert "bidirectional" not in table.attrs["description"]
        assert "bidirectional" not in f["general/notes"].asstr()[()]


@pytest.mark.parametrize("dataset", ["processing/ophys/Plane0_xoff", "processing/ophys/SummaryImages0/meanImg"])
def test_readback_refuses_missing_derived_interfaces(tmp_path, log, tiffs, run, dataset):
    from tank_lab_to_nwb.imaging.suite2p import convert_suite2p, verify_suite2p

    path = tmp_path / "missing.nwb"
    prepared = convert_suite2p(log, tiffs, run, path, file="session.mat")
    with h5py.File(path, "r+") as f:
        del f[dataset]
    with pytest.raises(ValueError, match="missing"):
        verify_suite2p(prepared, path, run, tiffs)


def test_numpy_scalars_nested_in_stat_lists_roundtrip(tmp_path, log, tiffs, run):
    from tank_lab_to_nwb.imaging.suite2p import convert_suite2p

    source = run / "plane0/stat.npy"
    stat = np.load(source, allow_pickle=True)
    stat[0]["med"] = [np.int64(1), np.int64(2)]
    np.save(source, stat)
    path = tmp_path / "nested.nwb"
    convert_suite2p(log, tiffs, run, path, file="session.mat")
    with h5py.File(path) as f:
        assert f["processing/ophys/ImageSegmentation/PlaneSegmentation0/stat_med"].asstr()[0] == "[1, 2]"


@pytest.fixture
def raw_file(tmp_path, log, tiffs):
    path = tmp_path / "raw.nwb"
    convert_session(log, tiffs, path, file="session.mat")
    return path


def _alignment(path):
    with h5py.File(path, "r") as file:
        table = file["processing/ophys/scanimage_alignment"]
        return dict(zip(table["key"].asstr()[()], table["value"].asstr()[()]))


def test_the_raw_file_it_goes_with_is_recorded(tmp_path, log, tiffs, run, raw_file):
    output = tmp_path / "s2p.nwb"
    convert_suite2p(log, tiffs, run, output, file="session.mat", raw_nwb=raw_file)
    values = _alignment(output)
    assert values["acquisition.id"] == _alignment(raw_file)["acquisition.id"]
    with h5py.File(raw_file, "r") as raw, h5py.File(output, "r") as file:
        assert values["companion.rawtiff.identifier"] == raw["identifier"].asstr()[()]
        assert values["companion.rawtiff.identifier"] != file["identifier"].asstr()[()]
    assert values["companion.rawtiff.sha256"] == hashlib.sha256(raw_file.read_bytes()).hexdigest()


def test_no_raw_file_means_no_companion_rows(tmp_path, log, tiffs, run):
    output = tmp_path / "s2p.nwb"
    convert_suite2p(log, tiffs, run, output, file="session.mat")
    assert not [key for key in _alignment(output) if key.startswith("companion.")]


def test_a_raw_file_from_another_acquisition_is_refused(tmp_path, log, tiffs, run, raw_file):
    with h5py.File(raw_file, "r+") as file:
        table = file["processing/ophys/scanimage_alignment"]
        keys = list(table["key"].asstr()[()])
        table["value"][keys.index("acquisition.id")] = "another"
    with pytest.raises(AlignmentError, match="another acquisition"):
        convert_suite2p(log, tiffs, run, tmp_path / "s2p.nwb", file="session.mat", raw_nwb=raw_file)


def test_a_raw_file_on_a_different_clock_is_refused(tmp_path, log, tiffs, run, raw_file):
    with h5py.File(raw_file, "r+") as file:
        times = file["processing/ophys/scanimage_frames/session_time"]
        times[3] = times[3] + 1e-9
    with pytest.raises(AlignmentError, match="clock"):
        convert_suite2p(log, tiffs, run, tmp_path / "s2p.nwb", file="session.mat", raw_nwb=raw_file)


@pytest.mark.parametrize("column", ["dc_over_voltage", "acquisition_number", "next_file_marker_timestamp", "page_index"])
def test_verify_catches_a_changed_header_column(tmp_path, log, tiffs, run, column):
    output = tmp_path / "s2p.nwb"
    prepared = convert_suite2p(log, tiffs, run, output, file="session.mat")
    with h5py.File(output, "r+") as file:
        data = file[f"processing/ophys/scanimage_frames/{column}"]
        data[4] = data[4] + 1
    with pytest.raises(RoundTripError):
        verify_suite2p(prepared, output, run, tiffs)


def test_verify_catches_a_changed_companion_record(tmp_path, log, tiffs, run, raw_file):
    output = tmp_path / "s2p.nwb"
    prepared = convert_suite2p(log, tiffs, run, output, file="session.mat", raw_nwb=raw_file)
    expected = read_companion(raw_file, prepared)
    with h5py.File(output, "r+") as file:
        table = file["processing/ophys/scanimage_alignment"]
        table["value"][list(table["key"].asstr()[()]).index("companion.rawtiff.sha256")] = "changed"
    with pytest.raises(RoundTripError):
        verify_suite2p(prepared, output, run, tiffs, companion=expected)


def test_a_multi_roi_session_is_refused_for_now(tmp_path, log, run):
    folder = tmp_path / "stack"
    folder.mkdir()
    rois = [dict(rows=3), dict(rows=3)]
    packets = {}
    trial = log["block"][0]["trial"][0]
    for iteration in range(1, 900, 2):
        time = (float(trial["start"].item()) + float(trial["time"][iteration - 1, 0])) * (1 - 5e-6) + 13.0
        frame = int(time // 0.02) + 1
        if frame <= 801:
            packets.setdefault(frame, []).append((time, (1, 1, iteration)))
    stack = write_stack(folder, 801, rois, frames_per_file=300, packets=packets)
    with pytest.raises(ValueError, match="multi-ROI"):
        convert_suite2p(log, stack, run, tmp_path / "out.nwb", file="session.mat")


def test_the_file_uses_the_schema_module_names_and_has_keywords(tmp_path, log, tiffs, run):
    output = tmp_path / "s2p.nwb"
    convert_suite2p(log, tiffs, run, output, file="session.mat")
    with h5py.File(output, "r") as file:
        assert set(file["processing"]) == {"behavior", "ophys"}
        assert "Suite2p" in list(file["general/keywords"].asstr()[()])


def test_the_report_lists_what_was_dropped_and_the_companion(tmp_path, log, tiffs, run, raw_file, monkeypatch):
    def with_log(_path, tiff_paths, root, output, **options):
        return convert_suite2p(log, tiff_paths, root, output, file="session.mat", **options)

    monkeypatch.setattr(cli, "convert_suite2p", with_log)
    path = cli.convert("session.mat", run, tmp_path / "out", tiffs, raw_nwb=raw_file)
    report = json.loads(path.with_suffix(".report.json").read_text())
    assert {"log.currentTrial", "log.currentIt"} <= set(report["coverage"]["dropped"])
    assert len(report["companion"]["sha256"]) == 64
    assert report["alignment"]["frames"] == 801


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda trace: trace[:, ::-1], id="reversed"),
        pytest.param(lambda trace: np.roll(trace, 1, axis=1), id="one frame late"),
        pytest.param(lambda trace: np.roll(trace, -1, axis=1), id="one frame early"),
    ],
)
def test_traces_that_do_not_match_the_registered_movie_are_refused(tmp_path, log, tiffs, run, change):
    np.save(run / "plane0/F.npy", change(np.load(run / "plane0/F.npy")))
    with pytest.raises(ValueError, match="trace"):
        convert_suite2p(log, tiffs, run, tmp_path / "bad.nwb", file="session.mat")


def test_the_evidence_records_trace_errors_with_controls(log, tiffs, run):
    evidence = prove_frame_mapping(prepare_session(log, tiffs, file="session.mat").acquisition, run)
    middle = next(sample for sample in evidence if sample["plane"] == 0 and sample["frame"] > 0)
    assert middle["trace_rmse"]["0"] == 0
    assert middle["trace_rmse"]["-1"] > 1 and middle["trace_rmse"]["1"] > 1


def test_a_trace_without_a_value_at_a_sampled_frame_is_not_measured_there(log, tiffs, run):
    trace = np.load(run / "plane0/F.npy")
    trace[0, 0] = np.nan
    np.save(run / "plane0/F.npy", trace)
    evidence = prove_frame_mapping(prepare_session(log, tiffs, file="session.mat").acquisition, run)
    first = next(sample for sample in evidence if sample["plane"] == 0 and sample["frame"] == 0)
    assert first["trace_rmse"] is None


def test_predicted_traces_use_normalized_weights_and_leave_out_overlaps():
    frame = np.arange(12, dtype="int16").reshape(3, 4)
    roi = dict(xpix=np.array([0, 1, 2]), ypix=np.array([1, 1, 1]), lam=np.array([1.0, 3.0, 8.0], dtype="float32"), overlap=np.array([False, False, True]))
    assert predict_traces(frame, [roi], allow_overlap=False).tolist() == [(4 * 1 + 5 * 3) / 4]
    assert predict_traces(frame, [roi], allow_overlap=True).tolist() == pytest.approx([(4 * 1 + 5 * 3 + 6 * 8) / 12], rel=1e-6)  # float32, as in Suite2p


def test_a_roi_without_overlap_recorded_counts_every_pixel():
    frame = np.ones((3, 4), dtype="int16")
    roi = dict(xpix=np.array([0, 1]), ypix=np.array([0, 0]), lam=np.array([1.0, 1.0], dtype="float32"))
    assert predict_traces(frame, [roi], allow_overlap=False).tolist() == [1.0]


def test_a_roi_entirely_in_overlap_has_a_zero_trace():
    frame = np.ones((3, 4), dtype="int16")
    roi = dict(xpix=np.array([0]), ypix=np.array([0]), lam=np.array([1.0], dtype="float32"), overlap=np.array([True]))
    assert predict_traces(frame, [roi], allow_overlap=False).tolist() == [0.0]


@pytest.mark.parametrize(
    ("errors", "passes"),
    [
        ({"0": 0.0, "1": 0.2}, True),
        ({"0": 0.0, "1": 0.1}, False),  # the control must be more than ten times worse, with a floor
        ({"0": 0.5, "-1": 5.0}, False),
        ({"0": 0.5, "-1": 5.1, "1": 6.0}, True),
        ({"0": 2.0, "1": 100.0}, True),
        ({"0": 2.1, "1": 100.0}, False),
        ({"0": 0.0}, True),  # one frame in the plane, so no neighbors
    ],
)
def test_a_match_must_be_close_and_stand_clear_of_its_controls(errors, passes):
    assert is_separated(errors, limit=2) is passes


def test_a_plane_with_no_pages_in_the_last_file_is_still_checked(tmp_path, log):
    folder = tmp_path / "short"
    folder.mkdir()
    tiffs = write_acquisition(folder, 7, frames_per_file=3)
    root = tmp_path / "s2p"
    for plane, count in enumerate((4, 3)):
        where = root / f"plane{plane}"
        where.mkdir(parents=True)
        movie = np.stack([frame_pixels(i) for i in range(plane + 1, 8, 2)])
        movie.tofile(where / "data.bin")
        np.save(where / "reg_outputs.npy", dict(xoff=np.zeros(count, dtype="int64"), yoff=np.zeros(count, dtype="int64")))
        np.save(where / "settings.npy", dict(registration=dict(do_bidiphase=False, nonrigid=False)))
        np.save(where / "stat.npy", [dict(xpix=np.array([2]), ypix=np.array([1]), lam=np.array([0.5], dtype="float32"))])
        np.save(where / "F.npy", movie[:, 1, 2].astype("float32")[None, :])
    evidence = prove_frame_mapping(read_acquisition(tiffs), root)
    assert {sample["plane"] for sample in evidence} == {0, 1}


def test_every_setting_must_be_in_the_file(tmp_path, log, tiffs, run):
    path = tmp_path / "out.nwb"
    prepared = convert_suite2p(log, tiffs, run, path, file="session.mat")
    with h5py.File(path, "r+") as f:
        keys = f["processing/ophys/suite2p_settings/key"]
        keys[list(keys.asstr()[()]).index("plane0.registration.do_bidiphase")] = "plane0.registration.renamed"
    with pytest.raises(RoundTripError, match="setting"):
        verify_suite2p(prepared, path, run, tiffs)


def test_a_changed_setting_value_is_caught(tmp_path, log, tiffs, run):
    path = tmp_path / "out.nwb"
    prepared = convert_suite2p(log, tiffs, run, path, file="session.mat")
    with h5py.File(path, "r+") as f:
        table = f["processing/ophys/suite2p_settings"]
        table["value"][list(table["key"].asstr()[()]).index("plane0.registration.do_bidiphase")] = "true"
    with pytest.raises(RoundTripError):
        verify_suite2p(prepared, path, run, tiffs)


def test_traces_must_point_at_their_own_plane_rois(tmp_path, log, tiffs, run):
    path = tmp_path / "out.nwb"
    prepared = convert_suite2p(log, tiffs, run, path, file="session.mat")
    with h5py.File(path, "r+") as f:
        f["processing/ophys/Fluorescence/Plane0/rois"].attrs["table"] = f["processing/ophys/ImageSegmentation/PlaneSegmentation1"].ref
    with pytest.raises(RoundTripError, match="rois"):
        verify_suite2p(prepared, path, run, tiffs)


def test_the_stored_mapping_evidence_is_checked(tmp_path, log, tiffs, run):
    path = tmp_path / "out.nwb"
    prepared = convert_suite2p(log, tiffs, run, path, file="session.mat")
    evidence = prove_frame_mapping(prepared.acquisition, run)
    verify_suite2p(prepared, path, run, tiffs, mapping=evidence)
    evidence[0]["rmse"]["0"] += 1
    with pytest.raises(RoundTripError, match="mapping"):
        verify_suite2p(prepared, path, run, tiffs, mapping=evidence)


def test_a_stat_key_absent_from_a_roi_is_not_confused_with_none(tmp_path, log, tiffs, run):
    folder = run / "plane0"
    roi = np.load(folder / "stat.npy", allow_pickle=True)[0]
    np.save(folder / "stat.npy", [dict(roi, note=None), dict(roi)])
    for key in ("F", "Fneu", "spks"):
        trace = np.load(folder / f"{key}.npy")
        np.save(folder / f"{key}.npy", np.concatenate([trace, trace]))
    np.save(folder / "iscell.npy", np.array([[0.0, 0.125], [1.0, 0.5]]))
    path = tmp_path / "out.nwb"
    prepared = convert_suite2p(log, tiffs, run, path, file="session.mat")
    with h5py.File(path, "r+") as f:
        column = f["processing/ophys/ImageSegmentation/PlaneSegmentation0/stat_note"]
        assert list(column.asstr()[()]) == ["null", ""]
        column[1] = "null"
    with pytest.raises(RoundTripError, match="note"):
        verify_suite2p(prepared, path, run, tiffs)


def test_a_zero_byte_artifact_is_archived(tmp_path, log, tiffs, run):
    (run / "plane0/empty.log").write_bytes(b"")
    path = tmp_path / "out.nwb"
    convert_suite2p(log, tiffs, run, path, file="session.mat")
    with h5py.File(path) as f:
        inventory = f["processing/ophys/source_inventory"]
        name = inventory["scratch_name"].asstr()[list(inventory["path"].asstr()[()]).index("plane0/empty.log")]
        assert f[f"scratch/{name}"].shape == (0,)


def test_a_movie_anywhere_is_listed_as_left_out(run):
    (run / "combined").mkdir()
    (run / "combined/data.bin").write_bytes(b"12")
    (run / "data.bin").write_bytes(b"12")
    inventory = cli.artifact_inventory(run)
    assert inventory["combined/data.bin"]["storage"].startswith("excluded:")
    assert inventory["data.bin"]["storage"].startswith("excluded:")


def _change_db(run, **changes):
    path = run / "plane0/db.npy"
    np.save(path, {**np.load(path, allow_pickle=True).item(), **changes})


@pytest.mark.parametrize(
    "changes",
    [dict(Ly=4), dict(Lx=5), dict(iplane=1), dict(nframes=400), dict(frames_per_file=[150, 150, 100])],
    ids=["rows", "columns", "plane", "frames", "frames per file"],
)
def test_a_run_that_disagrees_with_the_tiffs_is_refused(log, tiffs, run, changes):
    _change_db(run, **changes)
    with pytest.raises(ValueError, match="plane 0"):
        load_suite2p(run, prepare_session(log, tiffs, file="session.mat"))


def test_a_run_from_files_in_another_order_is_refused(log, tiffs, run):
    _change_db(run, file_list=[str(path) for path in reversed(tiffs)])
    with pytest.raises(ValueError, match="file order"):
        load_suite2p(run, prepare_session(log, tiffs, file="session.mat"))


@pytest.mark.parametrize(("probability", "passes"), [(0.0, True), (1.0, True), (float("nan"), False), (-0.1, False), (1.5, False)])
def test_cell_probability_must_be_a_probability(log, tiffs, run, probability, passes):
    np.save(run / "plane0/iscell.npy", np.array([[0.0, probability]]))
    prepared = prepare_session(log, tiffs, file="session.mat")
    if passes:
        assert load_suite2p(run, prepared)
    else:
        with pytest.raises(ValueError, match="probability"):
            load_suite2p(run, prepared)


@pytest.mark.parametrize(
    ("field", "value", "passes"),
    [("xpix", 0, True), ("xpix", 3, True), ("xpix", 4, False), ("xpix", -1, False), ("ypix", 0, True), ("ypix", 2, True), ("ypix", 3, False)],
)
def test_mask_pixels_must_lie_in_the_frame(log, tiffs, run, field, value, passes):
    stat = np.load(run / "plane0/stat.npy", allow_pickle=True)
    stat[0][field] = np.array([value])
    np.save(run / "plane0/stat.npy", stat)
    prepared = prepare_session(log, tiffs, file="session.mat")
    if passes:
        assert load_suite2p(run, prepared)
    else:
        with pytest.raises(ValueError, match="pixels"):
            load_suite2p(run, prepared)


def test_combined_settings_must_match_the_planes(run):
    combined = _write_combined(run)
    settings = np.load(run / "plane0/settings.npy", allow_pickle=True).item()
    np.save(combined / "settings.npy", settings)
    assert verify_combined(run)["roi_values_match"]
    np.save(combined / "settings.npy", dict(settings, version="other"))
    with pytest.raises(ValueError, match="combined settings"):
        verify_combined(run)


def _combined_db(run, save_path):
    folder = run / "plane1"
    image = np.arange(12, dtype="float32").reshape(3, 4)
    tiled = np.hstack([image, image]).astype("float64")
    merged = {}
    for name in ("db", "reg_outputs", "detect_outputs"):
        merged.update(np.load(folder / f"{name}.npy", allow_pickle=True).item())
    return {**merged, "meanImg": tiled, "max_proj": tiled, "Ly": 3, "Lx": 8, "yrange": [0, 3], "xrange": [0, 8], "save_path": save_path}


@pytest.mark.parametrize("save_path", ["runs/session/suite2p/combined", "/data/session/suite2p/combined"], ids=["relative", "absolute"])
def test_combined_db_may_use_any_root(run, save_path):
    combined = _write_combined(run)
    np.save(combined / "db.npy", _combined_db(run, save_path.replace("suite2p", run.name)))
    assert verify_combined(run)["all_metadata_and_tiled_images_match"]


def test_combined_db_elsewhere_is_refused(run):
    combined = _write_combined(run)
    np.save(combined / "db.npy", _combined_db(run, "elsewhere/plane0"))
    with pytest.raises(ValueError, match="storage path"):
        verify_combined(run)


def test_a_changed_value_in_combined_db_is_refused(run):
    combined = _write_combined(run)
    db = _combined_db(run, f"{run.name}/combined")
    db["nframes"] = 1
    np.save(combined / "db.npy", db)
    with pytest.raises(ValueError, match="combined db"):
        verify_combined(run)
