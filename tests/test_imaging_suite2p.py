"""Synthetic Suite2p output: all coordinates and counts are deliberately small."""

import h5py
import numpy as np
import pytest
from pynwb import validate

from tests.synthetic_tiff import frame_pixels
from tests.test_imaging_nwb import log, tiffs  # noqa: F401
from tank_lab_to_nwb.imaging.convert import prepare_session
from tank_lab_to_nwb.imaging.suite2p import load_suite2p


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
            F=np.arange(count, dtype="float32")[None, :],
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


def test_combined_is_only_tiling_and_padding(run):
    from tank_lab_to_nwb.imaging.convert_suite2p import verify_combined

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
        assert "open decision" in table.attrs["description"]


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
