"""The raw imaging NWB file: what's on disk equals the TIFFs and the log, read back independently."""

import warnings

import h5py
import numpy as np
import pytest
from pynwb import NWBHDF5IO, validate

from tank_lab_to_nwb.imaging.convert import convert_session, prepare_session
from tank_lab_to_nwb.imaging.verify import RoundTripError, verify_imaging
from tests.synthetic_log import make_block, make_log, make_trial, scalar
from tests.synthetic_tiff import FRAME_PERIOD, frame_pixels, software, write_acquisition

N_ITERATIONS = 900
STEP = 0.004  # seconds between iterations
INTERCEPT = 13.0  # the elapsed clock read zero 13 s after imaging started, 3 s before the session's first time


@pytest.fixture
def log():
    """One long trial, starting at elapsed 1 s."""
    trial = make_trial(start=1.0, n_frames=N_ITERATIONS, n_iti=0, duration=None)
    trial["time"] = (np.arange(N_ITERATIONS, dtype="float32") * STEP).reshape(-1, 1)
    trial["duration"] = np.array([[N_ITERATIONS * STEP]], dtype="float32")
    return make_log([make_block([trial], start=(2026, 7, 15, 12, 0, 1.0))], currentIt=scalar(0))


def _frame_packets(log, n_frames):
    """Packets every other iteration on the frame they fall in, after an exact line through the log's time."""
    trial = log["block"][0]["trial"][0]
    packets = {}
    for iteration in range(1, N_ITERATIONS + 1, 2):
        elapsed = float(trial["start"].item()) + float(trial["time"][iteration - 1, 0])
        time = elapsed * (1 - 5e-6) + INTERCEPT
        frame = int(time // FRAME_PERIOD) + 1
        if frame <= n_frames:
            packets.setdefault(frame, []).append((time, (1, 1, iteration)))
    return packets


@pytest.fixture
def tiffs(tmp_path, log):
    folder = tmp_path / "tiffs"
    folder.mkdir()
    return write_acquisition(folder, 801, frames_per_file=300, packets=_frame_packets(log, 801))


@pytest.fixture
def written(tmp_path, log, tiffs):
    path = tmp_path / "out.nwb"
    prepared = convert_session(log, tiffs, path, file="session.mat")
    return prepared, path, tiffs, log


def test_prepared_frame_times_put_imaging_before_the_session(log, tiffs):
    prepared = prepare_session(log, tiffs, file="session.mat")
    # Frame 0 falls before the session starts, so time zero moves to the first frame.
    assert prepared.frame_times[0] == pytest.approx(0.0, abs=2e-6)
    assert prepared.shift_microseconds > 0
    assert np.all(np.diff(prepared.frame_times) > 0)


def test_passes_pynwb_validate(written):
    errors = validate(path=str(written[1]))
    errors = errors[0] if isinstance(errors, tuple) else errors
    assert list(errors) == []


def test_no_silent_cast_warnings_on_write(tmp_path, log, tiffs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        convert_session(log, tiffs, tmp_path / "w.nwb", file="session.mat")


def test_each_plane_is_its_own_series_with_a_partial_last_volume(written):
    _, path, *_ = written
    with h5py.File(path, "r") as file:
        assert sorted(k for k in file["acquisition"]) == ["TwoPhotonSeriesPlane0", "TwoPhotonSeriesPlane1"]
        assert file["acquisition/TwoPhotonSeriesPlane0/data"].shape == (401, 4, 3)
        assert file["acquisition/TwoPhotonSeriesPlane1/data"].shape == (400, 4, 3)


def test_pixels_are_int16_and_transposed_to_time_x_y(written):
    _, path, *_ = written
    with h5py.File(path, "r") as file:
        data = file["acquisition/TwoPhotonSeriesPlane1/data"]
        assert data.dtype == np.int16
        # Frame numbers 2, 4, 6... are plane 1.
        np.testing.assert_array_equal(data[0], frame_pixels(2).T)
        np.testing.assert_array_equal(data[399], frame_pixels(800).T)


def test_compression_and_chunks_are_on_disk(written):
    _, path, *_ = written
    with h5py.File(path, "r") as file:
        data = file["acquisition/TwoPhotonSeriesPlane0/data"]
        assert data.compression == "gzip" and data.compression_opts == 4 and data.shuffle
        assert data.chunks[1:] == data.shape[1:]


def test_timestamps_follow_the_fit_and_each_plane_differs(written):
    prepared, path, *_ = written
    with h5py.File(path, "r") as file:
        zero = file["acquisition/TwoPhotonSeriesPlane0/timestamps"][()]
        one = file["acquisition/TwoPhotonSeriesPlane1/timestamps"][()]
    np.testing.assert_array_equal(zero, prepared.frame_times[prepared.acquisition.plane_frames(0)])
    np.testing.assert_array_equal(one, prepared.frame_times[prepared.acquisition.plane_frames(1)])
    assert zero[0] >= 0
    assert one[0] - zero[0] == pytest.approx(FRAME_PERIOD, abs=1e-6)


def test_frames_land_where_the_packets_say(written):
    """Independent of the fit's code: packet i, on the session clock, matches the behavior time of its iteration."""
    prepared, path, *_ = written
    with h5py.File(path, "r") as file:
        table = file["processing/scanimage/i2c_packets"]
        mapped, iteration = table["session_time"][()], table["iteration"][()]
        trial_start = file["intervals/trials/start_time"][0]
    expected = trial_start + (iteration - 1) * STEP
    np.testing.assert_allclose(mapped, expected, atol=2e-5)


def test_all_header_fields_and_packets_are_in_the_file(written):
    prepared, path, tiffs, _ = written
    with h5py.File(path, "r") as file:
        scanimage = file["processing/scanimage"]
        frames = scanimage["frames"]
        assert frames["frame_number"].shape == (801,)
        np.testing.assert_array_equal(frames["timestamp"][()], prepared.acquisition.frames["timestamp"])
        np.testing.assert_array_equal(frames["plane"][()], np.arange(801) % 2)
        assert frames["end_of_acquisition"].dtype.kind in "iub"
        assert scanimage["i2c_packets/block"].shape == prepared.acquisition.packets["block"].shape
        names = scanimage["files/file_name"].asstr()[()]
        assert list(names) == [tiff.name for tiff in tiffs]
        assert all("hStackManager" in text for text in scanimage["files/software"].asstr()[()])
        assert all("RoiGroups" in text for text in scanimage["files/artist"].asstr()[()])


def test_the_accuracy_is_recorded_in_the_file(written):
    _, path, *_ = written
    with h5py.File(path, "r") as file:
        table = file["processing/scanimage/alignment"]
        values = dict(zip(table["key"].asstr()[()], table["value"].asstr()[()]))
        meanings = dict(zip(table["key"].asstr()[()], table["meaning"].asstr()[()]))
    assert float(values["fit.slope"]) == pytest.approx(1 - 5e-6, abs=1e-9)
    assert float(values["fit.intercept"]) == pytest.approx(INTERCEPT, abs=1e-5)
    for key in ("fit.residual_p1_ms", "fit.residual_p99_ms", "fit.n_packets", "fit.n_rejected", "fit.method", "scanimage.epoch"):
        assert key in values
    assert "not corrected" in meanings["fit.send_latency"]


def test_session_start_moved_to_the_first_frame_and_the_original_is_kept(written):
    prepared, path, *_ = written
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        assert nwbfile.session_start_time == prepared.session.session_start_time
        settings = nwbfile.lab_meta_data["LabMetaData"].settings
        rows = dict(zip(settings["key"].data[:], settings["value"].data[:]))
    assert "derived.virmen_session_start" in rows


def test_placeholders_are_labelled(written):
    _, path, *_ = written
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        plane = io.read().imaging_planes["ImagingPlane0"]
        assert plane.indicator.startswith("PLACEHOLDER")
        assert plane.location == "root"
        assert "PLACEHOLDER" in plane.description
        assert np.isnan(plane.excitation_lambda)
        assert plane.imaging_rate == 25.0
        assert list(plane.grid_spacing) == [30.0, 40.0]
        assert plane.grid_spacing_unit == "micrometers"
        assert "10" in plane.description


def test_verify_passes_on_a_good_file(written):
    prepared, path, tiffs, log = written
    report = verify_imaging(prepared, path, tiffs)
    assert report["frames_checked"] == 801


@pytest.mark.parametrize(
    ("dataset", "change"),
    [
        ("acquisition/TwoPhotonSeriesPlane0/data", lambda d: d.__setitem__((5, 1, 1), d[5, 1, 1] + 1)),
        ("acquisition/TwoPhotonSeriesPlane1/timestamps", lambda d: d.__setitem__(3, d[3] + 1e-9)),
        ("processing/scanimage/frames/timestamp", lambda d: d.__setitem__(7, d[7] + 1e-9)),
        ("processing/scanimage/i2c_packets/iteration", lambda d: d.__setitem__(2, d[2] + 1)),
    ],
)
def test_verify_catches_a_changed_value(written, dataset, change):
    prepared, path, tiffs, _ = written
    with h5py.File(path, "r+") as file:
        change(file[dataset])
    with pytest.raises(RoundTripError):
        verify_imaging(prepared, path, tiffs)


def test_verify_catches_swapped_frames(written):
    prepared, path, tiffs, _ = written
    with h5py.File(path, "r+") as file:
        data = file["acquisition/TwoPhotonSeriesPlane0/data"]
        first, second = data[0].copy(), data[1].copy()
        data[0], data[1] = second, first
    with pytest.raises(RoundTripError, match="frame"):
        verify_imaging(prepared, path, tiffs)


def test_stub_cuts_only_the_pixels_and_their_timestamps(tmp_path, log, tiffs):
    path = tmp_path / "stub.nwb"
    prepared = convert_session(log, tiffs, path, file="session.mat", max_frames_per_plane=3)
    with h5py.File(path, "r") as file:
        assert file["acquisition/TwoPhotonSeriesPlane0/data"].shape == (3, 4, 3)
        assert file["acquisition/TwoPhotonSeriesPlane0/timestamps"].shape == (3,)
        assert file["processing/scanimage/frames/frame_number"].shape == (801,)
    assert verify_imaging(prepared, path, tiffs, max_frames_per_plane=3)["frames_checked"] == 6


def test_stub_of_zero_frames_is_refused(tmp_path, log, tiffs):
    with pytest.raises(ValueError, match="max_frames_per_plane"):
        convert_session(log, tiffs, tmp_path / "z.nwb", file="session.mat", max_frames_per_plane=0)


def test_a_log_with_no_matching_packets_is_refused(tmp_path, log):
    folder = tmp_path / "unsynced"
    folder.mkdir()
    tiffs = write_acquisition(folder, 10)
    with pytest.raises(ValueError, match="packets"):
        convert_session(log, tiffs, tmp_path / "x.nwb", file="session.mat")


def test_single_plane_session_writes(tmp_path, log):
    folder = tmp_path / "single"
    folder.mkdir()
    text = software(**{"SI.hStackManager.zs": "30", "SI.hStackManager.actualNumSlices": "1", "SI.hStackManager.numFramesPerVolumeWithFlyback": "1"})
    tiffs = write_acquisition(folder, 801, frames_per_file=300, packets=_frame_packets(log, 801), software_text=text)
    path = tmp_path / "single.nwb"
    prepared = convert_session(log, tiffs, path, file="session.mat")
    with h5py.File(path, "r") as file:
        assert list(file["acquisition"]) == ["TwoPhotonSeriesPlane0"]
        assert file["acquisition/TwoPhotonSeriesPlane0/data"].shape == (801, 4, 3)
    verify_imaging(prepared, path, tiffs)


def test_a_non_square_frame_keeps_its_orientation(tmp_path, log):
    folder = tmp_path / "wide"
    folder.mkdir()
    text = software(**{"SI.hRoiManager.linesPerFrame": "2", "SI.hRoiManager.pixelsPerLine": "6"})
    tiffs = write_acquisition(folder, 801, frames_per_file=300, packets=_frame_packets(log, 801), software_text=text, rows=2, cols=6)
    path = tmp_path / "wide.nwb"
    convert_session(log, tiffs, path, file="session.mat")
    with h5py.File(path, "r") as file:
        data = file["acquisition/TwoPhotonSeriesPlane0/data"]
        assert data.shape[1:] == (6, 2)
        np.testing.assert_array_equal(data[0], frame_pixels(1, 2, 6).T)


def test_the_trial_in_progress_is_kept_and_verified(tmp_path, log, tiffs):
    allocated = 6
    time = np.zeros((allocated, 1))
    time[:2, 0] = [0.0, 0.25]
    position = np.zeros((allocated, 3))
    position[:2] = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    log["currentIt"] = scalar(2)
    log["currentTrial"] = dict(start=scalar(30.0), time=time, position=position, viStart=np.array([[9]], dtype="uint32"))
    path = tmp_path / "progress.nwb"
    prepared = convert_session(log, tiffs, path, file="session.mat")
    with h5py.File(path, "r") as file:
        group = file["processing/in_progress_trial/position"]
        np.testing.assert_array_equal(group["data"][()], position[:2])
        assert group["data"].dtype == np.float64
        assert group["timestamps"][0] == pytest.approx(30.0 + prepared.clock_offset)
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        settings = io.read().lab_meta_data["LabMetaData"].settings
        rows = dict(zip(settings["key"].data[:], settings["value"].data[:]))
    assert rows["derived.in_progress_trial.trial"] == "2"
    assert rows["derived.in_progress_trial.viStart"] == "9"
    assert rows["derived.in_progress_trial.saved_iterations"] == "2"
    with h5py.File(path, "r+") as file:
        data = file["processing/in_progress_trial/position/data"]
        data[0, 0] = data[0, 0] + 1
    with pytest.raises(RoundTripError):
        verify_imaging(prepared, path, tiffs)


def test_frames_table_holds_the_pipeline_labels(written):
    prepared, path, *_ = written
    with h5py.File(path, "r") as file:
        frames = file["processing/scanimage/frames"]
        block, trial, iteration = frames["behavior_block"][()], frames["behavior_trial"][()], frames["behavior_iteration"][()]
    numbered = np.flatnonzero(block)
    assert numbered[0] > 0  # imaging started before the first trial
    assert set(block[numbered].tolist()) == {1} and set(trial[numbered].tolist()) == {1}
    assert np.all(np.diff(iteration[numbered]) >= 0)
    assert iteration[numbered[0]] == 1
