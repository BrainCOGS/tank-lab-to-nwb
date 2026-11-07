"""The raw imaging NWB file: what's on disk equals the TIFFs and the log, read back independently."""

import json
import warnings

import h5py
import numpy as np
import pytest
from pynwb import NWBHDF5IO, validate

from tank_lab_to_nwb.imaging import alignment
from tank_lab_to_nwb.imaging import convert_session as cli
from tank_lab_to_nwb.imaging.convert import _int32, convert_session, prepare_session
from tank_lab_to_nwb.imaging.scanimage import read_acquisition
from tank_lab_to_nwb.imaging.verify import RoundTripError, verify_imaging
from tank_lab_to_nwb.virmen.coerce import LossyConversionError
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
        ("processing/scanimage/frames/file_index", lambda d: d.__setitem__(4, d[4] + 1)),
        ("processing/scanimage/frames/page_index", lambda d: d.__setitem__(4, d[4] + 1)),
        ("processing/scanimage/frames/acquisition_number", lambda d: d.__setitem__(4, d[4] + 1)),
        ("processing/scanimage/frames/frame_number_acquisition", lambda d: d.__setitem__(4, d[4] + 1)),
        ("processing/scanimage/frames/acq_trigger_timestamp", lambda d: d.__setitem__(4, d[4] + 1)),
        ("processing/scanimage/frames/next_file_marker_timestamp", lambda d: d.__setitem__(4, d[4] + 1)),
        ("processing/scanimage/frames/end_of_acquisition", lambda d: d.__setitem__(4, d[4] + 1)),
        ("processing/scanimage/frames/end_of_acquisition_mode", lambda d: d.__setitem__(4, d[4] + 1)),
        ("processing/scanimage/frames/dc_over_voltage", lambda d: d.__setitem__(4, d[4] + 1)),
        ("processing/scanimage/i2c_packets/used_in_fit", lambda d: d.__setitem__(2, not d[2])),
        ("processing/scanimage/files/n_frames", lambda d: d.__setitem__(0, d[0] + 1)),
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
        assert group["timestamps"][0] == 30.0 + prepared.clock_offset + 0.0
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


def test_the_file_describes_its_method_well_enough_to_reproduce(written):
    _, path, *_ = written
    with h5py.File(path, "r") as file:
        table = file["processing/scanimage/alignment"]
        values = dict(zip(table["key"].asstr()[()], table["value"].asstr()[()]))
        meanings = dict(zip(table["key"].asstr()[()], table["meaning"].asstr()[()]))
    for detail in ("least squares", "median", "1.4826", "refit"):
        assert detail in values["fit.method"]
    assert "not frame-time error" in meanings["fit.kept_std_ms"]


def test_the_send_latency_claims_no_bound_the_source_does_not_give(written):
    _, path, *_ = written
    with h5py.File(path, "r") as file:
        table = file["processing/scanimage/alignment"]
        meaning = dict(zip(table["key"].asstr()[()], table["meaning"].asstr()[()]))["fit.send_latency"]
    assert "no upper bound" in meaning
    assert "12 ms" not in meaning


def test_the_label_columns_explain_their_limits(written):
    _, path, *_ = written
    with h5py.File(path, "r") as file:
        frames = file["processing/scanimage/frames"]
        block = frames["behavior_block"].attrs["description"]
        trial = frames["behavior_trial"].attrs["description"]
    assert "just before the trial began" in block and "late" in block
    assert "running when the log was saved" in trial


@pytest.mark.parametrize(
    "dataset",
    [
        "processing/scanimage/files/file_name",
        "processing/scanimage/files/software",
        "processing/scanimage/files/artist",
        "processing/scanimage/alignment/key",
        "processing/scanimage/alignment/value",
        "processing/scanimage/alignment/meaning",
    ],
)
def test_verify_catches_changed_text(written, dataset):
    prepared, path, tiffs, _ = written
    with h5py.File(path, "r+") as file:
        file[dataset][0] = "changed"
    with pytest.raises(RoundTripError):
        verify_imaging(prepared, path, tiffs)


def _alignment(path):
    with h5py.File(path, "r") as file:
        table = file["processing/scanimage/alignment"]
        return dict(zip(table["key"].asstr()[()], table["value"].asstr()[()]))


@pytest.mark.parametrize("value", [2**31, -(2**31) - 1, 1.5, float("nan")])
def test_a_header_value_that_does_not_fit_is_refused_not_wrapped(value):
    with pytest.raises(LossyConversionError, match="frame_number"):
        _int32([1, value], "frame_number")


def test_integer_columns_at_the_edge_of_the_range_are_exact():
    values = _int32([2**31 - 1, -(2**31), 0], "frame_number")
    assert values.dtype == np.int32
    assert values.tolist() == [2**31 - 1, -(2**31), 0]


def test_an_empty_integer_column_stays_empty():
    assert _int32([], "frame_number").size == 0


def test_conversion_writes_a_report_beside_the_file(written):
    report = json.loads(written[1].with_suffix(".report.json").read_text())
    assert report["alignment"]["frames"] == 801
    assert "dropped" in report["coverage"]


def test_the_report_lists_the_in_progress_trial_when_nothing_of_it_was_kept(written):
    dropped = json.loads(written[1].with_suffix(".report.json").read_text())["coverage"]["dropped"]
    assert not written[0].in_progress
    assert {"log.currentTrial", "log.currentIt"} <= set(dropped)


def _with_in_progress(log, **extra):
    block = log["block"][0]
    count = 2
    log["currentTrial"] = {
        "start": np.array([[30.0]]),
        "time": np.array([[0.0], [0.004], [0.0]]),
        "position": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]]),
        "viStart": np.array([[9.0]]),
        **extra,
    }
    log["currentIt"] = np.array([[float(count)]])
    return block


def test_the_report_omits_the_in_progress_trial_when_it_was_kept(tmp_path, log, tiffs):
    _with_in_progress(log)
    path = tmp_path / "kept.nwb"
    prepared = convert_session(log, tiffs, path, file="session.mat")
    assert prepared.in_progress
    dropped = json.loads(path.with_suffix(".report.json").read_text())["coverage"]["dropped"]
    assert "log.currentTrial" not in dropped and "log.currentIt" not in dropped


def test_empty_in_progress_fields_are_named_in_the_file(tmp_path, log, tiffs):
    _with_in_progress(log, cueCombo=np.empty((0, 0)), emptyText="")
    path = tmp_path / "kept.nwb"
    convert_session(log, tiffs, path, file="session.mat")
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        settings = io.read().lab_meta_data["LabMetaData"].settings
        rows = dict(zip(settings["key"].data[:], settings["value"].data[:]))
    assert set(json.loads(rows["derived.in_progress_trial.empty_fields"])) == {"cueCombo", "emptyText"}


def test_long_stretches_without_packets_are_recorded_like_the_pipelines_warning(tmp_path, log, monkeypatch):
    # This fixture's packets span only 2 s of frames, so lower the threshold rather than the realism of the fit.
    monkeypatch.setattr(alignment, "PACKET_GAP_SECONDS", 0.5)
    packets = _frame_packets(log, 801)
    folder = tmp_path / "gappy"
    folder.mkdir()
    numbers = sorted(packets)
    missing = set(numbers[20:70])  # 1 s of frames at 50 Hz
    tiffs = write_acquisition(folder, 801, frames_per_file=300, packets={number: value for number, value in packets.items() if number not in missing})
    path = tmp_path / "gappy.nwb"
    convert_session(log, tiffs, path, file="session.mat")
    values = _alignment(path)
    assert values["labels.gaps_over_2s"] == "1"
    assert float(values["labels.longest_gap_seconds"]) == pytest.approx(49 * FRAME_PERIOD + FRAME_PERIOD, abs=1e-9)


def test_a_gap_is_measured_from_its_first_frame_without_a_packet_to_the_next_packet(tmp_path):
    packet = [(0.0, (1, 1, 1))]
    paths = write_acquisition(tmp_path, 300, frames_per_file=400, packets={2: packet, 3: packet, 200: packet, 201: packet})
    # Frames 4 to 199 have no packet: from frame 4 at 0.06 s to frame 200 at 3.98 s.
    assert alignment.packet_gaps(read_acquisition(paths)).tolist() == [3.98 - 0.06]


@pytest.mark.parametrize("packets", [{}, {5: [(0.0, (1, 1, 1))]}, {5: [(0.0, (1, 1, 1))], 6: [(0.0, (1, 1, 2))]}])
def test_no_interior_gap_when_nothing_lies_between_packets(tmp_path, packets):
    paths = write_acquisition(tmp_path, 10, frames_per_file=400, packets=packets)
    assert alignment.packet_gaps(read_acquisition(paths)).size == 0


def test_short_stretches_without_packets_are_not_counted(written):
    values = _alignment(written[1])
    assert values["labels.gaps_over_2s"] == "0"
    assert float(values["labels.longest_gap_seconds"]) < 2.0


def test_the_acquisition_id_is_stable_and_names_only_this_acquisition(tmp_path, log, written):
    prepared, path, tiffs, _ = written
    again = tmp_path / "again.nwb"
    convert_session(log, tiffs, again, file="session.mat")
    first = _alignment(path)["acquisition.id"]
    assert first == _alignment(again)["acquisition.id"]
    assert len(first) == 64
    other = tmp_path / "other"
    other.mkdir()
    shorter = write_acquisition(other, 800, frames_per_file=300, packets=_frame_packets(log, 800))
    assert prepare_session(log, shorter, file="session.mat")
    elsewhere = tmp_path / "elsewhere.nwb"
    convert_session(log, shorter, elsewhere, file="session.mat")
    assert _alignment(elsewhere)["acquisition.id"] != first


class TestCommandLine:
    def test_converts_and_reports(self, tmp_path, log, tiffs, monkeypatch):
        monkeypatch.setattr(cli, "convert_session", convert_session_with_log(log))
        path = cli.convert("session.mat", tmp_path / "out", tiffs)
        assert path == tmp_path / "out" / "session.nwb"
        assert path.exists() and path.with_suffix(".report.json").exists()

    def test_stub_is_named_as_one(self, tmp_path, log, tiffs, monkeypatch):
        monkeypatch.setattr(cli, "convert_session", convert_session_with_log(log))
        path = cli.convert("session.mat", tmp_path / "out", tiffs, stub=5)
        assert path.name == "session_stub.nwb"

    def test_refuses_to_start_without_room_for_the_frames(self, tmp_path, tiffs, monkeypatch):
        usage = type("Usage", (), {"free": 0})
        monkeypatch.setattr(cli.shutil, "disk_usage", lambda _: usage)
        with pytest.raises(OSError, match="free"):
            cli.convert("session.mat", tmp_path / "out", tiffs)

    def test_a_stub_does_not_need_room_for_every_frame(self, tmp_path, log, tiffs, monkeypatch):
        usage = type("Usage", (), {"free": 0})
        monkeypatch.setattr(cli.shutil, "disk_usage", lambda _: usage)
        monkeypatch.setattr(cli, "convert_session", convert_session_with_log(log))
        assert cli.convert("session.mat", tmp_path / "out", tiffs, stub=1).exists()

    def test_the_usage_says_where_the_file_goes(self):
        assert "<output>/staging" not in cli.__doc__
        assert "<output>" in cli.__doc__


def convert_session_with_log(log):
    """The command line takes a path; these tests hand it the synthetic log already decoded."""

    def run(_path, tiffs, output, **options):
        return convert_session(log, tiffs, output, file="session.mat", **options)

    return run
