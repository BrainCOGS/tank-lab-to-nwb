"""Multi-ROI imaging, from a full ScanImage stack or from the lab's per-ROI split folders."""

import json

import h5py
import numpy as np
import pytest
from pynwb import validate

from tank_lab_to_nwb.imaging import convert_session as cli
from tank_lab_to_nwb.imaging.convert import convert_session, prepare_session
from tank_lab_to_nwb.imaging.scanimage import ScanImageError, find_tiffs, read_acquisition, read_field
from tank_lab_to_nwb.imaging.verify import RoundTripError, verify_imaging
from tests.synthetic_log import make_block, make_log, make_trial, scalar
from tests.synthetic_tiff import FLYBACK_ROWS, FRAME_PERIOD, frame_pixels, mroi_artist, mroi_software, software, split_stack, write_stack

THREE = [dict(rows=3, name="a"), dict(rows=2, name="b"), dict(rows=3, name="c")]
PAGE_ROWS = 3 + 2 + 3 + 2 * FLYBACK_ROWS
LINE_PERIOD = 6.3e-05
N_ITERATIONS = 900
STEP = 0.004
INTERCEPT = 13.0


@pytest.fixture
def log():
    trial = make_trial(start=1.0, n_frames=N_ITERATIONS, n_iti=0, duration=None)
    trial["time"] = (np.arange(N_ITERATIONS, dtype="float32") * STEP).reshape(-1, 1)
    trial["duration"] = np.array([[N_ITERATIONS * STEP]], dtype="float32")
    return make_log([make_block([trial], start=(2026, 7, 15, 12, 0, 1.0))], currentIt=scalar(0))


def _packets(log, n_frames):
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
def stack(tmp_path, log):
    folder = tmp_path / "originalStacks"
    folder.mkdir()
    return write_stack(folder, 801, THREE, frames_per_file=300, packets=_packets(log, 801))


@pytest.fixture
def split(tmp_path, stack):
    return split_stack(stack, tmp_path / "split", THREE)


def _page(number):
    return frame_pixels(number, PAGE_ROWS, 4)


class TestStack:
    def test_each_roi_is_a_field_with_its_own_rows(self, tmp_path):
        acquisition = read_acquisition(write_stack(tmp_path, 5, THREE))
        assert [(field.label, field.name, field.rows) for field in acquisition.fields] == [
            ("Plane0Roi1", "a", (0, 3)),
            ("Plane0Roi2", "b", (5, 7)),
            ("Plane0Roi3", "c", (9, 12)),
        ]

    def test_flyback_rows_are_what_is_left_of_the_page(self, tmp_path):
        layout = read_acquisition(write_stack(tmp_path, 5, THREE)).layout
        assert layout.flyback_rows == {0: ((3, 5), (7, 9))}

    def test_a_field_starts_when_its_first_row_was_scanned(self, tmp_path):
        acquisition = read_acquisition(write_stack(tmp_path, 5, THREE))
        for field in acquisition.fields:
            np.testing.assert_array_equal(field.timestamps, acquisition.frames["timestamp"] + field.rows[0] * LINE_PERIOD)

    def test_field_pixels_are_exactly_their_rows(self, tmp_path):
        acquisition = read_acquisition(write_stack(tmp_path, 5, THREE))
        field = acquisition.fields[1]
        got = read_field(acquisition, field, [0, 4])
        np.testing.assert_array_equal(got[0], _page(1)[5:7])
        np.testing.assert_array_equal(got[1], _page(5)[5:7])
        assert got.dtype == np.int16

    def test_geometry_comes_from_each_rois_header(self, tmp_path):
        rois = [dict(rows=3, center=[1.0, 2.0]), dict(rows=3, center=[3.0, -1.0])]
        fields = read_acquisition(write_stack(tmp_path, 5, rois)).fields
        assert fields[1].size_um == (120.0, 120.0)
        assert fields[1].pixel_size_um == (30.0, 40.0)
        # Top-left corner: (center - size / 2) x 15 um per degree, and the plane's z.
        assert fields[1].origin_um == ((3.0 - 4.0) * 15, (-1.0 - 4.0) * 15, 10.0)

    def test_rois_of_different_heights(self, tmp_path):
        rois = [dict(rows=5), dict(rows=1)]
        fields = read_acquisition(write_stack(tmp_path, 3, rois)).fields
        assert [field.rows for field in fields] == [(0, 5), (7, 8)]

    def test_different_rois_on_different_depths(self, tmp_path):
        rois = [dict(rows=3, z=10), dict(rows=3, z=10), dict(rows=4, z=50), dict(rows=4, z=50)]
        acquisition = read_acquisition(write_stack(tmp_path, 5, rois, zs=(10, 50), page_rows=10, frames_per_file=8))
        assert [(field.label, field.plane, field.rows) for field in acquisition.fields] == [
            ("Plane0Roi1", 0, (0, 3)),
            ("Plane0Roi2", 0, (7, 10)),
            ("Plane1Roi3", 1, (0, 4)),
            ("Plane1Roi4", 1, (6, 10)),
        ]
        assert acquisition.fields[0].frame_indices.tolist() == [0, 2, 4]
        assert acquisition.fields[2].frame_indices.tolist() == [1, 3]

    def test_rows_that_do_not_divide_into_even_gaps_are_refused(self, tmp_path):
        with pytest.raises(ScanImageError, match="flyback"):
            read_acquisition(write_stack(tmp_path, 3, [dict(rows=3), dict(rows=2), dict(rows=2)], page_rows=12))

    def test_rois_taller_than_the_page_are_refused(self, tmp_path):
        with pytest.raises(ScanImageError, match="flyback"):
            read_acquisition(write_stack(tmp_path, 3, [dict(rows=3), dict(rows=3)], page_rows=5))

    def test_a_roi_on_no_plane_is_refused(self, tmp_path):
        with pytest.raises(ScanImageError, match="no plane"):
            read_acquisition(write_stack(tmp_path, 3, [dict(rows=3, z=10), dict(rows=3, z=99)], page_rows=3))

    def test_a_plane_with_no_roi_is_refused(self, tmp_path):
        with pytest.raises(ScanImageError, match="no ROI"):
            read_acquisition(write_stack(tmp_path, 3, [dict(rows=3, z=10), dict(rows=3, z=10)], zs=(10, 50), page_rows=8))

    def test_a_roi_of_another_width_is_refused(self, tmp_path):
        rois = [dict(rows=3), dict(rows=3, cols=8)]
        with pytest.raises(ScanImageError, match="wide"):
            read_acquisition(write_stack(tmp_path, 3, rois, page_rows=8))

    def test_several_rois_without_mroi_enabled_are_refused(self, tmp_path):
        text = software(**{"SI.hRoiManager.mroiEnable": "0"})
        with pytest.raises(ScanImageError, match="ROI"):
            read_acquisition(write_stack(tmp_path, 3, THREE, software_text=text))

    def test_one_roi_keeps_the_plain_names(self, tmp_path):
        acquisition = read_acquisition(write_stack(tmp_path, 3, [dict(rows=3)], software_text=mroi_software()))
        assert [field.label for field in acquisition.fields] == ["Plane0"]
        assert acquisition.layout.flyback_rows == {0: ()}


class TestStackFile:
    @pytest.fixture
    def written(self, tmp_path, log, stack):
        path = tmp_path / "out.nwb"
        return convert_session(log, stack, path, file="session.mat"), path, stack

    def test_passes_pynwb_validate(self, written):
        assert not validate(path=str(written[1]))

    def test_each_field_is_a_series_on_its_own_plane(self, written):
        with h5py.File(written[1], "r") as file:
            acquisition = file["acquisition"]
            assert set(acquisition) == {"TwoPhotonSeriesPlane0Roi1", "TwoPhotonSeriesPlane0Roi2", "TwoPhotonSeriesPlane0Roi3", "FlybackLinesPlane0"}
            assert acquisition["TwoPhotonSeriesPlane0Roi2/data"].shape == (801, 4, 2)
            assert acquisition["TwoPhotonSeriesPlane0Roi2"].get("imaging_plane", getlink=True).path.endswith("ImagingPlane0Roi2")

    def test_pixels_are_transposed_like_a_single_roi(self, written):
        with h5py.File(written[1], "r") as file:
            np.testing.assert_array_equal(file["acquisition/TwoPhotonSeriesPlane0Roi2/data"][6], _page(7)[5:7].T)

    def test_flyback_rows_are_kept_so_every_page_can_be_rebuilt(self, written):
        with h5py.File(written[1], "r") as file:
            page = _page(7)
            rebuilt = np.vstack([
                file["acquisition/TwoPhotonSeriesPlane0Roi1/data"][6].T,
                file["acquisition/FlybackLinesPlane0/data"][6].T[:2],
                file["acquisition/TwoPhotonSeriesPlane0Roi2/data"][6].T,
                file["acquisition/FlybackLinesPlane0/data"][6].T[2:],
                file["acquisition/TwoPhotonSeriesPlane0Roi3/data"][6].T,
            ])
        np.testing.assert_array_equal(rebuilt, page)

    def test_each_series_has_its_rois_own_times(self, written):
        prepared, path, _ = written
        with h5py.File(path, "r") as file:
            first = file["acquisition/TwoPhotonSeriesPlane0Roi1/timestamps"][()]
            third = file["acquisition/TwoPhotonSeriesPlane0Roi3/timestamps"][()]
        assert first[0] == pytest.approx(prepared.frame_times[0], abs=1e-12)
        assert third[0] - first[0] == pytest.approx(9 * LINE_PERIOD, abs=1e-6)

    def test_the_rows_and_offsets_are_recorded(self, written):
        with h5py.File(written[1], "r") as file:
            table = file["processing/ophys/scanimage_alignment"]
            values = dict(zip(table["key"].asstr()[()], table["value"].asstr()[()]))
        assert values["roi.Plane0Roi2.name"] == "b"
        assert values["roi.Plane0Roi2.rows"] == "5:7"
        assert float(values["roi.Plane0Roi2.time_offset_seconds"]) == 5 * LINE_PERIOD
        assert "ScanImage" in values["acquisition.source"]

    @pytest.mark.parametrize("dataset", ["acquisition/TwoPhotonSeriesPlane0Roi2/data", "acquisition/FlybackLinesPlane0/data"])
    def test_verify_catches_a_changed_pixel(self, written, dataset):
        prepared, path, tiffs = written
        with h5py.File(path, "r+") as file:
            file[dataset][3, 1, 1] += 1
        with pytest.raises(RoundTripError):
            verify_imaging(prepared, path, tiffs)

    def test_verify_catches_a_changed_time(self, written):
        prepared, path, tiffs = written
        with h5py.File(path, "r+") as file:
            times = file["acquisition/TwoPhotonSeriesPlane0Roi3/timestamps"]
            times[3] += 1e-9
        with pytest.raises(RoundTripError, match="Roi3"):
            verify_imaging(prepared, path, tiffs)

    def test_a_stub_cuts_every_series(self, tmp_path, log, stack):
        path = tmp_path / "stub.nwb"
        convert_session(log, stack, path, file="session.mat", max_frames_per_plane=5)
        with h5py.File(path, "r") as file:
            assert file["acquisition/TwoPhotonSeriesPlane0Roi3/data"].shape[0] == 5
            assert file["acquisition/FlybackLinesPlane0/data"].shape[0] == 5


class TestSplit:
    def test_each_folder_is_a_field(self, split):
        acquisition = read_acquisition(split, allow_incomplete=True)
        assert [(field.label, field.name, field.plane, field.rows) for field in acquisition.fields] == [
            ("Plane0Roi1", "ROI01", 0, (0, 3)),
            ("Plane0Roi2", "ROI02", 0, (0, 2)),
            ("Plane0Roi3", "ROI03", 0, (0, 3)),
        ]

    def test_the_pixels_are_what_the_split_files_hold(self, split):
        acquisition = read_acquisition(split, allow_incomplete=True)
        got = read_field(acquisition, acquisition.fields[1], [6])[0]
        np.testing.assert_array_equal(got, np.clip(_page(7)[5:7], 0, None).astype("uint16"))
        assert got.dtype == np.uint16

    def test_frames_and_packets_match_the_stack(self, stack, split):
        whole, parts = read_acquisition(stack), read_acquisition(split, allow_incomplete=True)
        np.testing.assert_array_equal(parts.frames["frame_number"], whole.frames["frame_number"])
        for name in ("frame_index", "block", "trial", "iteration", "timestamp"):
            np.testing.assert_array_equal(parts.packets[name], whole.packets[name])

    def test_each_field_keeps_the_time_written_in_its_own_files(self, split):
        fields = read_acquisition(split, allow_incomplete=True).fields
        assert fields[2].timestamps[0] == 0.002

    def test_files_of_every_roi_are_listed(self, split):
        assert len(read_acquisition(split, allow_incomplete=True).files) == len(split)

    def test_several_depths_merge_back_into_frame_order(self, tmp_path):
        rois = [dict(rows=3, z=10), dict(rows=3, z=10), dict(rows=4, z=50), dict(rows=4, z=50)]
        stack = write_stack(tmp_path, 5, rois, zs=(10, 50), page_rows=10, frames_per_file=8)
        parts = split_stack(stack, tmp_path / "split", rois, zs=(10, 50))
        acquisition = read_acquisition(parts, allow_incomplete=True)
        assert acquisition.frames["frame_number"].tolist() == [1, 2, 3, 4, 5]
        assert [(field.label, field.frame_indices.tolist()) for field in acquisition.fields] == [
            ("Plane0Roi1", [0, 2, 4]),
            ("Plane0Roi2", [0, 2, 4]),
            ("Plane1Roi3", [1, 3]),
            ("Plane1Roi4", [1, 3]),
        ]

    def test_a_roi_left_off_the_list_is_an_error(self, split):
        with pytest.raises(ScanImageError, match="ROI03"):
            read_acquisition([path for path in split if "ROI03" not in path.name], allow_incomplete=True)

    def test_a_file_left_off_the_list_is_an_error(self, split):
        with pytest.raises(ScanImageError, match="wasn't given"):
            read_acquisition([path for path in split if not path.name.endswith("ROI02_z1_00002.tif")], allow_incomplete=True)

    def test_a_roi_with_fewer_files_is_an_error(self, split):
        doomed = [path for path in split if path.name.endswith("ROI02_z1_00003.tif")][0]
        doomed.unlink()
        with pytest.raises(ScanImageError, match="ROI"):
            read_acquisition([path for path in split if path != doomed], allow_incomplete=True)

    def test_a_stack_and_its_split_copies_together_are_an_error(self, stack, split):
        with pytest.raises(ScanImageError, match="either"):
            read_acquisition([*stack, *split], allow_incomplete=True)

    def test_rois_that_disagree_about_i2c_are_an_error(self, tmp_path, log):
        folder = tmp_path / "a"
        folder.mkdir()
        one = write_stack(folder, 12, THREE, frames_per_file=12, packets={2: [(0.03, (1, 1, 1))]})
        other = tmp_path / "b"
        other.mkdir()
        two = write_stack(other, 12, THREE, frames_per_file=12, packets={2: [(0.03, (1, 1, 3))]})
        parts_one = split_stack(one, tmp_path / "split", THREE)
        parts_two = split_stack(two, tmp_path / "other", THREE)
        bad = [parts_one[0], parts_two[1], parts_one[2]]
        (tmp_path / "mixed").mkdir()
        moved = []
        for path in bad:
            target = tmp_path / "mixed" / path.name
            target.write_bytes(path.read_bytes())
            moved.append(target)
        with pytest.raises(ScanImageError, match="I2C"):
            read_acquisition(moved, allow_incomplete=True)

    def test_a_roi_whose_clock_drifts_from_the_others_is_an_error(self, tmp_path):
        stack = write_stack(tmp_path, 12, THREE, frames_per_file=12)
        with pytest.raises(ScanImageError, match="time"):
            read_acquisition(_with_changed_time(tmp_path, stack), allow_incomplete=True)

    def test_headers_in_the_split_files_give_the_geometry(self, tmp_path):
        text = mroi_software(**{"SI.hRoiManager.mroiEnable": "0"})
        stack = write_stack(tmp_path, 6, THREE, frames_per_file=4)
        parts = split_stack(stack, tmp_path / "split", THREE, keep_tags=True, software_text=text, artist_text=mroi_artist(THREE))
        field = read_acquisition(parts, allow_incomplete=True).fields[1]
        assert field.name == "b"
        assert field.size_um == (120.0, 120.0)
        assert field.origin_um[2] == 10.0

    def test_without_headers_the_geometry_is_unknown_and_the_rate_comes_from_the_frames(self, split):
        acquisition = read_acquisition(split, allow_incomplete=True)
        assert acquisition.fields[0].size_um is None and acquisition.fields[0].pixel_size_um is None
        assert acquisition.layout.volume_rate == pytest.approx(1 / FRAME_PERIOD, rel=1e-3)

    def test_a_summed_width_is_refused(self, tmp_path):
        rois = [dict(rows=3, cols=8), dict(rows=3, cols=8)]
        text = mroi_software(**{"SI.hRoiManager.mroiEnable": "0", "SI.hRoiManager.pixelsPerLine": "8"})
        stack = write_stack(tmp_path, 6, [dict(rows=3), dict(rows=3)], page_rows=8, frames_per_file=4)
        parts = split_stack(stack, tmp_path / "split", [dict(rows=3), dict(rows=3)], keep_tags=True, software_text=text, artist_text=mroi_artist(rois))
        with pytest.raises(ScanImageError, match="wide"):
            read_acquisition(parts, allow_incomplete=True)


def _with_changed_time(tmp_path, stack):
    import tifffile

    parts = split_stack(stack, tmp_path / "split", THREE)
    path = next(path for path in parts if "ROI02" in path.name)
    with tifffile.TiffFile(path) as source:
        pages = [(page.asarray(), page.description) for page in source.pages]
    with tifffile.TiffWriter(path) as writer:
        for index, (pixels, text) in enumerate(pages):
            if index == 5:
                text = text.replace("frameTimestamps_sec = 0.101", "frameTimestamps_sec = 0.2")
            writer.write(pixels, description=text, contiguous=False, metadata=None)
    return parts


class TestSplitFile:
    @pytest.fixture
    def written(self, tmp_path, log, split):
        path = tmp_path / "out.nwb"
        return convert_session(log, split, path, file="session.mat", allow_incomplete=True), path, split

    def test_passes_pynwb_validate(self, written):
        assert not validate(path=str(written[1]))

    def test_series_hold_uint16_and_no_flyback(self, written):
        with h5py.File(written[1], "r") as file:
            assert set(file["acquisition"]) == {"TwoPhotonSeriesPlane0Roi1", "TwoPhotonSeriesPlane0Roi2", "TwoPhotonSeriesPlane0Roi3"}
            assert file["acquisition/TwoPhotonSeriesPlane0Roi2/data"].dtype == np.uint16
            np.testing.assert_array_equal(file["acquisition/TwoPhotonSeriesPlane0Roi2/data"][6], np.clip(_page(7)[5:7], 0, None).astype("uint16").T)

    def test_the_file_says_how_these_pixels_and_times_differ_from_the_stack(self, written):
        with h5py.File(written[1], "r") as file:
            table = file["processing/ophys/scanimage_alignment"]
            values = dict(zip(table["key"].asstr()[()], table["value"].asstr()[()]))
            meanings = dict(zip(table["key"].asstr()[()], table["meaning"].asstr()[()]))
            description = file["acquisition/TwoPhotonSeriesPlane0Roi1"].attrs["description"]
        assert "split" in values["acquisition.source"]
        assert "clipped" in meanings["acquisition.source"]
        assert "rounded" in meanings["acquisition.source"]
        assert "clipped" in description

    def test_verify_passes_and_catches_a_changed_pixel(self, written):
        prepared, path, tiffs = written
        report = verify_imaging(prepared, path, tiffs)
        assert report["frames_checked"] == 801 * 3
        with h5py.File(path, "r+") as file:
            file["acquisition/TwoPhotonSeriesPlane0Roi3/data"][4, 1, 1] += 1
        with pytest.raises(RoundTripError):
            verify_imaging(prepared, path, tiffs)

    def test_verify_catches_a_changed_time(self, written):
        prepared, path, tiffs = written
        with h5py.File(path, "r+") as file:
            times = file["acquisition/TwoPhotonSeriesPlane0Roi2/timestamps"]
            times[2] += 1e-9
        with pytest.raises(RoundTripError, match="Roi2"):
            verify_imaging(prepared, path, tiffs)


class TestCompleteness:
    def test_split_files_without_headers_cannot_confirm_they_are_all_there(self, split):
        with pytest.raises(ScanImageError, match="allow_incomplete"):
            read_acquisition(split)

    def test_a_full_last_file_can_be_accepted_and_is_recorded(self, tmp_path, log):
        text = mroi_software(**{"SI.hScan2D.logFramesPerFile": "300"})
        folder = tmp_path / "s"
        folder.mkdir()
        paths = write_stack(folder, 900, THREE, frames_per_file=300, software_text=text, packets=_packets(log, 900))
        with pytest.raises(ScanImageError, match="allow_incomplete"):
            prepare_session(log, paths, file="session.mat")
        path = tmp_path / "out.nwb"
        convert_session(log, paths, path, file="session.mat", allow_incomplete=True)
        with h5py.File(path, "r") as file:
            table = file["processing/ophys/scanimage_alignment"]
            values = dict(zip(table["key"].asstr()[()], table["value"].asstr()[()]))
        assert values["acquisition.completeness"].startswith("not confirmed")

    def test_a_complete_acquisition_says_so(self, tmp_path, log, stack):
        path = tmp_path / "out.nwb"
        convert_session(log, stack, path, file="session.mat")
        report = json.loads(path.with_suffix(".report.json").read_text())
        assert report["alignment"]["acquisition.completeness"].startswith("no gap found")


class TestFinding:
    def test_the_stack_is_preferred_when_both_exist(self, tmp_path, stack, split):
        (tmp_path / "ROI01_z1").mkdir()
        assert find_tiffs(tmp_path) == sorted(stack)

    def test_split_folders_are_found(self, tmp_path, stack):
        root = tmp_path / "session"
        parts = split_stack(stack, root, THREE)
        assert find_tiffs(root) == sorted(parts)

    def test_tiffs_in_the_folder_itself_are_found(self, tmp_path):
        paths = write_stack(tmp_path, 3, THREE)
        assert find_tiffs(tmp_path) == sorted(paths)

    def test_nothing_to_find_is_an_error(self, tmp_path):
        with pytest.raises(ScanImageError, match="no TIFF"):
            find_tiffs(tmp_path)

    def test_the_command_line_accepts_a_folder(self, tmp_path, log, stack, monkeypatch):
        def run(_path, tiffs, output, **options):
            return convert_session(log, tiffs, output, file="session.mat", **options)

        monkeypatch.setattr(cli, "convert_session", run)
        path = cli.convert("session.mat", tmp_path / "out", [tmp_path], allow_incomplete=False)
        assert path.exists()


class TestReviewFindings:
    """Cases an independent check of the real split data and of unusual layouts turned up."""

    def _field(self, rows, center=(1.0, 0.0), size=(8.0, 8.0), rotation=0):
        return {"pixelResolutionXY": [4, rows], "sizeXY": list(size), "centerXY": list(center), "rotationDegrees": rotation}

    # The splitter's replace changes an I2C packet stamped with exactly the frame's time.
    def test_a_packet_stamped_with_the_frame_time_is_recognized_as_the_splitters_change(self, tmp_path):
        packets = {2: [(FRAME_PERIOD, (1, 1, 1))], 3: [(0.05, (1, 1, 3))]}
        stack = write_stack(tmp_path, 8, THREE, frames_per_file=8, packets=packets)
        acquisition = read_acquisition(split_stack(stack, tmp_path / "split", THREE), allow_incomplete=True)
        assert acquisition.packets["iteration"].tolist() == [1, 3]
        assert acquisition.packets["timestamp"].tolist() == [0.02, 0.05]
        assert acquisition.split_packet_changes == 1

    def test_a_stack_has_no_such_changes(self, tmp_path):
        assert read_acquisition(write_stack(tmp_path, 3, THREE)).split_packet_changes == 0

    def test_the_change_is_recorded_in_the_file(self, tmp_path, log):
        folder = tmp_path / "s"
        folder.mkdir()
        packets = _packets(log, 801)
        number = sorted(packets)[10]
        packets[number] = [(FRAME_PERIOD * (number - 1), packets[number][0][1]), *packets[number][1:]]
        parts = split_stack(write_stack(folder, 801, THREE, frames_per_file=300, packets=packets), tmp_path / "split", THREE)
        path = tmp_path / "out.nwb"
        convert_session(log, parts, path, file="session.mat", allow_incomplete=True)
        with h5py.File(path, "r") as file:
            table = file["processing/ophys/scanimage_alignment"]
            values = dict(zip(table["key"].asstr()[()], table["value"].asstr()[()]))
        assert values["acquisition.split_packet_changes"] == "1"

    def test_any_other_packet_time_difference_is_still_refused(self, tmp_path):
        one = tmp_path / "a"
        two = tmp_path / "b"
        one.mkdir()
        two.mkdir()
        first = split_stack(write_stack(one, 8, THREE, frames_per_file=8, packets={2: [(0.03, (1, 1, 1))]}), tmp_path / "s1", THREE)
        second = split_stack(write_stack(two, 8, THREE, frames_per_file=8, packets={2: [(0.031, (1, 1, 1))]}), tmp_path / "s2", THREE)
        mixed = tmp_path / "mixed"
        mixed.mkdir()
        chosen = []
        for path in (first[0], second[1], first[2]):
            target = mixed / path.name
            target.write_bytes(path.read_bytes())
            chosen.append(target)
        with pytest.raises(ScanImageError, match="I2C"):
            read_acquisition(chosen, allow_incomplete=True)

    @pytest.mark.parametrize(("second_lag", "passes"), [(0.0011, True), (0.0012, False)])
    def test_rounding_of_two_stamps_is_allowed_and_no_more(self, tmp_path, second_lag, passes):
        import tifffile

        stack = write_stack(tmp_path, 6, THREE, frames_per_file=6)
        parts = split_stack(stack, tmp_path / "split", THREE)
        path = next(item for item in parts if "ROI02" in item.name)
        with tifffile.TiffFile(path) as source:
            pages = [(page.asarray(), page.description) for page in source.pages]
        with tifffile.TiffWriter(path) as writer:
            for index, (pixels, text) in enumerate(pages):
                base = (index) * FRAME_PERIOD
                lag = {0: 0.0009, 3: second_lag}.get(index, 0.001)
                old = text.split("frameTimestamps_sec = ")[1].split("\n")[0]
                writer.write(pixels, description=text.replace(f"frameTimestamps_sec = {old}", f"frameTimestamps_sec = {base + lag:.4f}"), contiguous=False, metadata=None)
        if passes:
            assert read_acquisition(parts, allow_incomplete=True)
        else:
            with pytest.raises(ScanImageError, match="lag"):
                read_acquisition(parts, allow_incomplete=True)

    # Geometry
    def test_scanfields_that_differ_by_depth_are_refused(self, tmp_path):
        rois = [dict(rows=3, z=[10, 50], scanfields=[self._field(3), self._field(3, center=(9.0, 8.0))]), dict(rows=3, z=[10, 50], scanfields=[self._field(3), self._field(3)])]
        with pytest.raises(ScanImageError, match="scanfields"):
            read_acquisition(write_stack(tmp_path, 4, rois, zs=(10, 50), page_rows=8))

    def test_identical_scanfields_are_one_scanfield(self, tmp_path):
        rois = [dict(rows=3, z=[10, 50], scanfields=[self._field(3), self._field(3)]), dict(rows=3, z=[10, 50], scanfields=[self._field(3), self._field(3)])]
        assert len(read_acquisition(write_stack(tmp_path, 4, rois, zs=(10, 50), page_rows=8)).fields) == 4

    def test_a_rotated_scanfield_is_refused(self, tmp_path):
        with pytest.raises(ScanImageError, match="rotat"):
            read_acquisition(write_stack(tmp_path, 3, [dict(rows=3, rotation=30), dict(rows=3)], page_rows=8))

    # Depths of split files
    def test_swapped_depth_folders_are_refused(self, tmp_path):
        rois = [dict(rows=3, z=10), dict(rows=3, z=50)]
        stack = write_stack(tmp_path, 6, rois, zs=(10, 50), page_rows=3, frames_per_file=8)
        parts = split_stack(stack, tmp_path / "split", rois, zs=(10, 50))
        swapped = tmp_path / "swapped"
        swapped.mkdir()
        chosen = []
        for path in parts:
            name = path.name.replace("_z1_", "_zX_").replace("_z2_", "_z1_").replace("_zX_", "_z2_")
            target = swapped / name
            target.write_bytes(path.read_bytes())
            chosen.append(target)
        with pytest.raises(ScanImageError, match="depth"):
            read_acquisition(chosen, allow_incomplete=True)

    def test_a_file_boundary_inside_a_volume_is_refused_when_the_splitter_restarted_its_cycle(self, tmp_path):
        rois = [dict(rows=3, z=10), dict(rows=3, z=50)]
        stack = write_stack(tmp_path, 7, rois, zs=(10, 50), page_rows=3, frames_per_file=3)
        parts = split_stack(stack, tmp_path / "split", rois, zs=(10, 50), file_local_depth=True)
        with pytest.raises(ScanImageError, match="depth|frame"):
            read_acquisition(parts, allow_incomplete=True)

    def test_settings_whose_depths_are_not_ascending_are_refused_for_split_files(self, tmp_path):
        rois = [dict(rows=3, z=50), dict(rows=3, z=10)]
        text = mroi_software(zs=(50, 10), **{"SI.hRoiManager.mroiEnable": "0"})
        stack = write_stack(tmp_path, 6, rois, zs=(50, 10), page_rows=3, frames_per_file=8)
        parts = split_stack(stack, tmp_path / "split", [rois[1], rois[0]], zs=(10, 50), keep_tags=True, software_text=text, artist_text=mroi_artist(rois))
        with pytest.raises(ScanImageError, match="depth"):
            read_acquisition(parts, allow_incomplete=True)

    # A ROI that's missing altogether
    def test_a_roi_named_in_the_settings_but_not_given_is_refused(self, tmp_path):
        text = mroi_software(**{"SI.hRoiManager.mroiEnable": "0"})
        rois = [dict(rows=3), dict(rows=3)]
        stack = write_stack(tmp_path, 6, rois, page_rows=8, frames_per_file=8)
        parts = split_stack(stack, tmp_path / "split", rois, keep_tags=True, software_text=text, artist_text=mroi_artist(rois))
        lone = tmp_path / "lone"
        lone.mkdir()
        target = lone / next(path for path in parts if "ROI02" in path.name).name
        target.write_bytes(next(path for path in parts if "ROI02" in path.name).read_bytes())
        with pytest.raises(ScanImageError, match="ROI 1"):
            read_acquisition([target], allow_incomplete=True)

    def test_without_settings_the_set_of_rois_is_said_to_be_unknown(self, split):
        assert "which ROIs" in read_acquisition(split, allow_incomplete=True).completeness

    # Completeness
    def test_more_frames_than_requested_is_refused(self, tmp_path):
        text = mroi_software(**{"SI.hStackManager.numVolumes": "2"})
        with pytest.raises(ScanImageError, match="requested"):
            read_acquisition(write_stack(tmp_path, 3, THREE, software_text=text))

    def test_a_short_last_file_says_only_that_no_gap_was_found(self, tmp_path):
        text = mroi_software(**{"SI.hScan2D.logFramesPerFile": "4"})
        completeness = read_acquisition(write_stack(tmp_path, 6, THREE, frames_per_file=4, software_text=text)).completeness
        assert completeness.startswith("no gap found")
        assert "not proof" in completeness

    def test_every_requested_volume_is_complete(self, tmp_path):
        text = mroi_software(**{"SI.hScan2D.logFramesPerFile": "4", "SI.hStackManager.numVolumes": "8"})
        completeness = read_acquisition(write_stack(tmp_path, 8, THREE, frames_per_file=4, software_text=text)).completeness
        assert completeness.startswith("complete")
