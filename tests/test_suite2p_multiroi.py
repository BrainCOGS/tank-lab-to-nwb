"""Suite2p output of multi-ROI sessions, built by hand: which plane folder is which field, and when its frames were scanned."""

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
from pynwb import validate

from tank_lab_to_nwb.imaging import convert_suite2p as cli
from tank_lab_to_nwb.imaging.alignment import AlignmentError
from tank_lab_to_nwb.imaging.convert import convert_session, prepare_session
from tank_lab_to_nwb.imaging.scanimage import read_acquisition, read_field
from tank_lab_to_nwb.imaging.suite2p import convert_suite2p, load_suite2p, match_planes, read_companion, verify_suite2p
from tank_lab_to_nwb.imaging.suite2p_mapping import prove_frame_mapping
from tank_lab_to_nwb.imaging.verify import RoundTripError
from tests.synthetic_tiff import mroi_software, split_stack, write_stack
from tests.test_multiroi import THREE, _packets, log  # noqa: F401

N_FRAMES = 801
COLS = 4


def pixels(frame_number, rows, cols):
    """Every pixel differs across frames and positions; each frame has negative and positive pixels, so the splitter's
    clipping at 0 changes some pixels but leaves the frame order visible."""
    return (np.arange(rows * cols, dtype="int64").reshape(rows, cols) * 97 + frame_number * 30 - 500).astype("int16")


def _write_stack(folder, log, rois=THREE, zs=(10,), n_frames=N_FRAMES, **options):
    folder.mkdir(parents=True)
    return write_stack(folder, n_frames, rois, zs=zs, frames_per_file=300, packets=_packets(log, n_frames), pixels=pixels, software_text=mroi_software(zs), **options)


def _write_plane(folder, movie, db):
    """One Suite2p plane folder: a one-pixel ROI at row 1, column 2, whose F is that pixel of the registered movie."""
    folder.mkdir(parents=True)
    count = len(movie)
    image = np.zeros(movie.shape[1:], dtype="float32")
    for key, value in dict(
        db=dict(nframes=count, Ly=movie.shape[1], Lx=movie.shape[2], **db),
        stat=[dict(xpix=np.array([2]), ypix=np.array([1]), lam=np.array([0.5], dtype="float32"), med=np.array([1.0, 2.0]))],
        settings=dict(version="1.1.0", registration=dict(do_bidiphase=False, nonrigid=False), extraction=dict(allow_overlap=False)),
        reg_outputs=dict(meanImg=image, yrange=[0, movie.shape[1]], xrange=[0, movie.shape[2]], xoff=np.zeros(count, dtype="int64"), yoff=np.zeros(count, dtype="int64")),
        detect_outputs=dict(max_proj=image),
        iscell=np.array([[1.0, 0.75]]),
        F=movie[:, 1, 2].astype("float32")[None, :],
        Fneu=np.ones((1, count), dtype="float32"),
        spks=np.zeros((1, count), dtype="float32"),
    ).items():
        np.save(folder / f"{key}.npy", value)
    movie.astype("int16").tofile(folder / "data.bin")


def _counts(acquisition, field, files):
    """Pages of this field in each of `files`, rows of acquisition.files."""
    return [int(np.sum(field.file_index == index)) for index in files]


def _registered(acquisition, field, split_copy=False):
    """What Suite2p writes to data.bin with no shifts: int16 pages unchanged, uint16 pages halved."""
    movie = read_field(acquisition, field, np.arange(field.frame_indices.size))
    if split_copy:
        movie = np.clip(movie, 0, None).astype("uint16")
    return (movie // 2).astype("int16") if movie.dtype == np.uint16 else movie


def stack_run(root, stack, log_=None, lines=True):
    """Suite2p's mesoscope route on the original stack: db lines crop each ROI's rows, one folder per plane and ROI."""
    acquisition = read_acquisition(stack)
    per_plane = {}
    for field in acquisition.fields:
        per_plane.setdefault(field.plane, []).append(field)
    nrois = len(next(iter(per_plane.values())))
    for field in acquisition.fields:
        iroi = per_plane[field.plane].index(field)
        db = dict(nplanes=acquisition.layout.n_planes, iplane=field.plane, file_list=[str(p) for p in stack], frames_per_file=_counts(acquisition, field, range(len(acquisition.files))))
        if lines:
            db.update(nrois=nrois, iroi=iroi, lines=list(range(*field.rows)), dy=0, dx=iroi * COLS)
        _write_plane(root / f"plane{field.plane * nrois + iroi if lines else field.plane}", _registered(acquisition, field), db)
    return root


def split_runs(base, split_files, stack=None):
    """The lab's route: one Suite2p run per split ROI folder, each with one plane. With `stack`, the movie is built from it."""
    acquisition = read_acquisition(stack) if stack else read_acquisition(split_files, allow_incomplete=True)
    roots = []
    for folder in sorted({path.parent for path in split_files}):
        files = sorted(path for path in split_files if path.parent == folder)
        number, depth = int(folder.name[3:5]), int(folder.name.split("_z")[1])
        if stack:
            field = next(f for f in acquisition.fields if f.number == number and f.plane == depth - 1)
        else:
            field = next(f for f in acquisition.fields if f.number == number and [acquisition.files[i].path for i in sorted(set(f.file_index.tolist()))] == files)
        movie = _registered(acquisition, field, split_copy=bool(stack))
        counts = _counts(acquisition, field, range(len(acquisition.files)) if stack else sorted(set(field.file_index.tolist())))
        root = base / folder.name / "suite2p"
        _write_plane(root / "plane0", movie, dict(nplanes=1, iplane=0, file_list=[str(p) for p in files], frames_per_file=counts))
        roots.append(root)
    return roots


@pytest.fixture
def stack(tmp_path, log):
    return _write_stack(tmp_path / "originalStacks", log)


@pytest.fixture
def split(tmp_path, stack):
    return split_stack(stack, tmp_path / "split", THREE)


def _field_series(path, name):
    with h5py.File(path) as file:
        return file[f"processing/ophys/{name}"][()]


class TestStackRoute:
    """Suite2p ran on the original stack with lines, and the converter is given the stack."""

    def test_each_folder_maps_to_the_field_its_rows_name(self, tmp_path, stack):
        root = stack_run(tmp_path / "s2p", stack)
        planes = match_planes(root, read_acquisition(stack))
        assert [(p.folder.name, p.field.label, p.split_copy) for p in planes] == [
            ("plane0", "Plane0Roi1", False),
            ("plane1", "Plane0Roi2", False),
            ("plane2", "Plane0Roi3", False),
        ]

    def test_writes_one_set_per_field_at_its_own_times(self, tmp_path, log, stack):
        root = stack_run(tmp_path / "s2p", stack)
        path = tmp_path / "out.nwb"
        prepared = convert_suite2p(log, stack, root, path, file="session.mat")
        assert not validate(path=str(path))
        with h5py.File(path) as file:
            ophys = file["processing/ophys"]
            assert sorted(ophys["ImageSegmentation"]) == ["PlaneSegmentation0Roi1", "PlaneSegmentation0Roi2", "PlaneSegmentation0Roi3"]
            assert sorted(file["general/optophysiology"]) == ["ImagingPlane0Roi1", "ImagingPlane0Roi2", "ImagingPlane0Roi3"]
            for field in prepared.acquisition.fields:
                series = ophys[f"Fluorescence/{field.label}"]
                np.testing.assert_array_equal(series["timestamps"][()], prepared.field_times(field))
                assert file[series["rois"].attrs["table"]].name.endswith(f"PlaneSegmentation{field.label.removeprefix('Plane')}")
                segmentation = ophys[f"ImageSegmentation/PlaneSegmentation{field.label.removeprefix('Plane')}"]
                assert segmentation.get("imaging_plane", getlink=True).path.endswith(f"ImagingPlane{field.label.removeprefix('Plane')}")
                np.testing.assert_array_equal(ophys[f"{field.label}_xoff/timestamps"][()], prepared.field_times(field))
            table = ophys["suite2p_fields"]
            assert list(table["field"].asstr()[()]) == ["Plane0Roi1", "Plane0Roi2", "Plane0Roi3"]
            assert list(table["folder"].asstr()[()]) == ["plane0", "plane1", "plane2"]
            assert list(table["roi_name"].asstr()[()]) == ["a", "b", "c"]
        # The ROIs are scanned at different times within a frame, so each field's times differ.
        first = [prepared.field_times(field)[0] for field in prepared.acquisition.fields]
        assert first[0] < first[1] < first[2]

    def test_each_files_first_middle_and_last_frame_is_checked_for_each_field(self, tmp_path, stack):
        root = stack_run(tmp_path / "s2p", stack)
        acquisition = read_acquisition(stack)
        evidence = prove_frame_mapping(acquisition, root)
        for field in acquisition.fields:
            samples = [e for e in evidence if e["field"] == field.label]
            assert sorted({(e["file_index"], e["page_index"]) for e in samples}) == [(0, 0), (0, 149), (0, 299), (1, 0), (1, 149), (1, 299), (2, 0), (2, 100), (2, 200)]
            assert all(e["rmse"]["0"] == 0 and min(v for k, v in e["rmse"].items() if k != "0") > 0 for e in samples)
            assert all(e["roi"] == field.number and e["plane"] == field.plane for e in samples)

    def test_a_field_whose_movie_is_one_frame_late_is_refused(self, tmp_path, stack):
        root = stack_run(tmp_path / "s2p", stack)
        movie = np.fromfile(root / "plane1/data.bin", dtype="int16").reshape(N_FRAMES, 2, COLS)
        np.roll(movie, 1, axis=0).tofile(root / "plane1/data.bin")
        with pytest.raises(ValueError, match="Plane0Roi2.*mapping"):
            prove_frame_mapping(read_acquisition(stack), root)

    def test_a_movie_from_another_rois_rows_is_refused(self, tmp_path, stack):
        """ROIs a and c have the same height: swapped movies have the right shape and length, but the wrong pixels."""
        root = stack_run(tmp_path / "s2p", stack)
        for name in ("data.bin", "F.npy"):
            first, last = (root / "plane0" / name).read_bytes(), (root / "plane2" / name).read_bytes()
            (root / "plane0" / name).write_bytes(last)
            (root / "plane2" / name).write_bytes(first)
        with pytest.raises(ValueError, match="Plane0Roi1.*mapping"):
            prove_frame_mapping(read_acquisition(stack), root)

    def test_whole_pages_holding_several_rois_are_refused(self, tmp_path, stack):
        root = tmp_path / "s2p"
        acquisition = read_acquisition(stack)
        movie = np.stack([np.zeros((acquisition.layout.frame_shape), dtype="int16")] * N_FRAMES)
        _write_plane(root / "plane0", movie, dict(nplanes=1, iplane=0, file_list=[str(p) for p in stack], frames_per_file=[300, 300, 201]))
        with pytest.raises(ValueError, match="3 ROIs.*lines"):
            match_planes(root, acquisition)

    @pytest.mark.parametrize("lines", [list(range(1, 4)), list(range(0, 2)), [0, 2, 3]], ids=["shifted", "short", "gap"])
    def test_lines_that_are_not_a_fields_rows_are_refused(self, tmp_path, stack, lines):
        root = stack_run(tmp_path / "s2p", stack)
        path = root / "plane0/db.npy"
        np.save(path, {**np.load(path, allow_pickle=True).item(), "lines": lines})
        with pytest.raises(ValueError, match="plane 0.*rows"):
            match_planes(root, read_acquisition(stack))

    def test_a_folder_number_that_disagrees_with_its_plane_and_roi_is_refused(self, tmp_path, stack):
        root = stack_run(tmp_path / "s2p", stack)
        path = root / "plane2/db.npy"
        np.save(path, {**np.load(path, allow_pickle=True).item(), "iroi": 0})
        with pytest.raises(ValueError, match="plane 2.*folder"):
            match_planes(root, read_acquisition(stack))

    def test_a_missing_plane_folder_is_refused(self, tmp_path, stack):
        root = stack_run(tmp_path / "s2p", stack)
        shutil.rmtree(root / "plane2")
        with pytest.raises(ValueError, match="plane folders"):
            match_planes(root, read_acquisition(stack))

    def test_an_extra_plane_folder_is_refused(self, tmp_path, stack):
        root = stack_run(tmp_path / "s2p", stack)
        shutil.copytree(root / "plane2", root / "plane3")
        with pytest.raises(ValueError, match="plane folders"):
            match_planes(root, read_acquisition(stack))

    def test_a_folder_named_like_a_plane_but_not_numbered_is_refused(self, tmp_path, stack):
        root = stack_run(tmp_path / "s2p", stack)
        (root / "plane_old").mkdir()
        with pytest.raises(ValueError, match="plane_old"):
            match_planes(root, read_acquisition(stack))

    @pytest.mark.parametrize(
        ("name", "change"),
        [("nframes", N_FRAMES - 1), ("Ly", 4), ("frames_per_file", [300, 300, 200])],
    )
    def test_a_frame_count_or_shape_mismatch_is_refused(self, tmp_path, log, stack, name, change):
        root = stack_run(tmp_path / "s2p", stack)
        path = root / "plane1/db.npy"
        np.save(path, {**np.load(path, allow_pickle=True).item(), name: change})
        with pytest.raises(ValueError, match="Plane0Roi2"):
            load_suite2p(root, prepare_session(log, stack, file="session.mat"))

    def test_traces_one_frame_short_are_refused(self, tmp_path, log, stack):
        root = stack_run(tmp_path / "s2p", stack)
        np.save(root / "plane2/F.npy", np.zeros((1, N_FRAMES - 1), dtype="float32"))
        with pytest.raises(ValueError, match="Plane0Roi3.*frame"):
            load_suite2p(root, prepare_session(log, stack, file="session.mat"))

    def test_verify_catches_a_field_stored_at_its_planes_times(self, tmp_path, log, stack):
        """A ROI's frames start when its first row is scanned, not when the page starts."""
        root = stack_run(tmp_path / "s2p", stack)
        path = tmp_path / "out.nwb"
        prepared = convert_suite2p(log, stack, root, path, file="session.mat")
        field = prepared.acquisition.fields[2]
        with h5py.File(path, "r+") as file:
            file[f"processing/ophys/Fluorescence/{field.label}/timestamps"][:] = prepared.frame_times[field.frame_indices]
        with pytest.raises(RoundTripError, match="times"):
            verify_suite2p(prepared, path, root, stack)


class TestSeveralPlanes:
    def test_one_roi_on_each_plane(self, tmp_path, log):
        rois = [dict(rows=3, name="a", z=10), dict(rows=3, name="b", z=50)]
        stack = _write_stack(tmp_path / "stack", log, rois, zs=(10, 50))
        acquisition = read_acquisition(stack)
        assert [f.label for f in acquisition.fields] == ["Plane0Roi1", "Plane1Roi2"]
        for lines in (True, False):
            root = stack_run(tmp_path / f"s2p{lines}", stack, lines=lines)
            assert [(p.folder.name, p.field.label) for p in match_planes(root, acquisition)] == [("plane0", "Plane0Roi1"), ("plane1", "Plane1Roi2")]
        path = tmp_path / "out.nwb"
        prepared = convert_suite2p(log, stack, root, path, file="session.mat")
        assert not validate(path=str(path))
        np.testing.assert_array_equal(_field_series(path, "Fluorescence/Plane1Roi2/timestamps"), prepared.field_times(acquisition.fields[1]))

    def test_two_rois_on_each_of_two_planes(self, tmp_path, log):
        rois = [dict(rows=3, name="a", z=10), dict(rows=3, name="b", z=10), dict(rows=3, name="c", z=50), dict(rows=3, name="d", z=50)]
        stack = _write_stack(tmp_path / "stack", log, rois, zs=(10, 50))
        root = stack_run(tmp_path / "s2p", stack)
        planes = match_planes(root, read_acquisition(stack))
        assert [(p.folder.name, p.field.label) for p in planes] == [("plane0", "Plane0Roi1"), ("plane1", "Plane0Roi2"), ("plane2", "Plane1Roi3"), ("plane3", "Plane1Roi4")]
        prove_frame_mapping(read_acquisition(stack), root)

    def test_split_runs_of_rois_on_two_planes_map_by_depth(self, tmp_path, log):
        rois = [dict(rows=3, name="a", z=10), dict(rows=3, name="b", z=50)]
        stack = _write_stack(tmp_path / "stack", log, rois, zs=(10, 50))
        split = split_stack(stack, tmp_path / "split", rois, zs=(10, 50))
        roots = split_runs(tmp_path / "s2p", split, stack=stack)
        planes = match_planes(roots, read_acquisition(stack))
        assert [(p.folder.parent.parent.name, p.field.label, p.split_copy) for p in planes] == [("ROI01_z1", "Plane0Roi1", True), ("ROI02_z2", "Plane1Roi2", True)]
        prove_frame_mapping(read_acquisition(stack), roots)

    def test_split_runs_are_refused_when_depth_folders_cannot_be_matched_to_planes(self, tmp_path, log):
        """The splitter numbers depths in ascending z; a stack whose planes run the other way can't be matched by name."""
        rois = [dict(rows=3, name="a", z=50), dict(rows=3, name="b", z=10)]
        stack = _write_stack(tmp_path / "stack", log, rois, zs=(50, 10))
        split = split_stack(stack, tmp_path / "split", rois, zs=(50, 10))
        roots = split_runs(tmp_path / "s2p", split)
        with pytest.raises(ValueError, match="ascending"):
            match_planes(roots, read_acquisition(stack))


class TestSplitRoute:
    """The lab's route: Suite2p ran once per split ROI folder."""

    def test_given_the_split_files_each_run_is_its_folders_field(self, tmp_path, log, split):
        roots = split_runs(tmp_path / "s2p", split)
        acquisition = read_acquisition(split, allow_incomplete=True)
        planes = match_planes(roots, acquisition)
        assert [(p.field.label, p.split_copy) for p in planes] == [("Plane0Roi1", False), ("Plane0Roi2", False), ("Plane0Roi3", False)]
        path = tmp_path / "out.nwb"
        prepared = convert_suite2p(log, split, roots, path, file="session.mat", allow_incomplete=True)
        assert not validate(path=str(path))
        for field in prepared.acquisition.fields:
            np.testing.assert_array_equal(_field_series(path, f"Fluorescence/{field.label}/timestamps"), prepared.field_times(field))
        with h5py.File(path) as file:
            table = file["processing/ophys/suite2p_fields"]
            assert list(table["folder"].asstr()[()]) == [f"ROI0{r}_z1/suite2p/plane0" for r in (1, 2, 3)]
            inventory = list(file["processing/ophys/source_inventory/path"].asstr()[()])
            assert "ROI02_z1/suite2p/plane0/F.npy" in inventory

    def test_given_the_stack_the_split_runs_use_the_stacks_times(self, tmp_path, log, stack, split):
        roots = split_runs(tmp_path / "s2p", split, stack=stack)
        path = tmp_path / "out.nwb"
        prepared = convert_suite2p(log, stack, roots, path, file="session.mat")
        assert not validate(path=str(path))
        assert [p.split_copy for p in match_planes(roots, prepared.acquisition)] == [True] * 3
        for field in prepared.acquisition.fields:
            np.testing.assert_array_equal(_field_series(path, f"Fluorescence/{field.label}/timestamps"), prepared.field_times(field))
        with h5py.File(path) as file:
            evidence = json.loads(file["scratch/suite2p_frame_mapping"].asstr()[()])
            assert {e["input"] for e in evidence} == {"split copy of the given TIFFs: negatives set to 0, as uint16"}
            assert set(file["processing/ophys/suite2p_fields/input"].asstr()[()]) == {"split copy of the given TIFFs: negatives set to 0, as uint16"}

    def test_a_split_run_whose_pixels_were_not_clipped_is_refused(self, tmp_path, stack, split):
        """The splitter's copy is checked, not assumed: a movie from unclipped rows fails the mapping at frames with negatives."""
        roots = split_runs(tmp_path / "s2p", split, stack=stack)
        acquisition = read_acquisition(stack)
        field = acquisition.fields[0]
        movie = read_field(acquisition, field, np.arange(N_FRAMES)).astype("int32")
        (movie // 2).astype("int16").tofile(roots[0] / "plane0/data.bin")
        with pytest.raises(ValueError, match="Plane0Roi1.*mapping"):
            prove_frame_mapping(acquisition, roots)

    def test_a_missing_roi_run_is_refused(self, tmp_path, log, split):
        roots = split_runs(tmp_path / "s2p", split)
        with pytest.raises(ValueError, match="Plane0Roi3"):
            match_planes(roots[:2], read_acquisition(split, allow_incomplete=True))

    def test_the_same_run_twice_is_refused(self, tmp_path, split):
        roots = split_runs(tmp_path / "s2p", split)
        with pytest.raises(ValueError, match="twice|more than once"):
            match_planes([roots[0], roots[0], roots[1], roots[2]], read_acquisition(split, allow_incomplete=True))

    def test_two_runs_of_one_roi_are_refused(self, tmp_path, split):
        roots = split_runs(tmp_path / "s2p", split)
        copy = tmp_path / "again" / "suite2p"
        shutil.copytree(roots[0], copy)
        with pytest.raises(ValueError, match="Plane0Roi1"):
            match_planes([*roots, copy], read_acquisition(split, allow_incomplete=True))

    def test_a_run_over_files_of_two_rois_is_refused(self, tmp_path, split):
        roots = split_runs(tmp_path / "s2p", split)
        path = roots[0] / "plane0/db.npy"
        db = np.load(path, allow_pickle=True).item()
        np.save(path, {**db, "file_list": [*db["file_list"][:2], str(sorted(p for p in split if "ROI02" in p.name)[2])]})
        with pytest.raises(ValueError, match="plane 0"):
            match_planes(roots, read_acquisition(split, allow_incomplete=True))

    def test_a_split_run_of_another_acquisition_is_refused(self, tmp_path, stack, split):
        roots = split_runs(tmp_path / "s2p", split, stack=stack)
        path = roots[1] / "plane0/db.npy"
        db = np.load(path, allow_pickle=True).item()
        np.save(path, {**db, "file_list": [name.replace("mouse_00001", "other_00001") for name in db["file_list"]]})
        with pytest.raises(ValueError, match="plane 0"):
            match_planes(roots, read_acquisition(stack))

    def test_a_split_run_missing_a_file_is_refused(self, tmp_path, stack, split):
        roots = split_runs(tmp_path / "s2p", split, stack=stack)
        path = roots[2] / "plane0/db.npy"
        db = np.load(path, allow_pickle=True).item()
        np.save(path, {**db, "file_list": db["file_list"][:2]})
        with pytest.raises(ValueError, match="plane 0"):
            match_planes(roots, read_acquisition(stack))

    def test_a_split_run_with_several_planes_is_refused(self, tmp_path, split):
        roots = split_runs(tmp_path / "s2p", split)
        path = roots[0] / "plane0/db.npy"
        np.save(path, {**np.load(path, allow_pickle=True).item(), "nplanes": 2})
        with pytest.raises(ValueError, match="plane"):
            match_planes(roots, read_acquisition(split, allow_incomplete=True))

    def test_split_and_original_names_in_one_run_are_refused(self, tmp_path, stack, split):
        roots = split_runs(tmp_path / "s2p", split, stack=stack)
        path = roots[0] / "plane0/db.npy"
        db = np.load(path, allow_pickle=True).item()
        np.save(path, {**db, "file_list": [db["file_list"][0], str(stack[1]), db["file_list"][2]]})
        with pytest.raises(ValueError, match="plane 0"):
            match_planes(roots, read_acquisition(stack))

    def test_a_stack_run_is_refused_against_split_tiffs(self, tmp_path, stack, split):
        """The split files have lost the stack's flyback rows and its page layout, so a stack run can't be placed in them."""
        root = stack_run(tmp_path / "s2p", stack)
        with pytest.raises(ValueError, match="plane 0"):
            match_planes(root, read_acquisition(split, allow_incomplete=True))


class TestCompanion:
    def test_the_raw_stack_file_is_accepted_and_recorded(self, tmp_path, log, stack, split):
        raw = tmp_path / "raw.nwb"
        convert_session(log, stack, raw, file="session.mat")
        roots = split_runs(tmp_path / "s2p", split, stack=stack)
        path = tmp_path / "out.nwb"
        convert_suite2p(log, stack, roots, path, file="session.mat", raw_nwb=raw)
        with h5py.File(path) as file:
            table = file["processing/ophys/scanimage_alignment"]
            assert "companion.rawtiff.sha256" in list(table["key"].asstr()[()])

    def test_a_raw_file_of_the_split_copies_is_refused_for_a_stack_conversion(self, tmp_path, log, stack, split):
        raw = tmp_path / "raw.nwb"
        convert_session(log, split, raw, file="session.mat", allow_incomplete=True)
        with pytest.raises(AlignmentError, match="another acquisition"):
            read_companion(raw, prepare_session(log, stack, file="session.mat"))

    def test_a_raw_file_whose_roi_times_differ_is_refused(self, tmp_path, log, stack):
        raw = tmp_path / "raw.nwb"
        prepared = convert_session(log, stack, raw, file="session.mat")
        with h5py.File(raw, "r+") as file:
            file["acquisition/TwoPhotonSeriesPlane0Roi3/timestamps"][0] += 1e-6
        with pytest.raises(AlignmentError, match="Plane0Roi3"):
            read_companion(raw, prepared)


def test_the_cli_takes_one_run_per_split_folder_and_reports_each_field(tmp_path, log, stack, split, monkeypatch):
    def with_log(_path, tiff_paths, root, output, **options):
        return convert_suite2p(log, tiff_paths, root, output, file="session.mat", **options)

    monkeypatch.setattr(cli, "convert_suite2p", with_log)
    roots = split_runs(tmp_path / "s2p", split, stack=stack)
    path = cli.convert(tmp_path / "session.mat", roots[0], tmp_path / "out", [stack[0].parent], extra=roots[1:])
    report = json.loads(path.with_suffix(".report.json").read_text())
    assert report["roi_counts"] == {"Plane0Roi1": 1, "Plane0Roi2": 1, "Plane0Roi3": 1}
    assert report["frame_counts"] == {"Plane0Roi1": N_FRAMES, "Plane0Roi2": N_FRAMES, "Plane0Roi3": N_FRAMES}
    assert {e["field"] for e in report["mapping"]} == {"Plane0Roi1", "Plane0Roi2", "Plane0Roi3"}


def to_pre_1_0(root):
    """Rewrite Suite2p 1.x plane folders as pre-1.0 Suite2p left them: one flat ops.npy, with filelist, the registration
    outputs and settings together, and no db.npy, settings.npy, reg_outputs.npy or detect_outputs.npy."""
    for folder in sorted(root.glob("plane*")):
        parts = {name: np.load(folder / f"{name}.npy", allow_pickle=True).item() for name in ("db", "settings", "reg_outputs", "detect_outputs")}
        ops = {"suite2p_version": "0.10.3", "nonrigid": False, "do_bidiphase": False, "bidiphase": 0, "two_step_registration": False, "allow_overlap": False, "block_size": [128, 128]}
        db = dict(parts["db"])
        ops.update(filelist=db.pop("file_list"), **db, **parts["reg_outputs"], **parts["detect_outputs"])
        ops["regPC"] = np.zeros((2, 2, 30, 40), dtype="float32")  # too big for the settings table; kept in the archived ops.npy
        np.save(folder / "ops.npy", ops)
        for name in parts:
            (folder / f"{name}.npy").unlink()
    return root


class TestPre10Layout:
    """Suite2p before 1.0 (the lab's mesoscope output may be from 2021) kept everything in ops.npy."""

    def test_split_runs_convert_and_keep_ops_settings(self, tmp_path, log, stack, split):
        roots = [to_pre_1_0(root) for root in split_runs(tmp_path / "s2p", split, stack=stack)]
        path = tmp_path / "out.nwb"
        prepared = convert_suite2p(log, stack, roots, path, file="session.mat")
        assert not validate(path=str(path))
        with h5py.File(path) as file:
            ophys = file["processing/ophys"]
            for field in prepared.acquisition.fields:
                np.testing.assert_array_equal(ophys[f"Fluorescence/{field.label}/timestamps"][()], prepared.field_times(field))
            settings = dict(zip(ophys["suite2p_settings/key"].asstr()[()], ophys["suite2p_settings/value"].asstr()[()]))
            assert settings["ROI01_z1/suite2p/plane0.suite2p_version"] == '"0.10.3"'
            assert "ROI01_z1/suite2p/plane0.regPC" not in settings and "ROI01_z1/suite2p/plane0.xoff" not in settings
            assert "ROI01_z1/suite2p/plane0/ops.npy" in list(ophys["source_inventory/path"].asstr()[()])
            assert "Plane0Roi1_xoff" in ophys and "SummaryImages0Roi1" in ophys

    def test_a_stack_run_with_lines_maps(self, tmp_path, stack):
        root = to_pre_1_0(stack_run(tmp_path / "s2p", stack))
        planes = match_planes(root, read_acquisition(stack))
        assert [(p.folder.name, p.field.label, p.meta["layout"]) for p in planes] == [
            ("plane0", "Plane0Roi1", "pre-1.0"),
            ("plane1", "Plane0Roi2", "pre-1.0"),
            ("plane2", "Plane0Roi3", "pre-1.0"),
        ]
        prove_frame_mapping(read_acquisition(stack), root)

    @pytest.mark.parametrize(
        "prefix",
        ["//server/share/session/{folder}\\", "Z:\\session\\{folder}\\", "C:/session/{folder}/"],
        ids=["unc-then-backslash", "drive-backslashes", "drive-slashes"],
    )
    def test_split_runs_recorded_on_windows_map(self, tmp_path, stack, split, prefix):
        """A run on a Windows rig records its file list with backslashes, sometimes after a forward-slash share path."""
        roots = [to_pre_1_0(root) for root in split_runs(tmp_path / "s2p", split, stack=stack)]
        for root in roots:
            path = root / "plane0/ops.npy"
            ops = np.load(path, allow_pickle=True).item()
            ops["filelist"] = [prefix.format(folder=root.parent.name) + Path(name).name for name in ops["filelist"]]
            np.save(path, ops)
        acquisition = read_acquisition(stack)
        planes = match_planes(roots, acquisition)
        assert [p.field.label for p in planes] == ["Plane0Roi1", "Plane0Roi2", "Plane0Roi3"]
        assert all(p.split_copy for p in planes)
        prove_frame_mapping(acquisition, roots)

    @pytest.mark.parametrize("key", ["filelist", "frames_per_file", "iplane", "xoff"])
    def test_ops_without_what_places_and_checks_the_plane_is_refused(self, tmp_path, stack, key):
        root = to_pre_1_0(stack_run(tmp_path / "s2p", stack))
        path = root / "plane1/ops.npy"
        ops = np.load(path, allow_pickle=True).item()
        del ops[key]
        np.save(path, ops)
        with pytest.raises(ValueError, match=f"pre-1.0.*{key}"):
            match_planes(root, read_acquisition(stack))


def test_a_folder_with_neither_db_nor_ops_is_refused(tmp_path, stack):
    root = stack_run(tmp_path / "s2p", stack)
    (root / "plane1/db.npy").unlink()
    with pytest.raises(ValueError, match="neither db.npy.*nor ops.npy"):
        match_planes(root, read_acquisition(stack))


def test_a_1x_folder_without_its_settings_is_refused(tmp_path, stack):
    root = stack_run(tmp_path / "s2p", stack)
    (root / "plane2/settings.npy").unlink()
    with pytest.raises(ValueError, match="settings"):
        match_planes(root, read_acquisition(stack))


@pytest.mark.parametrize("pre_1_0", [False, True], ids=["1.x", "pre-1.0"])
def test_a_plane_without_its_registered_movie_is_refused(tmp_path, stack, pre_1_0):
    root = stack_run(tmp_path / "s2p", stack)
    if pre_1_0:
        to_pre_1_0(root)
    (root / "plane0/data.bin").unlink()
    with pytest.raises(ValueError, match="Plane0Roi1.*data.bin.*can't be proven"):
        prove_frame_mapping(read_acquisition(stack), root)
