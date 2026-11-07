"""DANDI layout: dandi organize names the files from their metadata, then the session is put back if it dropped it."""

from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest
import yaml
from pynwb import NWBHDF5IO, NWBFile
from pynwb.behavior import Position, SpatialSeries
from pynwb.file import Subject

from tank_lab_to_nwb.virmen.layout import insert_description, insert_session_id, organize_for_dandi, write_stub_dandiset_yaml


def _write_nwb(path, subject_id="mouse-1", session_id="20260101"):
    nwbfile = NWBFile(
        session_description="d",
        identifier=str(uuid4()),
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        session_id=session_id,
    )
    nwbfile.subject = Subject(subject_id=subject_id, species="Mus musculus", sex="U", age="P90D")
    behavior = nwbfile.create_processing_module("behavior", "behavior")
    series = SpatialSeries(name="xy", data=np.zeros((2, 2)), reference_frame="r", timestamps=[0.0, 1.0], unit="cm")
    behavior.add(Position(spatial_series=series))
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)
    return path


class TestInsertSessionId:
    def test_inserts_after_the_subject(self):
        assert insert_session_id("sub-a_behavior.nwb", "20260101") == "sub-a_ses-20260101_behavior.nwb"

    def test_name_with_a_session_is_unchanged(self):
        name = "sub-a_ses-20260101_behavior.nwb"
        assert insert_session_id(name, "20260101") == name

    def test_subject_containing_ses_still_gets_a_session(self):
        # neuroconv's `"ses" not in stem` would skip this one.
        assert insert_session_id("sub-moses_behavior.nwb", "1") == "sub-moses_ses-1_behavior.nwb"

    def test_name_without_a_modality(self):
        assert insert_session_id("sub-a.nwb", "1") == "sub-a_ses-1.nwb"

    @pytest.mark.parametrize("session_id", [None, ""])
    def test_missing_session_id_fails(self, session_id):
        with pytest.raises(ValueError, match="session_id"):
            insert_session_id("sub-a_behavior.nwb", session_id)


class TestOrganize:
    @pytest.fixture
    def staging(self, tmp_path):
        folder = tmp_path / "staging"
        folder.mkdir()
        return folder

    def test_stub_yaml_is_marked_as_a_stub_and_loads(self, tmp_path):
        write_stub_dandiset_yaml(tmp_path)
        content = yaml.safe_load((tmp_path / "dandiset.yaml").read_text())
        assert "stub" in content["description"].lower()

    def test_stub_yaml_does_not_overwrite_an_existing_one(self, tmp_path):
        (tmp_path / "dandiset.yaml").write_text("identifier: DANDI:123456\n")
        write_stub_dandiset_yaml(tmp_path)
        assert "123456" in (tmp_path / "dandiset.yaml").read_text()

    def test_single_session_subject_keeps_its_session(self, staging, tmp_path):
        _write_nwb(staging / "whatever.nwb", subject_id="mouse-1", session_id="20260101")
        paths = organize_for_dandi(staging, tmp_path / "dandiset")
        assert [p.relative_to(tmp_path / "dandiset").as_posix() for p in paths] == [
            "sub-mouse-1/sub-mouse-1_ses-20260101_behavior.nwb"
        ]

    def test_two_subjects(self, staging, tmp_path):
        _write_nwb(staging / "a.nwb", subject_id="a", session_id="20260101")
        _write_nwb(staging / "b.nwb", subject_id="b", session_id="20260102")
        paths = organize_for_dandi(staging, tmp_path / "dandiset")
        assert sorted(p.name for p in paths) == ["sub-a_ses-20260101_behavior.nwb", "sub-b_ses-20260102_behavior.nwb"]

    def test_two_sessions_of_one_subject(self, staging, tmp_path):
        _write_nwb(staging / "a.nwb", subject_id="a", session_id="20260101")
        _write_nwb(staging / "b.nwb", subject_id="a", session_id="20260102")
        paths = organize_for_dandi(staging, tmp_path / "dandiset")
        assert sorted(p.name for p in paths) == ["sub-a_ses-20260101_behavior.nwb", "sub-a_ses-20260102_behavior.nwb"]

    def test_underscores_in_a_subject_are_sanitized_by_dandi(self, staging, tmp_path):
        _write_nwb(staging / "a.nwb", subject_id="lab_mouse_1", session_id="20260101")
        paths = organize_for_dandi(staging, tmp_path / "dandiset")
        assert paths[0].parent.name == "sub-lab-mouse-1"

    def test_copy_mode_leaves_staging_in_place(self, staging, tmp_path):
        source = _write_nwb(staging / "a.nwb")
        organize_for_dandi(staging, tmp_path / "dandiset", files_mode="copy")
        assert source.exists()

    def test_move_mode_empties_staging(self, staging, tmp_path):
        source = _write_nwb(staging / "a.nwb")
        organize_for_dandi(staging, tmp_path / "dandiset", files_mode="move")
        assert not source.exists()

    def test_a_file_of_another_session_is_kept(self, staging, tmp_path):
        _write_nwb(staging / "a.nwb", subject_id="a", session_id="20260101")
        organize_for_dandi(staging, tmp_path / "dandiset")
        (staging / "a.nwb").unlink()
        _write_nwb(staging / "b.nwb", subject_id="a", session_id="20260102")
        organize_for_dandi(staging, tmp_path / "dandiset")
        assert sorted(p.name for p in (tmp_path / "dandiset").rglob("*.nwb")) == [
            "sub-a_ses-20260101_behavior.nwb", "sub-a_ses-20260102_behavior.nwb"
        ]

    def test_empty_staging_fails(self, staging, tmp_path):
        with pytest.raises(ValueError, match="no NWB files"):
            organize_for_dandi(staging, tmp_path / "dandiset")

    def test_file_without_a_subject_fails(self, staging, tmp_path):
        nwbfile = NWBFile("d", str(uuid4()), datetime(2026, 1, 1, tzinfo=timezone.utc))
        with NWBHDF5IO(str(staging / "a.nwb"), "w") as io:
            io.write(nwbfile)
        with pytest.raises(Exception):
            organize_for_dandi(staging, tmp_path / "dandiset")


class TestInsertDescription:
    def test_goes_after_the_session(self):
        assert insert_description("sub-a_ses-1_behavior+ophys.nwb", "suite2p") == "sub-a_ses-1_desc-suite2p_behavior+ophys.nwb"

    def test_a_subject_containing_ses_does_not_confuse_it(self):
        assert insert_description("sub-moses_ses-1_ophys.nwb", "rawtiff") == "sub-moses_ses-1_desc-rawtiff_ophys.nwb"

    def test_needs_a_session(self):
        with pytest.raises(ValueError, match="session"):
            insert_description("sub-a_behavior.nwb", "suite2p")

    def test_a_second_description_is_refused(self):
        with pytest.raises(ValueError, match="already"):
            insert_description("sub-a_ses-1_desc-raw_ophys.nwb", "suite2p")

    @pytest.mark.parametrize("description", ["", "raw_tiff", "raw tiff", "raw-tiff", "../x"])
    def test_only_letters_and_digits_fit_in_a_name(self, description):
        with pytest.raises(ValueError, match="description"):
            insert_description("sub-a_ses-1_ophys.nwb", description)


def test_two_views_of_a_session_share_one_dandiset(tmp_path):
    dandiset = tmp_path / "dandiset"
    names = []
    for description in ("rawtiff", "suite2p"):
        staging = tmp_path / f"staging-{description}"
        staging.mkdir()
        _write_nwb(staging / "session.nwb")
        names += [path.name for path in organize_for_dandi(staging, dandiset, description=description)]
    assert names == ["sub-mouse-1_ses-20260101_desc-rawtiff_behavior.nwb", "sub-mouse-1_ses-20260101_desc-suite2p_behavior.nwb"]
    assert sorted(path.name for path in dandiset.rglob("*.nwb")) == names


def test_organizing_a_description_again_replaces_its_file(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    _write_nwb(staging / "session.nwb")
    organize_for_dandi(staging, tmp_path / "dandiset", description="suite2p")
    organize_for_dandi(staging, tmp_path / "dandiset", description="suite2p")
    assert len(list((tmp_path / "dandiset").rglob("*.nwb"))) == 1
