"""VirmenDataInterface, and the whole local pipeline: staging, dandi organize, the session fix, dandi validate."""

import os
import shutil
import subprocess

import numpy as np
import pytest
from pynwb import NWBHDF5IO

import tank_lab_to_nwb.virmen.convert as convert
import tank_lab_to_nwb.virmen.interface as interface_module
from tank_lab_to_nwb.virmen.convert_folder import convert_folder
from tank_lab_to_nwb.virmen.interface import VirmenDataInterface
from tank_lab_to_nwb.virmen.verify import verify_round_trip
from tests.synthetic_log import make_block, make_log, make_trial


@pytest.fixture
def interface(monkeypatch, tmp_path):
    monkeypatch.setattr(interface_module, "load_log", lambda path: make_log())
    return VirmenDataInterface(file_path=tmp_path / "session.mat")


def test_package_exports_it_lazily():
    import tank_lab_to_nwb

    assert tank_lab_to_nwb.VirmenDataInterface is VirmenDataInterface


def test_metadata_comes_from_the_log(interface):
    metadata = interface.get_metadata()
    assert metadata["Subject"]["subject_id"] == "mouse_1"
    assert metadata["NWBFile"]["session_id"] == "20260715"
    assert metadata["NWBFile"]["session_start_time"].utcoffset() is not None
    assert "PLACEHOLDER" in metadata["Subject"]["description"]


def test_original_timestamps_are_every_logged_frame(interface):
    np.testing.assert_array_equal(interface.get_original_timestamps(), [0.5 + i * 0.5 for i in range(5)])


def test_timestamps_default_to_the_original(interface):
    np.testing.assert_array_equal(interface.get_timestamps(), interface.get_original_timestamps())


def test_run_conversion_writes_exactly_the_normalized_values(interface, tmp_path):
    path = tmp_path / "out.nwb"
    interface.run_conversion(nwbfile_path=path, metadata=interface.get_metadata())
    verify_round_trip(interface.session, path)
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        assert type(io.read().trials).__name__ == "VirmenTrials"


def test_aligned_timestamps_move_series_trials_and_epochs(interface, tmp_path):
    aligned = interface.get_original_timestamps() + 100.0
    interface.set_aligned_timestamps(aligned)
    np.testing.assert_array_equal(interface.get_timestamps(), aligned)
    path = tmp_path / "out.nwb"
    interface.run_conversion(nwbfile_path=path, metadata=interface.get_metadata())
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        assert nwbfile.trials["start_time"].data[0] == 100.0 + interface.get_original_timestamps()[0]
        behavior = nwbfile.processing["behavior"]
        assert behavior["SensorDots"].timestamps[0] == aligned[0]
        assert behavior["Position"]["position"].timestamps[0] == aligned[0]


@pytest.mark.parametrize("length_change", [-1, 1])
def test_aligned_timestamps_of_the_wrong_length_fail(interface, length_change):
    size = interface.get_original_timestamps().size + length_change
    with pytest.raises(ValueError, match="aligned"):
        interface.set_aligned_timestamps(np.zeros(size))


class TestPipeline:
    @pytest.fixture
    def source(self, tmp_path, monkeypatch):
        folder = tmp_path / "behavior"
        (folder / "user" / "user_a" / "20260715_g0").mkdir(parents=True)
        (folder / "user" / "user_b" / "20260716_g0").mkdir(parents=True)
        (folder / "user" / "user_a" / "20260715_g0" / "a.mat").touch()
        (folder / "user" / "user_b" / "20260716_g0" / "b.mat").touch()

        def fake_log(path):
            log = make_log([make_block(make_trial())])
            log["animal"]["name"] = "user_a" if path.name == "a.mat" else "user_b"
            return log

        monkeypatch.setattr(convert, "load_log", fake_log)
        return folder

    def test_flat_staging_then_organized_names(self, source, tmp_path):
        output = tmp_path / "nwb"
        organized = convert_folder(source, output)
        assert sorted(p.relative_to(output / "dandiset").as_posix() for p in organized) == [
            "sub-user-a/sub-user-a_ses-20260715_behavior.nwb",
            "sub-user-b/sub-user-b_ses-20260715_behavior.nwb",
        ]
        assert sorted(p.name for p in (output / "staging").glob("*.nwb")) == ["a.nwb", "b.nwb"]
        assert (output / "staging" / "a.coverage.json").exists()
        assert (output / "dandiset" / "dandiset.yaml").exists()

    def test_converting_again_replaces_the_organized_files(self, source, tmp_path):
        output = tmp_path / "nwb"
        first = convert_folder(source, output)
        second = convert_folder(source, output)
        assert first == second
        assert len(list((output / "dandiset").rglob("*.nwb"))) == 2

    def test_only_selected_logs_are_converted(self, source, tmp_path):
        organized = convert_folder(source, tmp_path / "nwb", only=["a.mat"])
        assert [p.name for p in organized] == ["sub-user-a_ses-20260715_behavior.nwb"]

    def test_no_logs_fail(self, tmp_path):
        with pytest.raises(ValueError, match="no logs"):
            convert_folder(tmp_path, tmp_path / "nwb")

    def test_organized_folder_passes_dandi_validate(self, source, tmp_path):
        output = tmp_path / "nwb"
        convert_folder(source, output)
        dandi = shutil.which("dandi", path=os.path.dirname(os.sys.executable))
        result = subprocess.run([dandi, "validate", str(output / "dandiset")], capture_output=True, text=True,
                                env={**os.environ, "XDG_STATE_HOME": str(tmp_path / "state")})
        assert result.returncode == 0, result.stdout + result.stderr
