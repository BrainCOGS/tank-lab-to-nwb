"""TowersNWBConverter writes behavior only, with the subject from the registry, and sends ephys to its own path."""

import pytest

import tank_lab_to_nwb.virmen.interface as interface_module
from tank_lab_to_nwb.convert_towers_task.towersnwbconverter import TowersNWBConverter
from tests.synthetic_log import make_log


@pytest.fixture
def converter(monkeypatch, tmp_path):
    monkeypatch.setattr(interface_module, "load_log", lambda path: make_log())
    return TowersNWBConverter(dict(VirmenData=dict(file_path=str(tmp_path / "session.mat"))))


def test_subject_comes_from_the_registry_based_log(converter):
    metadata = converter.get_metadata()
    assert metadata["Subject"]["subject_id"] == "mouse_1"
    assert metadata["NWBFile"]["session_id"] == "20260715"


def test_writes_the_behavior_session(converter, tmp_path):
    path = tmp_path / "out.nwb"
    converter.run_conversion(nwbfile_path=path, metadata=converter.get_metadata(), overwrite=True)
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(path), "r") as io:
        nwbfile = io.read()
        assert nwbfile.subject.subject_id == "mouse_1" and len(nwbfile.trials) == 1


@pytest.mark.parametrize("key", ["KilosortProbe0", "SpikeGLXAP", "Kilosort"])
def test_ephys_is_sent_to_convert_session(monkeypatch, tmp_path, key):
    monkeypatch.setattr(interface_module, "load_log", lambda path: make_log())
    with pytest.raises(ValueError, match="tank_lab_to_nwb.ephys.convert_session"):
        TowersNWBConverter(dict(VirmenData=dict(file_path=str(tmp_path / "session.mat")), **{key: dict(folder_path=str(tmp_path))}))
