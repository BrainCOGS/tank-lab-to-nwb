"""TowersNWBConverter: behavior alone, or behavior with sorted units through the ephys converter, on the recording's clock."""

import json
from datetime import datetime, timedelta

import h5py
import numpy as np
import pytest
from pynwb import NWBHDF5IO

import tank_lab_to_nwb.virmen.interface as interface_module
from tank_lab_to_nwb.convert_towers_task.towersnwbconverter import TowersNWBConverter
from tank_lab_to_nwb.ephys.convert_session import SyncTimestampsError, convert
from tank_lab_to_nwb.ephys.session import SessionSyncError, build_session_sync
from tests.synthetic_ephys import NIDQ_RATE
from tests.synthetic_log import make_log
from tests.test_ephys_convert import named_log, session  # noqa: F401  (fixtures)


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
    with NWBHDF5IO(str(path), "r") as io:
        nwbfile = io.read()
        assert nwbfile.subject.subject_id == "mouse_1" and len(nwbfile.trials) == 1


def test_behavior_alone_moves_onto_sync_timestamps(monkeypatch, tmp_path):
    monkeypatch.setattr(interface_module, "load_log", lambda path: make_log())
    source_data = dict(VirmenData=dict(file_path=str(tmp_path / "session.mat")))
    frames = TowersNWBConverter(source_data).data_interface_objects["VirmenData"].get_original_timestamps()
    converter = TowersNWBConverter(source_data, sync_timestamps=frames + 100.0)
    path = tmp_path / "out.nwb"
    metadata = converter.get_metadata()
    metadata["Subject"]["date_of_birth"] = datetime(2026, 1, 2)  # naive, as the U19 handler passes it
    converter.run_conversion(nwbfile_path=path, metadata=metadata, overwrite=True)
    with NWBHDF5IO(str(path), "r") as io:
        nwbfile = io.read()
        assert nwbfile.trials["start_time"][0] == pytest.approx(frames[0] + 100.0)
        assert nwbfile.subject.date_of_birth.date().isoformat() == "2026-01-02"
        assert nwbfile.subject.date_of_birth.utcoffset() == timedelta(hours=-5)  # the session's zone, Eastern, in January


def test_ttl_source_is_refused(converter, tmp_path):
    source_data = dict(VirmenData=dict(file_path=str(tmp_path / "session.mat")))
    with pytest.raises(ValueError, match="ttl_source"):
        TowersNWBConverter(source_data, ttl_source=str(tmp_path / "run.nidq.bin"))
    with pytest.raises(ValueError, match="ttl_source"):
        TowersNWBConverter(source_data, str(tmp_path / "run.nidq.bin"))


@pytest.mark.parametrize("key, says", [("SpikeGLXAP", "raw bands"), ("SpikeGLXLFP", "raw bands"), ("SpikeGLXRecording", "raw bands"), ("Suite2pSegmentation", "tank_lab_to_nwb.imaging"), ("TiffImaging", "tank_lab_to_nwb.imaging"), ("Whatever", "VirmenData")])
def test_refused_keys(converter, tmp_path, key, says):
    with pytest.raises(ValueError, match=says):
        TowersNWBConverter(dict(VirmenData=dict(file_path=str(tmp_path / "session.mat")), **{key: dict(folder_path=str(tmp_path))}))


def test_a_run_folder_without_units_is_refused(converter, tmp_path):
    with pytest.raises(ValueError, match="no Kilosort key"):
        TowersNWBConverter(dict(VirmenData=dict(file_path=str(tmp_path / "session.mat")), SpikeGLX=dict(folder_path=str(tmp_path))))


def test_clock_options_need_units(converter, tmp_path):
    with pytest.raises(ValueError, match="probe_clock"):
        TowersNWBConverter(dict(VirmenData=dict(file_path=str(tmp_path / "session.mat"))), probe_clock="meta_rates")


# --- with sorted units, on the synthetic ephys session --------------------------------------------------------------


@pytest.fixture
def towers(session, named_log, monkeypatch):  # noqa: F811
    """Build the converter the way the U19 handler does, with the synthetic log behind VirmenData's file_path."""
    monkeypatch.setattr(interface_module, "load_log", lambda path: session["log"])

    def make(key="KilosortProbe0", spikeglx=True, **options):
        source_data = {"VirmenData": {"file_path": str(named_log)}, key: {"folder_path": str(session["sorter"].parent)}}
        if spikeglx:
            source_data["SpikeGLX"] = {"folder_path": str(session["raw"])}
        return TowersNWBConverter(source_data=source_data, **options)

    return make


def _iteration_times(session, **options):
    sync = build_session_sync(session["raw"], session["log"], file="log.mat", **options)
    return np.concatenate(sync.iteration_seconds) - sync.clock.shift


@pytest.mark.parametrize("key", ["KilosortProbe0", "Kilosort"])
def test_u19_call_shape_writes_verified_units_with_the_metadata_edits(session, towers, key):  # noqa: F811
    converter = towers(key, sync_timestamps=None)
    metadata = converter.get_metadata()
    metadata["NWBFile"]["session_description"] = "U19 pipeline export"
    metadata["NWBFile"]["experimenter"] = ["Last, First"]
    metadata.setdefault("Subject", {})["sex"] = "M"
    metadata["Subject"]["date_of_birth"] = datetime(2026, 1, 2)
    path = session["tmp"] / "export" / "session.nwb"
    converter.run_conversion(nwbfile_path=str(path), metadata=metadata, overwrite=True)
    assert sorted(p.name for p in path.parent.iterdir()) == ["session.nwb", "session.report.json"]
    report = json.loads(path.with_suffix(".report.json").read_text())
    assert report["verified"]["units"] == session["cluster_ids"].size
    assert report["verified"]["spikes"] == sum(v.size for v in session["trains"].values())
    assert "sync_timestamps" not in report
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        assert list(nwbfile.units["probe"][:]) == ["imec0"] * session["cluster_ids"].size
        assert len(nwbfile.trials) == 5
        assert nwbfile.session_description == "U19 pipeline export"
        assert list(nwbfile.experimenter) == ["Last, First"]
        assert nwbfile.subject.sex == "M" and nwbfile.subject.species == "Mus musculus"
        assert nwbfile.subject.date_of_birth.date().isoformat() == "2026-01-02"
        alignment = nwbfile.processing["ecephys"]["ephys_sync_alignment"].to_dataframe().set_index("key")["value"]
        assert (alignment["probe_clock"], alignment["iteration_time"]) == ("sync_edges", "pipeline_index")
        assert nwbfile.session_start_time.isoformat().startswith("2024-07-27T13:59:44")


def _datasets(path):
    """Every unit's spike times, the trials' times, each behavior series' timestamps and the electrode ids of a file."""
    found = {}
    with h5py.File(path, "r") as f:
        index = f["units/spike_times_index"][:]
        found["units"] = np.split(f["units/spike_times"][:], index[:-1])
        found["trials"] = (f["intervals/trials/start_time"][:], f["intervals/trials/stop_time"][:])
        f["processing/behavior"].visititems(lambda name, item: found.setdefault(f"behavior/{name}", item[:]) if name.endswith("timestamps") else None)
        found["electrodes"] = f["general/extracellular_ephys/electrodes/id"][:]
    return found


def test_the_class_and_convert_write_the_same_times(session, towers, named_log):  # noqa: F811
    options = dict(probe_clock="meta_rates", iteration_time="edge")
    by_function = convert(named_log, session["raw"], session["tmp"] / "function", {"imec0": session["sorter"].parent}, **options)
    by_class = session["tmp"] / "class" / "out.nwb"
    converter = towers(**options)
    converter.run_conversion(nwbfile_path=by_class, metadata=converter.get_metadata())
    a, b = _datasets(by_function), _datasets(by_class)
    assert a.keys() == b.keys() and any(k.startswith("behavior/") for k in a)
    assert len(a["units"]) == session["cluster_ids"].size
    assert all(np.array_equal(x, y) for x, y in zip(a["units"], b["units"]))
    for key in set(a) - {"units"}:
        assert np.array_equal(a[key], b[key]), key
    with NWBHDF5IO(str(by_class), "r", load_namespaces=True) as io:
        alignment = io.read().processing["ecephys"]["ephys_sync_alignment"].to_dataframe().set_index("key")["value"]
        assert (alignment["probe_clock"], alignment["iteration_time"]) == ("meta_rates", "edge")


def test_units_without_the_raw_run_folder_are_refused(towers):
    with pytest.raises(ValueError, match=r"SpikeGLX.*folder_path.*nidq"):
        towers(spikeglx=False)


def test_sync_timestamps_that_agree_are_recorded(session, towers):  # noqa: F811
    given = _iteration_times(session) + 0.4 / NIDQ_RATE
    path = session["tmp"] / "out.nwb"
    towers(sync_timestamps=given).run_conversion(nwbfile_path=path)
    check = json.loads(path.with_suffix(".report.json").read_text())["sync_timestamps"]
    assert check["agrees"] is True and check["iterations"] == given.size
    assert check["max_difference_samples"] == pytest.approx(0.4, abs=1e-3)


def test_sync_timestamps_that_disagree_stop_the_conversion(session, towers):  # noqa: F811
    times = _iteration_times(session)
    path = session["tmp"] / "out.nwb"
    late = times.copy()
    late[10] += 3 / NIDQ_RATE
    with pytest.raises(SyncTimestampsError, match=r"3\.0 samples"):
        towers(sync_timestamps=late).run_conversion(nwbfile_path=path)
    with pytest.raises(SyncTimestampsError, match=f"{times.size - 1} given, {times.size} recomputed"):
        towers(sync_timestamps=times[:-1]).run_conversion(nwbfile_path=path)
    assert not path.exists()


def test_a_refusal_of_the_ephys_code_reaches_the_caller_unchanged(session, towers):  # noqa: F811
    with pytest.raises(SessionSyncError, match="digital-word"):
        towers(session_date="2021-05-06").run_conversion(nwbfile_path=session["tmp"] / "out.nwb")


def test_an_existing_file_is_kept_without_overwrite(session, towers):  # noqa: F811
    path = session["tmp"] / "out.nwb"
    path.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        towers().run_conversion(nwbfile_path=path)
    assert path.read_bytes() == b"keep"


def test_neuroconv_options_left_empty_are_accepted_with_units(session, towers):  # noqa: F811
    path = session["tmp"] / "out.nwb"
    towers().run_conversion(nwbfile_path=path, conversion_options={})
    assert path.is_file()
    with pytest.raises(ValueError, match="conversion_options"):
        towers().run_conversion(nwbfile_path=path, overwrite=True, conversion_options={"VirmenData": {"stub": True}})


def test_an_edit_the_units_file_cannot_carry_is_refused(session, towers):  # noqa: F811
    converter = towers()
    metadata = converter.get_metadata()
    metadata["NWBFile"]["session_id"] = "another"
    with pytest.raises(ValueError, match="session_id"):
        converter.run_conversion(nwbfile_path=session["tmp"] / "out.nwb", metadata=metadata)


def test_dandiset_names_and_validates_the_file(session, towers):  # noqa: F811
    dandiset = session["tmp"] / "dandiset"
    path = session["tmp"] / "out.nwb"
    towers().run_conversion(nwbfile_path=path, dandiset=dandiset)
    report = json.loads(path.with_suffix(".report.json").read_text())
    assert report["dandi"]["validated"] is True
    assert [p.name for p in dandiset.rglob("*.nwb")] == [report["dandi"]["organized"].rsplit("/", 1)[-1]]


# --- raw bands, chosen by the same options as convert() ------------------------------------------------------------


def test_raw_bands_go_through_the_class_as_through_convert(session, towers, named_log):  # noqa: F811
    by_function = convert(named_log, session["raw"], session["tmp"] / "function", {"imec0": session["sorter"].parent}, raw_bands=("ap",))
    by_class = session["tmp"] / "class" / "out.nwb"
    towers(raw_bands=("ap",)).run_conversion(nwbfile_path=by_class)
    reports = [json.loads(path.with_suffix(".report.json").read_text()) for path in (by_function, by_class)]
    assert reports[0]["verified"]["raw"] == reports[1]["verified"]["raw"] and reports[1]["verified"]["raw"]["series"] == 1
    with h5py.File(by_function, "r") as a, h5py.File(by_class, "r") as b:
        series = "acquisition/ElectricalSeriesAPImec0"
        assert np.array_equal(a[f"{series}/data"][:], b[f"{series}/data"][:]) and a[f"{series}/data"].shape[0] > 0
        edges = "processing/ecephys/spikeglx_sync_edges"
        assert sorted(a[edges]) == sorted(b[edges]) and all(np.array_equal(a[edges][name][:], b[edges][name][:]) for name in a[edges])


def test_the_class_writes_no_raw_data_when_told(session, towers):  # noqa: F811
    path = session["tmp"] / "out.nwb"
    towers(raw_bands=(), sync_edges=False).run_conversion(nwbfile_path=path)
    with h5py.File(path, "r") as f:
        assert "ElectricalSeriesAPImec0" not in f["acquisition"] and "spikeglx_sync_edges" not in f["processing/ecephys"]
    assert "raw" not in json.loads(path.with_suffix(".report.json").read_text())["verified"]


def test_a_file_with_raw_bands_is_named_ecephys_for_dandi_through_the_class(session, towers):  # noqa: F811
    path = session["tmp"] / "out.nwb"
    towers(raw_bands=("ap",)).run_conversion(nwbfile_path=path, dandiset=session["tmp"] / "dandiset")
    organized = json.loads(path.with_suffix(".report.json").read_text())["dandi"]["organized"]
    assert "_desc-ecephys_" in organized


def test_raw_band_keys_name_the_option(converter, tmp_path):
    with pytest.raises(ValueError, match="raw_bands="):
        TowersNWBConverter(dict(VirmenData=dict(file_path=str(tmp_path / "session.mat")), SpikeGLXAP=dict(folder_path=str(tmp_path))))


def test_raw_options_need_units(converter, tmp_path):
    with pytest.raises(ValueError, match="raw_bands"):
        TowersNWBConverter(dict(VirmenData=dict(file_path=str(tmp_path / "session.mat"))), raw_bands=("ap",))
