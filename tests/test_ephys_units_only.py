"""Units-only files: a sorting nothing ties to a recording or to behavior, on the sorted binary's own clock, asked for by name."""

import json

import numpy as np
import probeinterface
import pytest
from pynwb import NWBHDF5IO

from tank_lab_to_nwb.ephys.convert_session import UnitsOnlyError, convert_units_only
from tank_lab_to_nwb.ephys.sorting import read_sorting
from tests.synthetic_ephys import DATA, imec_files, phy_folder

RATE = 3000.0367


@pytest.fixture
def clip(tmp_path, monkeypatch):
    monkeypatch.setenv("TANK_EPHYS_LINK_ROOT", str(tmp_path / "links"))
    ap = imec_files(tmp_path / "clip" / "run_g0", 0, n_samples=40_000, rate=RATE)
    folder, trains, cluster_ids = phy_folder(tmp_path / "job" / "kilosort2_output", kind="ks2", n_samples=39_000)
    probe = probeinterface.read_spikeglx(DATA / "np1.ap.meta")
    np.save(folder / "channel_map.npy", np.arange(8, dtype=np.int32))
    np.save(folder / "channel_positions.npy", probe.contact_positions[:8])
    (folder / "spikeinterface" / "spikeinterface_recording.json").write_text(json.dumps({"kwargs": {"folder_path": str(ap.parent)}}))
    (folder / "temp_wh.dat").write_bytes(b"\0" * (2 * 8 * 40_000))  # params.py says 8 channels
    return dict(ap=ap, folder=folder, trains=trains, ids=cluster_ids, tmp=tmp_path)


def test_times_are_the_sorted_binarys_own(clip):
    path = convert_units_only(clip["tmp"] / "out", {"imec0": clip["folder"]}, subject_id="mouse_1", session_id="clip1")
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        units = nwbfile.units.to_dataframe()
        for cluster, times in zip(units["cluster_id"], units["spike_times"]):
            assert np.array_equal(times, clip["trains"][int(cluster)].astype(np.int64) / RATE)
        assert len(nwbfile.electrodes) == 384 and nwbfile.trials is None
        clock = nwbfile.processing["ecephys"]["units_only_clock"].to_dataframe().set_index("key")["value"]
        assert "own" in clock["clock"] and "nothing ties" in clock["tied_to"].lower()
        assert clock["imec0.sorted_binary"].endswith("run_g0_t0.imec0.ap.bin")
        assert nwbfile.subject.subject_id == "mouse_1" and nwbfile.session_id == "clip1"
        assert "units only" in nwbfile.session_description.lower()


def test_the_binary_must_be_found_or_named(clip):
    (clip["folder"] / "spikeinterface" / "spikeinterface_recording.json").write_text(json.dumps({"kwargs": {"folder_path": str(clip["tmp"] / "gone")}}))
    with pytest.raises(UnitsOnlyError, match="binary the sorter read"):
        convert_units_only(clip["tmp"] / "out", {"imec0": clip["folder"]}, subject_id="m", session_id="s")
    path = convert_units_only(clip["tmp"] / "out", {"imec0": clip["folder"]}, binaries={"imec0": clip["ap"]}, subject_id="m", session_id="s")
    assert path.is_file()


def test_a_binary_of_another_length_is_refused(clip):
    (clip["folder"] / "temp_wh.dat").write_bytes(b"\0" * (2 * 8 * 200_000))
    with pytest.raises(Exception, match="samples"):
        convert_units_only(clip["tmp"] / "out", {"imec0": clip["folder"]}, subject_id="m", session_id="s")


def test_reader_says_what_it_ignored(clip):
    (clip["folder"] / "diagnostics.png").write_bytes(b"")
    result = read_sorting(clip["folder"])
    assert "diagnostics.png" in result.ignored and "pc_features.npy" in result.ignored
    assert "spike_times.npy" not in result.ignored


def test_units_only_is_never_a_fallback(clip, tmp_path):
    """The session converter without a sync file stops; it doesn't quietly write units alone."""
    from tank_lab_to_nwb.ephys.session import build_session_sync
    from tank_lab_to_nwb.ephys.spikeglx import SpikeGLXError
    from tests.synthetic_log import make_log

    with pytest.raises(SpikeGLXError, match="no \\*nidq.bin"):
        build_session_sync(clip["ap"].parent.parent, make_log(), file="log.mat")
