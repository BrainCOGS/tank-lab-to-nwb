"""A whole synthetic ephys session, from raw sync files, a log and a sorter folder to a verified NWB file."""

import json

import numpy as np
import probeinterface
import pytest
from pynwb import NWBHDF5IO

from tank_lab_to_nwb.ephys.convert import SampleOriginError
from tank_lab_to_nwb.ephys.convert_session import convert
from tank_lab_to_nwb.ephys.session import SessionSyncError, build_session_sync
from tests.synthetic_ephys import (
    DATA,
    NIDQ_RATE,
    imec_files,
    iteration_schedule,
    nidq_meta,
    phy_folder,
    sync_word,
    write_spikeglx,
)
from tests.synthetic_log import make_block, make_log, make_trial

# A low imec rate keeps the AP files small; the clock only uses the meta file's rate.
IMEC_RATE = 3000.0367
SLOPE, INTERCEPT = 1 + 4e-6, 2.5  # sync-stream seconds = SLOPE * elapsed + INTERCEPT
PROBE_DRIFT = -10e-6


def _log(schedule):
    trials = []
    for start, times in schedule:
        trial = make_trial(start=start, n_frames=times.size - 2, n_iti=2)
        trial["time"] = times.reshape(-1, 1)
        trial["sensorDots"] = np.zeros((times.size, 5), dtype="int16")
        trials.append(trial)
    log = make_log([make_block(trials, start=(2026, 7, 15, 12, 0, 3.0))])
    log["session"]["end"] = np.array([[2026, 7, 15, 12, 1, 30.0]])
    log["currentIt"] = np.array([[0.0]])
    return log


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setenv("TANK_EPHYS_LINK_ROOT", str(tmp_path / "links"))
    schedule = iteration_schedule(n_trials=5, iterations=(120, 160), gap=0.8)
    duration = schedule[-1][0] + float(schedule[-1][1][-1]) + 5.0
    n_nidq = int((SLOPE * duration + INTERCEPT + 2) * NIDQ_RATE)
    word, _ = sync_word(schedule, n_samples=n_nidq, slope=SLOPE, intercept=INTERCEPT)
    raw = tmp_path / "raw" / "run_g0"
    write_spikeglx(raw, "run_g0_t0.nidq", [word.view(np.int16)], nidq_meta(first_sample=0))
    n_ap = int(duration * IMEC_RATE)
    # probe seconds = (sync seconds - 0.03) * (1 + drift): the probe file starts 30 ms after the NIDQ's.
    sync_seconds = 0.03 + (np.arange(n_ap) / IMEC_RATE) / (1 + PROBE_DRIFT)
    sy = ((((sync_seconds - 0.3) % 1.0) < 0.5).astype(np.int16) << 6)
    imec_files(raw, 0, n_samples=n_ap, sync=sy, first_sample=int(0.03 * IMEC_RATE), rate=IMEC_RATE)
    sorter, trains, cluster_ids = phy_folder(tmp_path / "processed" / "job_id_1" / "kilosort4_output", n_samples=n_ap - 10, rate=30000.0)
    probe = probeinterface.read_spikeglx(DATA / "np1.ap.meta")
    np.save(sorter / "channel_map.npy", np.arange(8, dtype=np.int32))
    np.save(sorter / "channel_positions.npy", probe.contact_positions[:8] + [11.0, 0.0])
    return dict(log=_log(schedule), raw=raw, sorter=sorter, trains=trains, cluster_ids=cluster_ids, n_ap=n_ap, tmp=tmp_path)


def test_spikes_land_on_the_behavior_clock(session):
    sync = build_session_sync(session["raw"], session["log"], file="log.mat")
    assert (sync.pipeline.regular_sync_status, sync.pipeline.mode) == (1, "counter_bit0")
    assert abs(sync.clock.behavior.slope - SLOPE) < 2e-6
    fit = sync.clock.probes["imec0"]
    assert abs((fit.linear_slope - 1) * 1e6 - (-PROBE_DRIFT * 1e6)) < 3
    # A spike at a known NIDQ time lands at that time's elapsed reading plus the log's offset.
    nidq_seconds = np.array([3.0, 7.5, 11.0])
    probe_samples = (nidq_seconds - 0.03) * (1 + PROBE_DRIFT) * IMEC_RATE
    elapsed = (nidq_seconds - INTERCEPT) / SLOPE
    got = sync.clock.probe_samples_to_session("imec0", probe_samples)
    assert np.max(np.abs(got - (elapsed + sync.clock.clock_offset))) < 1e-4


def test_whole_conversion_reads_back(session, monkeypatch):
    import tank_lab_to_nwb.ephys.convert_session as cs

    real = cs.build_session_sync
    monkeypatch.setattr(cs, "build_session_sync", lambda raw, log, **kw: real(raw, session["log"], file="log.mat", **kw))
    path = convert(session["tmp"] / "log.mat", session["raw"], session["tmp"] / "out", {"imec0": session["sorter"].parent})
    report = json.loads(path.with_suffix(".report.json").read_text())
    assert report["verified"]["units"] == session["cluster_ids"].size
    assert report["verified"]["spikes"] == sum(v.size for v in session["trains"].values())
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        units = nwbfile.units.to_dataframe()
        assert list(units["probe"].unique()) == ["imec0"]
        assert list(units["cluster_id"]) == session["cluster_ids"].tolist()
        assert set(units["quality"]) <= {"good", "mua", "noise"}
        assert len(nwbfile.electrodes) == 384
        alignment = nwbfile.processing["ecephys"]["ephys_sync_alignment"].to_dataframe().set_index("key")["value"]
        assert alignment["pipeline.index_lag_seconds"] == repr(6 / NIDQ_RATE)
        sorting = nwbfile.processing["ecephys"]["sorting_imec0"].to_dataframe().set_index("key")["value"]
        assert sorting["sorter"] == "Kilosort 4" and sorting["sorter_version"] == "4.0.37"
        assert json.loads(sorting["channel_position_offset_um"]) == [11.0, 0.0]
        assert min(min(t) for t in units["spike_times"]) >= 0
        assert len(nwbfile.trials) == 5


def test_a_trimmed_sorting_is_refused(session):
    from tank_lab_to_nwb.ephys.convert import check_sample_origin
    from tank_lab_to_nwb.ephys.sorting import read_sorting
    from tank_lab_to_nwb.ephys.spikeglx import open_stream

    (session["sorter"] / "temp_wh.dat").write_bytes(b"\0" * (2 * 385 * (session["n_ap"] - 100)))
    with pytest.raises(SampleOriginError, match="moves sample 0"):
        check_sample_origin(read_sorting(session["sorter"]), open_stream(next(session["raw"].glob("*_imec0/*.ap.bin"))))


def test_old_digital_word_sessions_are_refused(session):
    with pytest.raises(SessionSyncError, match="digital-word"):
        build_session_sync(session["raw"], session["log"], file="log.mat", session_date="2021-05-06")


def test_unsynced_lines_are_refused(session):
    nidq = next(session["raw"].glob("*.nidq.bin"))
    data = np.fromfile(nidq, dtype="<u2")
    data &= np.uint16(~(1 << 2) & 0xFFFF)  # the iteration line never changes
    data.tofile(nidq)
    with pytest.raises(SessionSyncError):
        build_session_sync(session["raw"], session["log"], file="log.mat")
