"""A whole synthetic ephys session, from raw sync files, a log and a sorter folder to a verified NWB file."""

import json
import re
from pathlib import Path

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
from tank_lab_to_nwb.virmen.layout import DandiValidationError
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
    return dict(schedule=schedule, n_nidq=n_nidq, log=_log(schedule), raw=raw, sorter=sorter, trains=trains, cluster_ids=cluster_ids, n_ap=n_ap, tmp=tmp_path)


def test_iterations_take_their_pulses_and_spikes_their_samples(session):
    sync = build_session_sync(session["raw"], session["log"], file="log.mat", probe_clock="meta_rates")
    assert (sync.pipeline.regular_sync_status, sync.pipeline.mode, sync.vectors) == (1, "counter_bit0", "regular")
    for got, index in zip(sync.iteration_seconds, sync.iteration_idx_vector):
        assert np.array_equal(got, np.asarray(index) / NIDQ_RATE)
    frames = next(s.timestamps for s in sync.session.series if s.clock == "all")
    assert np.array_equal(frames, np.concatenate(sync.iteration_seconds))
    samples = np.array([0, 12345, session["n_ap"] - 1])
    assert np.array_equal(sync.clock.probe_samples_to_session("imec0", samples), samples / IMEC_RATE)
    # The trials table follows the frames: a trial starts at its first iteration's pulse.
    starts = next(c.data for c in sync.session.trials if c.name == "start_time")
    assert np.array_equal(starts, [t[0] for t in sync.iteration_seconds])


def test_sync_edges_moves_spikes_by_the_measured_drift(session):
    sync = build_session_sync(session["raw"], session["log"], file="log.mat")
    assert sync.clock.probe_clock == "sync_edges"
    fit = sync.clock.probes["imec0"]
    assert abs((fit.linear_slope - 1) * 1e6 - (-PROBE_DRIFT * 1e6)) < 3
    # A spike at a known NIDQ time lands at that time, whatever the probe's own rate says.
    nidq_seconds = np.array([3.0, 7.5, 11.0])
    probe_samples = (nidq_seconds - 0.03) * (1 + PROBE_DRIFT) * IMEC_RATE
    assert np.max(np.abs(sync.clock.probe_samples_to_session("imec0", probe_samples) - nidq_seconds)) < 1e-4


def test_virmen_borrowed_vectors_only_when_asked(session):
    # Tens of pulses missing in three trials: too many for the pipeline's fix path.
    skip = {(t, i) for t in (1, 2, 3) for i in range(20, 60)}
    word, _ = sync_word(session["schedule"], n_samples=session["n_nidq"], slope=SLOPE, intercept=INTERCEPT, skip=skip)
    word.view(np.int16).tofile(next(session["raw"].glob("*.nidq.bin")))
    with pytest.raises(SessionSyncError, match="allow_virmen_borrowed"):
        build_session_sync(session["raw"], session["log"], file="log.mat")
    sync = build_session_sync(session["raw"], session["log"], file="log.mat", allow_virmen_borrowed=True)
    assert sync.vectors == "virmen_borrowed"
    for got, index in zip(sync.iteration_seconds, sync.pipeline.iteration_idx_vector_from_virmen):
        assert np.array_equal(got, np.asarray(index) / NIDQ_RATE)


def test_whole_conversion_reads_back(session, monkeypatch):
    import tank_lab_to_nwb.ephys.convert_session as cs

    real = cs.build_session_sync
    monkeypatch.setattr(cs, "build_session_sync", lambda raw, log, **kw: real(raw, session["log"], file="log.mat", **kw))
    path = convert(session["tmp"] / "log.mat", session["raw"], session["tmp"] / "out", {"imec0": session["sorter"].parent})
    assert convert(session["tmp"] / "log.mat", session["raw"], session["tmp"] / "meta", {"imec0": session["sorter"].parent}, probe_clock="meta_rates", iteration_time="edge").is_file()
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
        assert (alignment["probe_clock"], alignment["iteration_time"], alignment["vectors"]) == ("sync_edges", "pipeline_index", "regular")
        assert nwbfile.session_start_time.isoformat().startswith("2024-07-27T13:59:44")
        sorting = nwbfile.processing["ecephys"]["sorting_imec0"].to_dataframe().set_index("key")["value"]
        assert sorting["sorter"] == "Kilosort 4" and sorting["sorter_version"] == "4.0.37"
        assert json.loads(sorting["channel_position_offset_um"]) == [11.0, 0.0]
        assert min(min(t) for t in units["spike_times"]) >= 0
        assert len(nwbfile.trials) == 5


@pytest.fixture
def named_log(session, monkeypatch):
    """convert() reads the log from a path; give it the synthetic one, and keep dandi's log files in the test's folder."""
    import tank_lab_to_nwb.ephys.convert_session as cs

    real = cs.build_session_sync
    monkeypatch.setattr(cs, "build_session_sync", lambda raw, log, **kw: real(raw, session["log"], file="log.mat", **kw))
    monkeypatch.setenv("XDG_STATE_HOME", str(session["tmp"] / "state"))
    return session["tmp"] / "log.mat"


def test_conversion_is_named_and_validated_for_dandi(session, named_log):
    dandiset = session["tmp"] / "dandiset"
    path = convert(named_log, session["raw"], session["tmp"] / "out", {"imec0": session["sorter"].parent}, raw_bands=(), dandiset=dandiset)
    report = json.loads(path.with_suffix(".report.json").read_text())
    organized = Path(report["dandi"]["organized"])
    assert report["dandi"]["validated"] is True
    assert list(dandiset.rglob("*.nwb")) == [organized]
    assert re.fullmatch(r"sub-[^_]+_ses-[^_]+_desc-units_behavior\+ecephys\.nwb", organized.name), organized.name
    assert path.is_file() and organized.stat().st_size == path.stat().st_size


def test_a_file_dandi_rejects_stops_the_conversion(session, named_log):
    dandiset = session["tmp"] / "dandiset"
    with pytest.raises(DandiValidationError, match="(?i)species"):
        convert(named_log, session["raw"], session["tmp"] / "out", {"imec0": session["sorter"].parent}, metadata={"subject": {"sex": "U"}}, raw_bands=(), dandiset=dandiset)
    report = json.loads((session["tmp"] / "out" / "log_units.report.json").read_text())
    assert report["dandi"]["validated"] is False and "species" in report["dandi"]["error"].lower()
    assert report["verified"]["units"] == session["cluster_ids"].size
    assert not list(dandiset.rglob("*.nwb"))


def test_a_trimmed_sorting_is_refused(session):
    from tank_lab_to_nwb.ephys.convert import check_sample_origin
    from tank_lab_to_nwb.ephys.sorting import read_sorting
    from tank_lab_to_nwb.ephys.spikeglx import open_stream

    # temp_wh.dat holds whole padded batches, so it can't be shorter than the file the sorter read.
    (session["sorter"] / "temp_wh.dat").write_bytes(b"\0" * (2 * 385 * (session["n_ap"] - 1)))
    with pytest.raises(SampleOriginError, match="moves sample 0"):
        check_sample_origin(read_sorting(session["sorter"]), open_stream(next(session["raw"].glob("*_imec0/*.ap.bin"))))


def test_old_digital_word_sessions_are_refused(session):
    with pytest.raises(SessionSyncError, match="digital-word"):
        build_session_sync(session["raw"], session["log"], file="log.mat", session_date="2021-05-06")


def test_a_log_with_more_trials_than_pulses_is_named(session):
    log = _log(session["schedule"] + iteration_schedule(n_trials=2, seed=9))
    with pytest.raises(SessionSyncError, match="5 trial pulses and the log 7 saved trials"):
        build_session_sync(session["raw"], log, file="log.mat")
