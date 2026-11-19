"""Raw bands on a session's clock (A's EphysClock): the LF-to-AP rule, clips against full files, and add_raw_ephys."""

from datetime import datetime, timezone
from types import SimpleNamespace

import json

import numpy as np
import pytest
from pynwb import NWBHDF5IO, NWBFile

from synthetic_spikeglx import write_s3_like
from tank_lab_to_nwb.ephys.clip import clip_recording
from tank_lab_to_nwb.ephys.clock import EphysClock
from tank_lab_to_nwb.ephys.raw import RawEphysError, add_raw_ephys, ap_samples_of_lf, probe_clock, sync_stream_clock
from tank_lab_to_nwb.ephys.spikeglx_layout import find_recording

N = 8


def _sync(recording, shift=0.25):
    """What build_session_sync would hand over, for the parts the raw writer uses."""
    rates = {p.device: p.ap.sampling_rate for p in recording.probes}
    clock = EphysClock(sync_kind="nidq", sync_rate=recording.sync.sampling_rate, probe_clock="meta_rates", probe_rates=rates, shift=shift)
    streams = {p.device: SimpleNamespace(bin_path=p.ap.bin_path, meta=p.ap.meta) for p in recording.probes}
    return SimpleNamespace(clock=clock, probe_streams=streams, sync_stream=SimpleNamespace(bin_path=recording.sync.bin_path, meta=recording.sync.meta))


def test_lf_to_ap_rule(tmp_path):
    """S3's numbers: NP1 LF counts on the AP counter / 12; CatGT LF is every 12th AP sample of the file."""
    recording = find_recording(write_s3_like(tmp_path, seconds=0.2))
    catgt, native = recording.probes[0], recording.probes[1]
    assert catgt.lf.is_catgt and not native.lf.is_catgt
    k = np.array([0, 1, 1000])
    np.testing.assert_array_equal(ap_samples_of_lf(catgt.lf, catgt.ap, k), 12 * k)
    # synthetic_spikeglx gives imec1 AP firstSample 83220095 and LF 6935007, as in S3: LF starts 11 AP samples early.
    assert 12 * native.lf.first_sample - native.ap.first_sample == -11
    np.testing.assert_array_equal(ap_samples_of_lf(native.lf, native.ap, k), 12 * k - 11)


def test_clip_maps_like_the_full_file(tmp_path):
    """A clock built on the full files gives a clip's samples the times the full file gives them, and the other way round."""
    full = find_recording(write_s3_like(tmp_path / "raw", seconds=2.0))
    folder, _ = clip_recording(full.folder, tmp_path / "clips", start_s=0.75, seconds=1.0)
    clip = find_recording(folder)
    for sync_on, other in ((full, clip), (clip, full)):
        sync = _sync(sync_on)
        for a, b in zip(full.probes, clip.probes):
            for band in ("ap", "lf"):
                s_full, s_clip = getattr(a, band), getattr(b, band)
                samples = np.arange(s_clip.clip_start, s_clip.clip_start + 50)  # original numbering, inside the clip
                on_full = probe_clock(sync, a.device, s_full, ap=a.ap)(samples)
                on_clip = probe_clock(sync, b.device, s_clip, ap=b.ap)(samples)
                np.testing.assert_allclose(on_clip, on_full, rtol=0, atol=1e-12)
        samples = np.arange(clip.sync.clip_start, clip.sync.clip_start + 50)
        np.testing.assert_allclose(sync_stream_clock(sync, clip.sync)(samples), sync_stream_clock(sync, full.sync)(samples), rtol=0, atol=1e-12)
    # With the clock built on the full files, original sample s of AP is s / rate + shift.
    sync = _sync(full)
    ap = clip.probes[0].ap
    assert probe_clock(sync, "imec0", ap)([ap.clip_start])[0] == pytest.approx(ap.clip_start / ap.sampling_rate + 0.25, abs=1e-12)


def _electrodes(nwbfile, recording):
    rows = {}
    for probe in recording.probes:
        device = nwbfile.create_device(name=f"Neuropixels{probe.device.capitalize()}")
        group = nwbfile.create_electrode_group(name=probe.device, description="stand-in", location="root", device=device)
        start = len(nwbfile.electrodes) if nwbfile.electrodes is not None else 0
        for _ in range(N):
            nwbfile.add_electrode(group=group, location="root")
        rows[probe.device] = np.arange(start, start + N)
    return rows


@pytest.mark.parametrize("bands", [("lf",), ("ap", "lf"), ()])
def test_add_raw_ephys(tmp_path, bands):
    full = find_recording(write_s3_like(tmp_path / "raw", seconds=1.0, n_neural=N))
    folder, _ = clip_recording(full.folder, tmp_path / "clips", start_s=0.25, seconds=0.5)
    sync = _sync(full)
    nwbfile = NWBFile(session_description="s", identifier="s", session_start_time=datetime(2024, 7, 27, tzinfo=timezone.utc))
    rows = _electrodes(nwbfile, full)
    notes = add_raw_ephys(nwbfile, sync, rows, raw_folder=folder, bands=bands)
    with NWBHDF5IO(tmp_path / "raw.nwb", "w") as io:
        io.write(nwbfile)
    clip = find_recording(folder)
    with NWBHDF5IO(tmp_path / "raw.nwb", "r") as io:
        read = io.read()
        names = set(read.acquisition) | (set(read.processing["ecephys"]["LFP"].electrical_series) if "LFP" in read.processing["ecephys"].data_interfaces else set())
        expected = {f"ElectricalSeries{b.upper()}{p.capitalize()}" for b in bands for p in ("imec0", "imec1")}
        assert names == expected and set(notes) >= expected
        if "lf" in bands:
            lf = read.acquisition["ElectricalSeriesLFImec1"]  # NP1's own LF
            stream = clip.probes[1].lf
            np.testing.assert_array_equal(lf.data[:], np.asarray(stream.memmap()[:, :N]))
            expected_start = (12 * (stream.first_sample + 0) - full.probes[1].ap.first_sample) / full.probes[1].ap.sampling_rate + 0.25
            assert lf.starting_time == pytest.approx(expected_start, abs=1e-9)
            assert list(lf.electrodes.data[:]) == list(rows["imec1"])
        edges = read.processing["ecephys"]["spikeglx_sync_edges"].to_dataframe()
        np.testing.assert_allclose(edges["time"], edges["sample"] / full.sync.sampling_rate + 0.25, rtol=0, atol=1e-12)


def test_raw_folder_must_hold_the_clocks_files(tmp_path):
    full = find_recording(write_s3_like(tmp_path / "a", seconds=0.2, n_neural=N))
    other = write_s3_like(tmp_path / "b", seconds=0.2, n_neural=N)
    for path in other.rglob("*.bin"):
        path.rename(path.with_name(path.name.replace("syn_g0", "oth_g0")))
    for path in other.rglob("*.meta"):
        path.rename(path.with_name(path.name.replace("syn_g0", "oth_g0")))
    for path in sorted(other.glob("syn_g0_imec*")):
        path.rename(path.with_name(path.name.replace("syn_g0", "oth_g0")))
    nwbfile = NWBFile(session_description="s", identifier="s", session_start_time=datetime(2024, 7, 27, tzinfo=timezone.utc))
    with pytest.raises(RawEphysError, match="not the file the clock was built on"):
        add_raw_ephys(nwbfile, _sync(full), _electrodes(nwbfile, full), raw_folder=other)


# The whole session through the units conversion's entry point, on its synthetic session (tests/test_ephys_convert.py).
from tests.test_ephys_convert import session  # noqa: E402,F401  (a pytest fixture)


@pytest.mark.parametrize("probe_clock", ["sync_edges", "meta_rates"])
def test_convert_writes_raw_ap_on_the_units_clock_and_rows(session, monkeypatch, probe_clock):  # noqa: F811
    import tank_lab_to_nwb.ephys.convert_session as cs
    from tank_lab_to_nwb.ephys.session import build_session_sync

    real = cs.build_session_sync
    monkeypatch.setattr(cs, "build_session_sync", lambda raw, log, **kw: real(raw, session["log"], file="log.mat", **kw))
    path = cs.convert(session["tmp"] / "log.mat", session["raw"], session["tmp"] / "out", {"imec0": session["sorter"].parent}, raw_bands=("ap", "lf"), probe_clock=probe_clock)
    assert path.name.endswith("_ecephys.nwb")
    sync = build_session_sync(session["raw"], session["log"], file="log.mat", probe_clock=probe_clock)
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        ap = nwbfile.acquisition["ElectricalSeriesAPImec0"]
        # The raw series points at the rows the units use, not a copy: the electrode table holds one probe's rows once.
        assert len(nwbfile.electrodes) == ap.data.shape[1]
        unit_rows = {int(r) for rows in nwbfile.units["electrodes"][:] for r in np.asarray(rows.index)}
        assert unit_rows <= set(int(r) for r in ap.electrodes.data[:])
        # Every sample is where the units' clock puts it.
        samples = np.array([0, ap.data.shape[0] // 2, ap.data.shape[0] - 1])
        expected = sync.clock.probe_samples_to_session("imec0", samples)
        got = ap.timestamps[samples] if ap.timestamps is not None else ap.starting_time + samples / ap.rate
        np.testing.assert_allclose(got, expected, rtol=0, atol=0.1 / 3000)
        assert "ElectricalSeriesLFImec0" not in nwbfile.acquisition  # this synthetic probe has no LF file
        edges = nwbfile.processing["ecephys"]["spikeglx_sync_edges"].to_dataframe()
        assert set(edges["channel"]) == {"XD0"}
    report = json.loads(path.with_suffix(".report.json").read_text())
    assert report["raw"]["imec0 lf"].startswith("No LF file")


def test_raw_folder_option_does_not_replace_the_recording_folder():
    """--raw-folder names where the raw bands come from; the clock is still built from the recording folder."""
    import tank_lab_to_nwb.ephys.convert_session as cs
    from pathlib import Path

    arguments = cs.session_parser().parse_args(["log.mat", "recording", "out", "imec0=sorted", "--raw-folder", "clips"])
    assert arguments.raw_folder == Path("recording")
    assert arguments.raw_from == Path("clips")
    arguments = cs.session_parser().parse_args(["log.mat", "recording", "out", "imec0=sorted"])
    assert arguments.raw_folder == Path("recording") and arguments.raw_from is None


def _converted_with_raw(session, monkeypatch):
    import tank_lab_to_nwb.ephys.convert_session as cs
    from tank_lab_to_nwb.ephys.session import build_session_sync

    real = cs.build_session_sync
    monkeypatch.setattr(cs, "build_session_sync", lambda raw, log, **kw: real(raw, session["log"], file="log.mat", **kw))
    path = cs.convert(session["tmp"] / "log.mat", session["raw"], session["tmp"] / "out", {"imec0": session["sorter"].parent}, raw_bands=("ap",))
    return path, build_session_sync(session["raw"], session["log"], file="log.mat")


def test_convert_reads_the_raw_series_back(session, monkeypatch):  # noqa: F811
    path, _ = _converted_with_raw(session, monkeypatch)
    checked = json.loads(path.with_suffix(".report.json").read_text())["verified"]["raw"]
    assert checked["series"] == 1 and checked["samples"] > 0 and checked["edges"] > 0


@pytest.mark.parametrize("damage", ["sample", "time", "edge", "electrode"])
def test_raw_read_back_catches_a_file_that_differs_from_its_sources(session, monkeypatch, damage):  # noqa: F811
    import h5py

    from tank_lab_to_nwb.ephys.raw import verify_raw_ephys
    from tank_lab_to_nwb.virmen.verify import RoundTripError

    path, sync = _converted_with_raw(session, monkeypatch)
    with h5py.File(path, "r") as f:
        rows = {"imec0": f["acquisition/ElectricalSeriesAPImec0/electrodes"][:]}
    verify_raw_ephys(path, sync, rows, bands=("ap",))
    with h5py.File(path, "r+") as f:
        series = f["acquisition/ElectricalSeriesAPImec0"]
        if damage == "sample":
            series["data"][-1, 0] = series["data"][-1, 0] + 1
        elif damage == "time":
            if "timestamps" in series:
                series["timestamps"][0] = series["timestamps"][0] + 1e-3
            else:
                series["starting_time"][()] = series["starting_time"][()] + 1e-3
        elif damage == "edge":
            f["processing/ecephys/spikeglx_sync_edges/sample"][5] += 1
        else:
            series["electrodes"][0], series["electrodes"][1] = series["electrodes"][1], series["electrodes"][0]
    with pytest.raises(RoundTripError):
        verify_raw_ephys(path, sync, rows, bands=("ap",))
