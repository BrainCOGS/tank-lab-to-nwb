"""Writing SpikeGLX AP/LF and sync edges to NWB, and reading them back exactly. Synthetic files only."""

from datetime import datetime, timezone

import h5py
import numpy as np
import pytest
from pynwb import NWBHDF5IO, NWBFile
from pynwb import validate as nwb_validate

from synthetic_spikeglx import VOLTS_PER_BIT, nidq_word, write_s3_like, write_s5_like
from tank_lab_to_nwb.ephys.clip import clip_recording
from tank_lab_to_nwb.ephys.raw import LinearClock, RawEphysError, add_electrical_series, add_sync_edges
from tank_lab_to_nwb.ephys.spikeglx_layout import find_recording

N = 8


def _nwbfile():
    return NWBFile(session_description="synthetic", identifier="synthetic", session_start_time=datetime(2026, 7, 15, tzinfo=timezone.utc))


def _electrodes(nwbfile, probes):
    """Stand-in for the units conversion's electrode table: one group per probe, N rows each. Returns rows per probe."""
    rows = {}
    for probe in probes:
        device = nwbfile.create_device(name=f"Neuropixels{probe.capitalize()}")
        group = nwbfile.create_electrode_group(name=f"Neuropixels{probe.capitalize()}", description="stand-in", location="unknown", device=device)
        start = len(nwbfile.electrodes) if nwbfile.electrodes is not None else 0
        for _ in range(N):
            nwbfile.add_electrode(group=group, location="unknown")
        rows[probe] = list(range(start, start + N))
    return rows


def _write(path, nwbfile):
    with NWBHDF5IO(path, "w") as io:
        io.write(nwbfile)
    assert nwb_validate(path=str(path)) == []


def test_ap_and_lf_read_back_exactly(tmp_path):
    recording = find_recording(write_s3_like(tmp_path / "raw", seconds=1.0, n_neural=N))
    nwbfile = _nwbfile()
    rows = _electrodes(nwbfile, ["imec0", "imec1"])
    clocks = {}
    for probe in recording.probes:
        for stream in (probe.ap, probe.lf):
            clocks[stream.bin_path.name] = clock = LinearClock(start=12.5, rate=stream.sampling_rate * (1 + 20e-6))
            add_electrical_series(nwbfile, stream, rows[probe.device], clock, buffer_bytes=4096)
    _write(tmp_path / "raw.nwb", nwbfile)
    with NWBHDF5IO(tmp_path / "raw.nwb", "r") as io:
        read = io.read()
        series = {
            "imec0.ap": read.acquisition["ElectricalSeriesAPImec0"],
            "imec1.ap": read.acquisition["ElectricalSeriesAPImec1"],
            "imec1.lf": read.acquisition["ElectricalSeriesLFImec1"],  # SpikeGLX's own LF band
            "imec0.lf": read.processing["ecephys"]["LFP"]["ElectricalSeriesLFImec0"],  # CatGT's, so processed
        }
        for probe in recording.probes:
            for stream, kind in ((probe.ap, "ap"), (probe.lf, "lf")):
                got = series[f"{probe.device}.{kind}"]
                assert got.data.shape == (stream.n_samples, N) and got.data.dtype == np.int16  # SY channel left out
                np.testing.assert_array_equal(got.data[:], np.asarray(stream.memmap()[:, :N]))
                probe_type = "NP1" if probe.device == "imec1" else "NP2"
                assert got.conversion == pytest.approx(VOLTS_PER_BIT[(probe_type, kind)], rel=1e-12) and got.unit == "volts"
                assert got.channel_conversion is None
                assert got.starting_time == 12.5 and got.rate == pytest.approx(stream.sampling_rate * (1 + 20e-6), rel=1e-12)
                assert got.timestamps is None
                assert list(got.electrodes.data[:]) == rows[probe.device]
                assert got.electrodes.table is read.electrodes


def test_clip_lands_at_its_place_on_the_clock(tmp_path):
    """A clip's first sample is mapped through the clock at its original sample number."""
    source = write_s5_like(tmp_path / "raw", seconds=3.0, n_neural=N)
    folder, _ = clip_recording(source, tmp_path / "clips", start_s=1.5, seconds=1.0)
    stream = find_recording(folder).probes[0].ap
    clock = LinearClock(start=100.0, rate=stream.sampling_rate)
    nwbfile = _nwbfile()
    series = add_electrical_series(nwbfile, stream, _electrodes(nwbfile, ["imec0"])["imec0"], clock)
    assert stream.clip_start == int(round(1.5 * stream.sampling_rate))
    assert series.starting_time == pytest.approx(100.0 + stream.clip_start / stream.sampling_rate, abs=1e-12)


def test_bent_clock_writes_timestamps(tmp_path):
    recording = find_recording(write_s5_like(tmp_path / "raw", seconds=0.5, n_neural=N))
    stream = recording.probes[0].ap

    def bent(samples):
        seconds = np.asarray(samples, dtype=np.float64) / stream.sampling_rate
        return 3.0 + seconds + 1e-3 * seconds**2  # 0.25 ms of curvature over 0.5 s, far more than a tenth of a sample

    nwbfile = _nwbfile()
    add_electrical_series(nwbfile, stream, _electrodes(nwbfile, ["imec0"])["imec0"], bent)
    _write(tmp_path / "bent.nwb", nwbfile)
    with NWBHDF5IO(tmp_path / "bent.nwb", "r") as io:
        got = io.read().acquisition["ElectricalSeriesAPImec0"]
        assert got.rate is None
        np.testing.assert_array_equal(got.timestamps[:], bent(np.arange(stream.n_samples)))


def test_stub_and_chunking(tmp_path):
    recording = find_recording(write_s5_like(tmp_path / "raw", seconds=0.5, n_neural=N))
    stream = recording.probes[0].ap
    nwbfile = _nwbfile()
    add_electrical_series(
        nwbfile, stream, _electrodes(nwbfile, ["imec0"])["imec0"], LinearClock(0.0, stream.sampling_rate), n_samples=5000, chunk_shape=(1000, 3), buffer_bytes=3000 * 2 * stream.n_channels
    )
    _write(tmp_path / "stub.nwb", nwbfile)
    with h5py.File(tmp_path / "stub.nwb", "r") as f:
        data = f["acquisition/ElectricalSeriesAPImec0/data"]
        assert data.shape == (5000, N) and data.chunks == (1000, 3)
        assert data.compression == "gzip" and data.shuffle
        np.testing.assert_array_equal(data[:], np.asarray(stream.memmap()[:5000, :N]))


def test_wrong_number_of_electrode_rows_is_refused(tmp_path):
    recording = find_recording(write_s5_like(tmp_path / "raw", seconds=0.1, n_neural=N))
    nwbfile = _nwbfile()
    rows = _electrodes(nwbfile, ["imec0"])["imec0"]
    with pytest.raises(RawEphysError, match="7 electrode rows for 8 neural channels"):
        add_electrical_series(nwbfile, recording.probes[0].ap, rows[:-1], LinearClock(0.0, 3e4))


def test_sync_stream_is_refused_as_neural_data(tmp_path):
    recording = find_recording(write_s3_like(tmp_path / "raw", seconds=0.1, n_neural=N))
    with pytest.raises(RawEphysError, match="not an imec stream"):
        add_electrical_series(_nwbfile(), recording.sync, [], LinearClock(0.0, 5e3))


def test_sync_edges_rebuild_the_digital_word(tmp_path):
    recording = find_recording(write_s3_like(tmp_path / "raw", seconds=2.0, n_neural=N))
    nidq = recording.sync
    clock = LinearClock(start=7.0, rate=nidq.sampling_rate)
    nwbfile = _nwbfile()
    add_sync_edges(nwbfile, [(nidq, clock)])
    _write(tmp_path / "sync.nwb", nwbfile)
    with NWBHDF5IO(tmp_path / "sync.nwb", "r") as io:
        table = io.read().processing["ecephys"]["spikeglx_sync_edges"].to_dataframe()
    assert set(table["channel"]) == {"XD0"} and set(table["stream"]) == {nidq.bin_path.name}
    np.testing.assert_allclose(table["time"], clock(table["sample"]), rtol=0, atol=0)
    # Rebuild every bit from its edges and compare with the word.
    word = nidq_word(np.arange(nidq.n_samples), nidq.sampling_rate)
    for bit in range(16):
        edges = table[table["bit"] == bit]
        assert edges["sample"].iloc[0] == 0
        rebuilt = np.repeat(edges["level"].to_numpy(), np.diff(np.r_[edges["sample"].to_numpy(), nidq.n_samples]))
        np.testing.assert_array_equal(rebuilt, (word >> bit) & 1)
    # Line 0 is the 1 Hz sync: rising edges a second apart on the nidq clock.
    rising = table[(table["bit"] == 0) & (table["level"] == 1)]["sample"].to_numpy()
    np.testing.assert_allclose(np.diff(rising[rising > 0]) / nidq.sampling_rate, 1.0, atol=1 / nidq.sampling_rate)


def test_onebox_sync_edges_include_the_sync_channel(tmp_path):
    recording = find_recording(write_s5_like(tmp_path / "raw", seconds=1.2, n_neural=N))
    nwbfile = _nwbfile()
    table = add_sync_edges(nwbfile, [(recording.sync, LinearClock(0.0, recording.sync.sampling_rate))]).to_dataframe()
    assert set(table["channel"]) == {"XD0", "SY0"}
    sy = table[(table["channel"] == "SY0") & (table["bit"] == 6)]
    assert list(sy["level"]) == [1, 0, 1]  # high from 0, low from 0.5 s, high again from 1 s
    assert sy["sample"].iloc[2] == int(np.ceil(recording.sync.sampling_rate))
