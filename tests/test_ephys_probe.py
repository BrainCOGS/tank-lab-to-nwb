"""Per-probe devices, electrode groups and electrodes from the meta file, and the probe's SY word."""

from datetime import datetime, timezone

import numpy as np
import pytest
from pynwb import NWBFile

from tank_lab_to_nwb.ephys import spikeglx as sg
from tank_lab_to_nwb.ephys.probe import ProbeError, add_probe_electrodes, check_channel_positions
from tank_lab_to_nwb.ephys.session import read_probe_sync_edges
from tests.synthetic_ephys import DATA, IMEC_RATE, imec_files


@pytest.fixture(autouse=True)
def links(tmp_path, monkeypatch):
    monkeypatch.setenv("TANK_EPHYS_LINK_ROOT", str(tmp_path / "links"))


def _nwbfile():
    return NWBFile(session_description="test", identifier="test", session_start_time=datetime(2024, 7, 27, tzinfo=timezone.utc))


def test_neo_cant_open_a_t0_ap_beside_a_tcat_lf(tmp_path):
    """Pins the library limitation that isolated_stream_folder works around."""
    import spikeinterface.extractors as se

    ap = imec_files(tmp_path / "run_g0", 0, "np2010.ap.meta", with_tcat_lf=True)
    with pytest.raises(KeyError):
        se.read_spikeglx(ap.parent, stream_id="imec0.ap")


def test_electrodes_of_two_probes_from_their_meta_files(tmp_path):
    import probeinterface

    session = tmp_path / "run_g0"
    np2 = imec_files(session, 0, "np2010.ap.meta", with_tcat_lf=True)
    np1 = imec_files(session, 1, "np1.ap.meta")
    nwbfile = _nwbfile()
    first = add_probe_electrodes(nwbfile, np2)
    second = add_probe_electrodes(nwbfile, np1)
    assert len(nwbfile.electrodes) == 768
    assert np.array_equal(first.rows, np.arange(384)) and np.array_equal(second.rows, np.arange(384, 768))
    assert (first.serial_number, first.part_number) == ("20403312961", "NP2010")
    assert (second.serial_number, second.part_number) == ("19108307162", "PRB_1_4_0480_1")
    assert first.device_name != second.device_name and {first.device_name, second.device_name} <= set(nwbfile.devices)
    assert len(set(first.group_names)) == 4 and len(set(second.group_names)) == 1
    assert set(first.group_names) | set(second.group_names) == set(nwbfile.electrode_groups)
    for electrodes, meta in ((first, "np2010.ap.meta"), (second, "np1.ap.meta")):
        probe = probeinterface.read_spikeglx(DATA / meta)
        assert np.array_equal(electrodes.positions, probe.contact_positions)
    again = add_probe_electrodes(nwbfile, np1)
    assert len(nwbfile.electrodes) == 768 and np.array_equal(again.rows, second.rows)


def test_channel_positions_are_checked(tmp_path):
    nwbfile = _nwbfile()
    electrodes = add_probe_electrodes(nwbfile, imec_files(tmp_path / "run_g0", 0, "np1.ap.meta"))
    channel_map = np.delete(np.arange(384), [191])  # the reference channel, left out of the lab's chanmap
    difference, offset = check_channel_positions(electrodes, channel_map, electrodes.positions[channel_map] + [0, 20])
    assert difference == 0 and np.array_equal(offset, [0, 20])
    with pytest.raises(ProbeError, match="another probe"):
        check_channel_positions(electrodes, channel_map, electrodes.positions[channel_map[::-1]])
    with pytest.raises(ProbeError, match="names channels up to 400"):
        check_channel_positions(electrodes, np.array([0, 400]), np.zeros((2, 2)))


def test_probe_sync_edges_through_spikeinterface_equal_the_direct_read(tmp_path):
    n = int(IMEC_RATE * 4.2)
    sync = ((((np.arange(n) / IMEC_RATE - 0.3) % 1.0) < 0.5).astype(np.int16) << 6) | np.int16(1 << 2)  # another status bit set too
    ap = imec_files(tmp_path / "run_g0", 0, "np2010.ap.meta", n_samples=n, sync=sync, with_tcat_lf=True)
    stream = sg.open_stream(ap)
    windows = sg.sampling_windows(stream, every_seconds=1.0, length_seconds=1.25)
    through_library = read_probe_sync_edges(ap, windows)
    direct = sg.sync_edges(stream, windows)
    assert np.array_equal(through_library, direct)
    assert np.array_equal(direct, np.flatnonzero(np.diff((sync >> 6) & 1) == 1) + 1)
