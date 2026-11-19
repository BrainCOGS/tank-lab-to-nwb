"""Spike times are written one probe at a time, so a four-probe session never holds every probe's spikes at once."""

import h5py
import numpy as np

from tank_lab_to_nwb.ephys.convert import ProbeSpikeTimes


def test_chunks_across_probe_boundaries_read_back_exactly(tmp_path):
    rng = np.random.default_rng(0)
    probes = [np.sort(rng.uniform(0, 100, n)) for n in (1003, 17, 2500)]
    calls = []

    def times_of(i):
        calls.append(i)
        return probes[i]

    data = ProbeSpikeTimes([p.size for p in probes], times_of, chunk=256)
    with h5py.File(tmp_path / "x.h5", "w") as f:
        dataset = f.create_dataset("t", shape=data.maxshape, dtype=data.dtype, chunks=data.chunk_shape)
        for chunk in data:
            dataset[chunk.selection] = chunk.data
    with h5py.File(tmp_path / "x.h5", "r") as f:
        assert np.array_equal(f["t"][:], np.concatenate(probes))
    # Each probe's times were made once, in order: only one probe is held at a time.
    assert calls == [0, 1, 2]
    assert data.peak_held <= max(p.size for p in probes)
