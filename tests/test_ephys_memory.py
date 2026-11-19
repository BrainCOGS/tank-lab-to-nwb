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


def test_reading_a_sorting_holds_few_copies_of_its_spikes(tmp_path):
    """rec 80's imec3 (29.4 M spikes) took 2.4 GB to read, 82 bytes a spike; a spike's own data is 16 bytes."""
    import tracemalloc

    from tank_lab_to_nwb.ephys.sorting import read_sorting
    from tests.synthetic_ephys import phy_folder

    folder, _, cluster_ids = phy_folder(tmp_path / "kilosort4_output")
    n = 2_000_000
    rng = np.random.default_rng(1)
    np.save(folder / "spike_times.npy", np.sort(rng.integers(0, 10**9, n)).astype(np.uint64))
    clusters = rng.choice(cluster_ids, n).astype(np.int32)
    np.save(folder / "spike_clusters.npy", clusters)
    np.save(folder / "spike_templates.npy", clusters.astype(np.uint32))
    tracemalloc.start()
    result = read_sorting(folder)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert result.n_spikes == n
    # The times, clusters, templates, the sort order and the times by cluster, plus argsort's buffer: 40 bytes a spike.
    assert peak / n < 42, f"{peak / n:.0f} bytes a spike"
