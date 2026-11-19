"""SpikeGLX files: the library read and the direct read give the same values, so neither is the source of a disagreement."""

import numpy as np
import pytest

from tank_lab_to_nwb.ephys import pipeline_sync as ps
from tank_lab_to_nwb.ephys import spikeglx as sg
from tank_lab_to_nwb.ephys.session import read_sync_word
from tests.synthetic_ephys import NIDQ_RATE, iteration_schedule, nidq_meta, sync_word, write_spikeglx


@pytest.fixture
def nidq_file(tmp_path):
    trials = iteration_schedule(n_trials=6)
    word, truth = sync_word(trials, n_samples=int(NIDQ_RATE * 40))
    # A mixed folder like S3's: CatGT tcat files beside the raw t0 ones make neo fail on the folder itself.
    path = write_spikeglx(tmp_path / "run_g0", "run_g0_t0.nidq", [word.view(np.int16)], nidq_meta())
    (tmp_path / "run_g0" / "run_g0_tcat.nidq.meta").write_text(path.with_suffix(".meta").read_text())
    return path, word, trials, truth


def test_library_and_direct_reads_agree(nidq_file, tmp_path, monkeypatch):
    path, word, _, _ = nidq_file
    monkeypatch.setenv("TANK_EPHYS_LINK_ROOT", str(tmp_path / "links"))
    stream = sg.open_stream(path)
    through_library = read_sync_word(stream)
    direct = sg.read_channel(stream, sg.digital_channel(stream))
    assert through_library.dtype == np.int16
    assert np.array_equal(through_library, direct)
    assert np.array_equal(direct.view(np.uint16), word)


def test_sync_edges_are_the_wave(nidq_file):
    path, _, _, _ = nidq_file
    stream = sg.open_stream(path)
    edges = sg.sync_edges(stream)
    expected = np.flatnonzero(np.diff(((np.arange(stream.n_samples) / NIDQ_RATE - 0.3) % 1.0 < 0.5).astype(np.int8)) == 1) + 1
    assert np.array_equal(edges, expected)
    windows = sg.sampling_windows(stream, every_seconds=10)
    assert set(sg.sync_edges(stream, windows)) <= set(edges)


def test_pipeline_lines_take_the_pipelines_sample_count(nidq_file):
    path, word, _, _ = nidq_file
    stream = sg.open_stream(path)
    trial, iteration = ps.pipeline_lines(word, "nidq", stream.rate, stream.meta["fileTimeSecs"])
    assert trial.size == int(stream.rate * float(stream.meta["fileTimeSecs"]))
    assert np.array_equal(trial, ((word[: trial.size] >> 1) & 1))
    assert np.array_equal(iteration, ((word[: trial.size] >> 2) & 1))


def test_onebox_lines_pick_the_busier_one_as_iteration():
    trial = np.zeros(100, dtype=np.uint16)
    trial[10:15] = 1
    iteration = (np.arange(100) // 3 % 2).astype(np.uint16)
    for trial_line, iteration_line in ((0, 1), (1, 0)):
        word = (trial << trial_line) | (iteration << iteration_line)
        got_trial, got_iteration = ps.pipeline_lines(word, "obx", 1.0, 100)
        assert np.array_equal(got_trial, trial) and np.array_equal(got_iteration, iteration)


def test_meta_and_file_size_must_agree(nidq_file):
    path, _, _, _ = nidq_file
    with open(path, "ab") as handle:
        handle.write(b"\0\0")
    with pytest.raises(sg.SpikeGLXError, match="incomplete"):
        sg.open_stream(path)


def test_analog_sync_is_refused(nidq_file):
    path, _, _, _ = nidq_file
    meta = path.with_suffix(".meta")
    meta.write_text(meta.read_text().replace("syncNiChanType=0", "syncNiChanType=1"))
    with pytest.raises(sg.SpikeGLXError, match="analog"):
        sg.sync_edges(sg.open_stream(path))


def test_no_sync_file_is_named(tmp_path):
    with pytest.raises(sg.SpikeGLXError, match="no \\*nidq.bin or \\*obx.bin"):
        sg.find_sync_stream(tmp_path)


def test_isolated_folder_holds_only_links(nidq_file, tmp_path):
    path, _, _, _ = nidq_file
    folder = sg.isolated_stream_folder(path, root=tmp_path / "links")
    assert sorted(p.name for p in folder.iterdir()) == ["run_g0_t0.nidq.bin", "run_g0_t0.nidq.meta"]
    assert all(p.is_symlink() and not p.readlink().is_absolute() for p in folder.iterdir())
    assert sg.isolated_stream_folder(path, root=tmp_path / "links") == folder
