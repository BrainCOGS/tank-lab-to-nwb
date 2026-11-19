"""Finding SpikeGLX streams, refusing layouts we don't handle, and cutting clips. Synthetic files only."""

import hashlib
import shutil

import numpy as np
import pytest

from synthetic_spikeglx import VOLTS_PER_BIT, neural_value, nidq_word, write_imec, write_nidq, write_s3_like, write_s5_like
from tank_lab_to_nwb.ephys.clip import clip_recording
from tank_lab_to_nwb.ephys.spikeglx_layout import SpikeGLXLayoutError, find_recording, read_meta


def test_s3_layout(tmp_path):
    recording = find_recording(write_s3_like(tmp_path))
    assert [p.device for p in recording.probes] == ["imec0", "imec1"]
    imec0, imec1 = recording.probes
    assert imec0.ap.probe_part_number == "NP2010" and imec0.lf.is_catgt and "CatGT" in imec0.lf_note
    assert imec1.ap.probe_part_number == "PRB_1_4_0480_1" and not imec1.lf.is_catgt
    assert recording.sync.kind == "nidq" and recording.sync.digital_word_channel == 0
    # The SY channel is the last saved channel and is not a neural channel.
    assert imec1.ap.n_channels == 9 and imec1.ap.sync_channel == 8
    np.testing.assert_array_equal(imec1.ap.neural_channels, np.arange(8))


def test_s5_layout(tmp_path):
    recording = find_recording(write_s5_like(tmp_path))
    assert all(p.lf is None and "no hardware LF" in p.lf_note for p in recording.probes)
    assert recording.sync.kind == "obx"
    assert recording.sync.digital_word_channel == 2 and recording.sync.sync_channel == 3


@pytest.mark.parametrize("probe_type, band", sorted(VOLTS_PER_BIT))
def test_gain_matches_spikeglx_definition(tmp_path, probe_type, band):
    if band == "lf":
        write_imec(tmp_path / "syn_g0", probe_type=probe_type, band="ap", seconds=0.01)
    path = write_imec(tmp_path / "syn_g0", probe_type=probe_type, band=band, catgt=probe_type != "NP1" and band == "lf", seconds=0.01)
    stream = find_recording(tmp_path / "syn_g0").probes[0]
    stream = stream.ap if band == "ap" else stream.lf
    assert stream.bin_path == path
    np.testing.assert_allclose(stream.volts_per_bit(), VOLTS_PER_BIT[(probe_type, band)], rtol=1e-12)


def _ap_only(tmp_path):
    run = tmp_path / "syn_g0"
    write_imec(run, seconds=0.01)
    return run


@pytest.mark.parametrize(
    "damage, message",
    [
        (lambda run: (run / "syn_g0_imec0" / "syn_g0_t0.imec0.ap.bin").rename(run / "syn_g0_imec0" / "syn_g0_t0.imec0.ap.cbin"), "mtscomp"),
        (lambda run: (run / "syn_g0_imec0" / "syn_g0_t0.imec0.ap.meta").unlink(), "no .meta"),
        (lambda run: open(run / "syn_g0_imec0" / "syn_g0_t0.imec0.ap.bin", "ab").write(b"\0\0"), "truncated or was changed"),
        (lambda run: write_imec(run, run="syn_g0_x", seconds=0.01) and shutil.move(run / "syn_g0_x_imec0" / "syn_g0_x_t0.imec0.ap.bin", run / "syn_g0_imec0" / "syn_g0_t1.imec0.ap.bin") and shutil.move(run / "syn_g0_x_imec0" / "syn_g0_x_t0.imec0.ap.meta", run / "syn_g0_imec0" / "syn_g0_t1.imec0.ap.meta"), "only a single t0"),
        (lambda run: [shutil.copy(run / "syn_g0_imec0" / f"syn_g0_t0.imec0.ap.{ext}", run / "syn_g0_imec0" / f"syn_g0_tcat.imec0.ap.{ext}") for ext in ("bin", "meta")] and [(run / "syn_g0_imec0" / f"syn_g0_t0.imec0.ap.{ext}").unlink() for ext in ("bin", "meta")], "written by CatGT"),
        (lambda run: [shutil.copy(run / "syn_g0_imec0" / f"syn_g0_t0.imec0.ap.{ext}", run / "syn_g0_imec0" / f"syn_g0_{t}.imec0.lf.{ext}") for ext in ("bin", "meta") for t in ("t0", "tcat")], "2 LF files"),
        (lambda run: shutil.rmtree(run / "syn_g0_imec0"), "no imec AP files"),
    ],
    ids=["cbin", "no-meta", "size-mismatch", "second-trigger", "catgt-ap", "two-lf", "no-probe"],
)
def test_refuses_unhandled_layouts(tmp_path, damage, message):
    run = _ap_only(tmp_path)
    damage(run)
    with pytest.raises(SpikeGLXLayoutError, match=message):
        find_recording(run)


def test_refuses_two_sync_streams(tmp_path):
    run = _ap_only(tmp_path)
    write_nidq(run, seconds=0.01)
    from synthetic_spikeglx import write_obx

    write_obx(run, seconds=0.01)
    with pytest.raises(SpikeGLXLayoutError, match="more than one sync stream"):
        find_recording(run)


def test_clip_keeps_sample_numbering_and_rewrites_meta(tmp_path):
    source = find_recording(write_s3_like(tmp_path / "raw", seconds=3.0))
    folder, written = clip_recording(source.folder, tmp_path / "clips", start_s=1.25, seconds=1.0)
    assert len(written) == 5
    clipped = find_recording(folder)
    for original, clip in zip(
        [s for p in source.probes for s in (p.ap, p.lf)] + [source.sync], [s for p in clipped.probes for s in (p.ap, p.lf)] + [clipped.sync]
    ):
        start = int(round(1.25 * original.sampling_rate))
        count = int(round(1.0 * original.sampling_rate))
        assert clip.clip_start == start and clip.n_samples == count
        assert clip.first_sample == original.first_sample + start
        meta = read_meta(clip.bin_path.with_suffix(".meta"))
        assert int(meta["fileSizeBytes"]) == clip.bin_path.stat().st_size
        assert float(meta["fileTimeSecs"]) == pytest.approx(count / original.sampling_rate, rel=1e-12)
        assert meta["fileSHA1"] == hashlib.sha1(clip.bin_path.read_bytes()).hexdigest().upper()
        assert meta["clipSourceFile"] == str(original.bin_path)
        # Everything else in the meta is untouched, and the ~ keys stay last.
        source_meta = read_meta(original.bin_path.with_suffix(".meta"))
        changed = {"fileSizeBytes", "fileTimeSecs", "fileSHA1", "firstSample", "clipStartSample", "clipSourceFile"}
        assert {k: v for k, v in meta.items() if k not in changed} == {k: v for k, v in source_meta.items() if k not in changed}
        tilde = [k.startswith("~") for k in meta]
        assert tilde == sorted(tilde)
        np.testing.assert_array_equal(np.asarray(clip.memmap()), np.asarray(original.memmap()[start : start + count]))
    # The values are those of the original samples.
    ap = clipped.probes[1].ap
    samples = np.arange(ap.n_samples) + ap.clip_start
    np.testing.assert_array_equal(np.asarray(ap.memmap()[:, :8]), neural_value(samples[:, None], np.arange(8)[None, :]))
    np.testing.assert_array_equal(np.asarray(clipped.sync.memmap()[:, 0]), nidq_word(np.arange(clipped.sync.n_samples) + clipped.sync.clip_start, clipped.sync.sampling_rate))


def test_clip_of_a_clip_counts_from_the_original(tmp_path):
    source = find_recording(write_s5_like(tmp_path / "raw", seconds=3.0))
    once, _ = clip_recording(source.folder, tmp_path / "once", start_s=1.0, seconds=1.5)
    twice, _ = clip_recording(once, tmp_path / "twice", start_s=0.5, seconds=0.5)
    original, clip = source.probes[0].ap, find_recording(twice).probes[0].ap
    expected = int(round(1.0 * original.sampling_rate)) + int(round(0.5 * original.sampling_rate))
    assert clip.clip_start == expected and clip.first_sample == original.first_sample + expected


def test_clip_outside_the_file_is_refused(tmp_path):
    run = write_s5_like(tmp_path / "raw", seconds=1.0)
    with pytest.raises(ValueError, match="outside"):
        clip_recording(run, tmp_path / "clips", start_s=0.5, seconds=1.0)


def test_spikeinterface_reads_a_clip(tmp_path):
    """The clips open in the standard reader, with the gains and start time it gives the original."""
    from spikeinterface.extractors import read_spikeglx

    source = write_s5_like(tmp_path / "raw", seconds=2.0)
    folder, _ = clip_recording(source, tmp_path / "clips", start_s=0.5, seconds=1.0)
    for stream_id in ("imec0.ap", "imec1.ap"):
        original = read_spikeglx(source, stream_id=stream_id)
        clip = read_spikeglx(folder, stream_id=stream_id)
        assert clip.get_num_channels() == 8
        np.testing.assert_array_equal(clip.get_channel_gains(), original.get_channel_gains())
        start = int(round(0.5 * original.get_sampling_frequency()))
        np.testing.assert_array_equal(clip.get_traces(start_frame=0, end_frame=10), original.get_traces(start_frame=start, end_frame=start + 10))
        assert clip.get_time_info()["t_start"] == pytest.approx(original.get_time_info()["t_start"] + start / original.get_sampling_frequency(), abs=1e-9)


def test_neo_cannot_open_t0_ap_beside_tcat_lf(tmp_path):
    """Why this package parses each file itself: neo 0.14 puts a tcat file in its own segment, so a folder holding t0 AP
    and CatGT's tcat LF (every NP2 probe with LF, S3 among them) fails to open. If this starts passing, neo fixed it."""
    from neo.rawio import SpikeGLXRawIO

    with pytest.raises(KeyError):
        SpikeGLXRawIO(write_s3_like(tmp_path, seconds=0.1)).parse_header()
