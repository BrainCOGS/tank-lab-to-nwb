"""Synthetic SpikeGLX sync lines and sorter output folders with known answers. No network, no DB, no raw data."""

import json
from pathlib import Path

import numpy as np

NIDQ_RATE = 5000.062709
IMEC_RATE = 30000.36957328386


def iteration_schedule(n_trials=12, iterations=(180, 260), period=0.012, gap=1.5, seed=0):
    """Per trial (start, per-iteration times) on ViRMEn's elapsed clock, float32-rounded like a saved log."""
    rng = np.random.default_rng(seed)
    trials, start = [], 3.0
    for _ in range(n_trials):
        n = int(rng.integers(*iterations))
        steps = period + rng.uniform(-0.002, 0.004, n)
        times = np.concatenate(([0.0], np.cumsum(steps[:-1]))).astype(np.float32)
        trials.append((float(np.float32(start)), times))
        start += float(times[-1]) + period + gap
    return trials


def sync_word(trials, n_samples, rate=NIDQ_RATE, slope=1.0 + 4e-6, intercept=12.345, skip=(), sync_line=0, sync_phase=0.3, latency=0.0, trial_line=1, counter_lines=(2, 3, 4, 5, 6, 7)):
    """A NIDQ digital word: the 1 Hz sync wave, a 5 ms trial pulse, and ViRMEn's iteration counter on its lines.

    Sync-stream seconds = slope * elapsed + intercept (+ latency). The counter restarts at 1 with each trial, as on the
    real rigs, and holds its value between trials. An iteration in `skip`, given as (trial index, iteration index), sends
    nothing, so the counter jumps over it. Returns (word, true edge sample per trial and iteration, -1 if skipped).
    """
    word = np.zeros(n_samples, dtype=np.uint16)
    seconds = np.arange(n_samples) / rate
    word |= ((((seconds - sync_phase) % 1.0) < 0.5).astype(np.uint16) << sync_line)
    counter = np.zeros(n_samples, dtype=np.int64)
    edges = []
    skip = set(skip)
    for t, (start, times) in enumerate(trials):
        at = np.ceil((slope * (start + times.astype("float64")) + intercept + latency) * rate).astype(np.int64)  # the first sample at or after the change
        trial_edges = np.where([(t, i) not in skip for i in range(times.size)], at, -1)
        edges.append(trial_edges)
        word[at[0] : at[0] + int(0.005 * rate)] |= np.uint16(1 << trial_line)
        for i, sample in enumerate(at):
            if (t, i) not in skip:
                counter[sample:] = i + 1
    for bit, line in enumerate(counter_lines):
        word |= (((counter >> bit) & 1).astype(np.uint16) << line)
    return word, edges


def write_spikeglx(folder, name, columns, meta):
    """Write int16 columns as a SpikeGLX .bin with a .meta holding `meta` plus the size fields."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    data = np.ascontiguousarray(np.stack(columns, axis=1).astype("<i2"))
    path = folder / f"{name}.bin"
    data.tofile(path)
    rate = float(meta.get("niSampRate") or meta.get("imSampRate") or meta.get("obSampRate"))
    full = dict(meta, nSavedChans=str(data.shape[1]), fileSizeBytes=str(data.nbytes), fileTimeSecs=repr(data.shape[0] / rate))
    path.with_suffix(".meta").write_text("".join(f"{key}={value}\n" for key, value in full.items()))
    return path


def nidq_meta(rate=NIDQ_RATE, first_sample=1000):
    """The keys of a real NIDQ meta file (S3's, 2024), with this file's rate and start."""
    return {
        "acqMnMaXaDw": "0,0,0,1", "appVersion": "20231207", "fileCreateTime": "2024-07-27T13:59:44", "fileName": "D:/run_g0/run_g0_t0.nidq.bin",
        "firstSample": str(first_sample), "gateMode": "Immediate", "nSavedChans": "1", "niAiRangeMax": "5", "niAiRangeMin": "-5", "niAiTermination": "Default",
        "niClockLine1": "Internal", "niClockSource": "PXI1Slot4_13ch_Int : 5000.000000", "niDev1": "PXI1Slot4", "niDev1ProductName": "PXIe-6341",
        "niMAChans1": "", "niMAGain": "1", "niMNChans1": "", "niMNGain": "200", "niMaxInt": "32768", "niMuxFactor": "32", "niSampRate": repr(rate),
        "niStartEnable": "false", "niXAChans1": "", "niXDBytes1": "1", "niXDChans1": "0:7", "snsMnMaXaDw": "0,0,0,1", "snsSaveChanSubset": "all",
        "syncNiChan": "0", "syncNiChanType": "0", "syncNiThresh": "1.1", "syncSourceIdx": "3", "syncSourcePeriod": "1", "trigMode": "Immediate",
        "typeImEnabled": "2", "typeNiEnabled": "1", "typeObEnabled": "0", "typeThis": "nidq", "userNotes": "", "~snsChanMap": "(0,0,32,0,1)(XD0;0:0)", "~snsShankMap": "(1,2,0)",
    }


def imec_sync_column(n_samples, rate, sync_to_probe, sync_phase=0.3):
    """An imec SY word: bit 6 high during the first half of each sync second, placed by `sync_to_probe` (sync s -> probe s)."""
    probe_seconds = np.arange(n_samples) / rate
    # Invert the probe's map numerically: sync seconds of each probe sample.
    grid = np.linspace(-5, probe_seconds[-1] + 5, 20001)
    sync_seconds = np.interp(probe_seconds, sync_to_probe(grid), grid)
    return ((((sync_seconds - sync_phase) % 1.0) < 0.5).astype(np.int16) << 6)


# ---- sorter folders ----


def _tsv(path, header, rows):
    path.write_text("\t".join(header) + "\n" + "".join("\t".join(str(v) for v in row) + "\n" for row in rows))


def phy_folder(folder, kind="ks4", n_units=5, n_channels=8, rate=30000.0, seed=1, n_samples=300_000, curated=False, with_cluster_info=None, with_pc_features=True, spikeinterface=False, ibl=False):
    """A fake Kilosort/phy folder of the given kind. Returns (folder, expected per-unit spike samples, cluster ids).

    kind: "ks2" (SpikeInterface Kilosort 2.0.2 layout), "ks3" (MATLAB Kilosort 3, `kilosort_output`), "ks4" (python Kilosort 4).
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cluster_ids = np.arange(n_units) * 2  # gaps, as after merges in phy
    trains = {int(c): np.sort(rng.choice(np.arange(10, n_samples), size=int(rng.integers(20, 200)), replace=False)).astype(np.uint64) for c in cluster_ids}
    times = np.concatenate(list(trains.values()))
    clusters = np.concatenate([np.full(len(v), c, dtype=np.int32) for c, v in trains.items()])
    order = np.argsort(times, kind="stable")
    np.save(folder / "spike_times.npy", times[order].reshape(-1, 1) if kind == "ks3" else times[order])
    np.save(folder / "spike_clusters.npy", clusters[order])
    np.save(folder / "spike_templates.npy", clusters[order].astype(np.uint32))
    np.save(folder / "amplitudes.npy", rng.uniform(5, 50, times.size).astype(np.float32))
    np.save(folder / "channel_map.npy", np.arange(n_channels, dtype=np.int32))
    positions = np.stack([np.tile([43.0, 11.0, 59.0, 27.0], n_channels // 4 + 1)[:n_channels], 20.0 * (np.arange(n_channels) // 2)], axis=1)
    np.save(folder / "channel_positions.npy", positions)
    np.save(folder / "templates.npy", rng.normal(size=(cluster_ids.max() + 1, 61 if kind == "ks4" else 82, n_channels)).astype(np.float32))
    np.save(folder / "whitening_mat_inv.npy", np.eye(n_channels))
    if with_pc_features:
        np.save(folder / "pc_features.npy", np.zeros((times.size, 3, 4), dtype=np.float32))
        np.save(folder / "pc_feature_ind.npy", np.zeros((cluster_ids.max() + 1, 4), dtype=np.uint32))
    labels = ["good", "mua", "noise", "good", "mua", "good", "mua"]
    ks = {int(c): labels[i % len(labels)] for i, c in enumerate(cluster_ids)}
    _tsv(folder / "cluster_KSLabel.tsv", ["cluster_id", "KSLabel"], [(c, ks[c]) for c in ks])
    _tsv(folder / "cluster_Amplitude.tsv", ["cluster_id", "Amplitude"], [(c, f"{10 + c:.1f}") for c in ks])
    _tsv(folder / "cluster_ContamPct.tsv", ["cluster_id", "ContamPct"], [(c, f"{c * 1.5:.1f}") for c in ks])
    # phy keeps cluster_group.tsv and cluster_info.tsv's group column the same; curated, a person called the first cluster noise.
    group = {c: ("noise" if (curated and i == 0) else ks[c]) for i, c in enumerate(ks)}
    _tsv(folder / "cluster_group.tsv", ["cluster_id", "group"], [(c, group[c]) for c in ks])
    if with_cluster_info:
        counts = {c: len(trains[c]) for c in ks}
        if with_cluster_info == "stale":
            counts[next(iter(counts))] += 1
        _tsv(folder / "cluster_info.tsv", ["cluster_id", "Amplitude", "ContamPct", "KSLabel", "amp", "ch", "depth", "fr", "group", "n_spikes", "sh"], [(c, 10 + c, c * 1.5, ks[c], 50.5, c % n_channels, 20.0 * (c % n_channels // 2), 1.25, group[c], counts[c], 0) for c in ks])
    dat = {"ks2": "recording.dat", "ks3": "/mnt/cup/x/temp_wh.dat", "ks4": "/mnt/cup/x/temp_wh.dat"}[kind]
    (folder / "params.py").write_text(f"dat_path = '{dat}'\nn_channels_dat = {n_channels + (0 if kind == 'ks2' else 1)}\ndtype = 'int16'\noffset = 0\nsample_rate = {rate!r}\nhp_filtered = {kind != 'ks2'}\n")
    if kind == "ks4":
        (folder / "kilosort4.log").write_text("2026-01-08 18:41:06,302 kilosort.run_kilosort INFO     Kilosort version 4.0.37\n2026-01-08 18:41:09,695 kilosort.run_kilosort INFO     Sorting [PosixPath('/mnt/cup/raw/x_t0.imec0.ap.bin')]\n")
    if kind == "ks3":
        (folder / "rez.mat").write_bytes(b"")
    if spikeinterface or kind == "ks2":
        (folder / "sorter_params.json").write_text("{}")
        (folder / "sorter_provenance.json").write_text(json.dumps({"sorter_version": "kilosort2@2.0.2", "sorter": "kilosort2", "mode": "container", "spikeinterface": "0.105.0"} if kind == "ks2" else {"sorter_version": "kilosort4@4.0.37", "sorter": "kilosort4", "mode": "native", "installed_version": "4.0.37"}))
        si = folder / "spikeinterface"
        si.mkdir(exist_ok=True)
        (si / "spikeinterface_log.json").write_text(json.dumps({"sorter_name": "kilosort2" if kind == "ks2" else "kilosort4", "error": False, "run_time": 12.5}))
        (si / "quality_metrics.csv").write_text("unit_id,firing_rate,isi_violations_ratio,snr\n" + "".join(f"{c},{1.0 + c},{0.01 * c},{5.0 + c}\n" for c in ks))
    if ibl:
        ibl_dir = folder.parent / "ibl_data"
        ibl_dir.mkdir(exist_ok=True)
        np.save(ibl_dir / "spikes.samples.npy", times[order].astype(np.int64))
        (ibl_dir / "cluster_metrics.csv").write_text("cluster_id,label,noise_cutoff,ks2_label\n" + "".join(f"{c},{0.5 + c / 100},{c * 0.1},{ks[c]}\n" for c in ks))
    return folder, trains, cluster_ids


DATA = Path(__file__).parent / "data" / "ephys"


def imec_files(session_folder, probe, meta_name="np1.ap.meta", n_samples=3000, sync=None, run="run_g0", with_tcat_lf=False, first_sample=1000, rate=None):
    """A probe folder as SpikeGLX writes it: `<run>_imec<N>/<run>_t0.imec<N>.ap.bin` with a real meta file's keys.

    `sync` is the SY column (int16, n_samples long), zero if None. `with_tcat_lf` adds a CatGT `tcat` LF file beside it,
    the layout neo 0.14.6 can't parse (S3's imec0).
    """
    folder = Path(session_folder) / f"{run}_imec{probe}"
    meta = dict(line.split("=", 1) for line in (DATA / meta_name).read_text().splitlines() if "=" in line)
    meta.pop("fileSHA1", None)
    meta["firstSample"] = str(first_sample)
    if rate is not None:
        meta["imSampRate"] = repr(rate)
    # neo names the stream from the meta's fileName, the name SpikeGLX gave the file.
    meta["fileName"] = f"D:/{run}/{run}_imec{probe}/{run}_t0.imec{probe}.ap.bin"
    columns = [np.zeros(n_samples, dtype=np.int16)] * 384 + [np.zeros(n_samples, dtype=np.int16) if sync is None else sync.astype(np.int16)]
    path = write_spikeglx(folder, f"{run}_t0.imec{probe}.ap", columns, meta)
    if with_tcat_lf:
        lf = dict(line.split("=", 1) for line in (DATA / "np2010_tcat.lf.meta").read_text().splitlines() if "=" in line)
        lf.pop("fileSHA1", None)
        lf["fileName"] = f"D:/{run}/{run}_imec{probe}/{run}_tcat.imec{probe}.lf.bin"
        write_spikeglx(folder, f"{run}_tcat.imec{probe}.lf", [np.zeros(n_samples // 12, dtype=np.int16)] * 385, lf)
    return path
