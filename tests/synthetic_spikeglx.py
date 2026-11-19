"""SpikeGLX .bin/.meta pairs built by hand, with no project data.

Every sample value is a known function of (sample index in the original file, channel), so a reader, a clip or an NWB
read-back can be checked exactly. Sync lines toggle at 1 Hz on the stream's own sample count.
"""

from pathlib import Path

import numpy as np

# Rates taken from real meta files (S3 and S5), so clips land on non-integer sample counts as they do in practice.
NP1_AP_RATE, NP1_LF_RATE = 30000.658158995815, 2500.0548465829847
NP2_AP_RATE, NP2_LF_RATE = 30000.36957328386, 2500.030797773655
NP2014_AP_RATE = 29999.846443514645
NIDQ_RATE, OBX_RATE = 5000.062709, 30303.030543933055

PROBES = {
    # part number, imroTbl header and entry, gain keys, voltage range and ADC maximum, as in the real files
    "NP1": dict(pn="PRB_1_4_0480_1", imro_head="(0,384)", imro_entry=lambda c: f"({c} 0 0 500 250 1)", range_max=0.6, max_int=512, extra={}),
    "NP2": dict(pn="NP2010", imro_head="(24,384)", imro_entry=lambda c: f"({c} 0 0 0 {c})", range_max=0.5, max_int=8192, extra={"imChan0apGain": "80"}),
    "NP2014": dict(pn="NP2014", imro_head="(2013,384)", imro_entry=lambda c: f"({c} 0 0 0 {c})", range_max=0.62, max_int=2048, extra={"imChan0apGain": "100"}),
}
SY_BIT = 64  # imec sync is bit 6 of the SY word
# Volts per bit of a neural channel, as SpikeGLX defines it: range / max_int / gain. NP1 LF gain is 250 here.
VOLTS_PER_BIT = {("NP1", "ap"): 0.6 / 512 / 500, ("NP1", "lf"): 0.6 / 512 / 250, ("NP2", "ap"): 0.5 / 8192 / 80, ("NP2", "lf"): 0.5 / 8192 / 80, ("NP2014", "ap"): 0.62 / 2048 / 100}


def neural_value(samples, channels):
    """The int16 value of neural channel `channels` at original sample `samples` (broadcasts)."""
    samples, channels = np.asarray(samples, dtype=np.int64), np.asarray(channels, dtype=np.int64)
    return (((samples * 7 + channels * 131) % 2001) - 1000).astype(np.int16)


def square(samples, rate, period_s=1.0, phase_s=0.0):
    """1 while the stream's own clock (sample / rate) is in the first half of each period."""
    t = np.asarray(samples, dtype=np.float64) / rate - phase_s
    return ((t % period_s) < period_s / 2).astype(np.int16)


def _write(bin_path, data, meta):
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ascontiguousarray(data, dtype=np.int16)
    data.tofile(bin_path)
    meta = {**meta, "fileSizeBytes": str(data.nbytes), "fileTimeSecs": repr(data.shape[0] / float(meta["_rate"]))}
    rate = meta.pop("_rate")
    del rate
    lines = [f"{k}={v}" for k, v in meta.items() if not k.startswith("~")] + [f"{k}={v}" for k, v in meta.items() if k.startswith("~")]
    bin_path.with_suffix(".meta").write_text("\n".join(lines) + "\n")
    return bin_path


def write_imec(run_folder, run="syn_g0", probe=0, band="ap", probe_type="NP1", n_neural=8, seconds=2.0, rate=None, first_sample=1000, catgt=False, sync=True):
    """One imec stream, `n_neural` channels plus SY, in `<run_folder>/<run>_imec<probe>/`."""
    spec = PROBES[probe_type]
    rate = rate or {("NP1", "ap"): NP1_AP_RATE, ("NP1", "lf"): NP1_LF_RATE, ("NP2", "ap"): NP2_AP_RATE, ("NP2", "lf"): NP2_LF_RATE, ("NP2014", "ap"): NP2014_AP_RATE}[(probe_type, band)]
    n = int(round(seconds * rate))
    samples = np.arange(n)[:, None]
    neural = neural_value(samples, np.arange(n_neural)[None, :])
    columns = [neural] + ([(square(samples[:, 0], rate) * SY_BIT)[:, None]] if sync else [])
    data = np.hstack(columns)
    native_lf = probe_type == "NP1" and band == "lf"
    offset = 384 if native_lf else 0
    saved = f"{offset}:{offset + n_neural - 1}" + (",768" if sync and probe_type == "NP1" else (",384" if sync else ""))
    names = [f"{'LF' if native_lf else 'AP'}{c};{c + offset}:{c + offset}" for c in range(n_neural)] + ([f"SY0;{768 if probe_type == 'NP1' else 384}:{768 if probe_type == 'NP1' else 384}"] if sync else [])
    trigger = "tcat" if catgt else "t0"
    counts = f"0,{n_neural},{int(sync)}" if native_lf else f"{n_neural},0,{int(sync)}"
    meta = {
        "_rate": rate,
        "acqApLfSy": "384,384,1" if probe_type == "NP1" else "384,0,1",
        "appVersion": "20231207",
        "fileCreateTime": "2026-07-15T11:59:50",
        "fileName": f"D:/data/{run}/{run}_imec{probe}/{run}_{trigger}.imec{probe}.{band}.bin",
        "fileSHA1": "0" * 40,
        "firstSample": str(first_sample),
        "imAiRangeMax": str(spec["range_max"]),
        "imAiRangeMin": str(-spec["range_max"]),
        "imMaxInt": str(spec["max_int"]),
        "imDatPrb_pn": spec["pn"],
        "imDatPrb_sn": f"1234567{probe}",
        "imDatPrb_slot": "2",
        "imDatPrb_port": str(probe + 1),
        "imDatPrb_dock": "1",
        "imSampRate": repr(rate),
        "nSavedChans": str(data.shape[1]),
        "snsApLfSy": counts,
        "snsSaveChanSubset": saved,
        "typeImEnabled": "2",
        "typeThis": "imec",
        **spec["extra"],
        "~imroTbl": spec["imro_head"] + "".join(spec["imro_entry"](c) for c in range(384)),
        "~snsChanMap": f"({counts.replace(',', ',', 2)})" + "".join(f"({name})" for name in names),
    }
    if catgt:
        meta["catGTCmdline0"] = f"<CatGT -dir=x -run={run.removesuffix('_g0')} -g=0 -t=0 -prb_fld -lf -prb={probe} -lffilter=biquad,2,0,500>"
    return _write(Path(run_folder) / f"{run}_imec{probe}" / f"{run}_{trigger}.imec{probe}.{band}.bin", data, meta)


def nidq_word(samples, rate):
    """Digital word as in S3: line 0 the 1 Hz sync, line 1 a trial pulse every 0.5 s, line 2 an iteration pulse every 10 ms."""
    return (square(samples, rate) | (square(samples, rate, 0.5, 0.1) << 1) | (square(samples, rate, 0.01) << 2)).astype(np.int16)


def write_nidq(run_folder, run="syn_g0", seconds=2.0, rate=NIDQ_RATE, first_sample=500):
    n = int(round(seconds * rate))
    data = nidq_word(np.arange(n), rate)[:, None]
    meta = {
        "_rate": rate,
        "fileName": f"D:/data/{run}/{run}_t0.nidq.bin",
        "fileSHA1": "0" * 40,
        "firstSample": str(first_sample),
        "nSavedChans": "1",
        "niAiRangeMax": "5",
        "niAiRangeMin": "-5",
        "niMAGain": "1",
        "niMNGain": "200",
        "niSampRate": repr(rate),
        "niXDChans1": "0:7",
        "snsMnMaXaDw": "0,0,0,1",
        "snsSaveChanSubset": "all",
        "syncNiChan": "0",
        "typeThis": "nidq",
        "~snsChanMap": "(0,0,32,0,1)(XD0;0:0)",
    }
    return _write(Path(run_folder) / f"{run}_t0.nidq.bin", data, meta)


def write_obx(run_folder, run="syn_g0", seconds=2.0, rate=OBX_RATE, first_sample=700):
    """OneBox as in S5: XA0, XA1 analog, XD0 digital word (line 0 trial, line 1 iteration), SY0 with the 1 Hz sync on bit 6."""
    n = int(round(seconds * rate))
    samples = np.arange(n)
    word = (square(samples, rate, 0.5, 0.1) | (square(samples, rate, 0.01) << 1)).astype(np.int16)
    data = np.stack([neural_value(samples, 0), neural_value(samples, 1), word, square(samples, rate) * SY_BIT], axis=1)
    meta = {
        "_rate": rate,
        "acqXaDwSy": "2,1,1",
        "fileName": f"C:/data/{run}/{run}_t0.obx0.obx.bin",
        "fileSHA1": "0" * 40,
        "firstSample": str(first_sample),
        "nSavedChans": "4",
        "obAiRangeMax": "5",
        "obAiRangeMin": "-5",
        "obMaxInt": "32768",
        "obSampRate": repr(rate),
        "snsSaveChanSubset": "all",
        "snsXaDwSy": "2,1,1",
        "typeThis": "obx",
        "~snsChanMap": "(2,1,1)(XA0;0:0)(XA1;1:1)(XD0;2:2)(SY0;3:3)",
    }
    return _write(Path(run_folder) / f"{run}_t0.obx0.obx.bin", data, meta)


def write_s3_like(root, seconds=2.0, n_neural=8):
    """Like S3: imec0 NP2 with CatGT LF, imec1 NP1 with its own LF, and a nidq stream."""
    run_folder = Path(root) / "syn_g0"
    write_imec(run_folder, probe=0, band="ap", probe_type="NP2", n_neural=n_neural, seconds=seconds, first_sample=83220203)
    write_imec(run_folder, probe=0, band="lf", probe_type="NP2", n_neural=n_neural, seconds=seconds, first_sample=6935016, catgt=True)
    write_imec(run_folder, probe=1, band="ap", probe_type="NP1", n_neural=n_neural, seconds=seconds, first_sample=83220095)
    write_imec(run_folder, probe=1, band="lf", probe_type="NP1", n_neural=n_neural, seconds=seconds, first_sample=6935007)
    write_nidq(run_folder, seconds=seconds, first_sample=13869887)
    return run_folder


def write_s5_like(root, seconds=2.0, n_neural=8):
    """Like S5: two NP2014 probes with AP only, and a OneBox stream instead of nidq."""
    run_folder = Path(root) / "syn_g0"
    for probe in (0, 1):
        write_imec(run_folder, probe=probe, band="ap", probe_type="NP2014", n_neural=n_neural, seconds=seconds, first_sample=919060)
    write_obx(run_folder, seconds=seconds, first_sample=928345)
    return run_folder
