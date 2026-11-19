# Raw SpikeGLX data in the NWB file

How `tank_lab_to_nwb/ephys/` writes the AP and LF bands and the sync lines. The clock and the electrode table belong to
the units conversion (`docs/ephys-timing.md`); this code takes both as arguments.

## What is written where

| Source | NWB location | Why there |
|---|---|---|
| AP band, `t0` | `acquisition/ElectricalSeriesAPImec<N>` | recorded data |
| LF band recorded by SpikeGLX (NP1) | `acquisition/ElectricalSeriesLFImec<N>` | recorded data |
| LF written by CatGT from AP (NP2) | `processing/ecephys/LFP/ElectricalSeriesLFImec<N>` | computed from AP, so processed; `filtering` holds the CatGT command |
| nidq / OneBox digital word, OneBox SY | `processing/ecephys/spikeglx_sync_edges` | the evidence the clock was fitted to |
| imec SY channel | not written as a neural channel | it is a bit field, not a voltage |

- Data stay int16 as SpikeGLX saved them. `conversion` is SpikeGLX's volts per bit (`imAiRangeMax / imMaxInt / gain`,
  with NP1 gains from `imroTbl`), taken from neo's per-file parser; per-channel gains that differ go in
  `channel_conversion`.
- Each series references the caller's electrode rows (`electrodes`: a `DynamicTableRegion` or the row indices), one per
  neural channel in file order, so AP, LF and units share one set of rows.
- NP2 probes without an LF file get no LF series: deriving one means filtering 83 GB per probe-hour, which is a
  processing decision and not part of this conversion.

## Time

The caller passes `clock(samples) -> seconds on the behavior clock`, where samples count from the start of the
**original** `.bin` (a clip adds its `clipStartSample`). `series_timing` checks the clock at points 1 s apart (at least
1,000 points): if it stays within 0.1 sample of the straight line through its first and last sample, the series gets
`starting_time` and `rate` (the fitted rate, not SpikeGLX's); otherwise explicit timestamps, computed and compressed a
block at a time.

| Storage | Exact when | Cost per probe-hour at 30 kHz |
|---|---|---|
| `starting_time` + SpikeGLX's `imSampRate` | never on the behavior clock | 0, but a 20 ppm rate difference is 72 ms off after an hour (50 ppm: 180 ms) |
| `starting_time` + fitted rate | the clock is linear (one fitted line per stream) | 0 |
| explicit float64 `timestamps` | always | 864 MB raw; 34 MB with gzip 4 + shuffle (measured on a linear 60 s ramp, `B/bench/timestamp_cost.txt`) |

The imaging conversion fits one line per session and writes per-frame timestamps because frames arrive irregularly;
ephys samples are regular, so a linear clock gives `rate` with no loss, and a non-linear one (piecewise per sync edge)
still costs only about 0.04% of the compressed AP.

## Sync edges

`spikeglx_sync_edges` has one row per change of each bit of the nidq or OneBox digital word (`XD0`) and the OneBox
`SY0` channel: `stream`, `channel`, `bit`, `level` (after the edge), `sample` (original file numbering, the stream's own
clock) and `time` (behavior clock). The first row per bit is its level at the first sample written, so each line can
be rebuilt exactly. On S3 a minute holds about 12,000 edges (mostly the 6-bit iteration counter on lines 2–7), so about
0.7 M rows per hour, a few MB: the full nidq trace is 36 MB per hour and the OneBox's 0.9 GB, mostly the unused analog
`XA` channels, which aren't written.

## Memory and chunking

Data are read from a memory map one buffer at a time (`BUFFER_BYTES`, 256 MB by default, spanning all channels so each
read is contiguous in the file), compressed chunk by chunk and written; memory does not grow with the file. Chunks are
1 s × 64 channels (3.8 MB at 30 kHz). Compression is gzip, the one filter built into every HDF5 library, so the files
open without plugins. The numbers behind these choices are in the phase-1 measurements (`B/MEASUREMENTS.md`).

## Layouts

`find_recording` accepts one `t0` per run, one gate, a `t0` AP per probe, at most one LF per probe (`t0` or CatGT
`tcat`) and at most one nidq or OneBox stream. It refuses `.cbin`, a `.bin` without `.meta`, a size that disagrees with
`fileSizeBytes`, CatGT AP, extra triggers or gates, and two sync streams, each with a message saying why.

neo 0.14 (and so SpikeInterface's `read_spikeglx` and neuroconv's SpikeGLX interfaces) cannot open a folder holding
`t0` AP beside `tcat` LF, which is every NP2 recording with LF; this package reads each file on its own for that reason.

## Clips

`python -m tank_lab_to_nwb.ephys.clip <run folder> <out> --start 600 --seconds 60` copies one window of every stream by
seeking, and rewrites `fileSizeBytes`, `fileTimeSecs`, `fileSHA1` and `firstSample`, adding `clipStartSample` and
`clipSourceFile`. Streams are cut at `round(start × rate)` of their own samples, so clips line up to within the streams'
rate-calibration difference (6.5 ms between S3's two probes at 600 s); the exact relation comes from the sync edges.
