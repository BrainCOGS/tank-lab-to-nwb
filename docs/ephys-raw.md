# Raw SpikeGLX data in the NWB file

`python -m tank_lab_to_nwb.ephys.convert_session ... [--raw-bands ap lf] [--raw-folder <clips>] [--raw-max-seconds S]`
writes the units file (`docs/ephys-timing.md`) with raw data beside it: LF and the sync edges by default (the file is
then `<name>_ecephys.nwb`), AP only when asked, nothing raw with an empty `--raw-bands` and `--no-sync-edges`.

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

The caller passes `clock(samples) -> seconds on the file's clock`, where samples count from the start of the
**original** `.bin` (a clip adds its `clipStartSample`). The units conversion defines that clock (`docs/ephys-timing.md`:
recording time, each probe mapped onto the sync stream by its 1 Hz edges). `series_timing` checks the clock at points
1 s apart (at least 1,000 points): if it stays within 0.1 sample of the straight line through its first and last
sample, the series gets `starting_time` and `rate` (the fitted rate, not SpikeGLX's); otherwise explicit timestamps,
computed and compressed a block at a time.

On the recording clock a probe's series is nearly a line: what remains is the drift between the probe and the sync
stream, which a single line misses by up to about 0.3 ms over S3's hour (measured by subagent A). That is 9 AP samples or
0.75 LF samples, so a `rate` series would put the raw trace up to 9 samples away from where the units' spike times say
a spike is. The rule above therefore writes timestamps for such a probe, which costs very little:

| Storage | Error on S3 over an hour | Cost per probe-hour at 30 kHz (2.5 kHz LF) |
|---|---|---|
| `starting_time` + SpikeGLX's `imSampRate` | 41.5 ms on imec0, 0.57 ms on imec1 (A's drift numbers) | 0 |
| `starting_time` + fitted rate | up to about 0.3 ms (9 AP samples) | 0 |
| explicit float64 `timestamps` | none (the clock itself) | 864 MB raw; 34 MB with gzip 4 + shuffle (2.9 MB for LF), which is 0.07% of the compressed AP (47 GB) |

The timestamp cost was measured on a linear 60 s ramp (`B/bench/timestamp_cost.txt`); a ramp bent by the drift
compresses the same way, because shuffle keeps the slowly changing high bytes together.

### LF on the AP's clock

The clock maps AP samples. An LF sample goes through the AP sample taken at the same instant, so LF, AP and spikes
share one map. That sample was measured on S3's SY edges (60 per probe), and it differs by where the LF came from:

- SpikeGLX's own LF (NP1) counts on the AP counter divided by 12: LF sample k is AP sample
  `12 × (LF firstSample + k) − AP firstSample`. On S3, every LF edge then lies 0–11 AP samples after its AP edge, as
  one LF period of quantization allows.
- CatGT's `tcat` LF keeps every 12th sample of the AP file: LF sample k is AP sample `12 k`. Its `firstSample` is the
  AP's divided by 12 and rounded down; using it would put the LF 11 AP samples (0.37 ms) early on S3.

Timing LF by `k / LF imSampRate` from the file start is right for CatGT LF, but 11 AP samples (0.37 ms) late for NP1.

## Sync edges

`spikeglx_sync_edges` has one row per change of each bit of the nidq or OneBox digital word (`XD0`) and the OneBox
`SY0` channel: `stream`, `channel`, `bit`, `level` (after the edge), `sample` (original file numbering, the stream's own
clock) and `time` (the file's clock). The first row per bit is its level at the first sample written, so each line can
be rebuilt exactly. On S3 a minute holds about 12,000 edges (mostly the 6-bit iteration counter on lines 2–7), so about
0.7 M rows per hour, a few MB: the full nidq trace is 36 MB per hour and the OneBox's 0.9 GB, mostly the unused analog
`XA` channels, which aren't written.

## Read-back check

After writing, `convert` reads the file's HDF5 directly (`raw.verify_raw_ephys`) and stops with `RoundTripError` if it
differs from the SpikeGLX files:

- each raw series: shape and dtype, its electrode rows against the units' rows, and at its start, middle and end (1 s
  each) every stored sample against the file's bytes and every time against the clock (exactly for timestamps, within
  the 0.1-sample tolerance for `starting_time` + `rate`);
- the sync edges: every row's time against the clock, and the digital word rebuilt from the rows alone against the
  file's bytes over the first and last million samples.

The whole of an AP file isn't reread: the writer copies blocks unchanged, so a fault would be in where blocks land,
which three windows across the file show. The counts go in the report under `verified.raw`.

## Memory and chunking

Data are read with positioned reads one buffer at a time (`BUFFER_BYTES`, 64 MB by default, spanning all channels so
each read is contiguous in the file), and the pages read are dropped from the page cache, which counts against a
cgroup's memory. Then they are compressed chunk by chunk and written. Memory does not grow with the file: writing a
60 s NP2 AP clip to NWB peaked at 242 MB RSS with 64 MB buffers (643 MB with 256 MB, at the same speed). Chunks are
1 s × 64 channels (3.8 MB at 30 kHz). Compression is gzip level 1 with the shuffle filter: gzip is the one filter built into every HDF5
library, so the files open without plugins, and shuffle makes it 1.2–1.3 times smaller and faster on NP2 AP. The numbers behind these choices are in the phase-1 measurements (`B/MEASUREMENTS.md`).

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
