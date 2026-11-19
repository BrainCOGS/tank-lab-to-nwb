# How behavior and spikes share a clock

An ephys session has one SpikeGLX run: a stream per Neuropixels probe (imec) and one on the NIDQ card or OneBox that
also records ViRMEn's digital outputs. ViRMEn's log has its own clocks (see [virmen-timing.md](virmen-timing.md)). The
converter follows the lab's definition of how they line up, from U19's `BehaviorSync` and this repository's
`SYNC_DATA_PLAN.md`, and offers two departures from it, each switched separately and recorded in the file.

## The rule

1. **The file's clock is the recording's.** t = 0 is the first sample of the sync file (`*nidq.bin`, or `*obx.bin` on
   a OneBox). `session_start_time` is that file's `fileCreateTime` (the rig's clock, to the second). Spikes aren't moved
   onto ViRMEn's clock; behavior is moved onto the recording's.
2. **Each iteration takes the time of its own pulse.** ViRMEn changes a digital line every iteration (bit 0 of its
   iteration counter, or one pulse per iteration) and pulses another at each trial's start. `pipeline_sync` is U19's
   `BehaviorSync.make` from those lines on, ported line for line, and gives each logged iteration a sample of the sync
   file: `iteration_idx_vector[trial][iteration]`. Its time is that sample over the sync file's rate. One for one:
   nothing is fitted, nothing is rejected, and no limit applies to how far a pulse sits from ViRMEn's logged time.
3. **Which vectors:** the regular ones when the pipeline's regular matching succeeds (`regular_sync_status` 1), the
   fixed ones when its fix path succeeds (`fixed_sync_status` 1), as `BehaviorSync.sync_data` stores them.
4. **The behavior goes onto those times** through `VirmenDataInterface`'s alignment (`virmen.interface.aligned_session`):
   every logged frame (iteration, ITI included) is replaced by its iteration's time. A time that isn't an iteration's,
   such as a trial's stop or a block's start, is placed linearly between the iterations around it, or keeps its
   distance from the first or last iteration outside them. Each trial starts at its first iteration's pulse.
5. **A spike's time is its sample over the probe's rate.**

### What the rule doesn't give, and what the converter does then

| Case | The pipeline gives | The converter |
|---|---|---|
| A saved trial past the last trial pulse (recording stopped early) | no iteration indices | stops, naming the trial |
| A trial whose synced iteration count differs from the log's | some iterations without indices | stops, naming the trial |
| The trial running when ViRMEn saved the log (`currentTrial`) | nothing: it isn't in the DB's trials | not written: it isn't in the behavior series either |
| Regular and fixed matching both fail (rec 80) | only `iteration_idx_vector_from_virmen`: each trial's start pulse plus ViRMEn's logged times, marked lower accuracy | stops, unless `allow_virmen_borrowed` (`--allow-virmen-borrowed`), which converts with those vectors and records `vectors: virmen_borrowed` |
| Sessions before 2021-06-01 (digital-word counter) | nothing: U19 refuses them | stops (rec 33) |

## The two departures

Both are recorded in `processing/ecephys/ephys_sync_alignment` (`probe_clock`, `iteration_time`, with their measured
sizes) and chosen with `convert(..., probe_clock=..., iteration_time=...)` or `--probe-clock`, `--iteration-time`.
The defaults are `PROBE_CLOCK` and `ITERATION_TIME` at the top of `tank_lab_to_nwb/ephys/clock.py`, one line each,
until the lab decides.

### `probe_clock`: drift between the probe and the sync file

| Value | Spike time | |
|---|---|---|
| `sync_edges` (default) | `map(sample / imSampRate)` | Departure. Every SpikeGLX stream records the same 1 Hz sync wave. Each probe's edges, read in 1.25 s windows every 30 s of its SY channel and paired with the sync file's, fit a least-squares continuous piecewise-linear map (knots about 300 s apart) from probe seconds to sync-file seconds. |
| `meta_rates` | `sample / imSampRate` | The rule as `SYNC_DATA_PLAN.md` states it: the calibrated rate of the AP meta file. |
| `params_rate` | `sample / params.py sample_rate` | What the DB's `CuratedClustering` stores: the nominal rate (30000) the sorter was given. |

Why `sync_edges` is the default: the calibrated rates in the meta files still disagree with each other. On S3 (rec 353,
66 minutes) the sync edges show:

| Probe | Rate difference left | Correction at the AP file's start, middle and end | Spikes in behavior put in another iteration than `meta_rates` puts them |
|---|---|---|---|
| imec0 (NP2010) | −10.38 ppm | see `A/RESULTS.md` | see `A/RESULTS.md` |
| imec1 (NP1) | −0.13 ppm | | |

A single straight line misses by up to 316 µs (imec0) and 183 µs (imec1): the rate difference wanders slowly over the
hour. After the piecewise map the edges' residual std is 57–60 µs, which is the NIDQ's 5 kHz sampling step alone
(200 µs / √12 = 58 µs), max 108–123 µs. An edge's time is taken half a sample before its first high sample, because
the level changed somewhere in the preceding sample period; without this a 5 kHz edge would read 100 µs late and a
30 kHz edge 17 µs late, and the map would carry the difference. Edges are paired by `firstSample` (a pair further than
250 ms apart is dropped; under 90 % pairing, a rate difference over 200 ppm, or an edge over 0.5 ms off the map stops
the conversion).

`params_rate` is far from both: on S3 imec1, `sample / 30000` is 87 ms from `sample / imSampRate` at the end of the file.
The DB's `CuratedClustersIteration` digitizes those spike times.

### `iteration_time`: the pipeline's index or the line's edge

| Value | Iteration time | |
|---|---|---|
| `pipeline_index` (default) | `iteration_idx_vector / rate` | The rule. |
| `edge` | `(edge - 0.5) / rate` | Departure: the edge the pipeline found the index from. |

In counter_bit0 mode the pipeline stores each later iteration's index 6 samples (1.20 ms at 5 kHz, 31 samples or
1.02 ms at a OneBox's 30.3 kHz) before the line's edge, and the first iteration's 1 sample before: it adds the trial's
start to an index found within a window that opens 1 ms after the trial pulse. On S3 this holds for all 367,223 later
iterations. `ephys_iteration_edges` keeps the pipeline's index, the edge sample and the time used, whichever is chosen.
With `edge`, an index the fix path borrowed from ViRMEn's times (no edge at it) keeps the index plus its trial's lag.

## Evidence that the rule is the pipeline's

From the raw sync file and the log alone, against each session's DB `BehaviorSync` row (all values compared exactly):

| Session | Path | Iteration indices equal | ViRMEn-borrowed equal | Trial vectors equal |
|---|---|---|---|---|
| S3, rec 353 | regular | 367,559 of 367,559 | 367,559 of 367,559 | yes |
| S2, rec 241 | regular | 195,901 of 195,901 | 195,901 of 195,901 | yes |
| S8a, rec 905 (OneBox) | fixed | 210,569 of 210,569 | 210,569 of 210,569 | yes |

More are in `A/RESULTS.md`. Every behavior frame's timestamp in the written file is checked to equal its index over
the rate (`verify_ephys`).

**The rate.** The DB stores `nidq_sampling_rate` as a MySQL float, rounded to 5000.06; the meta file has 5000.062709,
which is what the pipeline divided by when it computed its vectors. The converter uses the meta file's rate. Over S3's
66 minutes the two differ by up to 2.1 ms at the last iteration.

**CuratedClustersIteration** (U19 `ephys_sync`) assigns spikes to iterations with
`np.digitize(spike_times, iteration_times)`, where the iteration times come from the per-sample expansion of the
vectors (`get_full_vector_samples`) and its transitions by `np.diff`: each is one sample before the index, and the
session's first iteration is dropped. Its spike times are `sample / params.py sample_rate`, and it divides by the DB's
rounded rate. `pipeline_sync.curated_clusters_iteration_times` reproduces those times; how many spikes it puts in the
same iteration as this file is in `A/RESULTS.md`.

## The sorting's sample 0

`spike_times.npy` counts samples of whatever binary the sorter read; the converter takes it as the raw `t0` AP file's
sample 0 only when the sorter's files show it. Kilosort 4 names its input in `kilosort4.log` and `ops.npy`, and
`ops.npy` gives its length to within one batch; `temp_wh.dat` (read with stat only) holds whole padded batches. That
length must contain the raw file's sample count, and no spike may lie past its end. CatGT's `tcat` AP file starts at
the raw file's sample 0: its DB parameters trim nothing, and on S3 the CatGT sorts (jobs 728, 729) and the raw sorts
(1053, 1054) put spikes at the same samples (nearest-spike offset peaks at 0 with ±1–2 samples of detection jitter,
99.99 % within 2 ms).

## How far to trust it

| Source | Size | Seen in the file as |
|---|---|---|
| Iteration time | the sync file's sample period (0.2 ms at 5 kHz), plus 1.2 ms early under `pipeline_index` | `pipeline.index_lag_seconds` |
| Delay from ViRMEn's logged time to its digital output | not measured; under the rule it doesn't matter for spike alignment, because iterations are timed by their output | `ephys_iteration_edges` against ViRMEn's own times |
| Probe drift, `meta_rates` | up to tens of ms per hour (S3 imec0: 41 ms) | `<probe>.sync_correction_ms` |
| Probe drift, `sync_edges` | about 20 µs (60 µs edge scatter averaged over each segment) | `<probe>.map_residual_*` |
| When the recording started on the wall clock | the rig's clock, to the second | `sync.file_create_time` |

## What the file records

`processing/ecephys` holds `ephys_sync_alignment` (the rule, both choices, the pipeline's status and lag, each probe's
drift and correction, and how many spikes change iteration under the chosen `probe_clock`), `ephys_iteration_edges`
(each iteration: pipeline index, edge sample, edge time, the time used), `ephys_sync_maps` (each probe's map) and a
`sorting_<probe>` table per probe (sorter, version, input, why sample 0 is the raw file's, cluster tables read).

## The libraries, and the one direct read

neo, SpikeInterface and neuroconv do the reading: the NIDQ or OneBox digital word and each SY word come from
SpikeInterface's SpikeGLX reader, a chunk at a time (neo reads every channel of a block before picking one, which took
a 14-channel OneBox file to 3.4 GB), and the geometry, devices, groups and electrodes come from neuroconv and
probeinterface. Two library limits shape how:

- neo 0.14.6 can't open a folder holding a `t0` AP file beside a CatGT `tcat` LF file (`KeyError: (0, 'imec0.ap')`),
  337 of 704 raw folders. Each stream is read from a folder of relative links to just its `.bin` and `.meta`.
- SpikeInterface 0.105.4's Kilosort extractor drops the spikes of clusters missing from an inner merge of the tsv
  files, so trains are split from `spike_times.npy` with numpy. A test shows both give identical trains wherever the
  extractor keeps every spike.

`spikeglx.py` reads the same channels directly from the file's bytes. The conversion compares the two reads on the
first and last million samples, and the tests compare them on synthetic files.
