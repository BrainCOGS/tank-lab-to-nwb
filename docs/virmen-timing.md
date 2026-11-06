# How ViRMEn times become NWB times

A ViRMEn log has two clocks:

- **Wall clock:** `session.start`, `session.end` and `block.start`, read with MATLAB `clock`.
- **Elapsed clock:** `trial.start`, `trial.time` and `block.duration`, in seconds since the engine's timer started.

NWB needs one clock, so the converter adds an offset to the elapsed clock. The result is seconds since `session.start`.

## The rule

A **clock run** is a stretch of blocks whose elapsed clock keeps counting. One offset fits a whole run. A new run starts
only where a block's first trial starts before the previous trial ended: ViRMEn restarted and continued the same log.

Each run gets one offset, found in this order:

| Anchor (`derived.clock_anchor`) | When | Offset | Accuracy |
|---|---|---|---|
| `block_stamp` | Normal case | first block's stamp − its first `trial.start` | Stamp is 0–50 ms late |
| `engine_timestamps` | First block stamped during setup, and the fields exist | `initialTimestamp` + `timeElapsedFirstTrial` | Exact, to a few ms |
| `session_end_upper_bound` | First block stamped during setup, no fields | `session.end` − elapsed reading at the end | Upper bound; see `derived.clock_max_error` |

**Setup stamps.** The elapsed clock starts after the experiment's initialization code, but `ExperimentLog` creates its
first block during that code. Most experiments replace that block at the first trial, which pairs the stamp with the
trial. If the block is kept, its stamp predates the elapsed clock. The converter detects this because the stamp would put
the clock's zero before the session started, which can't happen.

`session.end` is read after the session's shutdown code, so it can only be late. That makes this anchor an upper bound.

**Epochs** use the same offset as trials, so every trial sits inside its epoch. Each block's raw stamp stays in the
`wall_clock_start` column. The raw elapsed clock is the stored time minus `derived.clock_offset`.

## Why not anchor every block on its own stamp?

Stamps are 0–50 ms late, and the delay varies. Per-block anchoring would make neighboring trials overlap by up to 34 ms at
53 of 161 block boundaries. One offset per run avoids that and keeps intervals exact, but leaves clock drift uncorrected:
up to 0.17 s per hour, and 5 ppm or less on most rigs. Which clock is closer to true time is unresolved.

## What the file tells you

- `derived.clock_offset`, `derived.clock_anchor` and `derived.clock_max_error` are per-block settings rows.
- `derived.clock_session_end_minus_block_end` compares the last block's end to `session.end`.
- The coverage report's `clock.needs_review` is true when a file uses the upper bound, or when that comparison is over 1 s.

Any file with `needs_review` true has times that a person should check against other evidence.

## Alignment

After alignment to another clock, such as ephys iteration pulses, those timestamps replace this mapping. Times outside the
frame range keep their distance from the nearest frame.
