# How imaging frames get onto the behavior clock

Each ScanImage frame has its own timestamp, on ScanImage's clock. ViRMEn's log has its own elapsed clock (see
[virmen-timing.md](virmen-timing.md)). The two computers share no clock signal, but ViRMEn tells ScanImage what it's doing.

## The link

In each iteration, `updateDAQSyncSignals` sends `[block, trial, iteration]` over I2C. ScanImage stamps each packet on its own
clock and stores it in the header of the frame being scanned:

```
I2CData = {{12.807774195, [1,0,1,0,1,0]} }
```

The six bytes are three little-endian uint16 values. So each packet is a pair of times for one event: the arrival on
ScanImage's clock, and the iteration's logged time on the elapsed clock, `trial.start + trial.time(iteration)`.

## The rule

One straight line per acquisition, from elapsed time to ScanImage time:

1. Match every packet to a logged iteration. A packet that isn't in the log stops the conversion: it's probably the wrong log.
2. Leave out each trial's first iteration. Trial setup runs between logging its time and sending its packet.
3. Fit by least squares. Reject packets further from the median residual than max(5 x MAD, 5 ms), with MAD scaled by 1.4826, refit,
   and repeat until the kept set stops changing. This removes the one packet per trial that end-of-trial work delays by hundreds of
   milliseconds.
4. A frame's time is then `(frameTimestamps_sec - intercept) / slope + clock_offset`.

The conversion stops if the clocks differ by over 100 ppm, over 5% of packets are rejected, the 1st or 99th percentile of the
residuals is beyond 10 ms, there are fewer than 100 packets, or ViRMEn's clock restarted during the session.

The packets stop when ViRMEn stops, and imaging usually runs longer on both ends, so frames outside the packets are extrapolated
along the same line.

## The trial missing from the log

ViRMEn saves the log with the running trial still unfinished, so its packets have no row in `log.block.trial`. It's in
`log.currentTrial`, with `log.currentIt` saying how many iterations are real, so those packets can be matched too. Its
per-iteration data goes in `processing/in_progress_trial` at full precision, because finished trials are rounded to single
precision and this one isn't. It has no row in the trials table.

## Time zero

Imaging usually starts before ViRMEn's `session.start`. NWB times shouldn't be negative, so `session_start_time` moves back to
the first frame, to the microsecond, and every behavior time moves forward by the same amount. The settings table records this
as `derived.session_start_shift_seconds` and `derived.virmen_session_start`. `derived.clock_offset` includes the shift.

## How far to trust it

| Source of error | Size | Seen in the file as |
|---|---|---|
| Packet scatter | about 1 ms. This is how far packets sit from the line, not frame-time error | `fit.kept_std_ms`, `fit.residual_*` |
| Clock rate difference | about 5 ppm, corrected by the slope | `fit.slope` |
| Delay from the logged time to the packet's arrival | A constant. Not measured, and the ViRMEn source sets no upper bound. Frame times run early by it | `fit.send_latency` |
| Time within a frame | A timestamp marks the frame's start, and scanning it takes up to one frame period | `fit.frame_time_meaning` |
| Display delay | The logged time and the packet come before the display updates, by an unmeasured amount | `fit.frame_time_meaning` |
| When the session started | Inherited from the behavior clock, and can be an upper bound. Moves frames and trials together, so it doesn't change their alignment | `derived.clock_anchor` |

Wall clocks aren't used. ScanImage's `epoch` and ViRMEn's clock came from different computers, and on the sessions measured they
disagreed by 0.15 to 0.96 s, with the sign changing between days.

## Relation to the pipeline

The lab's pipeline, `SyncImagingBehavior` in U19-pipeline-matlab, stores labels rather than a clock: the block, trial and iteration
of each frame's first packet, repeated across interior frames with no packet, and cleared for the trial missing from the log.
`processing/scanimage/frames` holds these as `behavior_block`, `behavior_trial` and `behavior_iteration`, so its results can be
compared.

Those labels say which header received a packet, not when the iteration happened. At a trial's onset, the first labelled frame can
have started just before the trial or run up to a frame late. To find a trial's frames, compare the frame timestamps with the trial's
start and stop times.

## What the converter refuses for now

More than one saved channel, averaged frames, saved or discarded flyback frames, several ROIs in one page, missing frames or files,
and several acquisitions in one file. Each stops with an error, because guessing would misplace frames.

Files missing from the end of an acquisition can't be seen from their numbering, so two more checks catch them. Any file of the
acquisition beside the given ones but not listed stops the conversion. So does a last file holding a full `logFramesPerFile` pages
when fewer than `numVolumes` volumes are present, because later files may exist. ScanImage's `endOfAcquisition` reads 0 on the last
frame of the data seen so far, so it can't settle this. An acquisition that really does end exactly on a file boundary without
reaching `numVolumes` is refused too, and would need a person to check it.

## Files for one session

A session's raw and processed files are separate NWB files in one dandiset, told apart by name:
`sub-<subject>_ses-<session>_desc-rawtiff_behavior+ophys.nwb` and `..._desc-suite2p_behavior+ophys.nwb`. Pass `description` to
`organize_for_dandi` to name them. They share a `subject_id`, a `session_id`, and `acquisition.id` in
`processing/scanimage/alignment`, a fingerprint of the acquisition itself.
