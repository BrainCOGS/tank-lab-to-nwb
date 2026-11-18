"""Wall-clock times of a session, and the offset from ViRMEn's elapsed clock onto them.

Python subtracts two datetimes that share a tzinfo as wall clocks, which is wrong across a DST change, so all
differences go through UTC.
"""

import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

EASTERN = ZoneInfo("America/New_York")


# How a clock run's offset was found.
BLOCK_STAMP = "block_stamp"
ENGINE_TIMESTAMPS = "engine_timestamps"
SESSION_END = "session_end_upper_bound"


class ClockAnchorError(ValueError):
    """Nothing in the log can tie the elapsed clock to the wall clock."""


@dataclass(frozen=True)
class ClockMap:
    """Per block: seconds to add to the elapsed clock, its reading when the block began, and how the offset was found.

    `max_errors` is how far too late the offset could be, or None if the anchor isn't a bound.
    """

    offsets: list
    block_start_readings: list
    anchors: list
    max_errors: list


class AmbiguousTimeWarning(UserWarning):
    """A wall-clock time is repeated or skipped by a DST change, so its true instant is uncertain."""


def datevec_to_datetime(datevec, timezone_info=EASTERN):
    """Convert a MATLAB [year month day hour minute second] vector. Seconds keep microsecond resolution."""
    values = np.asarray(datevec, dtype="float64").ravel()
    if values.size != 6 or not np.all(np.isfinite(values)):
        raise ValueError(f"not a datevec: {values.tolist()}")
    year, month, day, hour, minute = (int(v) for v in values[:5])
    naive = datetime(year, month, day, hour, minute) + timedelta(seconds=float(values[5]))
    aware = naive.replace(tzinfo=timezone_info)
    if aware.utcoffset() != aware.replace(fold=1).utcoffset():
        warnings.warn(f"{naive.isoformat()} is repeated or skipped by a DST change", AmbiguousTimeWarning, stacklevel=2)
    return aware


def seconds_between(start, end):
    """Real elapsed seconds from `start` to `end`, across any DST change."""
    return (end.astimezone(timezone.utc) - start.astimezone(timezone.utc)).total_seconds()


def map_clock(block_starts, first_trial_starts, previous_trial_ends, durations, session_end=None, elapsed_zero=None):
    """Map ViRMEn's elapsed clock onto seconds since the session started. All times are in seconds; see docs/virmen-timing.md.

    A clock run is a stretch of blocks whose elapsed clock keeps counting, so one offset fits all of it. A run ends where
    a block's first trial starts before the previous trial ended: the elapsed clock restarted.

    A block made at runtime is stamped in the same iteration as its first trial, so that pair gives the offset. A first
    block kept from setup is stamped before the elapsed clock starts, which would put the clock's zero before the session
    began. Its offset comes from `elapsed_zero` if known (exact), otherwise from `session_end`: read after the session's
    shutdown code, so it gives an upper bound.

    `previous_trial_ends` holds, per block, the elapsed time the previous block's last frame was logged (None for the first).
    """
    n_blocks = len(block_starts)
    run_starts = [i for i in range(n_blocks) if i == 0 or first_trial_starts[i] < previous_trial_ends[i]]
    offsets, readings, anchors, max_errors = [None] * n_blocks, list(first_trial_starts), [None] * n_blocks, [None] * n_blocks
    for run, first_block in enumerate(run_starts):
        last_block = run_starts[run + 1] - 1 if run + 1 < len(run_starts) else n_blocks - 1
        offset = block_starts[first_block] - first_trial_starts[first_block]
        anchor, max_error = BLOCK_STAMP, None
        if run > 0 and offset < offsets[first_block - 1] + previous_trial_ends[first_block]:
            raise ClockAnchorError(f"block {first_block + 1} restarted the clock but was stamped before the previous trial ended")
        if run == 0 and offset < 0:
            readings[first_block] = 0.0
            if elapsed_zero is not None:
                if elapsed_zero < 0:
                    raise ClockAnchorError(f"the elapsed clock started {-elapsed_zero} s before the session did")
                offset, anchor = elapsed_zero, ENGINE_TIMESTAMPS
            elif session_end is not None and last_block == n_blocks - 1:
                offset = session_end - (readings[last_block] + durations[last_block])
                if offset < block_starts[first_block]:
                    raise ClockAnchorError(f"the session end puts the elapsed clock's start {block_starts[first_block] - offset} s before the block stamp")
                anchor, max_error = SESSION_END, offset - block_starts[first_block]
            else:
                raise ClockAnchorError("the first block was stamped during setup and nothing else anchors the elapsed clock")
        for block in range(first_block, last_block + 1):
            offsets[block], anchors[block], max_errors[block] = offset, anchor, max_error
    return ClockMap(offsets, readings, anchors, max_errors)
