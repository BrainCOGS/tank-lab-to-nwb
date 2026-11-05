"""Wall-clock times of a session, and the offset from ViRMEn's elapsed clock onto them.

Python subtracts two datetimes that share a tzinfo as wall clocks, which is wrong across a DST change, so all
differences go through UTC.
"""

import warnings
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

EASTERN = ZoneInfo("America/New_York")


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


def clock_offsets(block_starts, first_trial_starts, previous_trial_ends):
    """Per block, what to add to ViRMEn's elapsed clock to get seconds since the session started.

    Each block's wall-clock start is stamped just before its first trial, so that pair ties the two clocks together. A
    block keeps the running offset unless its first trial starts before the previous trial ended: then the elapsed
    clock restarted (a log continued under a new log.version).
    """
    offsets = []
    for block_start, first_trial, previous_end in zip(block_starts, first_trial_starts, previous_trial_ends):
        restarted = previous_end is not None and first_trial < previous_end
        offsets.append(block_start - first_trial if not offsets or restarted else offsets[-1])
    return offsets
