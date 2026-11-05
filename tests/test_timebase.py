"""Wall-clock conversion and the session timeline. All differences are computed in UTC."""

from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from tank_lab_to_nwb.virmen.timebase import AmbiguousTimeWarning, clock_offsets, datevec_to_datetime, seconds_between

NY = ZoneInfo("America/New_York")


def test_datevec_uses_eastern_time_in_winter_and_summer():
    assert datevec_to_datetime([2026, 1, 15, 12, 0, 0]).utcoffset() == timedelta(hours=-5)
    assert datevec_to_datetime([2026, 7, 15, 12, 0, 0]).utcoffset() == timedelta(hours=-4)


def test_datevec_keeps_fractional_seconds_to_the_microsecond():
    assert datevec_to_datetime(np.array([[2026, 7, 15, 12, 0, 1.25]])).microsecond == 250000


def test_datevec_with_60_seconds_rolls_over():
    assert datevec_to_datetime([2026, 7, 15, 12, 0, 60.0]).minute == 1


@pytest.mark.parametrize("bad", [[2026, 7, 15], [np.nan, 7, 15, 12, 0, 0], [2026, 13, 1, 0, 0, 0], []])
def test_invalid_datevec_fails(bad):
    with pytest.raises(ValueError):
        datevec_to_datetime(bad)


def test_difference_across_the_fall_back_is_real_elapsed_time():
    # 00:30 to 03:30 on 2026-11-01 is 4 hours, though subtracting wall clocks gives 3.
    start = datevec_to_datetime([2026, 11, 1, 0, 30, 0])
    end = datevec_to_datetime([2026, 11, 1, 3, 30, 0])
    assert seconds_between(start, end) == 4 * 3600
    assert (end - start).total_seconds() == 3 * 3600  # the trap this guards against


def test_difference_across_spring_forward():
    start = datevec_to_datetime([2026, 3, 8, 1, 30, 0])
    end = datevec_to_datetime([2026, 3, 8, 3, 30, 0])
    assert seconds_between(start, end) == 3600


def test_zero_difference():
    start = datevec_to_datetime([2026, 7, 15, 12, 0, 0])
    assert seconds_between(start, start) == 0


def test_repeated_hour_warns():
    with pytest.warns(AmbiguousTimeWarning, match="2026-11-01"):
        datevec_to_datetime([2026, 11, 1, 1, 30, 0])


def test_unambiguous_hour_does_not_warn(recwarn):
    datevec_to_datetime([2026, 11, 1, 2, 30, 0])
    assert not [w for w in recwarn if w.category is AmbiguousTimeWarning]


class TestClockOffsets:
    def test_single_block_moves_the_first_trial_to_its_block_start(self):
        # The elapsed clock read 2.5 s when the block's wall clock read 0.05 s after the session start.
        assert clock_offsets(block_starts=[0.05], first_trial_starts=[2.5], previous_trial_ends=[None]) == [0.05 - 2.5]

    def test_later_blocks_on_the_same_clock_share_the_first_offset(self):
        offsets = clock_offsets([0.05, 80.0], [2.5, 82.47], [None, 82.4])
        assert offsets == [0.05 - 2.5, 0.05 - 2.5]

    def test_clock_reset_gets_a_new_offset(self):
        # The second version's clock restarted: its first trial starts before the previous trial ended.
        offsets = clock_offsets([0.05, 3000.0], [2.5, 1.0], [None, 2990.0])
        assert offsets == [0.05 - 2.5, 3000.0 - 1.0]

    def test_trial_starting_exactly_at_the_previous_end_is_not_a_reset(self):
        offsets = clock_offsets([0.0, 10.0], [1.0, 11.0], [None, 11.0])
        assert offsets == [-1.0, -1.0]

    def test_no_blocks(self):
        assert clock_offsets([], [], []) == []
