"""Wall-clock conversion and the session timeline. All differences are computed in UTC."""

from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from tank_lab_to_nwb.virmen.timebase import (
    BLOCK_STAMP,
    ENGINE_TIMESTAMPS,
    SESSION_END,
    AmbiguousTimeWarning,
    ClockAnchorError,
    datevec_to_datetime,
    map_clock,
    seconds_between,
)

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


class TestBlockStamp:
    """A block made at runtime is stamped in the same iteration as its first trial, so that pair ties the clocks."""

    def _map(self, starts, firsts, ends=None, durations=None, **kwargs):
        ends = [None, *firsts[:-1]] if ends is None else ends
        return map_clock(starts, firsts, ends, [10.0] * len(starts) if durations is None else durations, **kwargs)

    def test_single_block(self):
        clock = self._map([3.0], [1.0])
        assert clock.offsets == [2.0]
        assert clock.block_start_readings == [1.0]
        assert clock.anchors == [BLOCK_STAMP]
        assert clock.max_errors == [None]

    def test_later_blocks_on_the_same_clock_share_the_first_offset(self):
        # The later block's own stamp would give 2.03: a stamp is 0-50 ms late, so one offset keeps intervals exact.
        clock = self._map([3.0, 83.0], [1.0, 80.97], [None, 80.9])
        assert clock.offsets == [2.0, 2.0]
        assert clock.block_start_readings == [1.0, 80.97]

    def test_clock_reset_gets_a_new_offset(self):
        # A log continued after a restart: its first trial starts before the previous trial ended.
        clock = self._map([3.0, 3000.0], [1.0, 1.5], [None, 2990.0])
        assert clock.offsets == [2.0, 2998.5]
        assert clock.anchors == [BLOCK_STAMP, BLOCK_STAMP]

    def test_trial_starting_exactly_at_the_previous_end_is_not_a_reset(self):
        assert self._map([2.0, 12.0], [1.0, 11.0], [None, 11.0]).offsets == [1.0, 1.0]

    def test_no_blocks(self):
        clock = self._map([], [])
        assert clock.offsets == clock.anchors == clock.block_start_readings == clock.max_errors == []

    def test_first_trial_at_elapsed_zero(self):
        clock = self._map([0.5], [0.0])
        assert clock.offsets == [0.5] and clock.anchors == [BLOCK_STAMP]

    def test_stamp_and_first_trial_both_at_zero_is_not_a_setup_stamp(self):
        clock = self._map([0.0], [0.0])
        assert clock.offsets == [0.0] and clock.anchors == [BLOCK_STAMP]

    def test_session_end_and_engine_time_do_not_override_a_good_stamp(self):
        clock = self._map([3.0], [1.0], session_end=500.0, elapsed_zero=0.5)
        assert clock.offsets == [2.0] and clock.anchors == [BLOCK_STAMP]


class TestSetupStamp:
    """A block kept from setup is stamped before the elapsed clock starts: the stamp would put that zero before the session."""

    def test_engine_timestamps_give_the_exact_zero(self):
        clock = map_clock([0.054], [2.5], [None], [2733.95], session_end=2735.6, elapsed_zero=0.943)
        assert clock.offsets == [0.943]
        assert clock.block_start_readings == [0.0]
        assert clock.anchors == [ENGINE_TIMESTAMPS]
        assert clock.max_errors == [None]

    def test_session_end_is_an_upper_bound_without_engine_timestamps(self):
        clock = map_clock([0.05], [2.5], [None], [100.0], session_end=102.0)
        assert clock.offsets == [2.0]
        assert clock.block_start_readings == [0.0]
        assert clock.anchors == [SESSION_END]
        # The clock started somewhere between the stamp and this offset.
        assert clock.max_errors == [pytest.approx(1.95)]

    def test_session_end_uses_the_last_blocks_own_start_reading(self):
        clock = map_clock([0.05, 50.0], [2.5, 40.0], [None, 39.0], [39.0, 60.0], session_end=102.0)
        assert clock.offsets == [2.0, 2.0]
        assert clock.block_start_readings == [0.0, 40.0]
        assert clock.anchors == [SESSION_END, SESSION_END]

    def test_engine_timestamps_win_over_session_end(self):
        clock = map_clock([0.05], [2.5], [None], [100.0], session_end=999.0, elapsed_zero=0.9)
        assert clock.offsets == [0.9] and clock.anchors == [ENGINE_TIMESTAMPS]

    def test_elapsed_zero_of_exactly_zero_is_used(self):
        # 0.0 is a value, not a missing one.
        clock = map_clock([0.05], [2.5], [None], [100.0], session_end=102.0, elapsed_zero=0.0)
        assert clock.offsets == [0.0] and clock.anchors == [ENGINE_TIMESTAMPS]

    def test_no_anchor_fails(self):
        with pytest.raises(ClockAnchorError, match="setup"):
            map_clock([0.05], [2.5], [None], [100.0])

    def test_session_end_on_another_clock_run_fails(self):
        # session.end belongs to the last run, so it can't anchor the first one.
        with pytest.raises(ClockAnchorError, match="setup"):
            map_clock([0.05, 3000.0], [2.5, 1.0], [None, 2990.0], [2990.0, 100.0], session_end=3101.0)

    def test_session_end_earlier_than_the_stamp_allows_fails(self):
        with pytest.raises(ClockAnchorError, match="session end"):
            map_clock([50.0], [60.0], [None], [100.0], session_end=120.0)

    def test_elapsed_zero_before_the_session_start_fails(self):
        with pytest.raises(ClockAnchorError, match="before the session"):
            map_clock([0.05], [2.5], [None], [100.0], elapsed_zero=-1.0)


def test_later_clock_run_stamped_before_the_previous_run_ended_fails():
    with pytest.raises(ClockAnchorError, match="previous"):
        map_clock([3.0, 200.0], [1.0, 1.5], [None, 500.0], [500.0, 10.0])
