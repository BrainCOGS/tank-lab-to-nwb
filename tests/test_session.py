"""Normalizing a decoded log through the registry, including a regression test for each known converter bug."""

import json
from datetime import timedelta

import numpy as np
import pytest

from tank_lab_to_nwb.virmen.coerce import LossyConversionError
from tank_lab_to_nwb.virmen.loader import Enum
from tank_lab_to_nwb.virmen.session import ConversionError, normalize_log
from tests.synthetic_log import L, NIL, R, lr_cell, make_block, make_log, make_maze, make_trial, make_version, scalar, single


def _normalize(log):
    return normalize_log(log, file="session.mat")


def _column(table, name):
    return next(c for c in table if c.name == name)


def _names(table):
    return {c.name for c in table}


def _series(session, name):
    return next(s for s in session.series if s.name == name)


class TestFrameTimes:
    def test_multi_block_frames_add_only_the_trial_start(self):
        # Regression: each block's start offset was added twice, shifting frames of every block after the first.
        blocks = [
            make_block(make_trial(start=2.0), start=(2026, 7, 15, 12, 0, 0.5)),
            make_block(make_trial(start=12.0), start=(2026, 7, 15, 12, 0, 10.5)),
        ]
        session = _normalize(make_log(blocks))
        np.testing.assert_array_equal(_column(session.trials, "start_time").data, [10.5, 20.5])
        np.testing.assert_array_equal(_series(session, "position").timestamps, [10.5, 11.0, 11.5, 20.5, 21.0, 21.5])

    def test_first_trial_starts_at_its_block_wall_clock(self):
        session = _normalize(make_log([make_block(make_trial(start=2.5), start=(2026, 7, 15, 12, 0, 1.0))]))
        assert _column(session.trials, "start_time").data[0] == 11.0
        assert _column(session.epochs, "start_time").data[0] == 11.0

    def test_clock_reset_keeps_later_blocks_after_earlier_ones(self):
        # Regression: a second log.version restarts the clock, so block 2 overlapped block 1.
        blocks = [
            make_block([make_trial(start=2.0), make_trial(start=100.0)], start=(2026, 7, 15, 12, 0, 0.5)),
            make_block(make_trial(start=1.0), start=(2026, 7, 15, 12, 10, 0.5)),
        ]
        session = _normalize(make_log(blocks, version=[make_version(), make_version()]))
        starts = _column(session.trials, "start_time").data
        assert starts[2] == 610.5 and starts[1] < starts[2]
        assert np.all(np.diff(_series(session, "position").timestamps) > 0)

    def test_offsets_are_recorded_so_the_elapsed_clock_can_be_recovered(self):
        session = _normalize(make_log([make_block(make_trial(start=2.5), start=(2026, 7, 15, 12, 0, 1.0))]))
        rows = {key: value for key, value, _ in session.settings}
        assert json.loads(rows["derived.clock_offset"]) == [8.5]
        assert _column(session.trials, "start_time").data[0] - 8.5 == 2.5

    def test_position_stops_before_the_iti_and_sensor_dots_keep_it(self):
        session = _normalize(make_log([make_block(make_trial(n_frames=3, n_iti=2))]))
        assert _series(session, "position").data.shape == (3, 2)
        assert _series(session, "SensorDots").data.shape == (5, 5)
        assert len(_series(session, "SensorDots").timestamps) == 5

    def test_position_uses_the_unit_name_nwb_inspector_accepts(self):
        assert _series(_normalize(make_log()), "position").unit == "centimeters"

    def test_view_angle_and_angular_velocity_get_their_own_series_in_radians(self):
        session = _normalize(make_log())
        assert _series(session, "view_angle").unit == "radians"
        assert _series(session, "view_angle").data.ndim == 1
        assert _series(session, "AngularVelocity").unit == "radians/s"
        assert _series(session, "velocity").data.shape[1] == 2

    def test_frame_counts_that_disagree_fail(self):
        trial = make_trial(collision=np.zeros((2, 1), dtype=bool))
        with pytest.raises(ConversionError, match="collision"):
            _normalize(make_log([make_block(trial)]))

    def test_nan_trial_start_fails(self):
        with pytest.raises(ConversionError, match="start"):
            _normalize(make_log([make_block(make_trial(start=np.nan))]))

    @pytest.mark.parametrize("rows", [1, 3, 10])
    def test_raw_ball_input_is_a_recorded_drop_whatever_its_length(self, rows):
        session = _normalize(make_log([make_block(make_trial(rawBallInput=np.ones((rows, 2), dtype="float32")))]))
        assert "RawBallInput" not in {s.name for s in session.series}
        assert "log.block.trial.rawBallInput" in session.coverage.dropped

    def test_other_per_frame_fields_with_the_wrong_row_count_still_fail(self):
        with pytest.raises(ConversionError, match="collision"):
            _normalize(make_log([make_block(make_trial(collision=np.zeros((2, 1), dtype=bool)))]))


def _setup_stamped_log(**extra):
    """One block kept from setup: stamped 0.05 s after the session start, duration counted from elapsed 0."""
    trials = [make_trial(start=2.5, duration=single(5.0)), make_trial(start=7.5, duration=single(92.5))]
    block = make_block(trials, start=(2026, 7, 15, 11, 59, 50.05), duration=scalar(100.0))
    session = dict(start=np.array([[2026, 7, 15, 11, 59, 50.0]]), end=np.array([[2026, 7, 15, 12, 1, 32.0]]), blocks=np.array([[1.0, 1.0]]))
    return make_log([block], session=session, **extra)


def _settings(session):
    return {key: value for key, value, _ in session.settings}


class TestClockAnchor:
    def test_ordinary_block_uses_its_stamp_and_needs_no_review(self):
        session = _normalize(make_log())
        assert json.loads(_settings(session)["derived.clock_anchor"]) == ["block_stamp"]
        assert session.coverage.clock["needs_review"] is False

    def test_epoch_starts_with_its_first_trial_when_blocks_share_an_offset(self):
        # The second stamp is 30 ms late: keeping one offset, the epoch still starts with its own first trial.
        blocks = [
            make_block(make_trial(start=2.0), start=(2026, 7, 15, 12, 0, 0.0), duration=scalar(10.0)),
            make_block(make_trial(start=12.0), start=(2026, 7, 15, 12, 0, 10.03), duration=scalar(10.0)),
        ]
        session = _normalize(make_log(blocks))
        np.testing.assert_array_equal(_column(session.epochs, "start_time").data, _column(session.trials, "start_time").data)
        np.testing.assert_array_equal(_column(session.epochs, "stop_time").data, [20.0, 30.0])

    def test_wall_clock_stamp_is_kept_exactly_in_its_own_column(self):
        session = _normalize(make_log([make_block(make_trial(start=2.0), start=(2026, 7, 15, 12, 0, 0.25))]))
        assert _column(session.epochs, "wall_clock_start").data[0] == 10.25
        assert session.coverage.written["log.block.start"] == ["wall_clock_start"]

    def test_empty_block_keeps_its_stamp_as_its_start(self):
        blocks = [make_block(make_trial(start=2.0), start=(2026, 7, 15, 12, 0, 0.0)), make_block([], start=(2026, 7, 15, 12, 0, 30.0))]
        session = _normalize(make_log(blocks))
        assert _column(session.epochs, "start_time").data[1] == 40.0

    def test_setup_stamp_with_engine_timestamps_is_exact(self):
        # The engine started 10.5 s before the session object and the clock started 11.5 s after the engine: 1.0 s.
        log = _setup_stamped_log(initialTimestamp=np.array([[2026, 7, 15, 11, 59, 39.5]]), timeElapsedFirstTrial=scalar(11.5))
        session = _normalize(log)
        assert _column(session.trials, "start_time").data[0] == 3.5
        assert _column(session.epochs, "start_time").data[0] == 1.0
        assert _column(session.epochs, "stop_time").data[0] == 101.0
        assert _column(session.epochs, "wall_clock_start").data[0] == pytest.approx(0.05)
        assert _series(session, "position").timestamps[0] == 3.5
        rows = _settings(session)
        assert json.loads(rows["derived.clock_anchor"]) == ["engine_timestamps"]
        assert json.loads(rows["derived.clock_offset"]) == [1.0]
        assert json.loads(rows["derived.clock_session_end_minus_block_end"]) == pytest.approx(1.0)
        assert session.coverage.clock["needs_review"] is False

    def test_setup_stamp_without_engine_timestamps_is_marked_for_review(self):
        session = _normalize(_setup_stamped_log())
        assert _column(session.trials, "start_time").data[0] == 4.5
        assert _column(session.epochs, "start_time").data[0] == 2.0
        rows = _settings(session)
        assert json.loads(rows["derived.clock_anchor"]) == ["session_end_upper_bound"]
        assert json.loads(rows["derived.clock_max_error"]) == [pytest.approx(1.95)]
        assert session.coverage.clock["needs_review"] is True
        assert session.coverage.clock["max_error_seconds"] == pytest.approx(1.95)

    @pytest.mark.parametrize("missing", [dict(timeElapsedFirstTrial=np.zeros((0, 0))), dict(initialTimestamp=np.zeros((0, 0)))])
    def test_empty_engine_field_falls_back_to_session_end(self, missing):
        fields = dict(initialTimestamp=np.array([[2026, 7, 15, 11, 59, 39.5]]), timeElapsedFirstTrial=scalar(11.5))
        session = _normalize(_setup_stamped_log(**{**fields, **missing}))
        assert json.loads(_settings(session)["derived.clock_anchor"]) == ["session_end_upper_bound"]

    def test_setup_stamp_with_no_session_end_fails(self):
        log = _setup_stamped_log()
        log["session"]["end"] = np.zeros((0, 0))
        with pytest.raises(ConversionError, match="setup"):
            _normalize(log)

    def test_block_end_far_from_the_session_end_is_marked_for_review(self):
        log = make_log()
        log["session"]["end"] = np.array([[2026, 7, 15, 12, 0, 20.0]])
        session = _normalize(log)
        assert json.loads(_settings(session)["derived.clock_session_end_minus_block_end"]) == pytest.approx(9.0)
        assert session.coverage.clock["needs_review"] is True

    def test_clock_details_reach_the_coverage_report(self):
        report = _normalize(make_log()).coverage.as_dict()
        assert report["clock"]["anchors"] == ["block_stamp"]
        json.dumps(report)


class TestOptionalFields:
    @pytest.mark.parametrize("missing", ["cueCombo", "cueOffset"])
    def test_absent_trial_field_leaves_the_column_out(self, missing):
        # Regression: KeyError on trial.cueCombo and trial.cueOffset.
        trial = make_trial()
        del trial[missing]
        session = _normalize(make_log([make_block(trial)]))
        assert not any(name.startswith(missing) for name in _names(session.trials))

    def test_absent_easy_block_flag_leaves_the_column_out(self):
        # Regression: KeyError on block.easyBlockFlag.
        block = make_block(make_trial())
        del block["easyBlockFlag"]
        assert "easyBlockFlag" not in _names(_normalize(make_log([block])).epochs)

    def test_easy_block_flag_logged_as_double_or_logical(self):
        blocks = [make_block(make_trial(start=1.0), easyBlockFlag=scalar(1)),
                  make_block(make_trial(start=9.0), easyBlockFlag=np.array([[False]]), start=(2026, 7, 15, 12, 0, 9.0))]
        column = _column(_normalize(make_log(blocks)).epochs, "easyBlockFlag")
        assert column.data.dtype == np.bool_ and column.data.tolist() == [True, False]


class TestChoice:
    def test_enum_is_stored_by_name(self):
        session = _normalize(make_log([make_block([make_trial(choice=L), make_trial(start=9.0, choice=R), make_trial(start=19.0, choice=NIL)])]))
        assert _column(session.trials, "choice").data.tolist() == ["L", "R", "nil"]

    @pytest.mark.parametrize("raw, name", [(1.0, "L"), (2.0, "R")])
    def test_raw_number_is_mapped_to_its_name(self, raw, name):
        # Regression: choice is a plain single instead of the enum in 28 trials of one session.
        trial = make_trial(choice=single(raw))
        assert _column(_normalize(make_log([make_block(trial)])).trials, "choice").data.tolist() == [name]

    def test_other_raw_number_fails(self):
        with pytest.raises(LossyConversionError):
            _normalize(make_log([make_block(make_trial(choice=single(3.0)))]))

    def test_unknown_enum_member_fails(self):
        with pytest.raises(LossyConversionError, match="X"):
            _normalize(make_log([make_block(make_trial(choice=Enum("X", 9)))]))

    def test_session_performance_counts_matching_choices(self):
        trials = [make_trial(choice=L, trial_type=L), make_trial(start=9.0, choice=R, trial_type=L)]
        assert _normalize(make_log([make_block(trials)])).lab_meta["session_performance"] == 50.0

    def test_session_performance_is_left_out_when_no_trial_has_a_choice(self):
        session = _normalize(make_log([make_block(make_trial(choice=NIL))]))
        assert "session_performance" not in session.lab_meta


class TestLossless:
    def test_fractional_value_in_an_integer_field_names_the_trial(self):
        trials = [make_trial(), make_trial(start=9.0, iterations=single(2.5))]
        with pytest.raises(LossyConversionError, match=r"log\.block\.trial\.iterations.*element 1.*2\.5"):
            _normalize(make_log([make_block(trials)]))

    def test_unknown_code_fails(self):
        with pytest.raises(LossyConversionError, match="forcedChoice"):
            _normalize(make_log([make_block(make_trial(forcedChoice=single(2)))]))

    def test_double_in_a_float32_field_fails_when_inexact(self):
        with pytest.raises(LossyConversionError, match="rewardScale"):
            _normalize(make_log([make_block(make_trial(rewardScale=scalar(0.1)))]))

    def test_double_field_keeps_its_precision(self):
        value = 1 / 3
        session = _normalize(make_log([make_block(make_trial(), rewardMiL=scalar(value))]))
        assert _column(session.epochs, "rewardMiL").data[0] == value

    def test_vector_in_a_scalar_field_fails(self):
        with pytest.raises(ConversionError, match="iCueEntry"):
            _normalize(make_log([make_block(make_trial(iCueEntry=np.array([[1, 2]], dtype="uint16")))]))

    def test_empty_in_a_scalar_field_without_a_rule_fails(self):
        with pytest.raises(ConversionError, match="iCueEntry"):
            _normalize(make_log([make_block(make_trial(iCueEntry=np.zeros((0, 0), dtype="uint16")))]))

    def test_wrong_width_fails(self):
        with pytest.raises(ConversionError, match="startPosition"):
            _normalize(make_log([make_block(make_trial(startPosition=np.array([[1.0, 2.0]], dtype="float32")))]))

    def test_non_numeric_track_length_fails(self):
        maze = make_maze(variable={"lStart": "2*pi", "lCue": "1", "lMemory": "1"})
        with pytest.raises(ConversionError, match="lStart"):
            _normalize(make_log(version=make_version(mazes=[maze])))

    def test_track_lengths_parse_nan_and_inf(self):
        maze = make_maze(variable={"lStart": "NaN", "lCue": "Inf", "lMemory": "0"})
        other = make_maze(variable={"lStart": "1", "lCue": "2", "lMemory": "3"})  # a column of only NaN is left out
        session = _normalize(make_log(version=make_version(mazes=[maze, other])))
        assert np.isnan(_column(session.mazes, "lStart").data[0])
        assert np.isinf(_column(session.mazes, "lCue").data[0])
        assert _column(session.mazes, "lMemory").data[0] == 0


class TestRaggedAndFixed:
    def test_left_right_cells_split_into_typed_ragged_columns(self):
        trials = [make_trial(cuePos=lr_cell([1.5, 2.5], [])), make_trial(start=9.0, cuePos=lr_cell([], [3.5]))]
        session = _normalize(make_log([make_block(trials)]))
        left = _column(session.trials, "cuePos_left")
        assert left.data.dtype == np.float32
        assert left.data.tolist() == [1.5, 2.5] and left.index.tolist() == [2, 2]
        assert _column(session.trials, "cuePos_right").index.tolist() == [0, 1]
        assert left.index.dtype == np.uint32

    def test_cue_combo_matrix_splits_by_row(self):
        session = _normalize(make_log())
        assert _column(session.trials, "cueCombo_left").data.tolist() == [True, False]
        assert _column(session.trials, "cueCombo_right").data.tolist() == [False, True]

    def test_empty_cue_combo_gives_empty_cells(self):
        trials = [make_trial(cueCombo=np.zeros((0, 0), dtype=bool)), make_trial(start=9.0)]
        session = _normalize(make_log([make_block(trials)]))
        assert _column(session.trials, "cueCombo_left").index.tolist() == [0, 2]

    def test_scalar_trial_prob_becomes_a_pair(self):
        trials = [make_trial(trialProb=single(0.5)), make_trial(start=9.0)]
        column = _column(_normalize(make_log([make_block(trials)])).trials, "trialProb")
        assert column.data.tolist() == [[0.5, 0.5], [0.25, 0.75]] and column.data.dtype == np.float32

    def test_doorstop_markers_are_ragged_with_scalars_vectors_and_empties(self):
        # Regression: i* markers are sometimes vectors.
        trials = [make_trial(iDoorstopRetry=single(5)), make_trial(start=9.0, iDoorstopRetry=np.array([[6.0, 7.0]], dtype="float32")),
                  make_trial(start=19.0, iDoorstopRetry=np.zeros((1, 0), dtype="float32"))]
        column = _column(_normalize(make_log([make_block(trials)])).trials, "iDoorstopRetry")
        assert column.data.tolist() == [5, 6, 7] and column.index.tolist() == [1, 3, 3]

    def test_landmark_onset_is_doubly_ragged(self):
        cell = np.empty((2, 1), dtype=object)
        cell[0, 0], cell[1, 0] = np.array([[1, 2]], dtype="uint16"), np.array([[3]], dtype="uint16")
        column = _column(_normalize(make_log([make_block(make_trial(landmarkOnset=cell))])).trials, "landmarkOnset")
        assert column.data.tolist() == [1, 2, 3]
        assert column.index.tolist() == [2, 3] and column.index_of_index.tolist() == [2]

    def test_ragged_criteria_stay_ragged_in_maze_columns(self):
        # Regression: warmup values mix [], scalars and vectors in one table.
        mazes = [make_maze(criteria={"warmupMaze": np.zeros((0, 0))}),
                 make_maze(criteria={"warmupMaze": scalar(4)}),
                 make_maze(criteria={"warmupMaze": np.array([[4.0, 3.0]])})]
        column = _column(_normalize(make_log(version=make_version(mazes=mazes))).mazes, "warmupMaze")
        assert column.data.tolist() == [4, 4, 3] and column.index.tolist() == [0, 1, 3]


class TestAllEmptyRaggedColumns:
    # NWB Inspector 0.7.2 crashes on a ragged column with no values at all, and such a column says nothing anyway.
    def test_trial_column_is_left_out_and_reported(self):
        session = _normalize(make_log([make_block(make_trial(iDoorstopRetry=np.zeros((1, 0), dtype="float32")))]))
        assert "iDoorstopRetry" not in _names(session.trials)
        assert "log.block.trial.iDoorstopRetry" in session.coverage.omitted

    def test_one_side_of_a_pair_is_left_out_and_reported(self):
        session = _normalize(make_log([make_block(make_trial(cuePos=lr_cell([1.5], [])))]))
        assert "cuePos_left" in _names(session.trials)
        assert "cuePos_right" not in _names(session.trials)
        assert "log.block.trial.cuePos" in session.coverage.omitted

    def test_maze_column_is_left_out_and_reported(self):
        session = _normalize(make_log())
        assert "warmupMaze" not in _names(session.mazes)
        assert "log.version.mazes.criteria.warmupMaze" in session.coverage.omitted

    def test_column_with_a_value_in_any_row_is_kept(self):
        trials = [make_trial(iDoorstopRetry=np.zeros((1, 0), dtype="float32")), make_trial(start=9.0, iDoorstopRetry=single(5))]
        assert _column(_normalize(make_log([make_block(trials)])).trials, "iDoorstopRetry").index.tolist() == [0, 1]


class TestLicks:
    def test_two_row_licks_split(self):
        licks = np.array([[10, 20], [1, 2]], dtype="float32")
        session = _normalize(make_log([make_block(make_trial(licks=licks))]))
        assert _column(session.trials, "licks_iteration").data.tolist() == [10, 20]
        assert _column(session.trials, "licks_value").data.dtype == np.int32

    def test_all_nan_two_row_array_means_no_licks(self):
        trials = [make_trial(licks=np.full((2, 1), np.nan, dtype="float32")),
                  make_trial(start=9.0, licks=np.array([[5.0], [1.0]], dtype="float32"))]
        column = _column(_normalize(make_log([make_block(trials)])).trials, "licks_iteration")
        assert column.data.tolist() == [5] and column.index.tolist() == [0, 1]

    def test_partly_nan_two_row_array_fails(self):
        licks = np.array([[5.0, np.nan], [1.0, np.nan]], dtype="float32")
        with pytest.raises(ConversionError, match="licks"):
            _normalize(make_log([make_block(make_trial(licks=licks))]))

    def test_doorstop_preallocated_array_is_dropped_and_reported(self):
        session = _normalize(make_log([make_block(make_trial(licks=np.full((60000, 1), np.nan, dtype="float32")))]))
        assert "licks_iteration" not in _names(session.trials)
        assert "log.block.trial.licks" in session.coverage.omitted

    def test_doorstop_array_with_real_values_fails(self):
        licks = np.full((60000, 1), np.nan, dtype="float32")
        licks[7] = 1.0
        with pytest.raises(ConversionError, match="licks"):
            _normalize(make_log([make_block(make_trial(licks=licks))]))

    def test_all_nan_counter_licks_leave_the_column_out_and_are_reported(self):
        session = _normalize(make_log([make_block(make_trial(counter_licks=np.full((2, 1), np.nan, dtype="float32")))]))
        assert "counter_licks_iteration" not in _names(session.trials)
        assert "log.block.trial.counter_licks" in session.coverage.omitted


def _stimulus_table(rows):
    table = np.empty((len(rows), 8), dtype=object)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            table[r, c] = value if isinstance(value, str) else (np.array([[value]]) if isinstance(value, bool) else single(value))
    return table


class TestStimulusPairs:
    def _trial(self, start, pair, hits=0.5, ntotal=1, post=0.5, **extra):
        table = _stimulus_table([[1, 0.5, "L", 3, 9, hits, ntotal, post], [2, 0.5, "R", 9, 3, np.nan, 0, post]])
        return make_trial(start=start, pairNum=single(pair), stimulusTable=table, **extra)

    def test_regression_no_name_error_and_one_row_per_block_and_pair(self):
        blocks = [make_block([self._trial(1.0, 2), self._trial(9.0, 1)]),
                  make_block(self._trial(40.0, 1), start=(2026, 7, 15, 12, 0, 40.0))]
        session = _normalize(make_log(blocks))
        assert _column(session.stimulus_pairs, "block").data.tolist() == [1, 1, 2, 2]
        assert _column(session.stimulus_pairs, "Side").data.tolist() == ["L", "R", "L", "R"]
        region = _column(session.trials, "stimulus_pair")
        assert region.region and region.data.tolist() == [1, 0, 2]

    def test_running_columns_are_ragged_per_trial(self):
        session = _normalize(make_log([make_block([self._trial(1.0, 1), self._trial(9.0, 1, ntotal=2)])]))
        assert _column(session.trials, "NTotal").data.tolist() == [1, 0, 2, 0]
        assert _column(session.trials, "NTotal").index.tolist() == [2, 4]
        hits = _column(session.trials, "Hits")
        assert hits.data[0] == 0.5 and np.isnan(hits.data[1]) and hits.data.dtype == np.float32

    def test_hits_logged_as_logical_true_becomes_one(self):
        session = _normalize(make_log([make_block(self._trial(1.0, 1, hits=True))]))
        assert _column(session.trials, "Hits").data[0] == 1.0

    def test_static_columns_changing_within_a_block_fail(self):
        changed = make_trial(start=9.0, pairNum=single(1),
                             stimulusTable=_stimulus_table([[1, 0.9, "L", 3, 9, 0.5, 1, 0.5], [2, 0.1, "R", 9, 3, 0.5, 1, 0.5]]))
        with pytest.raises(ConversionError, match="stimulusTable"):
            _normalize(make_log([make_block([self._trial(1.0, 1), changed])]))

    def test_pair_number_outside_the_table_fails(self):
        with pytest.raises(ConversionError, match="pairNum"):
            _normalize(make_log([make_block(self._trial(1.0, 3))]))

    def test_session_without_a_table_has_no_pairs(self):
        assert _normalize(make_log()).stimulus_pairs == []

    def test_history_fields_are_dropped_and_reported(self):
        trial = self._trial(1.0, 1, hitHistory=np.array([[1.0]]), classHistory=np.array([[1.0]]), trialNum=single(1))
        session = _normalize(make_log([make_block(trial)]))
        for leaf in ("hitHistory", "classHistory", "trialNum"):
            assert f"log.block.trial.{leaf}" in session.coverage.dropped


class TestObjects:
    def test_conditioned_stimulus_members_become_columns(self):
        cs = dict(modality=Enum("visual", 2), side=Enum("R", 2), position=np.array([[1.5, 2.5]]),
                  isRewarded=np.array([[True]]), presented=np.array([[True]]))
        session = _normalize(make_log([make_block(make_trial(CS=cs))]))
        assert _column(session.trials, "CS_modality").data.tolist() == ["visual"]
        assert _column(session.trials, "CS_position").data.tolist() == [1.5, 2.5]
        assert _column(session.trials, "CS_isRewarded").data.dtype == np.bool_

    def test_unpresented_stimulus_logs_none_as_text(self):
        cs = dict(modality="none", side="none", position=np.zeros((1, 0)), isRewarded=np.array([[False]]), presented=np.array([[False]]))
        session = _normalize(make_log([make_block(make_trial(CS=cs))]))
        assert _column(session.trials, "CS_modality").data.tolist() == ["none"]

    def test_function_handles_need_no_matlab(self):
        # Regression: convert_function_handle_to_str required MATLAB and wrote temporary files to the working directory.
        session = _normalize(make_log())
        assert session.lab_meta["experiment_name"] == "experiment"
        assert session.lab_meta["protocol_name"] == "protocol"
        assert _column(session.epochs, "shapingProtocol").data.tolist() == ["protocol"]


class TestMetadata:
    def test_subject_id_is_the_animal_name(self):
        # Regression: Subject.subject_id received the whole animal dict.
        assert _normalize(make_log()).subject_id == "mouse_1"

    def test_session_id_is_the_session_date(self):
        assert _normalize(make_log()).session_id == "20260715"

    def test_session_end_uses_eastern_time(self):
        session = _normalize(make_log())
        assert session.lab_meta["session_end_time"].utcoffset() == timedelta(hours=-4)
        assert session.session_start_time.utcoffset() == timedelta(hours=-4)

    def test_rig_is_typed(self):
        rig = _normalize(make_log()).rig
        assert rig["hasDAQ"] is True
        assert rig["rewardChannel"].dtype == np.int32 and rig["rewardChannel"] == 0
        assert rig["sensorDotsPerRev"].tolist() == [2000.5]
        assert rig["colorAdjustment"].tolist() == [0.0, 0.4, 0.5]

    def test_empty_optional_attribute_is_left_out(self):
        session = _normalize(make_log())
        assert "stimulus_bank_path" not in session.lab_meta
        assert "motionBlurRange" not in session.animal

    def test_draw_method_cell_splits_in_either_orientation(self):
        log = make_log()
        log["animal"]["mainDrawMethod"] = log["animal"]["mainDrawMethod"].reshape(2, 1)
        animal = _normalize(log).animal
        assert animal["mainDrawMethod"] == "TRIAL_DRAWING" and animal["mainDrawMethodParameter"] == 1.0

    def test_pulse_with_only_a_delay_is_padded(self):
        log = make_log()
        log["animal"]["pulseDelayStim"] = scalar(28.0)
        pulse = _normalize(log).animal["pulseDelayStim"]
        assert pulse[0] == 28.0 and np.isnan(pulse[1])

    @pytest.mark.parametrize("value", ["", scalar(1)])
    def test_unset_software_params_write_nothing(self, value):
        log = make_log()
        log["animal"]["softwareParams"] = value
        assert not [row for row in _normalize(log).settings if "softwareParams" in row[0]]

    def test_set_software_params_go_to_settings(self):
        log = make_log()
        log["animal"]["softwareParams"] = dict(software_parameter_set_id=scalar(92), software_parameters=dict(lsrepoch="cue"))
        rows = {key: value for key, value, _ in _normalize(log).settings}
        assert rows["log.animal.softwareParams.software_parameter_set_id"] == "92.0"
        assert rows["log.animal.softwareParams.software_parameters.lsrepoch"] == "cue"

    def test_other_scalar_in_a_struct_field_fails(self):
        log = make_log()
        log["animal"]["softwareParams"] = scalar(7)
        with pytest.raises(ConversionError, match="softwareParams"):
            _normalize(log)


class TestSettings:
    def test_rows_have_key_value_meaning(self):
        rows = {key: (value, meaning) for key, value, meaning in _normalize(make_log()).settings}
        assert rows["log.version.variables.lCue"][0] == "300"
        assert rows["log.version.rig.laserIP"][0] == "0.0.0.0"
        assert rows["log.version.mazes(1).variable.cueDuration"][0] == "0.6"
        assert rows["log.version.mazes(2).variable.cueDuration"][0] == "0.6"
        assert rows["log.version.variables.lCue"][1]

    def test_doubles_round_trip_through_text(self):
        log = make_log()
        log["version"]["rig"]["delayCueDB"] = scalar(1 / 3)
        rows = {key: value for key, value, _ in _normalize(log).settings}
        assert json.loads(rows["log.version.rig.delayCueDB"]) == 1 / 3

    def test_singles_round_trip_through_text_exactly(self):
        log = make_log()
        log["version"]["rig"]["delayCueDB"] = single(0.1)
        rows = {key: value for key, value, _ in _normalize(log).settings}
        assert np.float32(json.loads(rows["log.version.rig.delayCueDB"])) == np.float32(0.1)

    def test_nan_and_arrays_are_kept(self):
        log = make_log()
        log["version"]["rig"]["aiChannels"] = np.array([[0.0, np.nan]])
        rows = {key: value for key, value, _ in _normalize(log).settings}
        assert rows["log.version.rig.aiChannels"] == "[[0.0, NaN]]"

    def test_later_version_differences_are_kept_under_its_index(self):
        second = make_version()
        second["rig"]["ballCircumference"] = scalar(70.0)
        rows = {key: value for key, value, _ in _normalize(make_log(version=[make_version(), second])).settings}
        assert rows["log.version(2).rig.ballCircumference"] == "70.0"

    def test_later_version_with_a_missing_rig_field_is_fine(self):
        first, second = make_version(), make_version()
        first["rig"]["squal"] = scalar(88.0)
        session = _normalize(make_log(version=[first, second]))
        assert session.rig["squal"] == 88.0

    def test_later_version_with_a_different_world_fails(self):
        with pytest.raises(ConversionError, match="log.version"):
            _normalize(make_log(version=[make_version(), make_version(name="other")]))


class TestCoverage:
    def test_registered_fields_are_accounted_for(self):
        coverage = _normalize(make_log()).coverage
        assert coverage.unregistered == []
        assert "log.block.trial.choice" in coverage.written
        assert "log.version.variables.lCue" in coverage.settings
        assert "log.block.trial.comment" in coverage.dropped
        assert "log.block" in coverage.structural

    def test_unregistered_trial_scalar_becomes_a_column_and_is_reported(self):
        session = _normalize(make_log([make_block(make_trial(newField=single(1.5)))]))
        assert _column(session.trials, "newField").data.dtype == np.float32
        assert session.coverage.unregistered == ["log.block.trial.newField"]

    def test_unregistered_trial_vector_becomes_a_ragged_column(self):
        trial = make_trial(newField=np.array([[1.0, 2.0]]))
        column = _column(_normalize(make_log([make_block(trial)])).trials, "newField")
        assert column.data.tolist() == [1.0, 2.0] and column.index.tolist() == [2]

    def test_unregistered_per_frame_field_becomes_a_series(self):
        trial = make_trial(newFrames=np.arange(3, dtype="float32").reshape(-1, 1))
        session = _normalize(make_log([make_block(trial)]))
        assert _series(session, "newFrames").data.shape == (3, 1)
        assert session.coverage.unregistered == ["log.block.trial.newFrames"]

    def test_unregistered_trial_text_becomes_a_column(self):
        column = _column(_normalize(make_log([make_block(make_trial(newField="abc"))])).trials, "newField")
        assert column.data.tolist() == ["abc"]

    def test_unregistered_block_field_becomes_an_epoch_column(self):
        session = _normalize(make_log([make_block(make_trial(), newBlockField=scalar(2.5))]))
        assert _column(session.epochs, "newBlockField").data.tolist() == [2.5]
        assert session.coverage.unregistered == ["log.block.newBlockField"]

    def test_unregistered_config_field_goes_to_settings(self):
        log = make_log()
        log["version"]["rig"]["newRigField"] = scalar(2.5)
        session = _normalize(log)
        rows = {key: value for key, value, _ in session.settings}
        assert rows["log.version.rig.newRigField"] == "2.5"
        assert session.coverage.unregistered == ["log.version.rig.newRigField"]

    def test_unregistered_cell_in_a_trial_goes_to_settings_per_trial(self):
        session = _normalize(make_log([make_block(make_trial(newCell=lr_cell([1.0], [2.0])))]))
        rows = {key: value for key, value, _ in session.settings}
        assert "log.block(1).trial(1).newCell" in rows


def test_single_trial_block_and_zero_trial_block():
    # A 1x1 trial struct arrives as a dict, an empty one as [].
    blocks = [make_block(make_trial()), make_block([], start=(2026, 7, 15, 12, 0, 30.0))]
    session = _normalize(make_log(blocks))
    assert len(_column(session.trials, "start_time").data) == 1
    assert len(_column(session.epochs, "start_time").data) == 2


def test_session_with_no_trials():
    session = _normalize(make_log([make_block([])]))
    assert len(_column(session.trials, "start_time").data) == 0
    assert _series(session, "position").data.shape == (0, 2)


class TestAllNanColumns:
    """A float column with no value in this session is left out and listed, not written as a column of NaN."""

    def _session(self, **trial_fields):
        trial = make_trial(**trial_fields)
        return normalize_log(make_log([make_block([trial, make_trial(start=9.0, **trial_fields)])]), file="x.mat")

    def test_an_all_nan_float_column_is_left_out_and_listed(self):
        session = self._session(trialID=np.array([[np.nan]], dtype="float32"))
        assert "trialID" not in [column.name for column in session.trials]
        assert "NaN" in session.coverage.omitted["log.block.trial.trialID"]
        assert "log.block.trial.trialID" not in session.coverage.written

    def test_a_partly_nan_column_stays(self):
        first = make_trial(trialID=np.array([[np.nan]], dtype="float32"))
        second = make_trial(start=9.0, trialID=np.array([[3.0]], dtype="float32"))
        session = normalize_log(make_log([make_block([first, second])]), file="x.mat")
        column = next(column for column in session.trials if column.name == "trialID")
        assert np.isnan(column.data[0]) and column.data[1] == 3.0
        assert "log.block.trial.trialID" not in session.coverage.omitted

    def test_an_all_zero_column_stays(self):
        session = self._session(trialID=np.array([[0.0]], dtype="float32"))
        assert "trialID" in [column.name for column in session.trials]

    def test_an_all_nan_maze_column_is_left_out_and_listed(self):
        session = _normalize(make_log(version=make_version(mazes=[make_maze(variable={"lStart": "NaN", "lCue": "Inf", "lMemory": "0"})])))
        assert "lStart" not in [column.name for column in session.mazes]
        assert any("lStart" in reason for reason in session.coverage.omitted.values())
