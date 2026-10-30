"""Authors: Cody Baker Szonja Weigl and Ben Dichter."""

from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import override
from zoneinfo import ZoneInfo
import re

import numpy as np
from hdmf.backends.hdf5.h5_utils import H5DataIO

# from ndx_tank_metadata import LabMetaDataExtension, MazeExtension, RigExtension, StimulusTable
from ndx_tank_metadata import LabMetaDataExtension, MazeExtension, RigExtension

# from neuroconv.basedatainterface import BaseData
from neuroconv.basetemporalalignmentinterface import BaseTemporalAlignmentInterface
from neuroconv.utils import dict_deep_update
from pydantic import FilePath
from neuroconv.utils.dict import DeepDict
from pynwb import NWBFile, TimeSeries
from pynwb.core import DynamicTable
from pynwb.behavior import CompassDirection, Position, SpatialSeries

from ..utils import (
    array_to_dt,
    check_module,
    convert_function_handle_to_str,
    convert_mat_file_to_dict,
    create_and_store_indexed_array,
    flatten_nested_dict,
)


class VirmenDataInterface(BaseTemporalAlignmentInterface):
    """Conversion class for Virmen behavioral data."""

    def __init__(self, file_path: FilePath, verbose: bool = True):
        """
        Parameters
        ----------
        file_path : FilePath
            Path to virman .mat file.
        verbose : bool, default: True
            Whether to output verbose text.
        """

        self.verbose = verbose
        file_path = Path(file_path)
        folder_path = file_path.parent
        super().__init__(
            file_path=file_path,
            folder_path=folder_path,
            verbose=verbose,
        )
        mat_file = self.source_data["file_path"]
        self._mat_dict = convert_mat_file_to_dict(mat_file)
        self._times = None
        self._trial_frame_boundaries = None  # Cache for trial frame boundaries

    @override
    def get_timestamps(self) -> np.ndarray:
        return self._times if self._times is not None else self.get_original_timestamps()

    @override
    def get_original_timestamps(self) -> np.ndarray:
        metadata_copy = deepcopy(self._mat_dict)

        session_start_time = self._get_session_start_time()

        if isinstance(metadata_copy["log"]["block"], dict):
            epochs: list[dict] = [metadata_copy["log"]["block"]]
        else:
            epochs: list[dict] = metadata_copy["log"]["block"]

        trials = [trial for epoch in epochs for trial in epoch["trial"] if not np.isnan(trial["start"])]
        # array_to_dt returns a naive datetime while _get_session_start_time is
        # tz-aware, so the subtraction below raised TypeError. It only ever
        # surfaced when get_original_timestamps() actually ran, which a caller
        # supplying sync_timestamps skips — imaging-only sessions do not supply
        # one, so they hit it immediately.
        session_tz = session_start_time.tzinfo
        epoch_start_dts = [array_to_dt(epoch["start"]).replace(tzinfo=session_tz) for epoch in epochs]
        epoch_start_nwb: list[float] = [
            (epoch_start_dt - session_start_time).total_seconds() for epoch_start_dt in epoch_start_dts
        ]

        # One offset for every block, not one per block. trial["start"] is
        # vr.timeElapsed (ExperimentLog.m:501), which the engine zeroes once at
        # firstTic (virmenEngine.m:124) and never resets -- obj.blockStart is
        # recorded per block but only ever read to compute block.duration. So
        # trial starts already run on a single session-wide clock, and the only
        # correction needed is that clock's zero (~block 1 start) against
        # session.start.
        #
        # Adding epoch_start_nwb[epoch_idx] instead double-counted: a second
        # block beginning 10 minutes in pushed its frames 600 s into the future,
        # away from both the trials table and any imaging series (which use
        # epoch_start_nwb[0], below and at :414).
        session_offset = epoch_start_nwb[0]

        frame_timestamps = []
        for epoch in epochs:
            for trial in epoch["trial"]:
                if not np.isnan(trial["start"]):
                    frame_timestamps.extend(trial["start"] + session_offset + trial["time"])

        frame_starts = frame_timestamps

        # Return this into an nd array

        return np.array(frame_starts)

    @override
    def set_aligned_timestamps(self, aligned_timestamps: np.ndarray) -> None:
        self._times = aligned_timestamps

    def _get_session_start_time(self):
        session_start_time = array_to_dt(self._mat_dict["log"]["session"]["start"]).replace(
            tzinfo=ZoneInfo("America/New_York")
        )
        return session_start_time

    def get_session_key(self) -> dict:
        """
        Extract session key information from the Virmen .mat file for DataJoint queries.

        Returns
        -------
        dict
            Dictionary containing 'subject_fullname' and 'session_date' for querying DataJoint.
        """
        subject_fullname = self._mat_dict["log"]["animal"]["name"]
        session_start_time = self._get_session_start_time()
        session_date = session_start_time.date()

        return {"subject_fullname": subject_fullname, "session_date": session_date}

    def _get_trial_frame_boundaries(self) -> list[tuple[int, int]]:
        """
        Get the start and end frame indices for each trial in the global timestamp array.
        
        Returns
        -------
        list of tuple
            List of (start_idx, end_idx) tuples for each trial's frames in the global timestamp array.
            Indices are inclusive of start, exclusive of end (Python slice convention).
        
        Notes
        -----
        This method caches its result since trial boundaries are based on Virmen structure
        and don't change when synchronized timestamps are applied.
        """
        if self._trial_frame_boundaries is not None:
            return self._trial_frame_boundaries
        
        metadata_copy = deepcopy(self._mat_dict)
        
        if isinstance(metadata_copy["log"]["block"], dict):
            epochs = [metadata_copy["log"]["block"]]
        else:
            epochs = metadata_copy["log"]["block"]
        
        trials = [trial for epoch in epochs for trial in epoch["trial"] if not np.isnan(trial["start"])]
        
        boundaries = []
        current_idx = 0
        for trial in trials:
            frame_count = len(trial["time"])
            boundaries.append((current_idx, current_idx + frame_count))
            current_idx += frame_count
        
        self._trial_frame_boundaries = boundaries
        return boundaries

    def _convert_trial_iteration_to_timestamp(self, trial_idx: int, iteration_num: float) -> float:
        """
        Convert a trial-local iteration number to a session-relative timestamp.
        
        Parameters
        ----------
        trial_idx : int
            Zero-based trial index
        iteration_num : float
            Iteration number (1-based Virmen convention) within the trial
            
        Returns
        -------
        float
            Timestamp in seconds from session start, or NaN if invalid
            
        Notes
        -----
        - Virmen iteration numbers are 1-based (first frame = 1)
        - Automatically uses synchronized timestamps if available via get_timestamps()
        - Returns NaN for invalid iteration numbers (≤0, NaN, or beyond trial length)
        """
        # Handle invalid iteration numbers
        if not np.isfinite(iteration_num) or iteration_num <= 0:
            return np.nan
        
        # Convert to 0-based frame index
        frame_idx = int(iteration_num) - 1
        
        # Get trial boundaries
        boundaries = self._get_trial_frame_boundaries()
        
        if trial_idx >= len(boundaries):
            return np.nan
        
        start_idx, end_idx = boundaries[trial_idx]
        trial_frame_count = end_idx - start_idx
        
        # Check if frame index is within trial
        if frame_idx >= trial_frame_count:
            return np.nan
        
        # Get global frame index and look up timestamp
        global_frame_idx = start_idx + frame_idx
        timestamps = self.get_timestamps()
        
        if global_frame_idx >= len(timestamps):
            return np.nan
        
        return float(timestamps[global_frame_idx])

    def _local_frames_to_timestamps(self, trial_idx: int, local_frame_indices) -> np.ndarray:
        """
        Map trial-local frame indices to session-relative timestamps.

        Parameters
        ----------
        trial_idx : int
            Zero-based trial index.
        local_frame_indices : array-like of int
            Zero-based frame indices within the trial.

        Returns
        -------
        np.ndarray
            Timestamps in seconds from session start for the requested frames.
            Uses synchronized (aligned) timestamps if available, otherwise Virmen
            internal timing. This keeps cue/lick event times in the same reference
            frame as the Position/Velocity series and the iteration markers.
        """
        boundaries = self._get_trial_frame_boundaries()
        start_idx, end_idx = boundaries[trial_idx]
        timestamps = self.get_timestamps()
        global_indices = start_idx + np.asarray(local_frame_indices, dtype=int)
        # Clamp to the trial's frame range to avoid leaking into adjacent trials
        global_indices = np.clip(global_indices, start_idx, end_idx - 1)
        return timestamps[global_indices]

    @classmethod
    def get_source_schema(cls):
        """Compile input schemas from each of the data interface classes."""
        return dict(required=["file_path"], properties=dict(file_path=dict(type="string")))

    def get_metadata(self) -> DeepDict:
        """
        Get metadata from the mat file and add it to the metadata dictionary.
        """

        super_metadata = super().get_metadata()

        local_log_copy = deepcopy(self._mat_dict["log"])
        metadata = deepcopy(self._mat_dict)

        # Note: Experimenter information should be provided via metadata override from external sources
        # (e.g., DataJoint queries in notebook/script) rather than queried here.
        # This keeps the interface modular and usable without database dependencies.

        # Commented out DataJoint database call:
        # try:
        #     import datajoint as dj
        #     subject = dj.create_virtual_module("subject", "u19_subject")
        #     lab = dj.create_virtual_module("lab", "u19_lab")
        #     subject_fullname = metadata["log"]["animal"]["name"]
        #     sub_info = (subject.Subject() * lab.User() & f"subject_fullname = '{subject_fullname}'").fetch1()
        #     owner = [sub_info["user_id"]]
        #     coowners = list(
        #         (subject.SubjectCoowners() & f"subject_fullname = '{subject_fullname}' and active = 1").fetch("coowner")
        #     )
        #     experimenter = owner + coowners
        # except Exception as e:
        #     if self.verbose:
        #         print(f"Could not import datajoint, setting experimenter to FAKE PERSON. Error: {e}")
        #     experimenter = []

        session_start_time = self._get_session_start_time()

        metadata_from_mat_dict = dict(
            Subject=dict(subject_id=local_log_copy["animal"]),
            NWBFile=dict(session_start_time=session_start_time),
        )

        metadata = dict_deep_update(super_metadata, metadata_from_mat_dict, copy=True)

        return metadata

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict):
        metadata_copy = deepcopy(self._mat_dict)
        nwbfile.add_epoch_column("label", "name of epoch")

        # ------------------------ Adding LabMetaDataExtension ----------------------- #

        experiment_metadata = metadata_copy["log"]["version"]

        if isinstance(metadata_copy["log"]["block"], dict):
            epochs = [metadata_copy["log"]["block"]]
        else:
            epochs = metadata_copy["log"]["block"]

        trials = [trial for epoch in epochs for trial in epoch["trial"] if not np.isnan(trial["start"])]

        # Extension of lab metadata
        experiment_metadata = metadata_copy["log"]["version"]
        subject_metadata = metadata_copy["log"]["animal"]
        rig_atrr = [
            "rig",
            "simulationMode",
            "hasDAQ",
            "hasSyncComm",
            "minIterationDT",
            "arduinoPort",
            "sensorDotsPerRev",
            "ballCircumference",
            "toroidXFormP1",
            "toroidXFormP2",
            "colorAdjustment",
            "soundAdjustment",
            "nidaqDevice",
            "nidaqPort",
            "nidaqLines",
            "syncClockChannel",
            "syncDataChannel",
            "rewardChannel",
            "rewardSize",
            "rewardDuration",
            "laserChannel",
            "rightPuffChannel",
            "leftPuffChannel",
            "webcam_name",
        ]

        rig = {k: v for k, v in experiment_metadata["rig"].items() if k in rig_atrr}
        rig.update((k, v.astype(np.int32)) for k, v in rig.items() if isinstance(v, np.ndarray) and v.dtype == np.uint8)

        # Wrap the rig_attr sensorDorsPerRev in a list to match the expected format since sometimes it is a scalar
        rig["sensorDotsPerRev"] = (
            [rig["sensorDotsPerRev"]]
            if "sensorDotsPerRev" in rig and isinstance(rig["sensorDotsPerRev"], (int, float))
            else rig.get("sensorDotsPerRev")
        )

        for maze in experiment_metadata["mazes"]:
            for k, v in maze["criteria"].items():
                if k.startswith("warmup") and isinstance(v, np.ndarray) and len(v) == 0:
                    maze["criteria"][k] = np.nan

        rig_extension = RigExtension(name="rig", **rig)

        maze_extension = MazeExtension(name="mazes", description="description of the mazes")

        for maze in experiment_metadata["mazes"]:
            flatten_maze_dict = flatten_nested_dict(maze)
            maze_extension.add_row(
                **dict((k, v) for k, v in flatten_maze_dict.items() if k in MazeExtension.mazes_attr)
            )

        num_trials = len(trials)
        session_end_time = (
            array_to_dt(metadata_copy["log"]["session"]["end"]).replace(tzinfo=ZoneInfo("America/New_York")).isoformat()
        )
        converted_metadata = convert_function_handle_to_str(mat_file_path=self.source_data["file_path"])

        lab_meta_data = dict(
            name="LabMetaData",
            experiment_name=converted_metadata["experiment_name"] if "experiment_name" in converted_metadata else "",
            world_file_name=experiment_metadata["name"],
            protocol_name=converted_metadata["protocol_name"] if "protocol_name" in converted_metadata else "",
            stimulus_bank_path=subject_metadata["stimulusBank"] if subject_metadata["stimulusBank"] else "",
            commit_id=experiment_metadata["repository"],
            location=experiment_metadata["rig"]["rig"],

            num_trials=num_trials,
            session_end_time=session_end_time,
            rig=rig_extension,
            mazes=maze_extension,
            # session_performance=0.,  # comment out to add (not in behavior file)
        )

        nwbfile.add_lab_meta_data(LabMetaDataExtension(**lab_meta_data))

        # --------------------------- Calculating Intervals -------------------------- #

        if isinstance(metadata_copy["log"]["block"], dict):
            epochs = [metadata_copy["log"]["block"]]
        else:
            epochs = metadata_copy["log"]["block"]

        trials = [trial for epoch in epochs for trial in epoch["trial"] if not np.isnan(trial["start"])]

        # ------------------------- Adding epochs information ------------------------- #

        session_start_time = self._get_session_start_time()

        # Calculate epoch times from Virmen data (default)
        epoch_start_dts = [array_to_dt(epoch["start"]).replace(tzinfo=ZoneInfo("America/New_York")) for epoch in epochs]
        epoch_durations_dts = [timedelta(seconds=epoch["duration"]) for epoch in epochs]
        epoch_start_nwb_virmen = [
            (epoch_start_dt - session_start_time).total_seconds() for epoch_start_dt in epoch_start_dts
        ]
        epoch_end_nwb_virmen = [
            (epoch_start_dt - session_start_time + epoch_duration).total_seconds()
            for epoch_start_dt, epoch_duration in zip(epoch_start_dts, epoch_durations_dts)
        ]

        # Check if we have synchronized timestamps - if so, recalculate from them
        all_timestamps = self.get_timestamps()
        use_sync_times = self._times is not None  # True if set_aligned_timestamps was called

        if use_sync_times:
            # Recalculate epoch start/end from synchronized frame timestamps
            epoch_start_nwb = []
            epoch_end_nwb = []
            current_frame_idx = 0

            for epoch in epochs:
                # Count frames in this epoch
                epoch_frame_count = sum(len(trial["time"]) for trial in epoch["trial"] if not np.isnan(trial["start"]))

                if epoch_frame_count > 0:
                    epoch_start = all_timestamps[current_frame_idx]
                    epoch_end = all_timestamps[current_frame_idx + epoch_frame_count - 1]
                    epoch_start_nwb.append(epoch_start)
                    epoch_end_nwb.append(epoch_end)
                    current_frame_idx += epoch_frame_count
                else:
                    # Fallback to Virmen times if no frames
                    epoch_idx = epochs.index(epoch)
                    epoch_start_nwb.append(epoch_start_nwb_virmen[epoch_idx])
                    epoch_end_nwb.append(epoch_end_nwb_virmen[epoch_idx])
        else:
            # Use Virmen internal timing
            epoch_start_nwb = epoch_start_nwb_virmen
            epoch_end_nwb = epoch_end_nwb_virmen

        for j, (start, end) in enumerate(zip(epoch_start_nwb, epoch_end_nwb)):
            nwbfile.add_epoch(start_time=start, stop_time=end, label="Epoch" + str(j + 1))

        epoch_maze_ids = [epoch["mazeID"] for epoch in epochs]
        epoch_main_maze_ids = [epoch["mainMazeID"] for epoch in epochs]
        epoch_easy_flag = [bool(epoch["easyBlockFlag"]) for epoch in epochs]
        epoch_first_trial = [epoch["firstTrial"] for epoch in epochs]
        epoch_num_trials = []
        for epoch in epochs:
            num_non_empty_trials = 0
            for trial in epoch["trial"]:
                if not np.isnan(trial["start"]):
                    num_non_empty_trials += 1
            epoch_num_trials.append(num_non_empty_trials)

        epoch_durations = [epoch["duration"] for epoch in epochs]
        epoch_reward_mil = [epoch["rewardMiL"] for epoch in epochs]
        epoch_stimulus_config = [epoch["stimulusConfig"] for epoch in epochs]

        nwbfile.add_epoch_column(name="maze_id", description="number of maze run in an epoch", data=epoch_maze_ids)
        nwbfile.add_epoch_column(
            name="main_maze_id", description='number of maze of "highest" level for subject', data=epoch_main_maze_ids
        )
        nwbfile.add_epoch_column(
            name="easy_epoch",
            description="True if block was flagged as easy (maze_id < main_maze_id)",
            data=epoch_easy_flag,
        )
        nwbfile.add_epoch_column(name="first_trial", description="first trial run in an epoch", data=epoch_first_trial)
        nwbfile.add_epoch_column(name="num_trials", description="number of trials in an epoch", data=epoch_num_trials)
        nwbfile.add_epoch_column(name="duration", description="duration of epoch in seconds", data=epoch_durations)
        nwbfile.add_epoch_column(name="reward_ml", description="reward in ml", data=epoch_reward_mil)
        nwbfile.add_epoch_column(
            name="stimulus_config", description="stimulus configuration number", data=epoch_stimulus_config
        )

        # Optional Block/Epoch level information

        epoch_columns = [
            (
                "lsrepoch",
                "Which task epochs the laser can be on in",
            ),
            (
                "P_on",
                "probability of a trial being a laser on trial",
            ),
            (
                "context",
                "Determines if pro or anti, 1=Pro, 2=Anti",
            ),
            (
                "stage",
                "Variable tracking pro/anti training stage, since mazeID is used to handle context switching",
            ),
        ]

        for column_name, desc in epoch_columns:
            data = [epoch[column_name] for epoch in epochs if column_name in epoch]
            if data:
                nwbfile.add_epoch_column(name=column_name, description=desc, data=data)

        # ------------------------- Adding trial information ------------------------- #
        # -------- This information stays the same throughout the specific trial ------ #

        trial_durations = [trial["duration"] for trial in trials]

        # Check if we should use synchronized timestamps or Virmen internal timing
        if use_sync_times:
            # Recalculate trial start/end times from synchronized timestamps
            trial_starts = []
            trial_ends = []

            current_frame_idx = 0
            for trial in trials:
                trial_frame_count = len(trial["time"])

                # Trial start is the timestamp of the first frame
                trial_start = all_timestamps[current_frame_idx]

                # Trial end is the timestamp of the last frame
                trial_end = all_timestamps[current_frame_idx + trial_frame_count - 1]

                trial_starts.append(trial_start)
                trial_ends.append(trial_end)

                current_frame_idx += trial_frame_count
        else:
            # Use Virmen internal timing (default)
            trial_starts = [trial["start"] + epoch_start_nwb[0] for trial in trials]
            trial_ends = [start_time + duration for start_time, duration in zip(trial_starts, trial_durations)]

        for k in range(len(trial_starts)):
            nwbfile.add_trial(start_time=trial_starts[k], stop_time=trial_ends[k])

        nwbfile.add_trial_column(name="duration", description="duration of trial in seconds", data=trial_durations)
        trial_idx = [trial_id for num_trials in epoch_num_trials for trial_id in np.arange(0, num_trials)]
        # ruff: noqa
        nwbfile.add_trial_column(name="trial_id", description="number of trial in block", data=trial_idx)

        trial_columns = [
            ("iterations", "number of iterations (frames) for entire trial"),
            ("iStartEntry", "iteration number when trial started"),
            ("iCueEntry", "iteration number when subject entered cue region"),
            ("iMemEntry", "iteration number when subject entered memory region"),
            ("iTurnEntry", "iteration number when subject entered turn region"),
            ("iArmEntry", "iteration number when subject entered arm region"),
            ("iBlank", "iteration number when screen is turned off"),
            ("excessTravel", "total distance traveled during the trial normalized to the length of the maze"),
            ("rewardScale", "multiplier of reward for each correct trial"),
            ("StartCycle", "The spatial frequency of the first stimulus shown to the mouse."),
            ("EndCycle", "The spatial frequency of the second stimulus shown to the mouse."),
            (
                "rule",
                'Specifies what the mouse should do to receive reward. Can be "StartCycle < EndCycle Left" or "StartCycle < EndCycle Right"',
            ),
            ("trialNum", "The trial number (not reliable, bugged)."),
            ("multibiasBeta", "Scalar value indicating the strength of the multibias."),
            ("multibiasTau", "Scalar value indicating the history dependence of the multibias."),
            ("pairNum", "Integer value indicating which stimulus pair was shown, is a row index into stimulusTable."),
            ("wallGuide", "Boolean, whether wallGuides were active or not on this trial."),
            ("alpha_plus", "Scalar, how much to multiply step_size by when moving the moon beacon trigger forward."),
            ("alpha_minus", "Scalar, how much to multiply step_size by when moving the moon beacon trigger back."),
            ("moonBeaconEnabled", "Boolean, whether the moon beacon is enabled or not."),
            (
                "moonBeaconPos",
                "Scalar [cm], current position of the moon beacon trigger relative to the moonBeaconTrigger.",
            ),
            ("moonBeaconTrigger", 'String indicating the trigger point for the moon. Can be "Sa" or "Sb".'),
            ("step_size", "Scalar [cm], how much to move the moon beacon triggerpoint on each trial."),
            ("lsrON", "Whether the laser is on for a given trial, 0=laser off, 1=laser on"),
            ("iLaserOn", "Virmen iteration when laser is turn on (if it is, otherwise 0)"),
            ("iLaserOff", "Virmen iteration when laser is turn off (if it is, otherwise 0)"),
            ("moonDistHint", "Distance from start at which moon beacon appears"),
            ("forcedChoice", "Whether a trial is a forced choice L-maze environment, 0=T-maze, 1=L-maze"),
        ]

        for column_name, desc in trial_columns:
            data = [trial[column_name] for trial in trials if column_name in trial]
            if data:
                nwbfile.add_trial_column(name=column_name, description=desc, data=data)

        if "trial_choice" in converted_metadata:
            nwbfile.add_trial_column(
                name="choice",
                description="choice (L=Left,R=Right,nil=Trial violation)",
                data=converted_metadata["trial_choice"],
            )
        if "trial_type" in converted_metadata:
            nwbfile.add_trial_column(
                name="trial_type", description="type of trial (L=Left,R=Right)", data=converted_metadata["trial_type"]
            )

        # --------------------- Processed cue timing and position -------------------- #
        # -------- This information stays the same throughout the specific trial ------ #

        left_cue_presence = [trial["cueCombo"][0] if len(trial["cueCombo"]) else trial["cueCombo"] for trial in trials]

        right_cue_presence = [trial["cueCombo"][1] if len(trial["cueCombo"]) else trial["cueCombo"] for trial in trials]

        # Cue onset/offset times are resolved by indexing into the aligned timestamp
        # source (synchronized if available, otherwise Virmen internal timing) so they
        # remain in the same reference frame as Position/Velocity and iteration markers.
        left_cue_onset = []
        for trial_idx, trial in enumerate(trials):
            if np.any(trial["cueOnset"][0]):
                indices = np.minimum(trial["cueOnset"][0], trial["cueOnset"][0] - 1)
                left_cue_onset_time = self._local_frames_to_timestamps(trial_idx, indices)
            else:
                left_cue_onset_time = trial["cueOnset"][0]
            left_cue_onset.append(left_cue_onset_time)

        right_cue_onset = []
        for trial_idx, trial in enumerate(trials):
            if np.any(trial["cueOnset"][1]):
                indices = np.minimum(trial["cueOnset"][1], trial["cueOnset"][1] - 1)
                right_cue_onset_time = self._local_frames_to_timestamps(trial_idx, indices)
            else:
                right_cue_onset_time = trial["cueOnset"][1]
            right_cue_onset.append(right_cue_onset_time)

        left_cue_offset = []
        for trial_idx, trial in enumerate(trials):
            if np.any(trial["cueOffset"][0]):
                indices = np.minimum(trial["cueOffset"][0], trial["cueOffset"][0] - 1)
                left_cue_offset_time = self._local_frames_to_timestamps(trial_idx, indices)
            else:
                left_cue_offset_time = trial["cueOffset"][0]
            left_cue_offset.append(left_cue_offset_time)

        right_cue_offset = []
        for trial_idx, trial in enumerate(trials):
            if np.any(trial["cueOffset"][1]):
                indices = np.minimum(trial["cueOffset"][1], trial["cueOffset"][1] - 1)
                right_cue_offset_time = self._local_frames_to_timestamps(trial_idx, indices)
            else:
                right_cue_offset_time = trial["cueOffset"][1]
            right_cue_offset.append(right_cue_offset_time)

        left_cue_position = [trial["cuePos"][0] if len(trial["cuePos"]) else trial["cuePos"] for trial in trials]

        right_cue_position = [trial["cuePos"][1] if len(trial["cuePos"]) else trial["cuePos"] for trial in trials]

        trial_columns = [
            (
                "left_cue_presence",
                "Indicates whether the nth cue appeared on the left",
            ),
            (
                "right_cue_presence",
                "Indicates whether the nth cue appeared on the right",
            ),
            ("left_cue_onset", "Onset times of left cues"),
            ("right_cue_onset", "Onset times of right cues"),
            ("left_cue_offset", "Offset times of left cues"),
            ("right_cue_offset", "Offset times of right cues"),
            ("left_cue_position", "Position of left cues"),
            ("right_cue_position", "Position of right cues"),
        ]

        if "baseCycles" in trials[0]:
            baseCycles = [trial["baseCycles"] if len(trial["baseCycles"]) else trial["baseCycles"] for trial in trials]

            trial_columns.append(
                ("baseCycles", "The base set of spatial frequencies from which StartCycle and EndCycle can be drawn.")
            )

        left_licks = []
        right_licks = []

        if "licks" in trials[0]:
            for trial_idx, trial in enumerate(trials):
                left_lick_frames = (trial["licks"][0][(trial["licks"][1] == 1)] - 1).astype(int)
                left_stuff = self._local_frames_to_timestamps(trial_idx, left_lick_frames)
                left_licks.append(left_stuff)
                right_lick_frames = (trial["licks"][0][(trial["licks"][1] == 2)] - 1).astype(int)
                right_stuff = self._local_frames_to_timestamps(trial_idx, right_lick_frames)
                right_licks.append(right_stuff)

            trial_columns.extend(
                [
                    ("left_licks", "Offset times of left cues"),
                    ("right_licks", "Offset times of right cues"),
                ]
            )

        # Create StimulusTable groups for each trial that has stimulus table data
        stimulus_processing_module = check_module(nwbfile, "stimulus", "contains stimulus table data for trials")

        for trial_idx, trial in enumerate(trials):
            if "stimulusTable" in trial and len(trial["stimulusTable"]) > 0:
                # Extract the 8 columns from the stimulus table
                stimulus_data = trial["stimulusTable"]

                # Create a StimulusTable for this trial
                stimulus_table = StimulusTable(
                    name=f"trial_{trial_idx}_stimulus_table", description=f"Stimulus table for trial {trial_idx}"
                )

                # Add the 8 columns according to the schema
                stimulus_table.add_column(
                    name="stimulusPairIndex",
                    description="First column description",
                    data=stimulus_data[:, 0].astype(int),
                )
                stimulus_table.add_column(
                    name="priorProb", description="Second column description", data=stimulus_data[:, 1].astype(float)
                )
                stimulus_table.add_column(
                    name="Side",
                    description="Third column description",
                    data=[str(side) for side in stimulus_data[:, 2]],
                )
                stimulus_table.add_column(
                    name="Sa", description="Fourth column description", data=stimulus_data[:, 3].astype(int)
                )
                stimulus_table.add_column(
                    name="Sb", description="Fifth column description", data=stimulus_data[:, 4].astype(int)
                )
                stimulus_table.add_column(
                    name="FracHit",
                    description="Sixth column description (can be NaN)",
                    data=stimulus_data[:, 5].astype(int),
                )
                stimulus_table.add_column(
                    name="Ntotal", description="Seventh column description", data=stimulus_data[:, 6].astype(int)
                )
                stimulus_table.add_column(
                    name="posteriorProb",
                    description="Eighth column description",
                    data=stimulus_data[:, 7].astype(float),
                )

                # Add the stimulus table to the processing module
                stimulus_processing_module.add_data_interface(stimulus_table)

        for variable_name, description in trial_columns:
            try:
                variable = locals()[variable_name]
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Variable {variable_name} not found, skipping. Error: {e}")
                continue
            try:
                create_and_store_indexed_array(
                    ndarray=variable,
                    array_name=variable_name,
                    description=description,
                    nwbfile=nwbfile,
                )
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to add indexed array for {variable_name}: {e}")
                # Re-raise to see what's happening
                raise

        # ------------------ Processed position, velocity, viewAngle ----------------- #
        # ---------- This information changes throughout the specific trial ----------- #

        if self.verbose:
            print("Adding behavioral timeseries data (Position, ViewAngle, Velocity, Collision)...")

        # Use aligned timestamps if available (from temporal alignment), otherwise use original timestamps
        all_timestamps = self.get_timestamps()

        pos_obj = Position(name="Position")
        view_angle_obj = CompassDirection(name="ViewAngle")

        pos_data = np.empty((0, 2))
        velocity_data = np.empty_like(pos_data)
        view_angle_data = []
        collision = []
        timestamp_indices = []  # Track which timestamps correspond to each frame

        # Derive seconds-from-session-start columns for iteration-based keys (e.g., iCueEntry -> iCueEntrySeconds)
        # These use synchronized timestamps if available, otherwise Virmen internal timing
        iteration_keys = sorted(
            {
                key
                for trial in trials
                for key in trial
                if isinstance(key, str) and re.match(r"^i[A-Z]", key)
            }
        )

        # Add session-relative timestamp columns for each iteration marker
        for key in iteration_keys:
            values_seconds = []
            for trial_idx, trial in enumerate(trials):
                iteration_val = trial.get(key, np.nan)
                timestamp = self._convert_trial_iteration_to_timestamp(trial_idx, iteration_val)
                values_seconds.append(timestamp)

            seconds_name = f"{key}Seconds"
            seconds_desc = (
                f"Time in seconds from session start when {key} occurred. "
                f"Uses synchronized timestamps if available, otherwise Virmen internal timing."
            )
            nwbfile.add_trial_column(name=seconds_name, description=seconds_desc, data=values_seconds)

        current_timestamp_idx = 0
        for trial in trials:
            # Track indices for this trial's frames
            trial_frame_count = len(trial["time"])
            trial_indices = list(range(current_timestamp_idx, current_timestamp_idx + trial_frame_count))
            timestamp_indices.extend(trial_indices)
            current_timestamp_idx += trial_frame_count

            # Padding exists since the time array is longer than all the other arrays.
            # The other arrays are all the same length
            padding = np.full((trial["time"].shape[0] - trial["position"].shape[0], 2), np.nan)
            trial_position = trial["position"][:, :-1]
            trial_velocity = trial["velocity"][:, :-1]
            trial_view_angle = trial["position"][:, -1]
            trial_collision = trial["collision"]
            pos_data = np.concatenate([pos_data, trial_position, padding], axis=0)
            velocity_data = np.concatenate([velocity_data, trial_velocity, padding], axis=0)
            view_angle_data = np.concatenate([view_angle_data, trial_view_angle, padding[:, 0]], axis=0)
            collision = np.concatenate([collision, trial_collision, padding[:, 0]], axis=0)

        # Slice timestamps to match the data length
        timestamps = all_timestamps[timestamp_indices]


        # Create a Time timeseries for explicit time values
        time = TimeSeries(
            name="Time",
            data=H5DataIO(timestamps, compression="gzip"),
            unit="s",
            resolution=np.nan,
            timestamps=H5DataIO(timestamps, compression="gzip"),
        )

        pos_obj.add_spatial_series(
            SpatialSeries(
                name="Position",
                data=H5DataIO(pos_data, compression="gzip"),
                reference_frame="(0,-80) is the start of the 'sample' region (or 'cue' region) which varies by maze and task.",  # noqa: E501
                description="The position of the animal by ViRMEN iteration.",
                unit="cm",
                # conversion=0.01,
                resolution=np.nan,
                timestamps=H5DataIO(timestamps, compression="gzip"),
            )
        )
        velocity_ts = TimeSeries(
            name="Velocity",
            data=H5DataIO(velocity_data, compression="gzip"),
            description="The velocity of the animal by ViRMEN iteration.",
            unit="cm/s",
            resolution=np.nan,
            timestamps=H5DataIO(timestamps, compression="gzip"),
        )

        view_angle_obj.add_spatial_series(
            SpatialSeries(
                name="ViewAngle",
                data=H5DataIO(view_angle_data, compression="gzip"),
                reference_frame="unknown",
                description="The velocity view angle of the animal by ViRMEN iteration in the unit of degrees.",
                unit="degrees",
                resolution=np.nan,
                timestamps=H5DataIO(timestamps, compression="gzip"),
            )
        )
        collision_ts = TimeSeries(
            name="Collision",
            data=H5DataIO(collision, compression="gzip"),
            unit="bool",
            resolution=np.nan,
            timestamps=H5DataIO(timestamps, compression="gzip"),
            description="boolean to indicate for each frame whether collision was detected",
        )

        behavioral_processing_module = check_module(nwbfile, "behavior", "contains processed behavioral data")
        behavioral_processing_module.add(pos_obj)
        behavioral_processing_module.add(velocity_ts)
        behavioral_processing_module.add(view_angle_obj)
        behavioral_processing_module.add(time)
        behavioral_processing_module.add(collision_ts)

        if self.verbose:
            print(f"✓ Added {len(behavioral_processing_module.data_interfaces)} behavioral data interfaces to NWB file")
            print(f"  Data samples: {len(pos_data)}, Timestamps: {len(timestamps)}")

        return nwbfile
