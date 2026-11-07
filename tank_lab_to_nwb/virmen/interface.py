"""The neuroconv interface to a ViRMEn behavior log."""

from dataclasses import replace
from pathlib import Path

import numpy as np
from neuroconv.basetemporalalignmentinterface import BaseTemporalAlignmentInterface
from neuroconv.utils import dict_deep_update
from pydantic import FilePath

from .convert import PLACEHOLDER_METADATA, add_session_to_nwbfile
from .loader import load_log
from .session import normalize_log


class VirmenDataInterface(BaseTemporalAlignmentInterface):
    """Conversion of a ViRMEn log, driven by the field registry in ndx-tank-metadata."""

    def __init__(self, file_path: FilePath, verbose: bool = False):
        super().__init__(file_path=file_path, verbose=verbose)
        file_path = Path(file_path)
        self.session = normalize_log(load_log(file_path), file=file_path)
        self._aligned = None

    @classmethod
    def get_source_schema(cls):
        return dict(required=["file_path"], properties=dict(file_path=dict(type="string")))

    def get_metadata(self):
        placeholders = {key: value for key, value in PLACEHOLDER_METADATA.items() if key != "subject"}
        from_log = dict(
            NWBFile=dict(session_start_time=self.session.session_start_time, session_id=self.session.session_id, **placeholders),
            Subject=dict(subject_id=self.session.subject_id, **PLACEHOLDER_METADATA["subject"]),
        )
        return dict_deep_update(super().get_metadata(), from_log)

    def _all_frames(self):
        """The clock of every logged frame, including the ITI: the longest one in the file."""
        return next((s.timestamps for s in self.session.series if s.clock == "all"), None)

    def get_original_timestamps(self) -> np.ndarray:
        return self._all_frames()

    def get_timestamps(self) -> np.ndarray:
        return self.get_original_timestamps() if self._aligned is None else self._aligned

    def set_aligned_timestamps(self, aligned_timestamps: np.ndarray) -> None:
        aligned_timestamps = np.asarray(aligned_timestamps, dtype="float64")
        if aligned_timestamps.size != self.get_original_timestamps().size:
            raise ValueError(f"expected {self.get_original_timestamps().size} aligned timestamps, got {aligned_timestamps.size}")
        if np.any(np.diff(aligned_timestamps) < 0):
            raise ValueError("aligned timestamps run backwards")
        self._aligned = aligned_timestamps

    def _aligned_session(self):
        """The session with every time moved onto the aligned clock."""
        original = self.get_original_timestamps()

        def move(times):
            # np.interp clamps outside the frames, and a block or trial can end after the last frame.
            times = np.asarray(times, dtype="float64")
            moved = np.interp(times, original, self._aligned)
            before, after = times < original[0], times > original[-1]
            moved[before] = times[before] + self._aligned[0] - original[0]
            moved[after] = times[after] + self._aligned[-1] - original[-1]
            return moved

        def move_table(columns):
            return [replace(c, data=move(c.data)) if c.name in ("start_time", "stop_time") else c for c in columns]

        series = [replace(s, timestamps=move(s.timestamps)) for s in self.session.series]
        return replace(self.session, trials=move_table(self.session.trials), epochs=move_table(self.session.epochs), series=series)

    def add_to_nwbfile(self, nwbfile, metadata=None, **conversion_options):
        add_session_to_nwbfile(nwbfile, self.session if self._aligned is None else self._aligned_session())
