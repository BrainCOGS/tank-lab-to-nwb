"""Normalize a decoded ViRMEn log through the field registry into typed, NWB-ready values.

Every value goes through coerce(), so this module either returns exactly what the log holds, in the declared dtype, or
raises. Nothing is cast later: the NWB writer receives arrays that already have their final dtype.
"""

import json
from dataclasses import dataclass, field as dataclass_field

import numpy as np
from ndx_tank_metadata.registry import Field, load_registry

from .coerce import LossyConversionError, coerce
from .loader import Enum, FunctionHandle, records
from .timebase import BLOCK_STAMP, SESSION_END, ClockAnchorError, ClockMap, datevec_to_datetime, map_clock, seconds_between

TRIAL = "log.block.trial."
BLOCK = "log.block."
RIG = "log.version.rig."
MAZE = "log.version.mazes."
ANIMAL = "log.animal."
ABSENT = object()
UNREGISTERED = "Not in the field registry; placed automatically by level and shape."
# Struct fields the training GUI leaves as '' or its popup default 1 when nothing is set.
_UNSET_AS_ONE = {ANIMAL + "softwareParams", ANIMAL + "stimulationProtocol"}
_CELL_STRUCTURES = {ANIMAL + "mainDrawMethod", ANIMAL + "warmupDrawMethod", TRIAL + "stimulusTable"}
_RAW_CHOICES = {1: "L", 2: "R"}


class ConversionError(LossyConversionError):
    """A log value doesn't have the shape or form its registry entry declares."""


@dataclass
class Column:
    name: str
    dtype: str
    description: str
    data: np.ndarray
    index: np.ndarray | None = None  # end of each row in data, or of each inner cell when doubly ragged
    index_of_index: np.ndarray | None = None
    region: bool = False


@dataclass
class Series:
    name: str
    data: np.ndarray
    timestamps: np.ndarray
    unit: str
    description: str
    clock: str  # "frames" (position rows) or "all" (time rows, including the ITI)
    container: str | None = None  # None, or the pynwb interface that holds this series
    spatial: bool = False


@dataclass
class Coverage:
    written: dict = dataclass_field(default_factory=dict)
    settings: list = dataclass_field(default_factory=list)
    dropped: dict = dataclass_field(default_factory=dict)
    omitted: dict = dataclass_field(default_factory=dict)
    structural: list = dataclass_field(default_factory=list)
    unregistered: list = dataclass_field(default_factory=list)
    clock: dict = dataclass_field(default_factory=dict)

    def as_dict(self):
        return {
            "written": dict(sorted(self.written.items())),
            "settings": sorted(set(self.settings)),
            "dropped": dict(sorted(self.dropped.items())),
            "omitted": dict(sorted(self.omitted.items())),
            "structural": sorted(set(self.structural)),
            "unregistered": sorted(set(self.unregistered)),
            "clock": self.clock,
        }


@dataclass
class NormalizedSession:
    file: str
    subject_id: str
    session_id: str
    session_start_time: object
    trials: list
    epochs: list
    mazes: list
    stimulus_pairs: list
    rig: dict
    animal: dict
    lab_meta: dict
    settings: list
    series: list
    coverage: Coverage


def _resolve(record, subpath):
    value = record
    for part in subpath.split("."):
        if not isinstance(value, dict) or part not in value:
            return ABSENT
        value = value[part]
    return value


# Beyond this, a block's end and session.end disagree by more than shutdown code explains (0.7 s at most in the 37 logs).
END_MISS_FOR_REVIEW = 1.0


def _is_empty(value):
    return value is ABSENT or value is None or (isinstance(value, str) and value == "") or (isinstance(value, np.ndarray) and value.size == 0)


def _ragged_index(lengths):
    return np.cumsum(np.asarray(lengths, dtype="uint32"), dtype="uint32") if len(lengths) else np.zeros(0, dtype="uint32")


def _concat(arrays, dtype):
    if not arrays:
        return coerce(np.zeros(0), dtype, file="", field="") if dtype != "text" else np.empty(0, dtype=object)
    return np.concatenate(arrays)


class _Normalizer:
    def __init__(self, log, file, registry):
        self.log = log
        self.file = file
        self.registry = registry
        self.coverage = Coverage()
        self.settings = []

    def fail(self, field_path, message, row=None):
        where = f"row {row}: " if row is not None else ""
        raise ConversionError(f"{self.file}: {field_path}: {where}{message}")

    def coerce(self, values, field, row=None):
        return coerce(values, field.dtype, file=self.file, field=field.path, row=row, codes=field.codes, missing=field.missing)

    # ---- one value ----

    def text_value(self, field, value, row=None):
        """A value of a text field: an enum, a function handle or a string."""
        if isinstance(value, (Enum, FunctionHandle)):
            return value.name
        if isinstance(value, str):
            return value
        if isinstance(value, np.ndarray) and value.size == 1 and field.transform == "enum_name" and "L" in (field.codes or {}):
            name = _RAW_CHOICES.get(value.item())  # one session logs choice as a plain 1 or 2
            if name is not None:
                return name
        self.coerce(np.asarray(value, dtype=object).reshape(-1), field, row)  # raises, naming the value
        self.fail(field.path, f"cannot be stored as text: {value!r}", row)

    def scalar_array(self, field, value, row=None):
        """A numeric scalar as a one-element array, or NaN for an allowed empty."""
        if field.transform == "str2double":
            try:
                return np.array([float(value)], dtype="float64")
            except (TypeError, ValueError):
                self.fail(field.path, f"is not a plain number: {value!r}", row)
        if _is_empty(value):
            if field.transform != "empty_to_missing":
                self.fail(field.path, "is empty, and the registry declares no missing value", row)
            return np.array([np.nan])
        if not isinstance(value, np.ndarray):
            self.fail(field.path, f"expected a number, got {value!r}", row)
        if value.size != 1:
            self.fail(field.path, f"expected a scalar, got shape {value.shape}", row)
        return value.ravel()

    def fixed_array(self, field, value, row=None):
        """A value of a fixed-width field as a 1-D array of that width."""
        width = field.shape[0]
        if not isinstance(value, np.ndarray):
            self.fail(field.path, f"expected {width} values, got {value!r}", row)
        flat = value.ravel()
        if flat.size == 1 and field.transform == "complementary_pair":
            flat = np.concatenate([flat, 1 - flat])
        elif flat.size == 1 and field.transform == "pad_pair":
            flat = np.concatenate([flat.astype("float64"), [np.nan]])
        if flat.size != width:
            self.fail(field.path, f"expected {width} values, got shape {value.shape}", row)
        return flat

    # ---- table columns ----

    def _ragged_column(self, field, name, cells):
        flats = [self.coerce(np.zeros(0) if _is_empty(cell) else np.asarray(cell).ravel(), field, row) for row, cell in enumerate(cells)]
        return Column(name, field.dtype, field.doc, _concat(flats, field.dtype), _ragged_index([len(flat) for flat in flats]))

    def _doubly_ragged_column(self, field, name, cells):
        flats, inner, outer = [], [], []
        for row, cell in enumerate(cells):
            members = [] if _is_empty(cell) else list(np.asarray(cell, dtype=object).flat)
            for member in members:
                flats.append(self.coerce(np.asarray(member).ravel(), field, row))
                inner.append(len(flats[-1]))
            outer.append(len(members))
        return Column(name, field.dtype, field.doc, _concat(flats, field.dtype), _ragged_index(inner), _ragged_index(outer))

    def _split_left_right(self, field, cell, row):
        if _is_empty(cell):
            return np.zeros(0), np.zeros(0)
        if isinstance(cell, np.ndarray) and cell.dtype == object and cell.size == 2:
            left, right = cell.flat
            return left, right
        if isinstance(cell, np.ndarray) and cell.ndim == 2 and cell.shape[0] == 2:
            return cell[0], cell[1]
        self.fail(field.path, f"expected a left/right pair, got {cell!r}", row)

    def _split_licks(self, field, cell, row):
        """(iteration, value) of a 2xN array, or None for a preallocated array nothing filled."""
        if _is_empty(cell):
            return np.zeros(0), np.zeros(0)
        all_nan = cell.dtype.kind == "f" and bool(np.isnan(cell).all())
        if cell.shape[0] != 2:
            if not all_nan:
                self.fail(field.path, f"a preallocated {cell.shape} array holds real values, so it can't be dropped", row)
            return None
        if all_nan:
            return np.zeros(0), np.zeros(0)
        if cell.dtype.kind == "f" and np.isnan(cell).any():
            self.fail(field.path, "is only partly NaN", row)
        return cell[0], cell[1]

    def _pair_columns(self, field, split):
        left, right = zip(*split) if split else ((), ())
        return [self._ragged_column(field, field.nwb_names[0], left), self._ragged_column(field, field.nwb_names[1], right)]

    def columns_for(self, field, cells):
        """The columns of one registered field, given its value in every row. Returns [] for an empty column."""
        if field.transform == "left_right":
            return self._pair_columns(field, [self._split_left_right(field, cell, row) for row, cell in enumerate(cells)])
        if field.transform == "licks":
            split = [self._split_licks(field, cell, row) for row, cell in enumerate(cells)]
            preallocated = any(part is None for part in split)
            split = [(np.zeros(0), np.zeros(0)) if part is None else part for part in split]
            if not any(len(part[0]) for part in split):
                reason = field.notes if preallocated else "All-NaN in every trial of this session, which means no licks."
                self.coverage.omitted[field.path] = reason
                return []
            return self._pair_columns(field, split)
        name = field.name
        if field.shape == ():
            if field.dtype == "text":
                strings = np.empty(len(cells), dtype=object)
                for row, cell in enumerate(cells):
                    if cell is ABSENT:
                        self.fail(field.path, "is missing", row)
                    strings[row] = self.text_value(field, cell, row)
                return [Column(name, field.dtype, field.doc, self.coerce(strings, field))]
            arrays = [self.scalar_array(field, cell, row) for row, cell in enumerate(cells)]
            data = np.concatenate(arrays) if arrays else np.zeros(0)
            return [Column(name, field.dtype, field.doc, self.coerce(data, field))]
        if field.shape == (None,):
            return [self._ragged_column(field, name, cells)]
        if field.shape == (None, None):
            return [self._doubly_ragged_column(field, name, cells)]
        rows = [self.fixed_array(field, cell, row) for row, cell in enumerate(cells)]
        data = np.stack(rows) if rows else np.zeros((0, field.shape[0]))
        return [Column(name, field.dtype, field.doc, self.coerce(data, field))]

    def table(self, rows, prefix, destination, special=(), account=True):
        """Columns of one table from its registered fields, and what happened to everything else in its rows."""
        columns = []
        names = []
        for row in rows:
            names += [name for name in row if name not in names]
        for field in self.registry:
            if field.destination != destination or not field.path.startswith(prefix) or field.path in special:
                continue
            cells = [_resolve(row, field.path[len(prefix):]) for row in rows]
            if all(cell is ABSENT for cell in cells):
                continue
            new = self.columns_for(field, cells)
            kept = [column for column in new if column.index is None or column.data.size]
            if len(kept) < len(new):
                self.coverage.omitted[field.path] = "Every cell of " + ", ".join(c.name for c in new if c not in kept) + " is empty in this session."
            kept = self._without_all_nan(field, kept)
            columns += kept
            if kept:
                self.coverage.written[field.path] = [column.name for column in kept]
        if account:
            for name in names:
                self._account_for(prefix + name, rows, name, columns, prefix)
        return columns

    def _without_all_nan(self, field, columns):
        """Leave out a float column with no value in this session, and say so."""
        kept = []
        for column in columns:
            data = np.asarray(column.data)
            if column.index is None and data.dtype.kind == "f" and data.size and np.isnan(data).all():
                self.coverage.omitted[field.path] = f"Every value of {column.name} is NaN in this session."
            else:
                kept.append(column)
        return kept

    def _account_for(self, path, rows, name, columns, prefix):
        field = self.registry.get(path)
        if field is None:
            self._place_unregistered(path, rows, name, columns, prefix)
        elif field.destination == "drop":
            self.coverage.dropped[path] = field.notes
        elif field.destination == "structural":
            self.coverage.structural.append(path)
            if path == TRIAL.rstrip("."):  # the trial table accounts for its own fields
                return
            members = []
            for row in rows:
                value = row.get(name)
                if isinstance(value, dict):
                    members += [member for member in value if member not in members]
            for member in members:
                if self.registry.get(f"{path}.{member}") is None:
                    self._unregistered_settings(f"{path}.{member}", prefix, rows, f"{name}.{member}")

    # ---- fields the registry doesn't know ----

    def _unregistered_settings(self, path, prefix, rows, subpath):
        self.coverage.unregistered.append(path)
        for row_index, row in enumerate(rows):
            value = _resolve(row, subpath)
            if value is not ABSENT:
                self._add_setting(f"{self._row_key(prefix, row_index)}.{subpath}", value, UNREGISTERED, path)

    def _row_key(self, prefix, row_index):
        if prefix == TRIAL:
            block_index, trial_index = self._trial_keys[row_index]
            return f"log.block({block_index + 1}).trial({trial_index + 1})"
        return f"log.block({row_index + 1})"

    def _place_unregistered(self, path, rows, name, columns, prefix):
        values = [row[name] for row in rows if name in row]
        numeric = all(isinstance(v, np.ndarray) and v.dtype.kind in "fiub" for v in values)
        complete = len(values) == len(rows)
        if complete and all(isinstance(v, str) for v in values):
            field = Field(path=path, destination="trial_column", name=name, dtype="text", shape=(), description=UNREGISTERED)
        elif complete and numeric:
            result = np.result_type(*values)
            dtype = {"b": "bool"}.get(result.kind) or (result.name if result.kind == "f" else "int32")
            dtype = "float64" if dtype not in ("bool", "float32", "float64", "int32") else dtype
            if prefix == TRIAL and all(v.ndim and v.shape[0] == len(self._frame_rows(row)) > 1 for v, row in zip(values, rows)):
                self._unregistered_series.append((path, name, dtype))
                self.coverage.unregistered.append(path)
                return
            shape = () if all(v.size == 1 for v in values) else (None,)
            field = Field(path=path, destination="trial_column", name=name, dtype=dtype, shape=shape, description=UNREGISTERED)
        else:
            self._unregistered_settings(path, prefix, rows, name)
            return
        self.coverage.unregistered.append(path)
        new = self.columns_for(field, [row[name] for row in rows])
        columns += new
        self.coverage.written[path] = [column.name for column in new]

    @staticmethod
    def _frame_rows(row):
        return row.get("position", np.zeros((0, 0)))

    # ---- settings ----

    def _add_setting(self, key, value, meaning, path):
        self.settings.append((key, encode_setting(value), meaning))
        self.coverage.settings.append(path)

    def _walk_settings(self, value, path, key):
        field = self.registry.get(path)
        if isinstance(value, list) and all(isinstance(item, dict) for item in value) and value:
            for index, item in enumerate(value):
                self._walk_settings(item, path, f"{key}({index + 1})")
            return
        if field is not None and field.destination == "drop":
            self.coverage.dropped[path] = field.notes
            return
        if isinstance(value, dict):
            if field is not None and field.destination == "structural":
                self.coverage.structural.append(path)
            for name, member in value.items():
                self._walk_settings(member, f"{path}.{name}", f"{key}.{name}")
            return
        if field is None:
            self.coverage.unregistered.append(path)
            self._add_setting(key, value, UNREGISTERED, path)
        elif field.destination == "settings":
            self._add_setting(key, value, field.doc, path)
        elif field.destination == "structural":
            self.coverage.structural.append(path)
            self._check_structural_leaf(path, value)

    def _check_structural_leaf(self, path, value):
        if _is_empty(value) or path in _CELL_STRUCTURES:
            return
        if path in _UNSET_AS_ONE and isinstance(value, np.ndarray) and value.size == 1 and value.item() == 1:
            return
        self.fail(path, f"expected a struct or nothing, got {value!r}")

    def collect_settings(self):
        for name, value in self.log.items():
            if name in ("block", "currentTrial"):
                field = self.registry.get(f"log.{name}")
                if field is not None and field.destination == "drop":
                    self.coverage.dropped[field.path] = field.notes
                elif field is not None:
                    self.coverage.structural.append(field.path)
                continue
            self._walk_settings(value, f"log.{name}", f"log.{name}")

    # ---- single-record attributes ----

    def attribute(self, field, value):
        """The typed value of an attribute, or ABSENT to leave it out."""
        if value is ABSENT:
            return ABSENT
        if _is_empty(value) and (field.transform == "empty_to_missing" or (field.transform == "datevec" and not field.required)):
            return ABSENT
        if field.dtype == "isodatetime":
            return datevec_to_datetime(value)
        if field.dtype == "text":
            text = self.text_value(field, value)
            self.coerce(np.array([text], dtype=object), field)
            return text
        if field.shape == ():
            result = self.coerce(self.scalar_array(field, value), field)[0]
            return bool(result) if field.dtype == "bool" else result
        if None in field.shape:
            if len(field.shape) == 2:
                if value.ndim != 2 or value.shape[0] != field.shape[0]:
                    self.fail(field.path, f"expected shape {field.shape}, got {value.shape}")
                return self.coerce(value, field)
            return self.coerce(value.ravel(), field)
        return self.coerce(self.fixed_array(field, value), field)

    def attributes(self, destination, values_of):
        attributes = {}
        for field in self.registry:
            if field.destination != destination:
                continue
            value = self.attribute(field, values_of(field))
            if value is not ABSENT:
                attributes[field.name] = value
                self.coverage.written[field.path] = [field.name]
        return attributes

    # ---- the session ----

    def normalize(self):
        log = self.log
        versions = records(log["version"])
        self.version = versions[0]
        session = log["session"]
        session = session if isinstance(session, dict) else session[0]
        start = datevec_to_datetime(session["start"])
        animal = log["animal"]
        blocks = records(log["block"])
        trials = [(block_index, trial_index, trial) for block_index, block in enumerate(blocks) for trial_index, trial in enumerate(records(block["trial"]))]
        self._trial_keys = [(b, t) for b, t, _ in trials]
        self._unregistered_series = []
        trial_rows = [trial for _, _, trial in trials]

        clock = self._clock(blocks, session, start)
        offsets = clock.offsets
        trial_columns = self._trial_columns(trials, trial_rows, offsets)
        pairs, region = self._stimulus_pairs(blocks, trials, trial_columns)
        if region is not None:
            trial_columns.append(region)
        epoch_columns = self._epoch_columns(blocks, start, clock)
        series = self._series(trial_rows, next(c for c in trial_columns if c.name == "start_time").data)

        # The settings walk accounts for the rest of each maze.
        mazes = self.table(records(self.version.get("mazes", [])), MAZE, "maze_column", account=False)
        rig = self.attributes("rig_attribute", lambda f: _resolve(self.version.get("rig", {}), f.path[len(RIG):]))
        animal_attributes = self.attributes("animal_attribute", lambda f: self._animal_value(animal, f))
        lab_meta = self.attributes("lab_meta_attribute", lambda f: self._lab_value(session, f))
        lab_meta.update(self._performance(trial_rows))

        self.collect_settings()
        self._check_later_versions(versions)
        self._record_clock(clock, session, start, epoch_columns)
        subject_field = self.registry["log.animal.name"]
        self.coverage.written[subject_field.path] = [subject_field.name]
        self.coverage.written["log.session.start"] = ["session_start_time"]
        return NormalizedSession(
            file=self.file, subject_id=animal["name"], session_id=start.strftime("%Y%m%d"), session_start_time=start,
            trials=trial_columns, epochs=epoch_columns, mazes=mazes, stimulus_pairs=pairs, rig=rig, animal=animal_attributes,
            lab_meta=lab_meta, settings=self.settings, series=series, coverage=self.coverage,
        )

    def _clock(self, blocks, session, session_start):
        """The clock map, one entry per block: a block without trials takes the running offset and has no reading."""
        starts, firsts, ends, durations, used = [], [], [], [], []
        previous_end = None
        for index, block in enumerate(blocks):
            block_trials = records(block["trial"])
            if not block_trials:
                continue
            for trial in block_trials:
                if np.isnan(np.asarray(trial["start"], dtype="float64")).any():
                    self.fail(TRIAL + "start", "is NaN")
            starts.append(seconds_between(session_start, datevec_to_datetime(block["start"])))
            firsts.append(float(block_trials[0]["start"].item()))
            ends.append(previous_end)
            durations.append(float(block["duration"].item()))
            used.append(index)
            last = block_trials[-1]
            previous_end = float(last["start"].item()) + float(np.asarray(last["time"], dtype="float64").ravel()[-1])
        try:
            found = map_clock(starts, firsts, ends, durations, self._seconds_since(session_start, session.get("end")), self._elapsed_zero(session_start))
        except ClockAnchorError as error:
            self.fail(BLOCK + "start", str(error))
        where = {index: position for position, index in enumerate(used)}
        offsets, readings, anchors, max_errors = [], [], [], []
        position = 0
        for index in range(len(blocks)):
            position = where.get(index, position)
            has_trials = index in where
            offsets.append(found.offsets[position] if used else 0.0)
            readings.append(found.block_start_readings[position] if has_trials else None)
            anchors.append(found.anchors[position] if used else BLOCK_STAMP)
            max_errors.append(found.max_errors[position] if used else None)
        return ClockMap(offsets, readings, anchors, max_errors)

    @staticmethod
    def _seconds_since(session_start, datevec):
        return None if _is_empty(datevec) else seconds_between(session_start, datevec_to_datetime(datevec))

    def _elapsed_zero(self, session_start):
        """When the elapsed clock read zero, from the engine's own timer. Only sessions that recorded video save the second."""
        engine_start = self._seconds_since(session_start, self.log.get("initialTimestamp"))
        until_zero = self.log.get("timeElapsedFirstTrial")
        return None if engine_start is None or _is_empty(until_zero) else engine_start + float(np.asarray(until_zero).item())

    def _record_clock(self, clock, session, session_start, epoch_columns):
        """Say in the file and the coverage report how its times were found, and whether a person should check them."""
        session_end = self._seconds_since(session_start, session.get("end"))
        stops = next(c for c in epoch_columns if c.name == "stop_time").data
        end_miss = None if session_end is None or not len(stops) else float(session_end - stops[-1])
        rows = (
            ("derived.clock_offset", [float(offset) for offset in clock.offsets],
             "Seconds added to ViRMEn's elapsed clock in each block to get seconds since session_start_time."),
            ("derived.clock_anchor", clock.anchors,
             "How each block's offset was found: block_stamp, its wall-clock stamp paired with its first trial; engine_timestamps, initialTimestamp + "
             "timeElapsedFirstTrial; session_end_upper_bound, session.end - block duration, because the block was stamped during setup and those weren't saved."),
            ("derived.clock_max_error", clock.max_errors,
             "How far too late each block's offset could be, in seconds, or null if its anchor isn't a bound. To be checked against other evidence."),
            ("derived.clock_session_end_minus_block_end", end_miss,
             "Seconds between the last block's end on the session clock and session.end, which is read after the session's shutdown code."),
        )
        self.settings.extend((key, json.dumps(value), meaning) for key, value, meaning in rows)
        errors = [error for error in clock.max_errors if error is not None]
        self.coverage.clock = dict(
            anchors=clock.anchors,
            offsets=[float(offset) for offset in clock.offsets],
            session_end_minus_block_end=end_miss,
            max_error_seconds=max(errors) if errors else None,
            needs_review=SESSION_END in clock.anchors or (end_miss is not None and abs(end_miss) > END_MISS_FOR_REVIEW),
        )

    def _trial_columns(self, trials, rows, offsets):
        special = {TRIAL + "start"}
        columns = self.table(rows, TRIAL, "trial_column", special=special | {f"{TRIAL}stimulusTable.{n}" for n in ("Hits", "NTotal", "PostProb")})
        start_field = self.registry[TRIAL + "start"]
        raw = np.array([float(trial["start"].item()) for _, _, trial in trials], dtype="float64")
        shift = np.array([offsets[block_index] for block_index, _, _ in trials], dtype="float64")
        start_time = raw + shift
        duration = np.array([float(trial["duration"].item()) for _, _, trial in trials], dtype="float64")
        columns.insert(0, Column("stop_time", "float64", "End of the trial, in seconds since session_start_time.", start_time + duration))
        columns.insert(0, Column("start_time", "float64", start_field.doc, start_time))
        self.coverage.written[start_field.path] = ["start_time", "stop_time"]
        return columns

    def _epoch_columns(self, blocks, session_start, clock):
        columns = self.table(blocks, BLOCK, "epoch_column", special={BLOCK + "start"})
        field = self.registry[BLOCK + "start"]
        stamps = np.array([seconds_between(session_start, datevec_to_datetime(block["start"])) for block in blocks], dtype="float64")
        # A block's elapsed start plus the shared offset; a block without trials has only its stamp.
        start_time = np.array([stamp if reading is None else reading + offset for stamp, reading, offset in zip(stamps, clock.block_start_readings, clock.offsets)], dtype="float64")
        duration = np.array([float(block["duration"].item()) for block in blocks], dtype="float64")
        columns.append(Column(field.name, "float64", field.doc, stamps))
        columns.insert(0, Column("stop_time", "float64", "End of the block, in seconds since session_start_time.", start_time + duration))
        columns.insert(0, Column("start_time", "float64", "Start of the block, in seconds since session_start_time, on the same clock as the trials.", start_time))
        self.coverage.written[field.path] = [field.name]
        return columns

    def _stimulus_pairs(self, blocks, trials, trial_columns):
        """StimulusPairs rows, the running trial columns (added to trial_columns), and the stimulus_pair region column."""
        if not any("stimulusTable" in trial for _, _, trial in trials):
            return [], None
        static_names = ("PairNum", "Prob", "Side", "Sa", "Sb")
        running_names = ("Hits", "NTotal", "PostProb")
        positions = {name: position for position, name in enumerate(static_names + running_names)}
        by_field = {name: self.registry[f"{TRIAL}stimulusTable.{name}"] for name in static_names + running_names}
        static_rows, block_of_row, first_row_of_block, seen = [], [], {}, {}
        region, running = [], {name: [] for name in running_names}
        for row, (block_index, _, trial) in enumerate(trials):
            table = trial["stimulusTable"]
            static = [[table[r, positions[name]] for name in static_names] for r in range(table.shape[0])]
            signature = repr(static)
            if block_index not in seen:
                seen[block_index] = signature
                first_row_of_block[block_index] = len(static_rows)
                static_rows += static
                block_of_row += [block_index + 1] * len(static)
            elif seen[block_index] != signature:
                self.fail(TRIAL + "stimulusTable", "its static columns changed within a block", row)
            pair = float(trial["pairNum"].item())
            numbers = [float(np.asarray(entry[0]).item()) for entry in static]
            if pair not in numbers:
                self.fail(TRIAL + "pairNum", f"{pair} is not in the stimulus table", row)
            region.append(first_row_of_block[block_index] + numbers.index(pair))
            for name in running_names:
                running[name].append(np.concatenate([np.asarray(table[r, positions[name]]).ravel() for r in range(table.shape[0])]))
        for name in running_names:
            trial_columns.append(self._ragged_column(by_field[name], name, running[name]))
            self.coverage.written[by_field[name].path] = [name]
        columns = []
        block_field = self.registry["derived.stimulus_pairs.block"]
        columns.append(Column("block", "int32", block_field.doc, self.coerce(np.array(block_of_row), block_field)))
        for position, name in enumerate(static_names):
            field = by_field[name]
            cells = [row[position] for row in static_rows]
            columns += self.columns_for(field, cells)
            self.coverage.written[field.path] = [name]
        region_field = self.registry["derived.trial.stimulus_pair"]
        self.coverage.written[region_field.path] = [region_field.name]
        return columns, Column(region_field.name, "region", region_field.doc, np.array(region, dtype="int32"), region=True)

    def _animal_value(self, animal, field):
        for kind in ("main", "warmup"):
            for member, wanted in (("method", str), ("parameter", np.ndarray)):
                if field.path == f"{ANIMAL}{kind}DrawMethod.{member}":
                    cell = animal.get(f"{kind}DrawMethod", ABSENT)
                    if cell is ABSENT:
                        return ABSENT
                    matches = [item for item in cell.flat if isinstance(item, wanted)]
                    if len(matches) != 1:
                        self.fail(field.path, f"expected {{method, parameter}}, got {cell!r}")
                    return matches[0]
        return _resolve(animal, field.path[len(ANIMAL):])

    def _lab_value(self, session, field):
        if field.path.startswith("derived."):
            return ABSENT
        if field.path.startswith("log.version."):
            return _resolve(self.version, field.path[len("log.version."):])
        if field.path.startswith("log.session."):
            return _resolve(session, field.path[len("log.session."):])
        return _resolve(self.log, field.path[len("log."):])

    @staticmethod
    def _performance(rows):
        """Percentage of trials whose choice matched their rewarded side, when the task has choices at all."""
        names = [getattr(row.get("choice"), "name", None) for row in rows]
        if not any(name in ("L", "R") for name in names):
            return {}
        correct = sum(name == getattr(row.get("trialType"), "name", None) for name, row in zip(names, rows))
        return {"session_performance": np.float64(100 * correct / len(rows))}

    def _check_later_versions(self, versions):
        """Keep what a later log.version changes in the rig; any other change isn't representable, so it fails."""
        for number, version in enumerate(versions[1:], start=2):
            for name, value in version.items():
                first = self.version.get(name, ABSENT)
                field = self.registry.get(f"log.version.{name}")
                if field is not None and field.destination in ("settings", "drop"):
                    continue
                if name == "rig":
                    for member, member_value in value.items():
                        first_value = first.get(member, ABSENT)
                        destination = getattr(self.registry.get(RIG + member), "destination", None)
                        if destination != "settings" and encode_setting(member_value) != encode_setting(first_value):
                            self._add_setting(f"log.version({number}).rig.{member}", member_value, "Differs from log.version(1).", RIG + member)
                elif name not in ("mazes", "variables", "worlds") and encode_setting(value) != encode_setting(first):
                    self.fail(f"log.version({number}).{name}", "differs from log.version(1), which the file can't represent")
                elif name == "mazes" and encode_setting(value) != encode_setting(first):
                    self.fail(f"log.version({number}).mazes", "differ from log.version(1), which the file can't represent")

    def _series(self, rows, start_times):
        frame_stamps, all_stamps = [], []
        for row, trial in enumerate(rows):
            n_frames = trial["position"].shape[0]
            time = coerce(np.asarray(trial["time"]).ravel(), "float32", file=self.file, field=TRIAL + "time", row=row).astype("float64")
            if time.size < n_frames:
                self.fail(TRIAL + "time", f"has {time.size} rows but position has {n_frames}", row)
            frame_stamps.append(start_times[row] + time[:n_frames])
            all_stamps.append(start_times[row] + time)
        frames = np.concatenate(frame_stamps) if frame_stamps else np.zeros(0)
        every = np.concatenate(all_stamps) if all_stamps else np.zeros(0)
        self.coverage.written[TRIAL + "time"] = ["timestamps"]
        series = []

        def stacked(field, width, rows_of):
            parts = []
            for row, trial in enumerate(rows):
                value = trial.get(field.path[len(TRIAL):], ABSENT)
                expected = rows_of(trial)
                if value is ABSENT:
                    self.fail(field.path, "is missing from this trial", row)
                value = self._frame_value(field, value, expected, row)
                parts.append(self.coerce(value.reshape(expected, -1) if width is None else value.reshape(expected, width), field, row))
            if parts:
                return np.concatenate(parts)
            return np.zeros((0, width or 1), dtype=field.dtype)

        by_frames = lambda trial: trial["position"].shape[0]  # noqa: E731
        by_time = lambda trial: np.asarray(trial["time"]).size  # noqa: E731
        present = set(rows[0]) if rows else {"position", "velocity"}

        if "position" in present:
            field = self.registry[TRIAL + "position"]
            data = stacked(field, 3, by_frames)
            series.append(Series("position", data[:, :2], frames, "centimeters", "Position in the virtual maze (x, y).", "frames", "Position", True))
            # A plain TimeSeries: the angle is unwrapped, and a radian SpatialSeries may not pass 2*pi.
            series.append(Series("view_angle", data[:, 2], frames, "radians", "View angle, unwrapped.", "frames"))
            self.coverage.written[field.path] = ["position", "view_angle"]
        if "velocity" in present:
            field = self.registry[TRIAL + "velocity"]
            data = stacked(field, 3, by_frames)
            series.append(Series("velocity", data[:, :2], frames, "cm/s", "Velocity in the virtual maze (x, y).", "frames"))
            series.append(Series("AngularVelocity", data[:, 2], frames, "radians/s", "Angular velocity of the view angle.", "frames"))
            self.coverage.written[field.path] = ["velocity", "AngularVelocity"]
        for leaf, unit, clock, rows_of, width in (
            ("collision", "n.a.", "frames", by_frames, 1),
            ("frictionEngagedVec", "n.a.", "frames", by_frames, 1),
            ("sensorDots", "dots", "all", by_time, 5),
        ):
            if leaf not in present:
                continue
            field = self.registry[TRIAL + leaf]
            data = stacked(field, width, rows_of)
            series.append(Series(field.name, data[:, 0] if width == 1 else data, frames if clock == "frames" else every, unit, field.doc, clock))
            self.coverage.written[field.path] = [field.name]
        for path, name, dtype in self._unregistered_series:
            field = Field(path=path, destination="timeseries", name=name, dtype=dtype, shape=(None,), description=UNREGISTERED)
            data = stacked(field, None, by_frames)
            series.append(Series(name, data, frames, "n.a.", UNREGISTERED, "frames"))
            self.coverage.written[path] = [name]
        return series

    def _frame_value(self, field, value, expected, row):
        """Check a per-frame array's row count."""
        if value.shape[0] != expected:
            self.fail(field.path, f"has {value.shape[0]} rows where {expected} are expected", row)
        return value


def _plain(value):
    """A settings value as JSON-compatible data. Floats keep every bit: single precision widens exactly to a double."""
    if isinstance(value, (Enum, FunctionHandle)):
        return value.name
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {name: _plain(member) for name, member in value.items()}
    if isinstance(value, list):
        return [_plain(member) for member in value]
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [_plain(member) for member in value.tolist()] if value.ndim == 1 else _plain_nested(value)
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot store a value of type {type(value).__name__} in settings")


def _plain_nested(cell):
    return [[_plain(member) for member in row] if isinstance(row, list) else _plain(row) for row in cell.tolist()]


def encode_setting(value):
    """Text for the settings table: strings as they are, everything else as lossless JSON."""
    if value is ABSENT:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (Enum, FunctionHandle)):
        return value.name
    if value is None:
        return ""
    if isinstance(value, np.ndarray) and value.size == 1 and value.dtype != object:
        return json.dumps(value.item())
    return json.dumps(_plain(value))


def normalize_log(log, file, registry=None):
    """Normalize a decoded log. Raises LossyConversionError rather than change a value."""
    return _Normalizer(log, str(file), registry or load_registry()).normalize()
