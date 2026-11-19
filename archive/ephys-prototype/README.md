# Ephys prototype (archive)

The notebooks and scripts in which the ephys conversion was first worked out, kept for reference. They are not
maintained and do not run against the current code: they build `TowersNWBConverter` with the old `sync_timestamps`
alignment and `KiloSortWithProbeInterface`, which `tank_lab_to_nwb.ephys` replaced (see `docs/ephys-timing.md`).

Each file sits below this folder at the path it had in the repository, unchanged:

- `notebooks/unified_virmen_kilosort_conversion*.ipynb`: the unified ViRMEn and Kilosort conversion notebooks.
- `attempting_forge_real_conversion.ipynb` and the notebooks under `tank_lab_to_nwb/convert_towers_task/`: conversion
  attempts, and `get_times_from_nidq.ipynb`, which reads a session's `BehaviorSync` row from the DB.
- `tank_lab_to_nwb/Untitled-1.ipynb`: one behavior log printed as a dictionary (3.9 MB, nearly all of it output).
- `validate_conversion.py`, `validate_sync_conversion.py`, `stub_spikeglx.py`: loose validation and stub scripts.
- `tank_lab_to_nwb/convert_towers_task/`: `kilosortinterface.py` (the old `KiloSortWithProbeInterface`),
  `towersnwbconverter-2024.py` and `towersconverter2.py` (earlier drafts of the converter), `attempting_conversion.py`.
- `KILOSORT_SYNC_IMPLEMENTATION.md`: the note on the constant `behavioral_start_offset` the prototype used.

`IMPLEMENTATION_PLAN.md` and `ITERATION_MARKER_SYNC_IMPLEMENTATION.md` at the repository root name some of these files
by their old paths.
