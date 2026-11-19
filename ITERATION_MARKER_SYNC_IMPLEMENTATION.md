## Iteration Marker Synchronization Implementation

### Overview

This implementation adds proper temporal alignment for trial-level iteration markers (frame indices) in the Virmen behavioral data. All iteration markers (variables starting with `i` followed by a capital letter, like `iCueEntry`, `iMemEntry`, etc.) now have corresponding `*Seconds` columns that provide session-relative timestamps.

### Key Features

1. **Automatic synchronization**: Uses externally-provided synchronized timestamps (from ephys NIDAQ) when available, falls back to Virmen internal timing otherwise
2. **Session-relative timestamps**: All `*Seconds` columns provide timestamps from NWB session start (not trial-relative)
3. **Cached performance**: Trial frame boundaries are computed once and cached
4. **Robust handling**: Properly handles invalid iteration numbers (0, NaN, negative, beyond trial length)
5. **Added `iStartEntry`**: New iteration marker for trial start

### Implementation Details

#### New Methods

**`_get_trial_frame_boundaries() -> list[tuple[int, int]]`**
- Returns list of `(start_idx, end_idx)` tuples for each trial's frames in the global timestamp array
- Cached after first call (structure doesn't change with sync)
- Uses Python slice convention (inclusive start, exclusive end)

**`_convert_trial_iteration_to_timestamp(trial_idx: int, iteration_num: float) -> float`**
- Converts trial-local iteration number (1-based Virmen) to session-relative timestamp
- Automatically uses `get_timestamps()` which returns synchronized or original timestamps
- Returns `NaN` for invalid iteration numbers

#### New Trial Columns

For each iteration marker (`iCueEntry`, `iMemEntry`, `iTurnEntry`, `iArmEntry`, `iBlank`, `iLaserOn`, `iLaserOff`, `iStartEntry`), a corresponding `*Seconds` column is created:

- `iCueEntrySeconds`: Time in seconds from session start when subject entered cue region
- `iMemEntrySeconds`: Time in seconds from session start when subject entered memory region  
- `iTurnEntrySeconds`: Time in seconds from session start when subject entered turn region
- `iArmEntrySeconds`: Time in seconds from session start when subject entered arm region
- `iBlankSeconds`: Time in seconds from session start when screen was turned off
- `iLaserOnSeconds`: Time in seconds from session start when laser turned on
- `iLaserOffSeconds`: Time in seconds from session start when laser turned off
- `iStartEntrySeconds`: Time in seconds from session start when trial started

### Usage Example

```python
from tank_lab_to_nwb.convert_towers_task.virmenbehaviordatainterface import VirmenDataInterface

# Create interface
interface = VirmenDataInterface(file_path="data.mat")

# Option 1: Use without synchronization (Virmen internal timing)
timestamps = interface.get_timestamps()  # Virmen internal timing

# Option 2: Apply external synchronization (e.g., from ephys NIDAQ)
sync_timestamps = get_sync_timestamps_from_datajoint(session_key)
interface.set_aligned_timestamps(sync_timestamps)
timestamps = interface.get_timestamps()  # Now returns synchronized timestamps

# Convert iteration markers to timestamps
# This automatically uses synchronized timestamps if available
trial_idx = 0
iteration_num = 42  # 1-based Virmen iteration number
timestamp = interface._convert_trial_iteration_to_timestamp(trial_idx, iteration_num)
```

### Synchronization Workflow

As shown in the notebooks (`unified_virmen_kilosort_conversion_v2.ipynb`, `unified_virmen_kilosort_conversion.ipynb`):

1. **Query DataJoint**: Get pre-computed synchronized timestamps from `BehaviorSync` table
2. **Create converter**: Initialize `TowersNWBConverter` with Virmen and Kilosort interfaces
3. **Apply sync**: Call `set_aligned_timestamps()` on the Virmen interface
4. **Run conversion**: All iteration marker `*Seconds` columns automatically use synchronized timestamps

```python
# From notebook workflow
sync_timestamps = get_sync_from_datajoint(session_key)
converter.temporally_align_data_interfaces()  # Calls set_aligned_timestamps internally
converter.run_conversion(nwbfile_path=output_path, metadata=metadata)
```

### Data Integrity

- **Original iteration markers preserved**: The original `iCueEntry`, `iMemEntry`, etc. columns remain unchanged (1-based frame indices)
- **New timestamp columns**: `iCueEntrySeconds`, `iMemEntrySeconds`, etc. provide real timestamps
- **Consistent timing**: Both behavioral position data and iteration markers use the same timestamp source
- **No recomputation**: Helper methods avoid recalculating trial boundaries on every call

### Testing

Run the test script to verify implementation:

```bash
python test_iteration_marker_sync.py
```

This tests:
- Trial frame boundary calculation and caching
- Conversion without synchronization (Virmen timing)
- Conversion with synchronization (external timestamps)
- Edge cases (invalid iteration numbers)

### Technical Notes

1. **Caching rationale**: Since `add_to_nwbfile()` is called only once during conversion, caching prevents redundant computation when processing hundreds of trials
2. **Naming convention**: Uses camelCase (`iCueEntrySeconds`) to match NWB/DANDI conventions
3. **Session-relative vs trial-relative**: Previous implementation provided trial-relative times; new implementation provides session-relative times matching other NWB timestamps
4. **Timezone handling**: Fixed timezone-aware datetime handling in `get_original_timestamps()`
