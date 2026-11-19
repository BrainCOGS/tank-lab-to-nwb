# ViRMEn-to-NWB Conversion System Implementation Plan

## Architecture Overview

Build a modular NWB conversion pipeline where:
- **VirmenDataInterface** converts MATLAB .mat behavioral files to NWB format
- **TowersNWBConverter** orchestrates multiple data modalities (behavior + neural recordings/imaging)
- **Database queries** happen externally in notebooks/scripts, NOT within data interfaces
- **Temporal synchronization** via pre-computed timestamps from U19 DataJoint database, with local fallback
- **Multi-modal support** for Kilosort, SLEAP, Suite2P via neuroconv's existing interfaces (as package extras)

---

## Core Design Principles

### 1. Database Independence with Fallback
Data interfaces work standalone without DataJoint. Database queries happen in external scripts; if database unavailable, perform calculations locally.

### 2. Temporal Alignment Strategy
- **Primary**: Pre-computed sync timestamps from DataJoint BehaviorSync table
- **Fallback**: Local calculation from behavioral data or TTL-based alignment
- **Strict validation**: Raise exception immediately if sync_timestamps length doesn't match expected length

### 3. Multi-Modal Integration
Seamlessly combine behavioral data (VirmenData) with optional neural data modalities. **Always log warnings** when optional modalities are missing.

### 4. MATLAB Handling
Use MATLAB script execution for converting function handles that cannot be processed by scipy.io.loadmat.

---

## Implementation Steps

### 1. VirmenDataInterface - Behavioral Data Conversion
**File**: `tank_lab_to_nwb/convert_towers_task/virmenbehaviordatainterface.py`

- Inherit from `BaseTemporalAlignmentInterface` (neuroconv)
- Implement temporal alignment methods:
  - `get_original_timestamps()` - calculate from Virmen trial times (local fallback)
  - `get_timestamps()` - return aligned timestamps if set, else original
  - `set_aligned_timestamps(aligned_timestamps)` - accept external synchronized timestamps with **strict length validation** (raise ValueError if length mismatch)
- Provide `get_session_key()` method returning `{subject_fullname, session_date}` for external database queries
- Parse MATLAB .mat files and convert to NWB structure:
  - Session/Subject metadata with timezone-aware datetimes (America/New_York)
  - LabMetaData extension via ndx-tank-metadata (experiment, protocol, rig, mazes)
  - Epochs (blocks) with maze_id, reward_ml, stimulus_config, optional laser/context columns
  - Trials with extensive columns: iterations, cue timing/position, choices, laser info, rule, etc.
  - Behavioral timeseries (Position, ViewAngle, Velocity, Collision)
- Handle variable-length arrays with NaN padding for indexed trial data
- Keep DataJoint code commented out with explanatory notes about external query pattern

### 2. TowersNWBConverter - Multi-Modal Orchestrator
**File**: `tank_lab_to_nwb/convert_towers_task/towersnwbconverter.py`

- Inherit from `NWBConverter` (neuroconv)
- Define `data_interface_classes` supporting multiple modalities (from neuroconv extras):
  - `VirmenData` (required) - custom behavioral interface
  - `SpikeGLXRecordingInterface` for AP/LFP (optional) - electrophysiology
  - `KiloSortSortingInterface` (optional) - spike sorting (neuroconv extra)
  - `Suite2pSegmentationInterface` (optional) - calcium imaging (neuroconv extra)
  - `TiffImagingInterface` (optional) - raw imaging
  - `SLEAPInterface` (optional) - pose tracking (neuroconv extra)
- Accept `sync_timestamps` parameter in `__init__()` for database-derived timestamps
- Accept optional `ttl_source` parameter for TTL-based fallback alignment
- Validate that `VirmenData` interface is present (raise ValueError if missing)
- **Log warnings for all missing optional interfaces** in `__init__()`
- Implement `temporally_align_data_interfaces()` method:
  - If `sync_timestamps` provided: apply via `set_aligned_timestamps()` to all temporal interfaces
  - If `sync_timestamps` is None and `ttl_source` provided: perform TTL-based alignment
  - If both None: warn and skip alignment (use original timestamps)
  - Gracefully skip missing optional interfaces during alignment
  - **Catch and re-raise exceptions** from `set_aligned_timestamps()` (length mismatches should fail immediately)

### 3. MATLAB Conversion Utilities
**File**: `tank_lab_to_nwb/utils.py`

- Implement `convert_function_handle_to_str(mat_file_path)` to execute MATLAB scripts extracting:
  - `experiment_name` (from code.version function handle)
  - `protocol_name` (from code.protocol function handle)
  - `trial_choice` data (per-trial choice outcomes)
  - `trial_type` data (per-trial types)
- Implement `mat_obj_to_dict(mat_struct)` for recursive nested struct conversion with special handling for shapingProtocol
- Provide `array_to_dt()` for MATLAB datenum to Python datetime conversion
- Handle graceful degradation when MATLAB not installed (return empty dict, log warning)

### 4. Integration Example Notebook
**File**: `notebooks/unified_virmen_kilosort_conversion.ipynb`

Demonstrate complete workflow with database integration:
- **Create single DataJoint connection** at notebook start, reuse throughout
- Query for session metadata using `VirmenDataInterface.get_session_key()`:
  - Experimenter list from `Subject * User * SubjectCoowners` tables
  - Subject sex, genotype from `Subject` table
  - Recording_id from session queries
- Extract synchronized timestamps from `BehaviorSync` or ephys tables using `recording_id`
- Build `source_data` dict with paths for all available modalities
- Initialize `TowersNWBConverter(source_data, sync_timestamps=...)`
- Call `converter.temporally_align_data_interfaces()` (handles length validation)
- Override metadata with database-derived information (experimenter, subject_sex)
- Run conversion with error handling
- Show fallback behavior when database unavailable (local timestamp calculation)

### 5. Package Configuration
**Files**: `pyproject.toml`, `setup.py`

- Core dependencies: `neuroconv`, `pynwb`, `ndx-tank-metadata`, `numpy`, `scipy`, `h5py`
- U19 integration: `u19-pipeline[pipeline]` (for DataJoint schema access)
- Optional extras for multi-modal support:
  - `neuroconv[kilosort]` - Kilosort interface
  - `neuroconv[suite2p]` - Suite2p interface
  - `neuroconv[sleap]` - SLEAP interface
- Development dependencies: `datajoint`, `pytest`, `ruff`
- Configure local editable installs for sibling projects (ndx-tank-metadata, u19-pipeline)

### 6. Testing Strategy
**File**: `tests/test_towers_conversion.py` (create new)

- **Assume database connections are available** for tests
- Test fixtures with real DataJoint queries using test database
- Test cases:
  - VirmenDataInterface standalone conversion
  - TowersNWBConverter with database-synchronized timestamps
  - Multi-modal conversion (VirmenData + Kilosort + Suite2p)
  - Timestamp length validation (expect ValueError on mismatch)
  - Missing optional interface warnings
  - MATLAB function handle conversion
  - Local fallback when sync_timestamps=None
- Use pytest fixtures for database connection management

---

## Modified/Created Files Summary

- `tank_lab_to_nwb/convert_towers_task/virmenbehaviordatainterface.py` - Behavioral interface with strict timestamp validation
- `tank_lab_to_nwb/convert_towers_task/towersnwbconverter.py` - Multi-modal converter with warning system
- `tank_lab_to_nwb/utils.py` - MATLAB conversion utilities with graceful degradation
- `notebooks/unified_virmen_kilosort_conversion.ipynb` - Complete example with database connection reuse
- `tank_lab_to_nwb/__init__.py` - Package exports
- `pyproject.toml` - Dependencies including neuroconv extras
- `tests/test_towers_conversion.py` - Test suite assuming database access

---

## Additional Context from Uncommitted Changes

### Data Mapping & NWB Structure

The conversion handles comprehensive behavioral data mapping:

#### Session/Subject Metadata
- Session start/end times (timezone-aware, America/New_York)
- Subject information from log.animal
- Experimenter information (queried externally from database)

#### LabMetaData Extension (ndx-tank-metadata)
- `experiment_name` - extracted via MATLAB function handle conversion
- `world_file_name` - name of the virtual world
- `protocol_name` - extracted via MATLAB function handle conversion
- `stimulus_bank_path` - path to stimulus bank
- `commit_id` - version control repository info
- `location` - rig identifier
- `num_trials` - total trial count
- `session_end_time` - ISO format timestamp
- `rig` - RigExtension with hardware configuration (DAQ, sensors, reward system, laser, etc.)
- `mazes` - MazeExtension with maze criteria and parameters

#### Epochs (Blocks)
Core epoch data:
- Start time, stop time, label
- `maze_id` - which maze was run
- `main_maze_id` - highest level maze for subject
- `easy_epoch` - flag for easy blocks
- `first_trial` - index of first trial
- `num_trials` - trial count per epoch
- `duration` - epoch duration in seconds
- `reward_ml` - reward volume
- `stimulus_config` - stimulus configuration number

Optional epoch columns (task-dependent):
- `lsrepoch` - laser epoch configuration
- `P_on` - laser trial probability
- `context` - pro/anti context (1=Pro, 2=Anti)
- `stage` - training stage tracker

#### Trials
Core trial data:
- Start time, stop time, duration
- `trial_id` - trial number within block

Extensive trial columns (task-dependent):
- Iteration markers: `iCueEntry`, `iMemEntry`, `iTurnEntry`, `iArmEntry`, `iBlank`
- Task parameters: `iterations`, `excessTravel`, `rewardScale`
- Stimulus info: `StartCycle`, `EndCycle`, `rule`, `baseCycles`, `pairNum`, `stimulusTable`
- Bias parameters: `multibiasBeta`, `multibiasTau`
- Environment: `wallGuide`, `forcedChoice`
- Moon beacon: `moonBeaconEnabled`, `moonBeaconPos`, `moonBeaconTrigger`, `step_size`, `alpha_plus`, `alpha_minus`, `moonDistHint`
- Laser: `lsrON`, `iLaserOn`, `iLaserOff`
- Outcomes: `choice` (L/R/nil), `trial_type` (L/R)

Cue information (processed from raw data):
- `left_cue_presence`, `right_cue_presence` - which cues appeared
- `left_cue_onset`, `right_cue_onset` - onset times
- `left_cue_offset`, `right_cue_offset` - offset times
- `left_cue_position`, `right_cue_position` - spatial positions
- `left_licks`, `right_licks` - lick timing arrays

#### Behavioral Timeseries
Continuous tracking data:
- Position (X, Y coordinates)
- ViewAngle (heading direction)
- Velocity (movement speed)
- Collision (collision events)

### Special Data Handling

1. **Variable-length arrays**: Padded with NaN for irregular lengths across trials
2. **Timezone handling**: All datetimes converted to America/New_York with ZoneInfo
3. **MATLAB datatypes**:
   - uint8 arrays converted to int32 where needed
   - Function handles converted via MATLAB script execution
   - Empty arrays handled specially (e.g., warmup criteria)
4. **Nested structures**: Recursively flattened for NWB compatibility
5. **Optional fields**: Gracefully skip missing task-specific columns

### Known Issues & TODOs

1. **Timestamp calculation uncertainty** (virmenbehaviordatainterface.py):
   - TODO exists about which exact time steps to use for frame timestamps
   - Current implementation: `trial["start"] + epoch_start_nwb[0] + trial["time"]`
   - May need validation with domain experts

2. **Session performance metric**:
   - Commented out in current implementation (not in behavior file)
   - May need alternative source or calculation

3. **Trial numbering**:
   - `trialNum` field noted as "not reliable, bugged"
   - Using calculated `trial_id` instead

---

## Usage Workflow

### Basic Workflow (with database)

```python
# 1. Create database connection (once, reuse throughout)
import datajoint as dj
subject = dj.create_virtual_module("subject", "u19_subject")
acquisition = dj.create_virtual_module("acquisition", "u19_acquisition")

# 2. Initialize VirmenDataInterface to get session key
from tank_lab_to_nwb import VirmenDataInterface
virmen_interface = VirmenDataInterface(source_data={"file_path": "path/to/virmen.mat"})
session_key = virmen_interface.get_session_key()

# 3. Query database for sync timestamps
recording_id = 123  # from session query
sync_timestamps = (acquisition.BehaviorSync() & {'recording_id': recording_id}).fetch1('sync_times')

# 4. Build source_data for all modalities
source_data = {
    "VirmenData": {"file_path": "path/to/virmen.mat"},
    "Kilosort": {"folder_path": "path/to/kilosort"},
    "Suite2pSegmentation": {"folder_path": "path/to/suite2p"},
}

# 5. Initialize converter with sync timestamps
from tank_lab_to_nwb import TowersNWBConverter
converter = TowersNWBConverter(source_data=source_data, sync_timestamps=sync_timestamps)

# 6. Apply temporal alignment
converter.temporally_align_data_interfaces()

# 7. Get metadata and override with database info
metadata = converter.get_metadata()
metadata["NWBFile"]["experimenter"] = experimenter_list  # from database
metadata["Subject"]["sex"] = subject_sex  # from database

# 8. Run conversion
converter.run_conversion(nwbfile_path="output.nwb", metadata=metadata, overwrite=True)
```

### Fallback Workflow (without database)

```python
# Initialize converter without sync_timestamps
source_data = {
    "VirmenData": {"file_path": "path/to/virmen.mat"},
}

converter = TowersNWBConverter(source_data=source_data, sync_timestamps=None)
# Uses original timestamps from Virmen data

metadata = converter.get_metadata()
converter.run_conversion(nwbfile_path="output.nwb", metadata=metadata)
```
