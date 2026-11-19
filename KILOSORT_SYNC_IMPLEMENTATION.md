# Kilosort Synchronization Implementation

## Overview

This document summarizes the implementation for synchronizing Kilosort spike times with Virmen behavioral data in the Tank Lab NWB conversion pipeline.

## Problem Statement

Kilosort stores spike times relative to the IMEC recording start (t=0), while behavioral data from Virmen is captured with an offset from the recording start. To properly analyze the relationship between neural activity and behavior, we need to account for this time offset.

## Implementation Components

### 1. TowersNWBConverter Enhancement

**File:** `tank_lab_to_nwb/convert_towers_task/towersnwbconverter.py`

**Changes:**
- Added `KiloSortSortingInterface` import from neuroconv
- Added `"Kilosort"` to `data_interface_classes` dictionary
- Updated temporal alignment comment to document that Kilosort requires recording registration

**Usage:**
```python
source_data = {
    "VirmenData": {"file_path": "/path/to/virmen.mat"},
    "Kilosort": {
        "folder_path": "/path/to/kilosort/output",
        "keep_good_only": False
    }
}

converter = TowersNWBConverter(source_data=source_data)
```

### 2. Behavioral Start Offset Calculation

**File:** `notebooks/unified_virmen_kilosort_conversion.ipynb`

**Cell #VSC-7e8abba1** calculates the time offset between IMEC recording start and first behavioral frame:

```python
# Calculate behavioral start offset
first_nidq_sample = sync_data['iteration_idx_vector'][0][0]
metadata_behavioral_offset = first_nidq_sample / nidq_rate

print(f"Behavioral start offset: {metadata_behavioral_offset:.6f} seconds")
```

This offset represents:
- Time from IMEC/NIDQ recording trigger (t=0) to first behavioral frame
- The value that must be subtracted from Kilosort spike times to align with behavioral timestamps

### 3. Metadata Storage

**File:** `notebooks/unified_virmen_kilosort_conversion.ipynb`

**Cell #VSC-a3c351b6** stores the offset in NWB metadata:

```python
if 'metadata_behavioral_offset' in globals() and metadata_behavioral_offset is not None:
    metadata["LabMetaData"]["behavioral_start_offset"] = metadata_behavioral_offset
```

The offset is stored in `LabMetaData` extension for easy retrieval when analyzing the NWB file.

### 4. Usage Examples

**File:** `notebooks/unified_virmen_kilosort_conversion.ipynb`

**Cell #VSC-1a2b509f** demonstrates how to retrieve and use the offset:

```python
# Retrieve offset from NWB file
with NWBHDF5IO(nwb_file_path, "r") as io:
    nwbfile = io.read()
    lab_meta = nwbfile.lab_meta_data['LabMetaData']
    behavioral_start_offset = lab_meta.behavioral_start_offset

    # Convert Kilosort spike times to behavioral reference
    spike_times_kilosort = nwbfile.units['spike_times'][0]
    spike_times_behavioral = spike_times_kilosort - behavioral_start_offset
```

## Time Reference System

### Hardware Synchronization
- IMEC and NIDQ recordings are triggered simultaneously
- Both start at t=0 (common hardware trigger)
- This provides the foundation for temporal alignment

### Time Bases

```
IMEC/NIDQ Recording Start (t=0)
    |
    |<-- behavioral_start_offset -->|
                                     First Behavioral Frame (behavioral t=0)
                                     |
                                     | Behavioral data continues...
                                     | (timestamps zero-referenced)
```

**Kilosort Time Base:**
- Spike times are in seconds relative to IMEC recording start
- Range: [0, recording_duration]

**Behavioral Time Base:**
- Frame timestamps are zero-referenced to first frame
- Range: [0, session_duration]

**Relationship:**
```
spike_time_behavioral = spike_time_kilosort - behavioral_start_offset
```

## Data Flow

1. **Database Sync Data** → `sync_data['iteration_idx_vector'][0][0]`
   - First NIDQ sample index for first behavioral frame

2. **Calculate Offset** → `first_nidq_sample / nidq_rate`
   - Convert sample index to seconds

3. **Store in Metadata** → `metadata["LabMetaData"]["behavioral_start_offset"]`
   - Persisted in NWB file

4. **Use for Alignment** → `spike_times - offset`
   - Applied during analysis to align spike times with behavioral events

## Design Decisions

### Why Not Use `.set_aligned_timestamps()`?

The `KiloSortSortingInterface` inherits from `BaseTemporalAlignmentInterface` and supports the `.set_aligned_timestamps()` method. However:

1. **Requires Recording Registration**: Would need to register the SpikeGLX recording with the interface
2. **Adds Complexity**: More infrastructure for minimal benefit
3. **Loss of Original Times**: Would lose the original IMEC-referenced timestamps
4. **Simple Alternative Exists**: Storing offset in metadata is simpler and more transparent

### Why Store Offset Instead of Modifying Times?

**Advantages:**
- Preserves original time references in the data
- Users can choose their preferred alignment method
- Transparent about time transformations
- Easier to verify and debug
- Maintains data provenance

**Trade-offs:**
- Users must apply offset manually during analysis
- Could lead to errors if offset is forgotten

We chose to prioritize data preservation and transparency over convenience.

## Verification Procedures

### 1. Check Offset Calculation
```python
# Verify offset makes sense
print(f"Behavioral start offset: {behavioral_start_offset:.3f}s")
print(f"First behavioral frame NIDQ sample: {first_nidq_sample}")
print(f"NIDQ sampling rate: {nidq_rate:.3f} Hz")
print(f"Expected offset: {first_nidq_sample / nidq_rate:.3f}s")
```

### 2. Verify Metadata Storage
```python
# Check LabMetaData contains offset
with NWBHDF5IO(nwb_file_path, "r") as io:
    nwbfile = io.read()
    assert hasattr(nwbfile.lab_meta_data['LabMetaData'], 'behavioral_start_offset')
    print(f"✓ Offset stored: {nwbfile.lab_meta_data['LabMetaData'].behavioral_start_offset:.6f}s")
```

### 3. Verify Time Alignment
```python
# Example: Find spikes during first trial
first_trial_start = nwbfile.trials['start_time'][0]
first_trial_stop = nwbfile.trials['stop_time'][0]

# Get spikes (in IMEC time)
spike_times_kilosort = nwbfile.units['spike_times'][0]

# Convert to behavioral time
spike_times_behavioral = spike_times_kilosort - behavioral_start_offset

# Find spikes in first trial
trial_spikes = spike_times_behavioral[
    (spike_times_behavioral >= first_trial_start) &
    (spike_times_behavioral < first_trial_stop)
]

print(f"Found {len(trial_spikes)} spikes in first trial")
```

## Documentation

### Files Updated
1. **SYNC_DATA_PLAN.md** - Added comprehensive Kilosort synchronization section
2. **towersnwbconverter.py** - Added Kilosort interface support
3. **unified_virmen_kilosort_conversion.ipynb** - Added offset calculation, metadata storage, and usage examples

### Key Concepts Documented
- Hardware synchronization (IMEC/NIDQ)
- Time reference systems
- Offset calculation method
- Metadata storage location
- Usage examples for analysis

## Future Enhancements

### Potential Improvements
1. **Automatic Offset Application**: Add helper method to converter for automatic alignment
2. **Validation Tools**: Create verification functions to check alignment quality
3. **Alternative Interfaces**: Support other spike sorting tools (e.g., Phy, Mountainsort)
4. **Unit Tests**: Add tests for offset calculation and metadata storage

### Known Limitations
1. Assumes IMEC and NIDQ are hardware-synchronized
2. Requires database access for sync_data
3. No automatic verification of alignment quality
4. User must manually apply offset during analysis

## Summary

This implementation provides a complete infrastructure for synchronizing Kilosort spike times with Virmen behavioral data:

✓ **Kilosort interface** added to TowersNWBConverter
✓ **Offset calculation** from database sync data
✓ **Metadata storage** in NWB LabMetaData
✓ **Usage examples** for analysis workflows
✓ **Comprehensive documentation** of time reference systems

The design prioritizes data preservation, transparency, and user flexibility while maintaining compatibility with the existing Tank Lab data pipeline.
