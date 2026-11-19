# Ephys-Behavior Synchronization Plan

## Overview

This document describes the synchronization strategy between behavioral data (Virmen) and electrophysiology data (NIDQ/IMEC) for Tank Lab NWB conversions.

## Synchronization Architecture

### Hardware Setup
- **NIDQ (National Instruments)**: Digital acquisition at ~5000 Hz recording TTL pulses
- **IMEC/Neuropixels**: Neural recording at 30000 Hz
- **Virmen**: Behavioral task engine generating trial and iteration pulses

### DataJoint BehaviorSync Table

The `u19_pipeline.ephys_pipeline.BehaviorSync` table stores synchronization data with the following key fields:

```python
nidq_sampling_rate    : float        # Sampling rate (e.g., 5000.062709 Hz)
iteration_index_nidq  : longblob     # Full-length array (DEPRECATED - stores [np.nan])
trial_index_nidq      : longblob     # Full-length array (DEPRECATED - stores [np.nan])
sync_data             : longblob     # Compact synchronization data (PRIMARY)
regular_sync_status   : tinyint      # =1 if pulse-based sync succeeded
fixed_sync_status     : tinyint      # =1 if pulse-fix method succeeded
virmen_sync_status    : tinyint      # =1 if Virmen time-based sync was used
```

## Sync_Data Structure (Compact Format)

The `sync_data` field contains a **dictionary with 4 subfields**, providing a compact representation of frame-to-sample mapping:

### Structure

```python
sync_data = {
    # Pulse-based synchronization (preferred, used when regular_sync_status=1 or fixed_sync_status=1)
    'trial_idx_vector': np.array([933517, 1579865, 1920738, ...]),  # 1D array of trial start indices
    'iteration_idx_vector': np.array([                              # Array of arrays (object dtype)
        array([933517, 933711, 936388, ...]),  # Trial 0: NIDQ sample indices for each frame
        array([1579865, 1580076, 1580485, ...]), # Trial 1: NIDQ sample indices for each frame
        ...  # One array per trial
    ], dtype=object),

    # Virmen time-based synchronization (fallback, used when pulse-based fails)
    'trial_idx_vector_from_virmen': np.array([...]),      # 1D array of trial start indices
    'iteration_idx_vector_from_virmen': list([...])       # List of arrays per trial
}
```

### Example Data

For a session with 282 trials:
- **trial_idx_vector**: Shape (282,) - One NIDQ sample index per trial
- **iteration_idx_vector**: 282 arrays of varying lengths
  - Trial 0: 1132 frames, starting at NIDQ samples [933517, 933711, 936388, ...]
  - Trial 1: 545 frames, starting at NIDQ samples [1579865, 1580076, 1580485, ...]

## Converting to Timestamps

### Method 1: Use Compact Data Directly (Recommended for NWB)

```python
# Fetch compact sync data
sync_data = (ep.BehaviorSync & {'recording_id': recording_id}).fetch1('sync_data')
nidq_rate = (ep.BehaviorSync & {'recording_id': recording_id}).fetch1('nidq_sampling_rate')

# For each trial, convert iteration indices to timestamps
# ALL TIMESTAMPS REMAIN IN IMEC TIME (t=0 = IMEC recording start)
for trial_num, iter_indices in enumerate(sync_data['iteration_idx_vector']):
    # Convert NIDQ sample indices to seconds (IMEC time reference)
    frame_timestamps = iter_indices / nidq_rate

    # DO NOT zero-reference - keep IMEC t=0 as universal reference
    # frame_timestamps already represents time since IMEC recording start

    # Store in NWB trials table or as timeseries
```

### Method 2: Use Full-Length Arrays (Current Approach)

```python
# Get expanded full-length arrays
all_vectors = ep.get_full_vectors_from_key({'recording_id': recording_id})

# All arrays have same length as NIDQ file (~5000 samples/sec)
time_vector = all_vectors['time_vector']                   # Continuous time in seconds
iteration_index_nidq = all_vectors['iteration_index_nidq'] # Frame number per NIDQ sample
trial_index_nidq = all_vectors['trial_index_nidq']         # Trial number per NIDQ sample

# For a specific Virmen frame, find its timestamp
frame_num = 100
trial_num = 5
nidq_samples = np.where((trial_index_nidq == trial_num) &
                        (iteration_index_nidq == frame_num))[0]
if len(nidq_samples) > 0:
    timestamp = time_vector[nidq_samples[0]]
```

## Storage Efficiency

| Format | Size (1-hour session) | Use Case |
|--------|----------------------|----------|
| **Compact (sync_data)** | ~0.1 MB | NWB metadata, efficient storage |
| **Full-length arrays** | ~144 MB | Direct sample-by-sample lookup |

## Synchronization Quality

The sync status flags indicate which method was used:

1. **regular_sync_status = 1**: All pulses detected, no corrections needed (highest quality)
2. **fixed_sync_status = 1**: Some pulses missed but successfully corrected using fix algorithm
3. **virmen_sync_status = 1**: Pulse-based sync failed, using Virmen timestamps (lower accuracy)

## NWB Conversion Strategy

### Current Implementation
- Uses `get_full_vectors_from_key()` to retrieve full-length `time_vector`
- Applies to all behavioral timeseries via `temporally_align_data_interfaces()`
- Results in synchronized behavioral data at NIDQ timestamps

### Recommended Enhancement
1. Store compact `sync_data` in NWB metadata/description
2. Add NIDQ device with precise sampling rate
3. Document sync quality (regular/fixed/virmen status)
4. Include sync method description in processing module
5. Optionally add iteration_idx_vector as structured metadata for frame-to-sample mapping

## Frame-to-IMEC-Sample Mapping

To align behavioral frames with neural spikes:

```python
# Get IMEC sampling rate
imec_rate = (ep.BehaviorSync.ImecSamplingRate &
             {'recording_id': recording_id, 'insertion_number': probe_num}
            ).fetch1('ephys_sampling_rate')  # Typically 30000 Hz

# Get scaling factor
scaling_factor = imec_rate / nidq_rate  # e.g., 30000 / 5000 = 6

# For a given frame:
nidq_sample_idx = sync_data['iteration_idx_vector'][trial_num][frame_num]
imec_sample_idx = int(nidq_sample_idx * scaling_factor)
imec_timestamp = imec_sample_idx / imec_rate  # in seconds
```

## Kilosort Spike Time Synchronization

### Background

**Kilosort stores spike times** as seconds relative to the start of the IMEC recording. Since IMEC and NIDQ are hardware-synchronized (same trigger), they share a common start time.

**However**, behavioral frames don't necessarily start at recording time=0. There's typically a delay between when the recording starts and when the first behavioral frame is captured.

### Time Reference Points

```
IMEC/NIDQ Recording Start (t=0) ← UNIVERSAL ZERO REFERENCE FOR ALL DATA
    |
    |<-- behavioral_start_offset (~25s) -->|
    |                                       First Behavioral Frame (t=~25s in IMEC time)
    |                                       |
    | Spike times (IMEC reference)          | Behavioral data (IMEC reference)
    | Neural data continues...              | Position, velocity, etc.
```

### Calculating the Time Offset

```python
# Fetch sync data
sync_data = (ep.BehaviorSync & {'recording_id': recording_id}).fetch1('sync_data')
nidq_rate = (ep.BehaviorSync & {'recording_id': recording_id}).fetch1('nidq_sampling_rate')

# Calculate offset from IMEC start to first behavioral frame
first_nidq_sample = sync_data['iteration_idx_vector'][0][0]  # First frame's NIDQ sample index
behavioral_start_offset = first_nidq_sample / nidq_rate  # Convert to seconds

print(f"Behavioral data starts {behavioral_start_offset:.3f}s into IMEC recording")
```

### Alignment Strategy for NWB

**REQUIRED APPROACH: IMEC t=0 as Universal Reference**
- **Behavioral timestamps**: Remain in IMEC time (t=0 = IMEC recording start)
- **Kilosort spike times**: Remain in IMEC time (t=0 = IMEC recording start)
- **All timestamps share common t=0** (IMEC recording start)
- **behavioral_start_offset** stored in metadata documents when behavioral data began
- **No user conversion needed** - all data is already aligned to same reference
- **Preserves original time references** and maintains data provenance

**Why IMEC t=0?**
- Hardware-synchronized trigger provides precise, absolute time reference
- Avoids negative timestamps (behavioral data starts after t=0)
- Simplifies cross-modal analysis (no offset calculations needed)
- Matches original data acquisition system conventions
- Eliminates ambiguity about which data stream defines "zero"

### Implementation in NWB

**Using Synchronized Timestamps (IMEC Reference):**

```python
from neuroconv.datainterfaces import KiloSortSortingInterface

# 1. Get sync data and convert to IMEC-referenced timestamps
sync_data = (ep.BehaviorSync & {'recording_id': recording_id}).fetch1('sync_data')
nidq_rate = (ep.BehaviorSync & {'recording_id': recording_id}).fetch1('nidq_sampling_rate')

# Convert all behavioral frames to IMEC time (NO zero-referencing)
sync_timestamps = []
for trial_iter_indices in sync_data['iteration_idx_vector']:
    trial_timestamps = trial_iter_indices / nidq_rate  # Already in IMEC time
    sync_timestamps.extend(trial_timestamps)
sync_timestamps = np.array(sync_timestamps)

# 2. Initialize converter with IMEC-referenced timestamps
converter = TowersNWBConverter(
    source_data={
        "VirmenData": {"file_path": virmen_file},
        "KilosortProbe0": {"folder_path": kilosort_folder},
    },
    sync_timestamps=sync_timestamps  # Already in IMEC time
)

# 3. Apply timestamps to behavioral data (keeps IMEC reference)
converter.temporally_align_data_interfaces()

# 4. Kilosort data automatically uses IMEC time (no modification needed)
# Both behavioral and spike data now share t=0 = IMEC recording start

# 5. Document when behavioral data started (for reference)
metadata['LabMetaData']['behavioral_start_offset'] = sync_timestamps[0]
```

### Verifying Synchronization

```python
# Load NWB file
with NWBHDF5IO(nwb_file_path, 'r') as io:
    nwbfile = io.read()

    # Get behavioral timestamps (in IMEC time)
    position = nwbfile.processing['behavior']['Position']['SpatialSeries']
    behavioral_times = position.timestamps[:]  # t=0 is IMEC recording start

    # Get spike times (in IMEC time)
    units = nwbfile.units
    spike_times = units['spike_times'][0]  # First unit, t=0 is IMEC recording start

    # Check when behavioral data started
    behavioral_start_offset = nwbfile.lab_meta_data['LabMetaData'].behavioral_start_offset
    print(f"Behavioral data started at t={behavioral_start_offset:.3f}s (IMEC time)")
    print(f"First behavioral frame at t={behavioral_times[0]:.3f}s")
    print(f"First spike at t={spike_times[0]:.3f}s")

    # Both timestamps are already in same reference frame (IMEC t=0)
    # NO conversion needed - can directly compare

    # Find spikes during first behavioral frame
    first_frame_duration = behavioral_times[1] - behavioral_times[0]
    spikes_in_first_frame = spike_times[
        (spike_times >= behavioral_times[0]) &
        (spike_times < behavioral_times[0] + first_frame_duration)
    ]

    print(f"Found {len(spikes_in_first_frame)} spikes during first behavioral frame")

    # Verify no negative spike times (all data should be t >= 0)
    assert np.all(spike_times >= 0), "ERROR: Negative spike times detected!"
    assert np.all(behavioral_times >= 0), "ERROR: Negative behavioral times detected!"
    print("✓ All timestamps are positive (correctly referenced to IMEC t=0)")
```
