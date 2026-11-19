#!/usr/bin/env python
"""Test script to verify iteration marker synchronization implementation."""

from pathlib import Path
import numpy as np
from tank_lab_to_nwb.convert_towers_task.virmenbehaviordatainterface import VirmenDataInterface

# Load test Virmen file
virmen_file = Path("/Users/ct5868/code/tank-lab-to-nwb-clean/jorge_pwmv2_cohort1_185A-Rig1_jyanar_ya014_T_20240722_0.mat")

if not virmen_file.exists():
    print(f"Test file not found: {virmen_file}")
    exit(1)

print("="*70)
print("TESTING ITERATION MARKER SYNCHRONIZATION")
print("="*70)

# Create interface
interface = VirmenDataInterface(file_path=virmen_file, verbose=True)

print("\n1. Testing _get_trial_frame_boundaries():")
boundaries = interface._get_trial_frame_boundaries()
print(f"   Number of trials: {len(boundaries)}")
print(f"   First 5 trial boundaries: {boundaries[:5]}")
print(f"   Last 5 trial boundaries: {boundaries[-5:]}")

# Verify boundaries are contiguous
total_frames = 0
for i, (start, end) in enumerate(boundaries):
    if i > 0 and start != boundaries[i-1][1]:
        print(f"   ✗ ERROR: Gap between trial {i-1} and {i}")
    total_frames += (end - start)
print(f"   Total frames across all trials: {total_frames}")

# Get original timestamps
original_timestamps = interface.get_original_timestamps()
print(f"\n2. Testing get_original_timestamps():")
print(f"   Total timestamps: {len(original_timestamps)}")
print(f"   First 5 timestamps: {original_timestamps[:5]}")
print(f"   Last 5 timestamps: {original_timestamps[-5:]}")
print(f"   Timestamps match frame count: {len(original_timestamps) == total_frames}")

# Test conversion without synchronized timestamps
print(f"\n3. Testing _convert_trial_iteration_to_timestamp() (Virmen timing):")
test_cases = [
    (0, 1),      # First trial, first frame
    (0, 10),     # First trial, 10th frame
    (0, 0),      # Invalid: iteration 0
    (0, -1),     # Invalid: negative
    (0, np.nan), # Invalid: NaN
    (0, 10000),  # Invalid: beyond trial length
]

for trial_idx, iteration_num in test_cases:
    timestamp = interface._convert_trial_iteration_to_timestamp(trial_idx, iteration_num)
    print(f"   Trial {trial_idx}, Iteration {iteration_num}: {timestamp:.6f}s" if np.isfinite(timestamp) else f"   Trial {trial_idx}, Iteration {iteration_num}: NaN")

# Test with synchronized timestamps (simulate external sync)
print(f"\n4. Testing with synchronized timestamps:")
# Create fake synchronized timestamps (just offset by 100 seconds to simulate IMEC time)
fake_sync_timestamps = original_timestamps + 100.0

interface.set_aligned_timestamps(fake_sync_timestamps)
print(f"   Applied synchronized timestamps (offset +100s)")

# Re-test conversion
print(f"   Testing conversion with sync timestamps:")
for trial_idx, iteration_num in [(0, 1), (0, 10)]:
    timestamp = interface._convert_trial_iteration_to_timestamp(trial_idx, iteration_num)
    expected = fake_sync_timestamps[boundaries[trial_idx][0] + int(iteration_num) - 1]
    match = np.isclose(timestamp, expected)
    print(f"   Trial {trial_idx}, Iteration {iteration_num}: {timestamp:.6f}s (expected {expected:.6f}s) {'✓' if match else '✗'}")

# Verify caching
print(f"\n5. Testing boundary caching:")
boundaries2 = interface._get_trial_frame_boundaries()
print(f"   Cached result identical: {boundaries is boundaries2}")

print("\n" + "="*70)
print("TESTING COMPLETE")
print("="*70)
