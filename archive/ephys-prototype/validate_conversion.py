"""Offline validation of the TowersNWBConverter (no DataJoint required).

Exercises: multi-probe Kilosort discovery (highest job id), full conversion,
units-per-probe attribution, and behavioral timestamp reporting.
"""

from pathlib import Path
from collections import Counter

import numpy as np
from pynwb import NWBHDF5IO

from tank_lab_to_nwb.convert_towers_task.towersnwbconverter import TowersNWBConverter
from tank_lab_to_nwb.convert_towers_task.virmenbehaviordatainterface import VirmenDataInterface

virmen_file_path = Path(
    "/Users/ct5868/code/testing_yanar/jorge_pwmv2_cohort1_185A-Rig1_jyanar_ya008_T_20240525_0.mat"
)
kilosort_base = Path(
    "/Users/ct5868/code/testing_yanar/known_kilosort_structure/ya008_20240525_g0"
)


def discover_kilosort_probes(base_path):
    probes = {}
    for probe_dir in sorted(base_path.glob("*_imec*")):
        probe_id = probe_dir.name.split("_imec")[-1]
        job_dirs = sorted(probe_dir.glob("job_id_*"), key=lambda x: int(x.name.split("_")[-1]))
        if not job_dirs:
            continue
        highest_job = job_dirs[-1]
        outputs = list(highest_job.glob("kilosort*_output"))
        if not outputs:
            continue
        probes[f"KilosortProbe{probe_id}"] = outputs[0]
        print(f"  {f'KilosortProbe{probe_id}'}: {highest_job.name} -> {outputs[0].name}")
    return probes


print("Discovering Kilosort probes (highest job id per probe):")
probes = discover_kilosort_probes(kilosort_base)
assert probes, "No probes discovered"

source_data = {"VirmenData": {"file_path": str(virmen_file_path)}}
for name, path in probes.items():
    source_data[name] = {"folder_path": str(path), "keep_good_only": False}

# Verify behavioral timestamps directly from the interface
vi = VirmenDataInterface(file_path=virmen_file_path, verbose=False)
ts = vi.get_original_timestamps()
print(f"\nBehavioral frames: {len(ts)}")
print(f"First/last behavioral timestamp (Virmen internal): {ts[0]:.4f}s / {ts[-1]:.4f}s")

converter = TowersNWBConverter(source_data=source_data, sync_timestamps=None, ttl_source=None)
print(f"\nActive interfaces: {list(converter.data_interface_objects.keys())}")

for name, iface in converter.data_interface_objects.items():
    if name.startswith("Kilosort"):
        s = iface.sorting_extractor
        n = len(s.get_unit_ids())
        sr = s.get_sampling_frequency()
        sample_unit = s.get_unit_ids()[0]
        st = s.get_unit_spike_train(sample_unit) / sr
        print(f"  {name}: {n} units, fs={sr}Hz, unit {sample_unit} sample spike times(s): {st[:3]}")

metadata = converter.get_metadata()
metadata["Subject"]["species"] = "Mus musculus"
metadata["Subject"]["sex"] = "U"

out = Path("/Users/ct5868/code/tank-lab-to-nwb-clean/output")
out.mkdir(exist_ok=True)
nwb_path = out / f"{metadata['NWBFile']['session_id']}_validation.nwb"

print(f"\nRunning conversion -> {nwb_path}")
converter.run_conversion(nwbfile_path=str(nwb_path), metadata=metadata, overwrite=True)
print("Conversion complete.")

with NWBHDF5IO(str(nwb_path), "r") as io:
    nwb = io.read()
    print("\n=== VERIFICATION ===")
    print(f"Trials: {len(nwb.trials)}")
    print(f"Trial columns incl *Seconds: {[c for c in nwb.trials.colnames if c.endswith('Seconds')]}")
    pos = nwb.processing["behavior"]["Position"]["Position"]
    bt = pos.timestamps[:]
    print(f"Behavioral timestamps: {bt[0]:.4f}s .. {bt[-1]:.4f}s ({len(bt)} frames)")
    if nwb.units is not None:
        print(f"Total units: {len(nwb.units)}")
        if "probe_id" in nwb.units.colnames:
            print(f"Units per probe_id: {dict(Counter(nwb.units['probe_id'][:]))}")
        if "electrode_group" in nwb.units.colnames:
            print(f"Units per electrode_group: {dict(Counter(eg.name for eg in nwb.units['electrode_group'][:]))}")
        u0 = nwb.units["spike_times"][0]
        print(f"Unit 0 sample spike times (s): {u0[:3]}")
# Equivalence check: in the non-sync case, the refactored cue-onset helper must match
# the legacy Virmen arithmetic for a single-epoch session.
mat = vi._mat_dict
blk = mat["log"]["block"]
epochs_raw = [blk] if isinstance(blk, dict) else blk
if len(epochs_raw) == 1:
    sst = vi._get_session_start_time()
    from tank_lab_to_nwb.utils import array_to_dt
    from zoneinfo import ZoneInfo
    e0 = epochs_raw[0]
    e0_off = (array_to_dt(e0["start"]).replace(tzinfo=ZoneInfo("America/New_York")) - sst).total_seconds()
    trials_raw2 = [t for t in e0["trial"] if not np.isnan(t["start"])]
    max_err = 0.0
    for ti, tr in enumerate(trials_raw2):
        if np.any(tr["cueOnset"][0]):
            idx = np.minimum(tr["cueOnset"][0], tr["cueOnset"][0] - 1)
            legacy = tr["start"] + e0_off + tr["time"][idx]
            new = vi._local_frames_to_timestamps(ti, idx)
            max_err = max(max_err, float(np.max(np.abs(legacy - new))))
    print(f"\nCue-onset equivalence (non-sync, single-epoch) max abs diff: {max_err:.3e}s")
    assert max_err < 1e-9, "Refactored cue-onset times diverge from legacy arithmetic"

print("\nVALIDATION PASSED")
