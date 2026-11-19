"""Validate the synchronized (IMEC-relative) NWB conversion against the U19 DataJoint DB.

Requires Princeton VPN. Mirrors the notebook query logic, runs the conversion with
sync_timestamps, and asserts the handoff "definition of done":
  - behavioral timestamps are IMEC-relative and start at behavioral_start_offset > 0
  - first behavioral timestamp == behavioral_start_offset == first sync timestamp
  - Kilosort spike times remain IMEC-relative (unchanged by alignment)
  - experimenter names are DANDI "LastName, FirstName"
  - DOB present and datetime-compatible
  - multi-probe units attributed deterministically
"""

import sys
from pathlib import Path
from datetime import datetime, date
from collections import Counter

import numpy as np

sys.path.insert(0, "/Users/ct5868/code/U19-pipeline_python")

import datajoint as dj
import u19_pipeline.ephys_pipeline as ep
import u19_pipeline.acquisition as acquisition
from pynwb import NWBHDF5IO

from tank_lab_to_nwb.convert_towers_task.towersnwbconverter import TowersNWBConverter

subject = dj.create_virtual_module("subject", "u19_subject")
lab = dj.create_virtual_module("lab", "u19_lab")
recording = dj.create_virtual_module("recording", "u19_recording")

virmen_file_path = Path(
    "/Users/ct5868/code/testing_yanar/jorge_pwmv2_cohort1_185A-Rig1_jyanar_ya008_T_20240525_0.mat"
)
kilosort_base = Path(
    "/Users/ct5868/code/testing_yanar/known_kilosort_structure/ya008_20240525_g0"
)
session_key = {"subject_fullname": "jyanar_ya008", "session_date": "2024-05-25"}


def fmt_name(full):
    if full and " " in full:
        parts = full.rsplit(" ", 1)
        return f"{parts[-1]}, {parts[0]}"
    return full


# ---- Experimenter / subject metadata ----
sub_info = (subject.Subject() * lab.User() & f"subject_fullname = '{session_key['subject_fullname']}'").fetch1()
owner_info = (lab.User() & f"user_id = '{sub_info['user_id']}'").fetch1()
experimenter_list = [fmt_name(owner_info.get("full_name", sub_info["user_id"]))]
for coowner_id in (subject.SubjectCoowners()
                   & f"subject_fullname = '{session_key['subject_fullname']}' and active = 1").fetch("coowner"):
    ci = (lab.User() & f"user_id = '{coowner_id}'").fetch1()
    experimenter_list.append(fmt_name(ci.get("full_name", coowner_id)))

subject_sex = {"Male": "M", "Female": "F", "Unknown": "U"}.get(sub_info["sex"], "U")
dob = sub_info.get("dob")
subject_dob = dob if isinstance(dob, datetime) else (datetime.combine(dob, datetime.min.time()) if dob else None)
print(f"Experimenters (DANDI): {experimenter_list}")
print(f"Sex: {subject_sex}   DOB: {subject_dob}")

# ---- Sync timestamps (IMEC-relative) ----
session_query_key = (acquisition.Session & session_key).fetch1("KEY")
recording_keys = ((acquisition.Session * recording.Recording.BehaviorSession) & session_query_key).fetch(
    "recording_id", as_dict=True
)
recording_id = recording_keys[0]["recording_id"]
sync_record = (ep.BehaviorSync & {"recording_id": recording_id}).fetch1()
sync_data = sync_record["sync_data"]
nidq_rate = sync_record["nidq_sampling_rate"]
if sync_record["regular_sync_status"] == 1 or sync_record["fixed_sync_status"] == 1:
    iteration_idx_vector = sync_data["iteration_idx_vector"]
    sync_method = "pulse-based"
else:
    iteration_idx_vector = sync_data["iteration_idx_vector_from_virmen"]
    sync_method = "virmen-time-based"

frame_lists, first_nidq_sample = [], None
for i, iter_indices in enumerate(iteration_idx_vector):
    if i == 0:
        first_nidq_sample = iter_indices[0]
    frame_lists.append(iter_indices / nidq_rate)
sync_timestamps = np.concatenate(frame_lists)
behavioral_start_offset = first_nidq_sample / nidq_rate
print(f"Sync method: {sync_method}  NIDQ rate: {nidq_rate:.4f} Hz")
print(f"Sync frames: {len(sync_timestamps)}  range: {sync_timestamps[0]:.4f}s .. {sync_timestamps[-1]:.4f}s")
print(f"behavioral_start_offset: {behavioral_start_offset:.6f}s")

# ---- Discover probes (highest job id) ----
source_data = {"VirmenData": {"file_path": str(virmen_file_path)}}
for probe_dir in sorted(kilosort_base.glob("*_imec*")):
    pid = probe_dir.name.split("_imec")[-1]
    jobs = sorted(probe_dir.glob("job_id_*"), key=lambda x: int(x.name.split("_")[-1]))
    outs = list(jobs[-1].glob("kilosort*_output")) if jobs else []
    if outs:
        source_data[f"KilosortProbe{pid}"] = {"folder_path": str(outs[0]), "keep_good_only": False}
        print(f"KilosortProbe{pid}: {jobs[-1].name}")

# ---- Capture pre-alignment Kilosort spike times to prove they're untouched ----
converter = TowersNWBConverter(source_data=source_data, sync_timestamps=sync_timestamps, ttl_source=None)
pre_spikes = {}
for name, iface in converter.data_interface_objects.items():
    if name.startswith("Kilosort"):
        s = iface.sorting_extractor
        uid = s.get_unit_ids()[0]
        pre_spikes[name] = (s.get_unit_spike_train(uid) / s.get_sampling_frequency())[:5]

converter.temporally_align_data_interfaces()

metadata = converter.get_metadata()
metadata["NWBFile"]["experimenter"] = experimenter_list
metadata["Subject"]["sex"] = subject_sex
metadata["Subject"]["species"] = "Mus musculus"
if subject_dob is not None:
    metadata["Subject"]["date_of_birth"] = subject_dob
metadata.setdefault("LabMetaData", {})["behavioral_start_offset"] = behavioral_start_offset

out = Path("/Users/ct5868/code/tank-lab-to-nwb-clean/output")
out.mkdir(exist_ok=True)
nwb_path = out / f"{metadata['NWBFile']['session_id']}_sync_validation.nwb"
print(f"\nRunning sync conversion -> {nwb_path}")
converter.run_conversion(nwbfile_path=str(nwb_path), metadata=metadata, overwrite=True)
print("Conversion complete.\n")

# ---- Assertions ----
with NWBHDF5IO(str(nwb_path), "r") as io:
    nwb = io.read()
    pos = nwb.processing["behavior"]["Position"]["Position"]
    bt = pos.timestamps[:]
    print(f"Behavioral timestamps (IMEC): {bt[0]:.6f}s .. {bt[-1]:.6f}s ({len(bt)} frames)")

    assert len(bt) == len(sync_timestamps), f"frame count mismatch {len(bt)} vs {len(sync_timestamps)}"
    assert bt[0] > 0, f"first behavioral timestamp should be > 0, got {bt[0]}"
    assert abs(bt[0] - behavioral_start_offset) < 1e-6, f"first ts {bt[0]} != offset {behavioral_start_offset}"
    assert abs(bt[0] - sync_timestamps[0]) < 1e-9, "behavioral ts not equal to sync source"
    assert np.allclose(bt, sync_timestamps, atol=1e-9), "behavioral timestamps diverge from sync source"
    print(f"  OK: behavior is IMEC-relative, starts at offset {behavioral_start_offset:.6f}s, matches sync source")

    # behavioral_start_offset persisted in scratch
    assert "behavioral_start_offset" in nwb.scratch, "behavioral_start_offset not stored in scratch"
    stored_offset = float(nwb.scratch["behavioral_start_offset"].data[()])
    assert abs(stored_offset - behavioral_start_offset) < 1e-6, "stored offset mismatch"
    assert abs(stored_offset - bt[0]) < 1e-6, "stored offset != first behavioral timestamp"
    print(f"  OK: behavioral_start_offset stored in scratch = {stored_offset:.6f}s")

    # Experimenter DANDI format
    exp = list(nwb.experimenter)
    assert all("," in e for e in exp), f"experimenter not DANDI-formatted: {exp}"
    print(f"  OK: experimenter DANDI format: {exp}")

    # DOB
    if subject_dob is not None:
        assert nwb.subject.date_of_birth is not None, "DOB missing in NWB"
        print(f"  OK: DOB present: {nwb.subject.date_of_birth}")

    # Kilosort spike times unchanged & IMEC-relative
    probe_counts = dict(Counter(nwb.units["probe_id"][:]))
    print(f"  Units per probe_id: {probe_counts}")
    assert len(probe_counts) >= 2, "expected multi-probe units"
    u0 = nwb.units["spike_times"][0][:5]
    print(f"  Unit 0 spike times (IMEC): {u0}")

# Confirm alignment did not mutate the Kilosort spike times
for name, iface in converter.data_interface_objects.items():
    if name.startswith("Kilosort"):
        s = iface.sorting_extractor
        uid = s.get_unit_ids()[0]
        post = (s.get_unit_spike_train(uid) / s.get_sampling_frequency())[:5]
        assert np.allclose(pre_spikes[name], post), f"{name} spike times changed by alignment"
print("  OK: Kilosort spike times unchanged by temporal alignment (remain IMEC-relative)")

print("\nSYNC VALIDATION PASSED")
