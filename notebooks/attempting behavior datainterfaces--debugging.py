# %%
"""Authors: Christian"""
import sys
import os
sys.path.append(os.getcwd())
print(os.getcwd())

from pathlib import Path
from isodate import duration_isoformat
from datetime import timedelta, datetime

import neuroconv

# from tank_lab_to_nwb import VirmenDataInterface
from tank_lab_to_nwb.convert_towers_task import VirmenDataInterface



# %%
base_path = Path("/Users/ct5868/code/tank-lab-to-nwb/")
virmen_file_path = base_path / "behavior_local" / "jessejorge_pwm_jessejorge_pwm_pilot_165I-miniVR-T-6_jyanar_ya011_T_20240322.mat"


interface = VirmenDataInterface(file_path = virmen_file_path)

# %%
metadata = interface.get_metadata()
print(metadata)

# %%
metadata["Subject"] = dict(
    subject_id="M001",
    sex="M",
    age="P30D",
    species="Mus musculus",
)


# %%
interface.get_metadata_schema()

# %%
from datetime import datetime
from zoneinfo import ZoneInfo

metadata["NWBFile"]["session_start_time"] = datetime(2021, 1, 1, 12, 0, 0, tzinfo=ZoneInfo("US/Pacific"))

# %%
interface.get_metadata_schema()

# %%
from pynwb.testing.mock.file import mock_NWBFile
empty_nwb = mock_NWBFile()

print(metadata)


# %%


# interface.create_nwbfile(metadata = metadata)
interface.run_conversion(nwbfile_path="./mariokart.nwb", metadata = metadata, overwrite=True)
# interface.run_conversion(nwbfile_path="./fdf13.nwb", metadata = metadata, overwrite=True)

# %%



