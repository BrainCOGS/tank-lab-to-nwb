# tank-lab-to-nwb

NWB conversion scripts and tutorials. A collaboration with the [Tank lab](https://pni.princeton.edu/faculty/david-tank), funded by the Simons Foundation.

## Install

```bash
pip install tank-lab-to-nwb
```

## Ephys-Behavior Synchronization

This converter integrates with the U19 DataJoint pipeline to access pre-computed synchronized timestamps between Virmen behavioral data and NIDQ/IMEC electrophysiology recordings.

### Synchronization Data Structure

The `BehaviorSync` table stores compact synchronization data in the `sync_data` field:
- **trial_idx_vector**: NIDQ sample indices where each trial starts
- **iteration_idx_vector**: Frame-by-frame NIDQ sample indices for each trial (array of arrays)
- Fallback versions using Virmen timestamps when pulse-based sync fails

See [SYNC_DATA_PLAN.md](SYNC_DATA_PLAN.md) for detailed documentation on the synchronization architecture.

## Usage

There are two ways to go about converting Neuropixel and Virmen behavior data.

(1) The **primary processing pipeline** synchronizes the task data with the electrophysiology data
   through TTL pulse and writes the spiking output to the same NWB file.

The required arguments for the use of the relevant functions are denoted in the comments of their
respective sections of the conversion script. These include the file or
folder locations of the data to be converted to NWB format, as well as several optional fields
such as Subject information (species/age/weight).

After editing the conversion script `convert_towers.py` with the proper path for Neuropixel and behavior data,
the conversion can be executed from the terminal:

```bash
cd tank-lab-to-nwb
python tank_lab_to_nwb/convert_towers_task/convert_towers.py
```

Alternatively, the conversion can be done using a custom tailored jupyter notebook `spikeinterface_pipeline.ipynb`
that can be launched from the terminal:

```bash
jupyter notebook notebooks/spikeinterface_pipeline.ipynb
```

(2) The NWBFile can be inspected by reading it from a python script:

```python
from pynwb import NWBHDF5IO

file_path = 'TowersTask_stub.nwb'
io = NWBHDF5IO(file_path, 'r')
nwb = io.read()
print(nwb)
```

Alternatively, the NWB data can be visualized with `nwb-jupyter-widgets` in a jupyter notebook:

```bash
jupyter notebook notebooks/towers_task_custom_widget.ipynb
```

```python
from pynwb import NWBHDF5IO
from nwbwidgets import nwb2widget
from tank_lab_to_nwb.nwbwidgets import custom_timeseries_widget_for_behavior
from nwbwidgets.view import default_neurodata_vis_spec
import pynwb


file_path = 'TowersTask_stub.nwb'
io = NWBHDF5IO(file_path, 'r')
nwb = io.read()

default_neurodata_vis_spec[pynwb.TimeSeries] = custom_timeseries_widget_for_behavior
nwb2widget(nwb)
```

## Background

### Behavioral data mapping

The behavioral data is contained in `.mat`files similar to the form:

`PoissonBlocksReboot4_cohort4_Bezos3_E65_T_20180202.mat`
This matlab file contains a struct (`log`) which contains several fields relevant for conversion.
The list of fields that are extracted from this struct can be found below.

#### NWBFile

Location in Virmen (.mat) file  | Location in NWB file | Description
------------- | ------------- | -------------
`log.session.start` | `nwb.session_start_time` | `datetime` when session started
  [not in file] | `nwb.session_description` | additional information about session (optional)
 [name of file] | `nwb.session_id`| unique identifier of the session

#### Subject

Location in Virmen (.mat) file  | Location in NWB file | Description
------------- | ------------- | -------------
`log.session.start` | `nwb.subject.age` | age (days) in isoformat (optional)
  [not in file] | `nwb.species` | information about the species (optional)
 `log.animal.name` | `nwb.subject_id`| identifier of the subject
 [not in file] | `nwb.genotype`| information about the genotype (optional)
 [not in file] | `nwb.sex`| information about the sex of the subject (optional)

#### LabMetaData

The lab specific metadata is populated in `tank_lab_to_nwb/convert_towers_task/virmenbehaviordatainterface.py`
using the custom extension [ndx-tank-metadata](https://github.com/catalystneuro/ndx-tank-metadata)
built for extending the NWB LabMetaData schema with the required fields:

Location in Virmen (.mat) file  | Location in NWB file | Description
------------- | ------------- | -------------
`log.version.code` | `nwb.lab_meta_data['LabMetaData'].experiment_name` | name of experiment run
  `log.version.name` | `nwb.lab_meta_data['LabMetaData'].world_file_name` | name of world run
 `log.animal.protocol` | `nwb.lab_meta_data['LabMetaData'].protocol_name`| name of protocol run
 `log.animal.stimulusBank` | `nwb.lab_meta_data['LabMetaData'].stimulus_bank_path`| path of stimulus bank file
 `log.version.repository` | `nwb.lab_meta_data['LabMetaData'].commit_id`| commit id for session run
 `log.session.end` | `nwb.lab_meta_data['LabMetaData'].session_end_time`| `datetime` when session ended
 `log.version.rig.rig` | `nwb.lab_meta_data['LabMetaData'].location`| name of rig where session was run
 [not in file] | `nwb.lab_meta_data['LabMetaData'].num_trials`| number of trials in the session
 [not in file] | `nwb.lab_meta_data['LabMetaData'].session_performance`| performance of correct responses in % (optional)*
 `log.version.rig` | `nwb.lab_meta_data['LabMetaData'].rig.fields`| rig information
 `log.version.mazes` | `nwb.lab_meta_data['LabMetaData'].mazes.to_dataframe()`| maze information

* session_performance can be edited from `virmenbehaviordatainterface.py`.

##### Rig

Rig information is converted from a `log.version.rig` struct object to a dictionary like as in this example:

```python
{'rig': 'NPX',
 'simulationMode': 1,
 'hasDAQ': 1,
 'hasSyncComm': 0,
 'minIterationDT': 0.01,
 'arduinoPort': 'COM18',
 'sensorDotsPerRev': array([2469.2, 2469.2, 2469.2, 2469.2]),
 'ballCircumference': 63.8,
 'toroidXFormP1': 0.3879,
 'toroidXFormP2': 0.392,
 'colorAdjustment': array([0. , 0.4, 0.5]),
 'soundAdjustment': 0.2,
 'nidaqDevice': 1,
 'nidaqPort': 1,
 'nidaqLines': array([ 0, 11], dtype=int32),
 'syncClockChannel': 5,
 'syncDataChannel': 6,
 'rewardChannel': 0,
 'rewardSize': 0.004,
 'rewardDuration': 0.05,
 'laserChannel': 1,
 'rightPuffChannel': 2,
 'leftPuffChannel': 3}
 ```

##### Mazes

Maze information is converted to a [DynamicTable](https://hdmf.readthedocs.io/en/stable/tutorials/dynamictable.html)
object that can be converted to a pandas dataframe by calling `.to_dataframe()` as in this example:

   id |   world |   lStart |   lCue |   lMemory |   cueDuration |   cueVisibleAt |   cueProbability | ... | blockPerform
-----|--------|---------|-------|----------|--------------|---------------|-----------------|-----------------|-----------------
0 |       1 |        5 |     45 |        10 |           nan |            inf |              inf | ... | 0.7
1 |       1 |       30 |    120 |        20 |           nan |            inf |              inf | ... | 0.7
2 |       1 |       30 |    220 |        20 |           nan |            inf |              inf | ... | 0.7
3 |       1 |       30 |    300 |        20 |           nan |            inf |              inf | ... | 0.7
4 |       1 |       30 |    380 |        20 |           nan |            inf |              inf | ... | 0.7

#### Epochs

Location in Virmen (.mat) file  | Location in NWB file | Description
------------- | ------------- | -------------
index of `log.block` structure | `nwb.intervals['epochs'].id` | number of epoch in session
`log.block.mazeID`| `nwb.intervals['epochs'].maze_id` | number of maze in epoch
`log.block.mainMazeID` | `nwb.intervals['epochs'].main_maze_id` | number of maze of "highest" level for subject
`log.block.easyBlockFlag` | `nwb.intervals['epochs'].easy_epoch` | 1 if block was flagged as easy (maze_id < main_maze_id)
`log.block.firstTrial` | `nwb.intervals['epochs'].first_trial` | first trial run in an epoch
[not in file] | `nwb.intervals['epochs'].num_trials`| number of trials for each epoch
`log.block.start` | `nwb.intervals['epochs'].start_time` | datetime when epoch started with respect to the start time of the session
`log.block.duration` | `nwb.intervals['epochs'].duration` | epoch duration in seconds
`log.block.rewardMiL` | `nwb.intervals['epochs'].reward_ml`| ml of reward in an epoch

#### Trials

Location in Virmen (.mat) file  | Location in NWB file | Description
------------- | ------------- | -------------
[not in file] | `nwb.intervals['trials'].id` | unique identifier of trial for all epochs
index of `log.block.trial` | `nwb.intervals['trials'].trial_id` | identifier of trial within an epoch
`log.block.trial.trialType`| `nwb.intervals['trials'].trial_type` | type of trial (L=Left, R=Right)
`log.block.trial.choice` | `nwb.intervals['trials'].choice` | (L=Left, R=Right, nil=Trial violation)
`log.block.trial.start` | `nwb.intervals['trials'].start_time` | start time of trial with respect to the start time of the epoch
`log.block.trial.duration` | `nwb.intervals['trials'].duration` | duration of trial in seconds
`log.block.trial.iterations`| `nwb.intervals['trials'].iterations`| number of frames in a trial
`log.block.trial.iCueEntry` | `nwb.intervals['trials'].iCueEntry` | iteration # when subject entered cue region
`log.block.trial.iMemEntry` | `nwb.intervals['trials'].iMemEntry` | iteration # when subject entered memory region
`log.block.trial.iTurnEntry` | `nwb.intervals['trials'].iTurnEntry` | iteration # when subject entered turn region
`log.block.trial.iArmEntry` | `nwb.intervals['trials'].iArmEntry` | iteration # when subject entered arm region
`log.block.trial.iBlank` | `nwb.intervals['trials'].iBlank` | iteration # when screen if turned off
`log.block.trial.cueCombo` | `nwb.intervals['trials'].left_cue_presence, nwb.intervals['trials'].right_cue_presence` | indicates if nth cue appeared on left or right
`log.block.trial.cuePosition` | `nwb.intervals['trials'].left_cue_position, nwb.intervals['trials'].right_cue_position` | position in maze for each cue
`log.block.trial.cueOnset` | `nwb.intervals['trials'].left_cue_onset, nwb.intervals['trials'].right_cue_onset` | iteration number when cues appeared in trial
`log.block.trial.cueOffset` | `nwb.intervals['trials'].left_cue_offset, nwb.intervals['trials'].right_cue_offset` | iteration number when cues disappeared in trial
`log.block.trial.excessTravel` | `nwb.intervals['trials'].excessTravel`| parameter that measures extra distance run by subject
`log.block.trial.rewardScale` | `nwb.intervals['trials'].rewardScale`| Multiplier of reward for each correct trial

#### Behavior

##### Position, ViewAngle, Velocity, Collision

Location in Virmen (.mat) file  | Location in NWB file | Description
------------- | ------------- | -------------
`log.block.trial.time` | `nwb.processing['behavior'].data_interfaces['Time']`| time vector for each frame measured by Virmen
`log.block.trial.position` | `nwb.processing['behavior'].data_interfaces['Position'].spatial_series`| position matrix for each frame (X(cm), Y(cm))
`log.block.trial.position` | `nwb.processing['behavior'].data_interfaces['ViewAngle'].spatial_series`| viewAngle for each frame (degrees)
`log.block.trial.velocity` | `nwb.processing['behavior'].data_interfaces['Velocity']`| velocity matrix for each frame (X(cm/s), Y(cm/s))
`log.block.trial.collision` |`nwb.processing['behavior'].data_interfaces['Collision']`| for each frame 1= collision detected
