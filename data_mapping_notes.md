# Notes on tank-lab-to-nwb data mapping

## Open questions for lab meeting are in straight braces []

## ToDo tasks are in curly {}

All positional information is contained in .mat files similar to the form: PoissonBlocksReboot4_cohort4_Bezos3_E65_T_20180202.mat

This matlab file contains a struct named "log" which contains several fields relevant for conversion...

### NWBFile

* log.session.start is an array of datetime information defining the start of the experiment
* log.animal.protocol may be worth including in session description for the NWBFile
* [do we have information on the experimenter name?]

### Subject

As per meeting on 10/29/2020, much of the subject information in the .mat file is filled with defaults and cannot therefore be trusted.

{To retrieve true subject information we need to work some sort of access to their subject database.}

## Recording

The code currently supports read in of the SpikeGLX data and converts it via the RecordingExtractor pipeline; however, it writes the entirety of the data (including pre-TTL pulse stuff).

{Use the approach very similar to Alessio's notebook where we make a SubRecordingExtractor clipping the start_frame to start with the virmen session_start_time.}

### Position (processed, not acquisition)

* The field log.block is a struct of length n_epochs
* The sub-field log.block(j).trial is a struct of length n_trials for epoch j

* log.block(j).trial(k).time is a vector of timestamps for the positional recording; unfortunately, these do appear to be irregularly sampled and so must be specified in the NWBFile. Also, they always begin indexed at zero with respect to the *trial* start time (see "Intervals" below to learn how to access this)
* log.block(j).trial(k).position is a 3-d vector of length *less* than log.block(j).trial(k).time; the positional recording stops when the subject completes the maze.

{Add Nan padding to SpatialSeries to avoid NwbWidget interpolation; with compression, they should not take much space.}

### Intervals

#### See Position for introduction to block and trial structure within the struct

* log.block(j).start contains an array of datetime information for the start of epoch j
* log.block(j).duration contains the number of seconds epoch j lasted for
* log.block(j).trial(k).start contains a float of the time difference (in seconds) between the start of the first epoch (not the session start time!) and the start of trial k
* log.block(j).trial(k).duration contains the duration (in seconds) of trial k of epoch j

*There are slight overlaps between certain epochs/trials end/start times, but this is not strictly in violation of the nwb schema and so is not an issue.*

Since trials are concatenated in NWBFiles distinct from but in line with the actual epochs, the trial intervals will have to be pulled and assembled in order.

{Include Epoch and Trial column information such as MazeID and cue info}
