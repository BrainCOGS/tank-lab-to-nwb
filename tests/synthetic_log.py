"""Decoded ViRMEn logs built by hand, with no project data."""

import numpy as np

from tank_lab_to_nwb.virmen.loader import Enum, FunctionHandle

L, R, NIL = Enum("L", 1), Enum("R", 2), Enum("nil", 4294967295)


def scalar(value, dtype="float64"):
    return np.array([[value]], dtype=dtype)


def single(value):
    return scalar(value, "float32")


def lr_cell(left, right, dtype="float32"):
    cell = np.empty((1, 2), dtype=object)
    cell[0, 0] = np.array([left], dtype=dtype).reshape(1, -1)
    cell[0, 1] = np.array([right], dtype=dtype).reshape(1, -1)
    return cell


def port_cell(*snapshots):
    """A trial's state_ports: one 2x5 float32 reading per lick, or ExperimentLog's initial {[]} with none."""
    members = [np.asarray(snapshot, dtype="float32") for snapshot in snapshots] or [np.zeros((0, 0))]
    cell = np.empty((1, len(members)), dtype=object)
    for position, member in enumerate(members):
        cell[0, position] = member
    return cell


def ports(left_contact=False, right_contact=False):
    """A reading whose newest two columns show contact (0) on the chosen port after a reading without (1)."""
    snapshot = np.ones((2, 5))
    for row, contact in enumerate((left_contact, right_contact)):
        if contact:
            snapshot[row, :2] = 0
    return snapshot


def make_trial(start=1.0, n_frames=3, n_iti=2, choice=L, trial_type=L, **extra):
    n_time = n_frames + n_iti
    trial = dict(
        start=single(start),
        duration=single(n_time * 0.5),
        time=(np.arange(n_time, dtype="float32") * 0.5).reshape(-1, 1),
        position=np.arange(n_frames * 3, dtype="float32").reshape(-1, 3),
        velocity=np.arange(n_frames * 3, dtype="float32").reshape(-1, 3) + 0.5,
        collision=np.zeros((n_frames, 1), dtype=bool),
        sensorDots=np.zeros((n_time, 5), dtype="int16"),
        iterations=single(n_frames),
        viStart=np.array([[2]], dtype="uint32"),
        choice=choice,
        trialType=trial_type,
        trialProb=np.array([[0.25], [0.75]], dtype="float32"),
        trialID=single(1),
        cueCombo=np.array([[True, False], [False, True]]),
        cuePos=lr_cell([10.5], [20.5]),
        cueOnset=lr_cell([1], [2], "uint16"),
        cueOffset=lr_cell([2], [3], "uint16"),
        iCueEntry=np.array([[1]], dtype="uint16"),
        iMemEntry=np.array([[2]], dtype="uint16"),
        excessTravel=single(0.1),
        rewardScale=single(1.5),
        comment="",
    )
    trial.update(extra)
    return trial


def make_block(trials, start=(2026, 7, 15, 12, 0, 1.0), **extra):
    block = dict(
        start=np.array([start], dtype="float64"),
        duration=scalar(10.0),
        mazeID=scalar(3),
        mainMazeID=scalar(3),
        firstTrial=scalar(1),
        rewardMiL=scalar(0.1),
        easyBlockFlag=np.array([[False]]),
        frozenStimuli=np.array([[False]]),
        shapingProtocol=FunctionHandle("protocol"),
        stimulusBank="bank.mat",
        stimulusConfig=scalar(0),
        stimulusSet=scalar(1),
        versionIndex=scalar(1),
        trial=trials,
    )
    block.update(extra)
    return block


def make_maze(**extra):
    maze = dict(
        world=scalar(1),
        color={},
        turnHint_Mem=np.array([[False]]),
        visible={"tri_turnHint": np.array([[True]])},
        variable={"lStart": "30", "lCue": "200", "lMemory": "100", "cueDuration": "0.6"},
        criteria={"numTrials": scalar(100), "maxBias": scalar(np.inf), "warmupMaze": np.zeros((0, 0))},
    )
    maze.update(extra)
    return maze


def make_version(**extra):
    version = dict(
        name="world",
        code=FunctionHandle("experiment"),
        repository="abc123",
        codeVersion=scalar(4.0),
        mazeVersion=scalar(1.0),
        rig=dict(
            rig="rig-1",
            rig_type="miniVR",
            hasDAQ=np.array([[True]]),
            ballCircumference=scalar(63.8),
            sensorDotsPerRev=scalar(2000.5),
            colorAdjustment=np.array([[0.0], [0.4], [0.5]]),
            nidaqLines=np.array([[0.0, 11.0]]),
            rewardChannel=scalar(0),
            laserIP="0.0.0.0",
        ),
        mazes=[make_maze(), make_maze(world=scalar(2))],
        variables={"lCue": "300", "worldXRange": "[-50 50]"},
        worlds={"name": "Tmaze", "objects": np.empty((0, 0), dtype=object)},
    )
    version.update(extra)
    return version


def make_log(blocks=None, **extra):
    draw = np.empty((1, 2), dtype=object)
    draw[0, 0], draw[0, 1] = "TRIAL_DRAWING", scalar(1)
    blocks = blocks if blocks is not None else [make_block(make_trial())]
    n_trials = sum(len(b["trial"]) if isinstance(b["trial"], list) else 1 for b in blocks)
    log = dict(
        session=dict(start=np.array([[2026, 7, 15, 11, 59, 50.0]]), end=np.array([[2026, 7, 15, 12, 0, 11.0]]), blocks=np.array([[1.0, 1.0]])),
        animal=dict(
            name="mouse_1",
            protocol=FunctionHandle("protocol"),
            experiment="experiment.mat",
            stimulusBank="",
            stimulusSet=scalar(1),
            virmenSensor=Enum("BottomVelocity", 1),
            autoAdvance=np.array([[True]]),
            mainMazeID=scalar(3),
            manipulationType="none",
            motionBlurRange=np.zeros((0, 0)),
            pulseDelayStim=np.array([[28.0, 5.0]]),
            rewardFactor=np.array([[2.0, 1.5], [1.0, 1.2]]),
            mainDrawMethod=draw,
            warmupDrawMethod=draw,
            waterAlloc=scalar(1.5),
            softwareParams="",
            stimulationProtocol=scalar(1),
            color=np.array([[0.1, 0.2, 0.3]]),
        ),
        version=make_version(),
        block=blocks,
        numTrials=scalar(n_trials),
        numTrialsTry=scalar(n_trials),
        currentTrial=make_trial(),
        currentIt=scalar(3),
        label="T",
    )
    log.update(extra)
    return log
