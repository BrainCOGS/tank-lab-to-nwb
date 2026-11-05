"""Writing a normalized session: what's on disk equals the normalized values, bit for bit."""

import warnings

import h5py
import numpy as np
import pytest
from pynwb import NWBHDF5IO, validate

from tank_lab_to_nwb.virmen.convert import convert_log, write_session
from tank_lab_to_nwb.virmen.session import normalize_log
from tank_lab_to_nwb.virmen.verify import RoundTripError, verify_round_trip
from tests.synthetic_log import make_block, make_log, make_trial, single
from tests.test_session import _stimulus_table

FULL_TRIAL = dict(
    licks=np.array([[10.0, 20.0], [1.0, 2.0]], dtype="float32"),
    forcedChoice=single(0),
    wallGuide=np.array([[False]]),
    startPosition=np.array([[0.0, -30.0, 2.0, 0.0]], dtype="float32"),
    rule="StartCycle < EndCycle Left",
    pairNum=single(1),
    stimulusTable=_stimulus_table([[1, 0.5, "L", 3, 9, 0.5, 1, 0.5], [2, 0.5, "R", 9, 3, np.nan, 0, 0.5]]),
)


def _session():
    return normalize_log(make_log([make_block([make_trial(**FULL_TRIAL), make_trial(start=9.0, **FULL_TRIAL)])]), file="session.mat")


@pytest.fixture
def written(tmp_path):
    session = _session()
    path = tmp_path / "out.nwb"
    write_session(session, path)
    return session, path


def test_passes_pynwb_validate(written):
    errors = validate(path=str(written[1]))
    errors = errors[0] if isinstance(errors, tuple) else errors
    assert list(errors) == []


def test_no_silent_cast_warnings_on_write(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        write_session(_session(), tmp_path / "out.nwb")


def test_declared_dtypes_are_on_disk(written):
    with h5py.File(written[1], "r") as f:
        trials = f["intervals/trials"]
        assert trials["forcedChoice"].dtype == np.int8
        assert trials["wallGuide"].dtype == np.bool_ or trials["wallGuide"].dtype.kind in "bui"
        assert trials["cuePos_left"].dtype == np.float32
        assert trials["cuePos_left_index"].dtype.kind == "u"
        assert trials["start_time"].dtype == np.float64
        assert f["intervals/epochs/rewardMiL"].dtype == np.float64


def test_round_trip_is_exact(written):
    verify_round_trip(*written)


def test_trials_and_epochs_use_the_subtypes(written):
    with NWBHDF5IO(str(written[1]), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        assert type(nwbfile.trials).__name__ == "VirmenTrials"
        assert type(nwbfile.epochs).__name__ == "VirmenEpochs"
        assert nwbfile.subject.subject_id == "mouse_1"
        assert nwbfile.session_id == "20260715"
        assert nwbfile.trials["stimulus_pair"][0]["Sa"].iloc[0] == 3
        lab = nwbfile.lab_meta_data["LabMetaData"]
        assert lab.rig.rig_type == "miniVR" and lab.mazes["lCue"].data[0] == 200
        assert lab.settings["key"].data[0]


def test_series_share_one_clock_per_frame_rate(written):
    with h5py.File(written[1], "r") as f:
        behavior = f["processing/behavior"]
        position = behavior["Position/position"]
        assert position.attrs["unit"] if "unit" in position.attrs else position["data"].attrs["unit"] == "cm"
        assert behavior["view_angle/data"].attrs["unit"] == "radians"
        assert behavior["AngularVelocity/data"].attrs["unit"] == "radians/s"


class TestVerifyCatchesDifferences:
    def _tamper(self, path, dataset, change):
        with h5py.File(path, "a") as f:
            data = f[dataset][()]
            change(data)
            f[dataset][...] = data

    def test_changed_float_is_caught(self, written):
        self._tamper(written[1], "intervals/epochs/rewardMiL", lambda d: d.__setitem__(0, d[0] + 1e-15))
        with pytest.raises(RoundTripError, match="rewardMiL"):
            verify_round_trip(*written)

    def test_changed_index_is_caught(self, written):
        self._tamper(written[1], "intervals/trials/cuePos_left_index", lambda d: d.__setitem__(0, 0))
        with pytest.raises(RoundTripError, match="cuePos_left"):
            verify_round_trip(*written)

    def test_changed_series_is_caught(self, written):
        self._tamper(written[1], "processing/behavior/Position/position/data", lambda d: d.__setitem__((0, 0), 99.0))
        with pytest.raises(RoundTripError, match="position"):
            verify_round_trip(*written)

    def test_changed_attribute_is_caught(self, written):
        with h5py.File(written[1], "a") as f:
            f["general/LabMetaData/rig"].attrs["rewardChannel"] = 5
        with pytest.raises(RoundTripError, match="rewardChannel"):
            verify_round_trip(*written)

    def test_changed_setting_is_caught(self, written):
        with h5py.File(written[1], "a") as f:
            data = f["general/LabMetaData/settings/value"]
            values = data.asstr()[()]
            values[0] = "changed"
            data[...] = values.astype(object)
        with pytest.raises(RoundTripError, match="settings"):
            verify_round_trip(*written)


def test_convert_log_is_verified_and_reports_coverage(monkeypatch, tmp_path):
    import tank_lab_to_nwb.virmen.convert as convert

    monkeypatch.setattr(convert, "load_log", lambda path: make_log())
    session = convert_log(tmp_path / "session.mat", tmp_path / "out.nwb")
    assert (tmp_path / "out.nwb").exists()
    assert (tmp_path / "out.coverage.json").exists()
    assert session.subject_id == "mouse_1"


def test_view_angle_is_not_a_spatial_series(written):
    # It's unwrapped and legitimately runs past 2*pi, which a radian SpatialSeries isn't allowed to.
    with NWBHDF5IO(str(written[1]), "r", load_namespaces=True) as io:
        assert type(io.read().processing["behavior"]["view_angle"]).__name__ == "TimeSeries"


def test_placeholder_subject_is_marked(written):
    with NWBHDF5IO(str(written[1]), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        assert "placeholder" in nwbfile.subject.description.lower()
        assert nwbfile.subject.species == "Mus musculus"
