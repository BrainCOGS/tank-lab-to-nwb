"""The mat-io loader decodes logs into plain values, and a decode failure stops the conversion."""

import warnings

import numpy as np
import pytest
from matio.utils.matclass import MatConvertWarning

from tank_lab_to_nwb.virmen import loader
from tank_lab_to_nwb.virmen.loader import Enum, FunctionHandle, LogLoadError, decode, records


def test_conversion_warning_becomes_an_error(monkeypatch, tmp_path):
    def fails_to_decode(path):
        warnings.warn("Failed to convert MCOS object of class table", MatConvertWarning)
        return {"log": None}

    monkeypatch.setattr(loader.matio, "load_from_mat", fails_to_decode)
    with pytest.raises(LogLoadError, match="session.mat"):
        loader.load_log(tmp_path / "session.mat")


def test_file_without_a_log_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(loader.matio, "load_from_mat", lambda path: {"other": 1})
    with pytest.raises(LogLoadError, match="no 'log'"):
        loader.load_log(tmp_path / "session.mat")


def test_canonical_empty_is_an_empty_array():
    from matio.utils.matclass import MatlabCanonicalEmpty

    assert decode(MatlabCanonicalEmpty()).size == 0


def test_empty_char_is_an_empty_string():
    assert decode(np.array([], dtype="<U1")) == ""


def test_char_row_is_a_string():
    assert decode(np.array(["abc"])) == "abc"


def test_char_matrix_keeps_every_row():
    assert decode(np.array(["ab", "cd"])) == ["ab", "cd"]


def test_numeric_keeps_dtype_and_shape():
    value = np.array([[1.5]], dtype="float32")
    decoded = decode(value)
    assert decoded.dtype == np.float32 and decoded.shape == (1, 1)


def test_struct_scalar_is_a_dict():
    value = np.zeros((1, 1), dtype=[("a", "O")])
    value[0, 0]["a"] = np.array([[0.0]])
    decoded = decode(value)
    assert decoded["a"].tolist() == [[0.0]]


def test_struct_array_is_a_list():
    value = np.zeros((1, 2), dtype=[("a", "O")])
    for i in range(2):
        value[0, i]["a"] = np.array([[float(i)]])
    assert [r["a"].item() for r in decode(value)] == [0.0, 1.0]


def test_empty_struct_array_is_an_empty_list():
    assert decode(np.zeros((0, 0), dtype=[("a", "O")])) == []


def test_cell_keeps_shape_and_decodes_elements():
    cell = np.empty((1, 2), dtype=object)
    cell[0, 0] = np.array(["x"])
    cell[0, 1] = np.array([[1.0]])
    decoded = decode(cell)
    assert decoded.shape == (1, 2) and decoded[0, 0] == "x" and decoded[0, 1].item() == 1.0


@pytest.mark.parametrize("value, expected", [({"a": 1}, [{"a": 1}]), ([{"a": 1}, {"a": 2}], [{"a": 1}, {"a": 2}]), ([], [])])
def test_records_gives_a_list_for_scalar_and_array_structs(value, expected):
    assert records(value) == expected


def test_records_of_an_empty_matrix_is_empty():
    assert records(np.zeros((0, 0))) == []


def test_enum_and_function_are_plain_values():
    assert Enum("L", 1).name == "L" and FunctionHandle("f").name == "f"
