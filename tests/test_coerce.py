"""Coercion to a registry dtype never loses data: it converts exactly, or raises and names what failed."""

import numpy as np
import pytest

from tank_lab_to_nwb.virmen.coerce import LossyConversionError, coerce

WHERE = dict(file="session.mat", field="log.block.trial.x")


def _coerce(values, dtype, **kwargs):
    return coerce(np.asarray(values), dtype, **WHERE, **kwargs)


class TestIntegerTargets:
    def test_fractional_value_fails(self):
        with pytest.raises(LossyConversionError, match="1.5"):
            _coerce([1.0, 1.5], "int32")

    def test_error_names_the_file_field_index_and_value(self):
        with pytest.raises(LossyConversionError) as error:
            _coerce([1.0, 2.0, 2.5], "int32", row=7)
        message = str(error.value)
        assert "session.mat" in message and "log.block.trial.x" in message
        assert "row 7" in message and "element 2" in message and "2.5" in message

    @pytest.mark.parametrize("value", [300, -129, 128])
    def test_out_of_range_for_int8_fails(self, value):
        with pytest.raises(LossyConversionError, match=str(value)):
            _coerce([value], "int8")

    @pytest.mark.parametrize("value", [127, -128, 0])
    def test_range_limits_of_int8_pass(self, value):
        result = _coerce([float(value)], "int8")
        assert result.dtype == np.int8 and result[0] == value

    def test_uint32_nil_does_not_fit_int32(self):
        with pytest.raises(LossyConversionError, match="4294967295"):
            _coerce(np.array([4294967295], dtype="uint32"), "int32")

    def test_nan_without_a_sentinel_fails(self):
        with pytest.raises(LossyConversionError, match="nan"):
            _coerce([1.0, np.nan], "int32")

    def test_nan_with_a_sentinel_becomes_the_sentinel(self):
        result = _coerce([1.0, np.nan], "int8", missing=-1)
        assert result.tolist() == [1, -1]

    @pytest.mark.parametrize("value", [np.inf, -np.inf])
    def test_inf_fails_even_with_a_sentinel(self, value):
        with pytest.raises(LossyConversionError, match="inf"):
            _coerce([value], "int8", missing=-1)

    def test_large_double_stays_exact(self):
        assert _coerce([2.0**31 - 1], "int32")[0] == 2**31 - 1

    def test_just_past_the_range_fails(self):
        with pytest.raises(LossyConversionError):
            _coerce([2.0**31], "int32")

    def test_bool_source_converts(self):
        assert _coerce([True, False], "int8").tolist() == [1, 0]

    def test_text_source_fails(self):
        with pytest.raises(LossyConversionError):
            _coerce(["1"], "int32")

    def test_unknown_code_fails(self):
        with pytest.raises(LossyConversionError, match="2"):
            _coerce([0.0, 2.0], "int8", codes={0: "off", 1: "on"})

    def test_known_codes_pass_including_zero(self):
        assert _coerce([0.0, 1.0], "int8", codes={0: "off", 1: "on"}).tolist() == [0, 1]

    def test_sentinel_must_not_trip_the_code_check(self):
        result = _coerce([np.nan, 1.0], "int8", codes={-1: "not specified", 1: "on"}, missing=-1)
        assert result.tolist() == [-1, 1]


class TestBoolTargets:
    def test_two_fails(self):
        with pytest.raises(LossyConversionError, match="2"):
            _coerce([0.0, 2.0], "bool")

    @pytest.mark.parametrize("values", [[0.0, 1.0], [False, True], [0, 1]])
    def test_zero_and_one_pass(self, values):
        result = _coerce(values, "bool")
        assert result.dtype == np.bool_ and result.tolist() == [False, True]

    def test_nan_fails(self):
        with pytest.raises(LossyConversionError, match="nan"):
            _coerce([np.nan], "bool")

    def test_fraction_fails(self):
        with pytest.raises(LossyConversionError, match="0.5"):
            _coerce([0.5], "bool")


class TestFloatTargets:
    def test_float64_is_bit_for_bit(self):
        values = np.array([0.1, 1 / 3, 1e-300, -0.0, np.nan, np.inf, 5e-324], dtype="float64")
        result = _coerce(values, "float64")
        assert result.dtype == np.float64
        assert result.tobytes() == values.tobytes()

    def test_float32_source_stays_bit_for_bit(self):
        values = np.array([0.1, np.nan, -0.0, 1e38], dtype="float32")
        result = _coerce(values, "float32")
        assert result.dtype == np.float32
        assert result.tobytes() == values.tobytes()

    def test_double_into_float32_fails_when_inexact(self):
        with pytest.raises(LossyConversionError, match="0.1"):
            _coerce(np.array([0.1], dtype="float64"), "float32")

    def test_double_into_float32_passes_when_exact(self):
        result = _coerce(np.array([0.5, np.nan, np.inf], dtype="float64"), "float32")
        assert result.dtype == np.float32
        assert result[0] == 0.5 and np.isnan(result[1]) and np.isinf(result[2])

    def test_double_overflowing_float32_fails(self):
        with pytest.raises(LossyConversionError):
            _coerce(np.array([1e300], dtype="float64"), "float32")

    def test_single_widens_to_float64_exactly(self):
        values = np.array([0.1, np.nan], dtype="float32")
        result = _coerce(values, "float64")
        assert result.dtype == np.float64
        assert result[0] == float(values[0]) and np.isnan(result[1])

    def test_integer_widens_to_float64(self):
        assert _coerce(np.array([3], dtype="uint16"), "float64").tolist() == [3.0]

    def test_huge_integer_inexact_in_float64_fails(self):
        with pytest.raises(LossyConversionError):
            _coerce(np.array([2**53 + 1], dtype="int64"), "float64")

    def test_integer_into_float32_fails_when_inexact(self):
        with pytest.raises(LossyConversionError):
            _coerce(np.array([2**24 + 1], dtype="int32"), "float32")

    def test_text_source_fails(self):
        with pytest.raises(LossyConversionError):
            _coerce(["1.5"], "float64")


class TestText:
    def test_strings_pass_as_object_array(self):
        result = _coerce(["a", ""], "text")
        assert result.dtype == object and result.tolist() == ["a", ""]

    def test_number_into_text_fails(self):
        with pytest.raises(LossyConversionError):
            _coerce([1.0], "text")

    def test_unknown_text_code_fails(self):
        with pytest.raises(LossyConversionError, match="X"):
            _coerce(["L", "X"], "text", codes={"L": "left", "R": "right"})


@pytest.mark.parametrize("dtype, expected", [("int8", "int8"), ("bool", "bool"), ("float32", "float32"), ("text", "object")])
def test_empty_input_keeps_the_declared_dtype(dtype, expected):
    result = _coerce(np.zeros(0), dtype) if dtype != "text" else _coerce(np.array([], dtype=object), dtype)
    assert result.size == 0 and result.dtype == np.dtype(expected)


def test_multidimensional_error_reports_the_index():
    with pytest.raises(LossyConversionError, match=r"\(1, 0\)"):
        _coerce([[1.0, 2.0], [2.5, 3.0]], "int32")


def test_shape_is_kept():
    assert _coerce([[1.0, 2.0], [3.0, 4.0]], "int32").shape == (2, 2)


def test_unknown_dtype_is_rejected():
    with pytest.raises(ValueError, match="float16"):
        _coerce([1.0], "float16")
