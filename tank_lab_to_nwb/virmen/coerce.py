"""Convert values to a registry dtype without losing data.

hdmf casts to the declared dtype on write and only warns (1.5 becomes 1 in an int8 column), so every value is converted
and checked here first. The result is exact, or a LossyConversionError names the file, field, index and value.
"""

import numpy as np

_INTEGERS = ("int8", "int16", "int32")
_FLOATS = ("float32", "float64")
_NUMERIC_KINDS = "fiub"


class LossyConversionError(ValueError):
    """A value cannot be stored in its declared dtype without changing it."""


def _fail(values, position, reason, dtype, file, field, row):
    """Raise for the element at flat position `position`."""
    index = np.unravel_index(position, values.shape)
    index = index[0] if len(index) == 1 else tuple(int(i) for i in index)
    value = values[np.unravel_index(position, values.shape)]
    value = value.item() if hasattr(value, "item") else value
    where = f"row {row}, element {index}" if row is not None else f"element {index}"
    raise LossyConversionError(f"{file}: {field}: {where}: value {value!r} {reason} (declared dtype {dtype})")


def _fail_on(mask, values, reason, dtype, file, field, row):
    if mask.any():
        _fail(values, int(np.argmax(mask.ravel())), reason, dtype, file, field, row)


def _check_numeric_source(values, dtype, file, field, row):
    if values.dtype.kind not in _NUMERIC_KINDS:
        if values.size == 0:
            return
        _fail(values, 0, f"is not numeric ({values.dtype})", dtype, file, field, row)


def _to_integer(values, dtype, missing, file, field, row):
    _check_numeric_source(values, dtype, file, field, row)
    info = np.iinfo(dtype)
    if values.dtype.kind == "f":
        nan = np.isnan(values)
        if missing is None:
            _fail_on(nan, values, "is NaN and no missing sentinel is declared", dtype, file, field, row)
        finite = ~nan
        _fail_on(np.isinf(values), values, "is infinite", dtype, file, field, row)
        with np.errstate(invalid="ignore"):
            _fail_on(finite & (values != np.round(values)), values, "is not an integer", dtype, file, field, row)
            _fail_on(finite & ((values < info.min) | (values > info.max)), values, "is out of range", dtype, file, field, row)
        result = np.where(nan, missing if missing is not None else 0, values).astype(dtype)
        return result
    if values.size and not np.can_cast(values.dtype, dtype, "safe"):
        as_objects = values.astype(object)
        low, high = int(info.min), int(info.max)
        outside = np.fromiter((not low <= int(v) <= high for v in as_objects.ravel()), bool, values.size)
        _fail_on(outside.reshape(values.shape), values, "is out of range", dtype, file, field, row)
    return values.astype(dtype)


def _to_bool(values, dtype, file, field, row):
    _check_numeric_source(values, dtype, file, field, row)
    if values.dtype.kind != "b":
        _fail_on(~((values == 0) | (values == 1)), values, "is not 0 or 1", dtype, file, field, row)
    return values.astype(bool)


def _to_float(values, dtype, file, field, row):
    _check_numeric_source(values, dtype, file, field, row)
    if values.dtype == np.dtype(dtype):
        return values.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        result = values.astype(dtype)
        # Tolerance zero: the conversion must give back exactly the source value. float32 to float64 always does.
        same = result.astype("float64") == values.astype("float64") if values.dtype.kind == "f" else None
    if values.dtype.kind == "f":
        both_nan = np.isnan(result) & np.isnan(values)
        _fail_on(~(same | both_nan), values, "is not exactly representable", dtype, file, field, row)
    else:
        back = result.astype(values.dtype)
        _fail_on((back != values) | (result.astype("float64") != result), values, "is not exactly representable", dtype, file, field, row)
    return result


def _to_text(values, dtype, file, field, row):
    if values.dtype.kind not in "UO" and values.size:
        _fail(values, 0, f"is not text ({values.dtype})", dtype, file, field, row)
    for position, value in enumerate(values.ravel().tolist()):
        if not isinstance(value, str):
            _fail(values, position, "is not text", dtype, file, field, row)
    result = np.empty(values.shape, dtype=object)
    result[...] = values
    return result


def _check_codes(result, values, codes, missing, dtype, file, field, row):
    known = set(codes) | ({missing} if missing is not None else set())
    if result.dtype == object:
        unknown = np.fromiter((v not in known for v in result.ravel()), bool, result.size)
    else:
        unknown = ~np.isin(result.ravel(), list(known))
    if unknown.any():
        _fail(values, int(np.argmax(unknown)), f"is not a known code ({sorted(map(str, codes))})", dtype, file, field, row)


def coerce(values, dtype, *, file, field, row=None, codes=None, missing=None):
    """Return `values` as an array of the registry `dtype`, exactly, or raise LossyConversionError."""
    values = np.asarray(values)
    if dtype in _INTEGERS:
        result = _to_integer(values, dtype, missing, file, field, row)
    elif dtype == "bool":
        result = _to_bool(values, dtype, file, field, row)
    elif dtype in _FLOATS:
        result = _to_float(values, dtype, file, field, row)
    elif dtype == "text":
        result = _to_text(values, dtype, file, field, row)
    else:
        raise ValueError(f"{file}: {field}: cannot coerce to dtype {dtype!r}")
    if codes:
        _check_codes(result, values, codes, missing, dtype, file, field, row)
    return result
