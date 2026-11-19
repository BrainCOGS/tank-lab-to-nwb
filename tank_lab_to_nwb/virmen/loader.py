"""Load a ViRMEn log with mat-io into plain Python values.

Structs become dicts (or lists of dicts for struct arrays), char arrays become strings, cells stay object arrays with
their MATLAB shape, numeric arrays keep their dtype and shape, and MATLAB objects become dicts of their properties.
"""

import warnings
from collections import namedtuple
from pathlib import Path

import matio
import numpy as np
from matio.utils.matclass import MatConvertWarning

Enum = namedtuple("Enum", "name value")
FunctionHandle = namedtuple("FunctionHandle", "name")


class LogLoadError(ValueError):
    """A log can't be read or fully decoded."""


def _enum_integer(member):
    value = member.value
    if isinstance(value, dict):  # mat-io holds a property map, e.g. {'uint32.Data': array([[1]])}
        value = next(iter(value.values()))
    return np.asarray(value).ravel()[0].item()


def _decode_enum(array):
    members = [Enum(member.name, _enum_integer(member)) for member in array.flat]
    if len(members) == 1:
        return members[0]
    cell = np.empty(array.shape, dtype=object)
    for position, member in enumerate(members):
        cell.flat[position] = member
    return cell


def _decode_function(array):
    handle = array.flat[0]["function_handle"].flat[0]
    return FunctionHandle(str(handle["function"].flat[0]))


def _decode_table(frame):
    rows = [{str(column): decode(value) for column, value in row.items()} for row in frame.to_dict("records")]
    return rows[0] if len(rows) == 1 else rows


def decode(value):
    """Turn one mat-io value into plain Python values."""
    kind = type(value).__name__
    if value is None or isinstance(value, (str, dict)):
        return str(value) if isinstance(value, np.str_) else value
    if kind == "EmptyMatStruct":
        return {}
    if kind == "MatlabCanonicalEmpty":
        return np.zeros((0, 0))
    if kind == "MatlabFunction":
        return _decode_function(value)
    if kind == "MatlabEnumerationArray":
        return _decode_enum(value)
    if kind == "MatlabOpaque":
        return {str(name): decode(member) for name, member in value.properties.items()}
    if kind == "DataFrame":
        return _decode_table(value)
    if np.isscalar(value):
        return np.array([[value]])
    if not isinstance(value, np.ndarray):
        raise LogLoadError(f"cannot decode a value of type {kind}")
    if value.dtype.names is not None:
        items = [{name: decode(item[name]) for name in value.dtype.names} for item in value.flat]
        return items[0] if len(items) == 1 else items
    if value.dtype.kind in "US":
        strings = [str(item) for item in value.flat]
        return strings[0] if len(strings) == 1 else ("" if not strings else strings)
    if value.dtype.kind == "O":
        cell = np.empty(value.shape, dtype=object)
        for position, item in enumerate(value.flat):
            cell.flat[position] = decode(item)
        return cell
    return value


def records(value):
    """A struct field as a list of its elements, whether MATLAB stored it as a scalar, an array or []."""
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray) and value.size == 0:
        return []
    raise LogLoadError(f"expected a struct, got {type(value).__name__}")


def load_log(path):
    """Load a log file and return its decoded `log` struct. A failed decode raises instead of dropping data."""
    path = Path(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatConvertWarning)
        try:
            contents = matio.load_from_mat(path)
        except MatConvertWarning as warning:
            raise LogLoadError(f"{path}: mat-io could not decode an object: {warning}") from warning
    if "log" not in contents:
        raise LogLoadError(f"{path}: no 'log' variable")
    return decode(contents["log"])
