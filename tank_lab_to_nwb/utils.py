"""Authors: Ben Dichter, Cody Baker."""

import os
import sys
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from shutil import which

import numpy as np
from pynwb import NWBFile
from scipy.io import loadmat, matlab

try:
    from typing import ArrayLike
except ImportError:
    from typing import Sequence, Union

    from numpy import ndarray

    # adapted from numpy typing
    ArrayLike = Union[bool, int, float, complex, list, ndarray, Sequence]


def check_module(nwbfile: NWBFile, name, description=None):
    """
    Check if processing module exists. If not, create it. Then return module.

    Parameters
    ----------
    nwbfile: pynwb.NWBFile
    name: str
    description: str | None (optional)

    Returns
    -------
    pynwb.processing
    """
    if name in nwbfile.processing:
        return nwbfile.processing[name]
    else:
        if description is None:
            description = name
        return nwbfile.create_processing_module(name, description)


def find_discontinuities(tt, factor=10000):
    """Find discontinuities in a timeseries. Returns the indices before each discontinuity."""
    dt = np.diff(tt)
    before_jumps = np.where(dt > np.median(dt) * factor)[0]

    if len(before_jumps):
        out = np.array([tt[0], tt[before_jumps[0]]])
        for i, j in zip(before_jumps, before_jumps[1:]):
            out = np.vstack((out, [tt[i + 1], tt[j]]))
        out = np.vstack((out, [tt[before_jumps[-1] + 1], tt[-1]]))
        return out
    else:
        return np.array([[tt[0], tt[-1]]])


def mat_obj_to_dict(mat_struct):
    """Recursive function to convert nested matlab struct objects to dictionaries.
    
    Handles special cases like shapingProtocol and other mat_struct objects that don't 
    convert cleanly by extracting string representations of function names.
    """
    dict_from_struct = {}
    for field_name in mat_struct.__dict__["_fieldnames"]:
        dict_from_struct[field_name] = mat_struct.__dict__[field_name]
        if isinstance(dict_from_struct[field_name], matlab.mio5_params.mat_struct):
            # Try to convert nested mat_struct objects; if it fails, store as string representation
            try:
                dict_from_struct[field_name] = mat_obj_to_dict(dict_from_struct[field_name])
            except (AttributeError, KeyError):
                # For protocol functions and other unconvertible objects, extract string representation
                dict_from_struct[field_name] = str(dict_from_struct[field_name])
        elif isinstance(dict_from_struct[field_name], matlab.MatlabFunction):
            dict_from_struct[field_name] = str(dict_from_struct[field_name])
        elif isinstance(dict_from_struct[field_name], np.ndarray):
            try:
                dict_from_struct[field_name] = mat_obj_to_array(dict_from_struct[field_name])
            except TypeError:
                continue
    return dict_from_struct


def mat_obj_to_array(mat_struct_array):
    """Construct array from matlab cell arrays.
    Recursively converts array elements if they contain mat objects."""
    if has_struct(mat_struct_array):
        array_from_cell = [mat_obj_to_dict(mat_struct) for mat_struct in mat_struct_array]
        array_from_cell = np.array(array_from_cell)
    else:
        array_from_cell = mat_struct_array

    return array_from_cell


def has_struct(mat_struct_array):
    """Determines if a matlab cell array contains any mat objects."""
    return any(isinstance(mat_struct, matlab.mio5_params.mat_struct) for mat_struct in mat_struct_array)


def convert_mat_file_to_dict(mat_file_name):
    """
    Convert mat-file to dictionary object.

    It calls a recursive function to convert all entries
    that are still matlab objects to dictionaries.
    """
    data = loadmat(mat_file_name, struct_as_record=False, squeeze_me=True)
    for key in data:
        if isinstance(data[key], matlab.mio5_params.mat_struct):
            data[key] = mat_obj_to_dict(data[key])
    return data


def array_to_dt(array):
    """Convert array of floats to datetime object."""
    dt_input = [int(x) for x in array]
    dt_input.append(round(np.mod(array[-1], 1) * 10**6))
    return datetime(*dt_input)


def create_indexed_array(ndarray):
    """Creates an indexed array from an irregular array of arrays.
    Returns the flat array and its indices."""
    flat_array = []
    array_indices = []
    for array in ndarray:
        if isinstance(array, Iterable):
            flat_array.extend(array)
            array_indices.append(len(array))
        else:
            flat_array.append(array)
            array_indices.append(1)
    array_indices = np.cumsum(array_indices, dtype=np.uint64)

    return flat_array, array_indices


def create_and_store_indexed_array(ndarray, array_name, description, nwbfile: NWBFile):
    array_data, array_indices = create_indexed_array(ndarray)

    nwbfile.add_trial_column(name=array_name, description=description, index=array_indices, data=array_data)


def flatten_nested_dict(nested_dict):
    """Recursively flattens a nested dictionary."""
    flatten_dict = {}
    for k, v in nested_dict.items():
        if isinstance(v, dict):
            if v:
                flatten_sub_dict = flatten_nested_dict(v).items()
                flatten_dict.update(dict(flatten_sub_dict))
            else:
                flatten_dict[k] = np.array([])
        else:
            flatten_dict[k] = v

    return flatten_dict




def fetch_coowners(subject_fullname: str):
    if subject_fullname is None:
        return None
    return (subject.SubjectCoowners() & f"subject_fullname = '{subject_fullname}' and active = 1").fetch("coowner")


def fetch_owner(subject_fullname: str):
    if subject_fullname is None:
        return None
    return (subject.Subject() & f"subject_fullname = '{subject_fullname}'").fetch()
