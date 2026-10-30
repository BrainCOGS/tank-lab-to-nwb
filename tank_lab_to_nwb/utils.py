"""Authors: Ben Dichter, Cody Baker."""

import warnings
import sys
import subprocess
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from shutil import which
from tempfile import TemporaryDirectory

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


def convert_function_handle_to_str(mat_file_path):
    """Executes a matlab script which converts function handle values to str
    if matlab is installed on the system."""
    matlab_class = """
    classdef Choice < uint32

        enumeration
            L(1)
            R(2)
            nil(inf)
        end

        methods (Static)
            function choices = all()
                choices = enumeration('Choice')';
                choices = choices(1:end-1);
            end

            function num = count()
                num = numel(enumeration('Choice'));
            end
        end

        methods
            function opp = opposite(obj)
                numValues   = numel(Choice.all());
                assert(numValues == 2);     % the concept of "opposite" only works for sets of 2

                flipped     = double(obj);
                flipped     = numValues+1 - flipped;
                opp         = obj;
                sel         = opp >= 1 & opp <= numValues;
                opp(sel)    = flipped(sel);
            end
        end

    end
    """
    matlab_code = r"""
    str_func = char(log.version.code);
    code_version = 'code_version.txt';
    fid = fopen(code_version, 'wt');
    fprintf(fid, str_func);
    fclose(fid);

    str_func = char(log.animal.protocol);
    protocol = 'protocol.txt';
    fid = fopen(protocol, 'wt');
    fprintf(fid, str_func);
    fclose(fid);

    choice_data = [];
    for i = 1 : size(log.block, 2)
        for j = 1 : size(log.block(i).trial, 2)
            choice_data = [choice_data; string(log.block(i).trial(j).choice)];
        end
    end

    choice = 'trial_choice.txt';
    fid = fopen(choice, 'wt');
    fprintf(fid,'%s\n', choice_data);
    fclose(fid);

    trial_type_data = [];
    for i = 1 : size(log.block, 2)
        for j = 1 : size(log.block(i).trial, 2)
            trial_type_data = [trial_type_data; string(log.block(i).trial(j).trialType)];
        end
    end

    trial_type = 'trial_type.txt';
    fid = fopen(trial_type, 'wt');
    fprintf(fid,'%s\n', trial_type_data);
    fclose(fid);

    quit;
    """

    metadata = {}

    if which("matlab") is None:
        # Every value this function produces (experiment_name, protocol_name,
        # trial_choice, trial_type) is optional to its caller, which already
        # falls back to "" for each. Raising here took down the entire
        # conversion on any host without MATLAB, including headless export
        # servers that have no reason to have it installed.
        warnings.warn(
            "MATLAB was not found on PATH. Code version, animal protocol, trial type "
            "and choice will be omitted from the NWB file; everything else converts "
            "normally. Install MATLAB on this host if you need those four fields."
        )
        return metadata

    # Everything below runs inside a scratch directory. It used to run in the
    # process's cwd, writing (and then unlinking) Choice.m,
    # convert_function_to_txt.m and four .txt files there. That litters whatever
    # directory the conversion was launched from, and silently deletes any file
    # already sitting there under one of those six fairly ordinary names.
    with TemporaryDirectory(prefix="tank_lab_to_nwb_matlab_") as scratch_dir:
        scratch = Path(scratch_dir)
        (scratch / "Choice.m").write_text(matlab_class)

        convert_script_path = scratch / "convert_function_to_txt.m"
        convert_script_path.write_text(f"filePath = '{mat_file_path}';\nload(filePath);{matlab_code}")

        if "win" in sys.platform and sys.platform != "darwin":
            matlab_argv = ["matlab", "-nosplash", "-wait", "-log", "-r", "convert_function_to_txt"]
        else:
            matlab_argv = ["matlab", "-nosplash", "-nodisplay", "-log", "-batch", "convert_function_to_txt"]

        try:
            # cwd=scratch is what puts the generated .m files on MATLAB's path
            # and keeps its output files out of the caller's directory.
            subprocess.run(matlab_argv, cwd=scratch, check=True)

            outputs = {
                "experiment_name": ("code_version.txt", "line"),
                "protocol_name": ("protocol.txt", "line"),
                "trial_choice": ("trial_choice.txt", "lines"),
                "trial_type": ("trial_type.txt", "lines"),
            }
            for key, (filename, mode) in outputs.items():
                text = (scratch / filename).read_text()
                metadata[key] = text.splitlines()[0] if mode == "line" else text.splitlines()

        except (subprocess.SubprocessError, OSError) as e:
            warnings.warn(
                f"There was an error while trying to execute {convert_script_path}: {e}. "
                f"Code version, animal protocol, trial type and choice will be omitted."
            )

    return metadata


def fetch_coowners(subject_fullname: str):
    if subject_fullname is None:
        return None
    return (subject.SubjectCoowners() & f"subject_fullname = '{subject_fullname}' and active = 1").fetch("coowner")


def fetch_owner(subject_fullname: str):
    if subject_fullname is None:
        return None
    return (subject.Subject() & f"subject_fullname = '{subject_fullname}'").fetch()
