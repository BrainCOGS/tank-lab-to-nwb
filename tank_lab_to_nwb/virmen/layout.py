"""Lay out NWB files for DANDI with dandi-cli's organize, and check them with `dandi validate`, locally. Nothing here uploads.

This follows neuroconv's automatic_dandi_upload: organize names the files from their subject_id and session_id, then
the session is added to any name that lacks it, because organize drops it for a subject with a single session.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from dandi.organize import FileOperationMode, organize
from pynwb import NWBHDF5IO

_STUB = """\
identifier: DANDI:000000
name: Local stub, not a real dandiset
description: Local stub so dandi organize and dandi validate can run. Nothing here has been uploaded.
license:
- spdx:CC-BY-4.0
contributor:
- schemaKey: Person
  name: Placeholder, Contributor
  roleName:
  - dcite:ContactPerson
  email: placeholder@example.com
"""


def write_stub_dandiset_yaml(dandiset_dir):
    """Create a stub dandiset.yaml, keeping any that already exists."""
    path = Path(dandiset_dir) / "dandiset.yaml"
    if not path.exists():
        path.write_text(_STUB)
    return path


def insert_session_id(filename, session_id):
    """Add ses-<session_id> after the subject in a DANDI filename, unless it already has a session."""
    if not session_id:
        raise ValueError(f"{filename}: session_id is required to name the file")
    if re.search(r"_ses-", filename):
        return filename
    return re.sub(r"^(sub-[^_.]+)", rf"\1_ses-{session_id}", filename)


def _check_description(description):
    if not re.fullmatch(r"[A-Za-z0-9]+", description):
        raise ValueError(f"description {description!r}: a DANDI name allows only letters and digits here")


def insert_description(filename, description):
    """Add desc-<description> after the session, so several files of one session can share a dandiset."""
    _check_description(description)
    if "_desc-" in filename:
        raise ValueError(f"{filename}: already has a description")
    named, count = re.subn(r"(_ses-[^_.]+)", rf"\1_desc-{description}", filename, count=1)
    if not count:
        raise ValueError(f"{filename}: needs a session before a description")
    return named


def _read_session_id(path):
    with NWBHDF5IO(str(path), mode="r", load_namespaces=True) as io:
        return io.read().session_id


class DandiValidationError(ValueError):
    """`dandi validate` found an error in a file, so it would be refused on upload."""


def validate_for_dandi(paths):
    """Run `dandi validate` on files or a dandiset folder, return what it printed, and raise if it reports an error.

    This is dandi's own command, run with this interpreter, so the answer is the one an upload would give. DANDI_NO_ET
    turns off its look-up of the newest NWB Inspector version, which is the command's only use of the network.
    """
    paths = [str(path) for path in paths]
    command = [sys.executable, "-c", "from dandi.cli.command import main; main()", "validate", *paths]
    result = subprocess.run(command, capture_output=True, text=True, env={**os.environ, "DANDI_NO_ET": "1"})
    if result.returncode:
        raise DandiValidationError(f"dandi validate found errors in {', '.join(Path(path).name for path in paths)}:\n{(result.stdout.strip() or result.stderr.strip())}")
    return result.stdout.strip()


def organize_for_dandi(staging_dir, dandiset_dir, files_mode="copy", description=None, validate=False):
    """Organize the NWB files of a flat folder, or one NWB file, into `dandiset_dir` and return the organized paths.

    `description` tells apart files that would otherwise share a name, such as a session's raw and processed files. Each
    view has its own staging folder, so organize never sees them together and adds no description of its own.

    organize runs in a fresh folder beside `dandiset_dir`, because it renames files to avoid any it finds already there.
    Each result then replaces the file of the same name, so converting again doesn't leave duplicates.

    With `validate`, the named files are checked by `dandi validate` in that fresh folder first. If it reports an error,
    DandiValidationError is raised and nothing enters `dandiset_dir`, so a file already there under the same name stays.
    The sources must then be copied, not moved, so a rejected file is still where it was.
    """
    if description is not None:
        _check_description(description)
    if validate and files_mode != "copy":
        raise ValueError(f'files_mode {files_mode!r}: validating needs "copy", so a rejected file stays in staging')
    staging_dir, dandiset_dir = Path(staging_dir), Path(dandiset_dir)
    if not (staging_dir.is_file() or list(staging_dir.glob("*.nwb"))):
        raise ValueError(f"{staging_dir}: no NWB files to organize")
    dandiset_dir.mkdir(parents=True, exist_ok=True)
    write_stub_dandiset_yaml(dandiset_dir)
    organized = []
    with TemporaryDirectory(dir=dandiset_dir.parent, prefix=".organizing-") as fresh:
        fresh = Path(fresh)
        shutil.copy(dandiset_dir / "dandiset.yaml", fresh / "dandiset.yaml")
        organize(paths=[str(staging_dir)], dandiset_path=str(fresh), files_mode=FileOperationMode(files_mode))
        named = []
        for path in sorted(fresh.rglob("*.nwb")):
            name = insert_session_id(path.name, _read_session_id(path))
            if description is not None:
                name = insert_description(name, description)
            named.append(path.rename(path.with_name(name)))
        if validate:
            validate_for_dandi(named)
        for path in named:
            target = dandiset_dir / path.relative_to(fresh)
            target.parent.mkdir(parents=True, exist_ok=True)
            path.replace(target)
            organized.append(target)
    return organized
