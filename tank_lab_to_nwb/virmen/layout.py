"""Lay out NWB files for DANDI with dandi-cli's organize, locally. Nothing here uploads.

This follows neuroconv's automatic_dandi_upload: organize names the files from their subject_id and session_id, then
the session is added to any name that lacks it, because organize drops it for a subject with a single session.
"""

import re
import shutil
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


def organize_for_dandi(staging_dir, dandiset_dir, files_mode="copy", description=None):
    """Organize the NWB files of a flat folder into `dandiset_dir` and return the organized paths.

    `description` tells apart files that would otherwise share a name, such as a session's raw and processed files. Each
    view has its own staging folder, so organize never sees them together and adds no description of its own.

    organize runs in a fresh folder beside `dandiset_dir`, because it renames files to avoid any it finds already there.
    Each result then replaces the file of the same name, so converting again doesn't leave duplicates.
    """
    if description is not None:
        _check_description(description)
    staging_dir, dandiset_dir = Path(staging_dir), Path(dandiset_dir)
    if not list(staging_dir.glob("*.nwb")):
        raise ValueError(f"{staging_dir}: no NWB files to organize")
    dandiset_dir.mkdir(parents=True, exist_ok=True)
    write_stub_dandiset_yaml(dandiset_dir)
    organized = []
    with TemporaryDirectory(dir=dandiset_dir.parent, prefix=".organizing-") as fresh:
        fresh = Path(fresh)
        shutil.copy(dandiset_dir / "dandiset.yaml", fresh / "dandiset.yaml")
        organize(paths=[str(staging_dir)], dandiset_path=str(fresh), files_mode=FileOperationMode(files_mode))
        for path in sorted(fresh.rglob("*.nwb")):
            name = insert_session_id(path.name, _read_session_id(path))
            if description is not None:
                name = insert_description(name, description)
            target = dandiset_dir / path.parent.relative_to(fresh) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            path.replace(target)
            organized.append(target)
    return organized
