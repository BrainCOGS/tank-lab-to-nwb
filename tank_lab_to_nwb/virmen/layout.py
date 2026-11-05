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


def _read_session_id(path):
    with NWBHDF5IO(str(path), mode="r", load_namespaces=True) as io:
        return io.read().session_id


def organize_for_dandi(staging_dir, dandiset_dir, files_mode="copy"):
    """Organize the NWB files of a flat folder into `dandiset_dir` and return the organized paths.

    organize runs in a fresh folder beside `dandiset_dir`, because it renames files to avoid any it finds already there.
    Each result then replaces the file of the same name, so converting again doesn't leave duplicates.
    """
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
            target = dandiset_dir / path.parent.relative_to(fresh) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            path.replace(target)
            organized.append(target)
    return organized
