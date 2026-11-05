"""Convert a folder of ViRMEn logs and lay them out for DANDI, locally. Nothing is uploaded.

Usage: python -m tank_lab_to_nwb.virmen.convert_folder <logs folder> <output folder> [--only NAME.mat ...]

Logs are found at <user>/<user>_<subject>/<date>_g0/*.mat. NWB files go to <output>/staging, and dandi organize names
them under <output>/dandiset from each file's subject_id and session_id. Run `dandi validate <output>/dandiset` after.
"""

import argparse
from pathlib import Path

from . import convert
from .layout import organize_for_dandi


def convert_folder(source, output, only=None):
    """Convert the logs under `source` into `output`/staging, and return the organized paths under `output`/dandiset."""
    source, output = Path(source), Path(output)
    logs = sorted(source.glob("*/*/*_g0/*.mat"))
    if only:
        logs = [log for log in logs if log.name in only]
    if not logs:
        raise ValueError(f"{source}: no logs to convert")
    staging = output / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    for log in logs:
        convert.convert_log(log, staging / f"{log.stem}.nwb")
    return organize_for_dandi(staging, output / "dandiset", files_mode="copy")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--only", nargs="*", help="Convert only logs with these file names.")
    arguments = parser.parse_args()
    for path in convert_folder(arguments.source, arguments.output, only=arguments.only):
        print(path)


if __name__ == "__main__":
    main()
