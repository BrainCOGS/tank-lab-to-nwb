"""Convert one ephys session: a ViRMEn log, the raw SpikeGLX folder for the clock, and one sorter folder per probe. No DB.

Usage: python -m tank_lab_to_nwb.ephys.convert_session <log.mat> <raw run folder> <output folder> imec0=<sorter folder> [imec1=<folder> ...] [--date YYYY-MM-DD]

A sorter folder is the one holding spike_times.npy (`kilosort_output`, `kilosort4_output`, ...); a job folder holding
exactly one such folder is accepted too. Only the raw folder's sync file and short windows of each AP file's SY channel
are read, never the AP samples. The NWB file goes in <output> with a report beside it. Run organize_for_dandi with
description="units", then `dandi validate`, afterwards.
"""

import argparse
import json
from pathlib import Path

from .convert import conversion_report, write_ephys_nwbfile
from .session import build_session_sync
from .sorting import SortingError, find_sorter_folders
from .verify import verify_ephys


def resolve_sorter_folder(path):
    path = Path(path)
    if (path / "spike_times.npy").is_file():
        return path
    found = find_sorter_folders(path)
    if len(found) > 1:
        raise SortingError(f"{path}: holds several sorter folders ({', '.join(p.name for p in found)}); name one")
    return found[0]


def convert(log, raw_folder, output, sorter_folders, session_date=None, metadata=None, name=None):
    """Convert, verify and report, and return the path of the file."""
    log, output = Path(log), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sorter_folders = {probe: resolve_sorter_folder(folder) for probe, folder in sorter_folders.items()}
    sync = build_session_sync(raw_folder, log, probes=sorted(sorter_folders), session_date=session_date)
    path = output / f"{name or log.stem}_units.nwb"
    prepared = write_ephys_nwbfile(sync, sorter_folders, path, metadata)
    checked = verify_ephys(sync, prepared, path)
    report = conversion_report(sync, prepared)
    report["verified"] = checked
    path.with_suffix(".report.json").write_text(json.dumps(report, indent=1, default=str))
    return path


def _probe_folder(text):
    probe, equals, folder = text.partition("=")
    if not equals or not probe.startswith("imec"):
        raise argparse.ArgumentTypeError(f"{text!r}: expected imec<N>=<sorter folder>")
    return probe, Path(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path)
    parser.add_argument("raw_folder", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("sorted", type=_probe_folder, nargs="+", help="imec<N>=<sorter folder>, one per probe.")
    parser.add_argument("--date", help="Recording date, YYYY-MM-DD; sessions before 2021-06-01 are refused.")
    parser.add_argument("--name", help="Stem of the output file; default the log's.")
    arguments = parser.parse_args()
    print(convert(arguments.log, arguments.raw_folder, arguments.output, dict(arguments.sorted), session_date=arguments.date, name=arguments.name))


if __name__ == "__main__":
    main()
