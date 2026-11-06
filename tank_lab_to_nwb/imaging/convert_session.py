"""Convert one imaging session: a ViRMEn log and the TIFFs of one ScanImage acquisition. Nothing is uploaded.

Usage: python -m tank_lab_to_nwb.imaging.convert_session <log.mat> <output folder> <tiff> [<tiff> ...] [--stub FRAMES]

The NWB file goes to <output>/staging with an alignment report beside it. With --stub, each plane keeps only that many
frames, and the file is named *_stub.nwb. Run organize_for_dandi, then `dandi validate`, on the staging folder afterwards.
"""

import argparse
import json
import shutil
from pathlib import Path

from .convert import alignment_summary, convert_session


def convert(log, output, tiffs, stub=None):
    """Convert, verify and report, and return the path of the file. Refuses to start without room for raw frames at their full size."""
    log, output = Path(log), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_bytes = sum(Path(tiff).stat().st_size for tiff in tiffs)
    free = shutil.disk_usage(output).free
    if stub is None and free < source_bytes:
        raise OSError(f"{output}: {free / 1e9:.1f} GB free, and these TIFFs hold {source_bytes / 1e9:.1f} GB")
    path = output / f"{log.stem}{'_stub' if stub else ''}.nwb"
    prepared = convert_session(log, tiffs, path, max_frames_per_plane=stub)
    report = dict(alignment=alignment_summary(prepared), coverage=prepared.session.coverage.as_dict())
    report["coverage"]["dropped"].pop("log.currentTrial", None)
    report["coverage"]["dropped"].pop("log.currentIt", None)
    path.with_suffix(".report.json").write_text(json.dumps(report, indent=1))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("tiffs", type=Path, nargs="+")
    parser.add_argument("--stub", type=int, help="Keep only this many frames per plane.")
    arguments = parser.parse_args()
    print(convert(arguments.log, arguments.output, arguments.tiffs, stub=arguments.stub))


if __name__ == "__main__":
    main()
