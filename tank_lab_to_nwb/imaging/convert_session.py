"""Convert one imaging session: a ViRMEn log and the TIFFs of one ScanImage acquisition. Nothing is uploaded.

Usage: python -m tank_lab_to_nwb.imaging.convert_session <log.mat> <output folder> <tiff or folder> [...] [--stub FRAMES] [--allow-incomplete]

A folder means its originalStacks when it has them, then TIFFs in the folder itself, then per-ROI split folders. With
--allow-incomplete, TIFFs that can't be shown to be the whole acquisition are converted anyway, and the file says so.

The NWB file goes in <output> with a report beside it. With --stub, each plane keeps only that many
frames, and the file is named *_stub.nwb. Run organize_for_dandi with description="rawtiff", then `dandi validate`, afterwards.
"""

import argparse
import shutil
from pathlib import Path

from .convert import convert_session
from .scanimage import find_tiffs


def _expand(sources):
    return [path for source in map(Path, sources) for path in (find_tiffs(source) if source.is_dir() else [source])]


def convert(log, output, tiffs, stub=None, allow_incomplete=False):
    """Convert, verify and report, and return the path of the file. Refuses to start without room for raw frames at their full size."""
    log, output, tiffs = Path(log), Path(output), _expand(tiffs)
    output.mkdir(parents=True, exist_ok=True)
    source_bytes = sum(Path(tiff).stat().st_size for tiff in tiffs)
    free = shutil.disk_usage(output).free
    if stub is None and free < source_bytes:
        raise OSError(f"{output}: {free / 1e9:.1f} GB free, and these TIFFs hold {source_bytes / 1e9:.1f} GB")
    path = output / f"{log.stem}{'_stub' if stub else ''}.nwb"
    convert_session(log, tiffs, path, max_frames_per_plane=stub, allow_incomplete=allow_incomplete)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("tiffs", type=Path, nargs="+")
    parser.add_argument("--stub", type=int, help="Keep only this many frames per plane.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Convert TIFFs that can't be shown to be the whole acquisition, and record that in the file.")
    arguments = parser.parse_args()
    print(convert(arguments.log, arguments.output, arguments.tiffs, stub=arguments.stub, allow_incomplete=arguments.allow_incomplete))


if __name__ == "__main__":
    main()
