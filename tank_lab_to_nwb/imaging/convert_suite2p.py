"""Convert a trusted Suite2p run with TIFF/behavior timing. Nothing here uploads.

python -m tank_lab_to_nwb.imaging.convert_suite2p LOG SUITE2P OUTPUT TIFF [TIFF ...] [--extra-suite2p DIR ...] [--raw-nwb RAW.nwb] [--allow-incomplete]

SUITE2P is a Suite2p output folder, the one holding plane0, plane1... A multi-ROI session run the lab's way, once per split
ROI folder, has one such folder per run: give the first as SUITE2P and each other with --extra-suite2p. A TIFF argument may be
a folder: its originalStacks when it has them, then TIFFs in the folder itself, then per-ROI split folders.

OUTPUT/staging/suite2p.nwb and suite2p.report.json are created. With --raw-nwb, the raw TIFF file of this acquisition is
checked and recorded. Name it with virmen.layout.organize_for_dandi(staging, dandiset, description="suite2p"), which can
share a dandiset with the raw file.
"""

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

from .convert import conversion_report
from .convert_session import _expand
from .suite2p import _relative, _sources, convert_suite2p, match_planes, source_base
from .suite2p_artifacts import artifact_inventory, verify_combined


def convert(log, root, output, tiffs, raw_nwb=None, extra=(), allow_incomplete=False):
    """Write and verify, then report source inventory and measured alignment beside NWB."""
    roots = [Path(root), *map(Path, extra)]
    output, tiffs = Path(output), _expand(tiffs)
    staging = output / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    source_bytes = sum(p.stat().st_size for p in _sources(roots))
    free = shutil.disk_usage(output).free
    if free < source_bytes * 3:
        raise OSError(f"Need {source_bytes * 3} bytes of free disk; found {free}")
    path = staging / "suite2p.nwb"
    base = source_base(roots)
    combined = {_relative(folder, base) if len(roots) > 1 else ".": verify_combined(folder) for folder in roots}
    prepared = convert_suite2p(log, tiffs, roots if len(roots) > 1 else roots[0], path, raw_nwb=raw_nwb, allow_incomplete=allow_incomplete)
    with h5py.File(path) as f:
        evidence = json.loads(f["scratch/suite2p_frame_mapping"].asstr()[()])
        table = f["processing/ophys/scanimage_alignment"]
        rows = dict(zip(table["key"].asstr()[()], table["value"].asstr()[()]))
        companion = {key.removeprefix("companion.rawtiff."): value for key, value in rows.items() if key.startswith("companion.rawtiff.")} or None
    planes = match_planes(roots, prepared.acquisition)
    report = dict(
        **conversion_report(prepared),
        companion=companion,
        combined=combined if len(roots) > 1 else combined["."],
        artifacts=artifact_inventory(roots),
        fields={p.field.label: dict(folder=_relative(p.folder, base), plane=p.field.plane, roi=p.field.number, roi_name=p.field.name, input=p.input) for p in planes},
        mapping=evidence,
        roi_counts={p.field.label: len(np.load(p.folder / "iscell.npy")) for p in planes},
        frame_counts={p.field.label: int(p.field.frame_indices.size) for p in planes},
        output_bytes=path.stat().st_size,
        verification="Exact behavior, headers, timestamps, traces, masks, stat values, images, registration series and archived artifact bytes checked by independent HDF5 read path.",
    )
    path.with_suffix(".report.json").write_text(json.dumps(report, indent=1))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path)
    parser.add_argument("suite2p", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("tiffs", type=Path, nargs="+")
    parser.add_argument("--extra-suite2p", type=Path, action="append", default=[], help="Another Suite2p output folder of this acquisition, such as one per split ROI folder.")
    parser.add_argument("--raw-nwb", type=Path, help="The raw TIFF NWB file of this acquisition, to check and record.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Convert TIFFs that can't be shown to be the whole acquisition, such as split copies without ScanImage settings, and record that.")
    args = parser.parse_args()
    print(convert(args.log, args.suite2p, args.output, args.tiffs, raw_nwb=args.raw_nwb, extra=args.extra_suite2p, allow_incomplete=args.allow_incomplete))


if __name__ == "__main__":
    main()
