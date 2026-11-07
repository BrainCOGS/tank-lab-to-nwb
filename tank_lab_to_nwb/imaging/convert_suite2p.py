"""Convert a trusted Suite2p run with TIFF/behavior timing. Nothing here uploads.

python -m tank_lab_to_nwb.imaging.convert_suite2p LOG SUITE2P OUTPUT TIFF [TIFF ...] [--raw-nwb RAW.nwb]
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
from .suite2p import _sources, convert_suite2p
from .suite2p_artifacts import artifact_inventory, verify_combined


def convert(log, root, output, tiffs, raw_nwb=None):
    """Write and verify, then report source inventory and measured alignment beside NWB."""
    root, output = Path(root), Path(output)
    staging = output / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    source_bytes = sum(p.stat().st_size for p in _sources(root))
    free = shutil.disk_usage(output).free
    if free < source_bytes * 3:
        raise OSError(f"Need {source_bytes * 3} bytes of free disk; found {free}")
    path = staging / "suite2p.nwb"
    combined = verify_combined(root)
    prepared = convert_suite2p(log, tiffs, root, path, raw_nwb=raw_nwb)
    with h5py.File(path) as f:
        evidence = json.loads(f["scratch/suite2p_frame_mapping"].asstr()[()])
        table = f["processing/ophys/scanimage_alignment"]
        rows = dict(zip(table["key"].asstr()[()], table["value"].asstr()[()]))
        companion = {key.removeprefix("companion.rawtiff."): value for key, value in rows.items() if key.startswith("companion.rawtiff.")} or None
    report = dict(
        **conversion_report(prepared),
        companion=companion,
        combined=combined,
        artifacts=artifact_inventory(root),
        mapping=evidence,
        roi_counts=[len(np.load(p / "iscell.npy")) for p in sorted(root.glob("plane*"))],
        frame_counts=[len(prepared.acquisition.plane_frames(p)) for p in range(prepared.acquisition.layout.n_planes)],
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
    parser.add_argument("--raw-nwb", type=Path, help="The raw TIFF NWB file of this acquisition, to check and record.")
    args = parser.parse_args()
    print(convert(args.log, args.suite2p, args.output, args.tiffs, raw_nwb=args.raw_nwb))


if __name__ == "__main__":
    main()
