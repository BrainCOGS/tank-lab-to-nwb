"""Convert a trusted Suite2p run with TIFF/behavior timing. Nothing here uploads.

python -m tank_lab_to_nwb.imaging.convert_suite2p LOG SUITE2P OUTPUT TIFF [TIFF ...]
OUTPUT/staging/suite2p.nwb and suite2p.report.json are created. Use the existing
virmen.layout.organize_for_dandi on staging and a separate OUTPUT/dandiset.
"""

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

from .convert import alignment_summary
from .suite2p import _sources, convert_suite2p


def artifact_inventory(root):
    """List every artifact and every dictionary field, without changing their values."""

    def describe(value):
        if isinstance(value, dict):
            return {str(k): describe(v) for k, v in value.items()}
        if isinstance(value, np.ndarray):
            return dict(shape=list(value.shape), dtype=str(value.dtype))
        return type(value).__name__

    result = {}
    for path in _sources(root):
        entry = dict(bytes=path.stat().st_size, storage="byte-exact NWB scratch archive")
        if path.suffix == ".npy":
            value = np.load(path, allow_pickle=True)
            if value.shape == ():
                entry["fields"] = describe(value.item())
            elif path.stem == "stat":
                entry["fields"] = {
                    k: describe(next(row[k] for row in value if k in row))
                    for k in sorted(set().union(*(set(row) for row in value)))
                }
            else:
                entry["array"] = describe(value)
        result[str(path.relative_to(root))] = entry
    for path in sorted(Path(root).glob("plane*/data.bin")):
        result[str(path.relative_to(root))] = dict(
            bytes=path.stat().st_size,
            storage="excluded: registered movie stays external; sampled reconstruction evidence stored in NWB",
        )
    return result


def convert(log, root, output, tiffs):
    """Write and verify, then report source inventory and measured alignment beside NWB."""
    root, output = Path(root), Path(output)
    staging = output / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    source_bytes = sum(p.stat().st_size for p in _sources(root))
    free = shutil.disk_usage(output).free
    if free < source_bytes * 3:
        raise OSError(f"Need {source_bytes * 3} bytes of free disk; found {free}")
    path = staging / "suite2p.nwb"
    prepared = convert_suite2p(log, tiffs, root, path)
    with h5py.File(path) as f:
        evidence = json.loads(f["scratch/suite2p_frame_mapping"].asstr()[()])
    report = dict(
        alignment=alignment_summary(prepared),
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
    args = parser.parse_args()
    print(convert(args.log, args.suite2p, args.output, args.tiffs))


if __name__ == "__main__":
    main()
