"""What a Suite2p output folder holds: every artifact's fields, and that combined/ adds nothing beyond the planes."""

from pathlib import Path

import numpy as np

from .suite2p import _sources


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
        elif path.suffix == ".mat":
            from scipy.io import whosmat

            entry["fields"] = {name: dict(shape=list(shape), matlab_class=kind) for name, shape, kind in whosmat(path)}
        result[str(path.relative_to(root))] = entry
    for path in sorted(Path(root).rglob("data.bin")):
        result[str(path.relative_to(root))] = dict(
            bytes=path.stat().st_size,
            storage="excluded: registered movie stays external; sampled reconstruction evidence stored in NWB",
        )
    return result


def verify_combined(root):
    """Check that combined adds only tiled coordinates/images and zero-padded traces.

    Every artifact is archived even if a new upstream metadata key is not recognized.
    """
    root = Path(root)
    combined = root / "combined"
    if not combined.exists():
        return {"present": False}
    folders = sorted(root.glob("plane*"), key=lambda p: int(p.name[5:]))
    dbs = []
    for folder in folders:
        db = np.load(folder / "db.npy", allow_pickle=True).item()
        for source in ("reg_outputs", "detect_outputs"):
            db = {**db, **np.load(folder / f"{source}.npy", allow_pickle=True).item()}
        dbs.append(db)
    rows, cols = dbs[0]["Ly"], dbs[0]["Lx"]
    ncols = int(np.ceil(np.sqrt(rows * cols * len(dbs)) / cols))
    cursor = 0
    stat = np.load(combined / "stat.npy", allow_pickle=True)
    cells = np.load(combined / "iscell.npy")
    matrices = {key: np.load(combined / f"{key}.npy", mmap_mode="r") for key in ("F", "Fneu", "spks")}
    image_keys = ("meanImg", "meanImgE", "Vcorr", "max_proj")
    tiled = {
        key: np.zeros((int(np.ceil(len(dbs) / ncols)) * rows, ncols * cols)) for key in image_keys if key in dbs[0]
    }

    def same(a, b, label):
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                raise ValueError(label + " keys differ")
            for key in a:
                same(a[key], b[key], label + "." + key)
        elif a is None or b is None:
            if a is not b:
                raise ValueError(label + " differs")
        else:
            aa, bb = np.asarray(a), np.asarray(b)
            nan = aa.dtype.kind in "fc" and bb.dtype.kind in "fc"
            if aa.shape != bb.shape or not np.array_equal(aa, bb, equal_nan=nan):
                raise ValueError(label + " differs")

    for plane, (folder, db) in enumerate(zip(folders, dbs)):
        dy, dx = (plane // ncols) * rows, (plane % ncols) * cols
        source_stat = np.load(folder / "stat.npy", allow_pickle=True)
        count = len(source_stat)
        same(cells[cursor : cursor + count], np.load(folder / "iscell.npy"), "combined iscell")
        for index, roi in enumerate(source_stat):
            transformed = dict(roi)
            transformed["xpix"] = roi["xpix"] + dx
            transformed["ypix"] = roi["ypix"] + dy
            transformed["med"] = np.asarray(roi["med"]) + np.array([dy, dx])
            transformed["iplane"] = plane
            same(stat[cursor + index], transformed, "combined stat")
        for key, matrix in matrices.items():
            original = np.load(folder / f"{key}.npy", mmap_mode="r")
            same(matrix[cursor : cursor + count, : original.shape[1]], original, "combined " + key)
            if np.any(matrix[cursor : cursor + count, original.shape[1] :] != 0):
                raise ValueError("combined " + key + " padding differs")
        for key, image in tiled.items():
            if key in ("Vcorr", "max_proj"):
                yr, xr = db["yrange"], db["xrange"]
                image[dy + yr[0] : dy + yr[1], dx + xr[0] : dx + xr[1]] = db[key]
            else:
                image[dy : dy + rows, dx : dx + cols] = db[key]
        cursor += count
    if cursor != len(stat) or cells.shape[0] != cursor or any(a.shape[0] != cursor for a in matrices.values()):
        raise ValueError("combined ROI count differs")
    report = {"present": True, "roi_values_match": True, "rois": cursor, "zero_padding_verified": True}
    if (combined / "settings.npy").exists():
        same(
            np.load(combined / "settings.npy", allow_pickle=True).item(),
            np.load(folders[0] / "settings.npy", allow_pickle=True).item(),
            "combined settings",
        )
    if (combined / "db.npy").exists():
        actual = np.load(combined / "db.npy", allow_pickle=True).item()
        expected = {
            **dbs[-1],
            **tiled,
            "Ly": next(iter(tiled.values())).shape[0],
            "Lx": next(iter(tiled.values())).shape[1],
        }
        expected["yrange"] = [0, expected["Ly"]]
        expected["xrange"] = [0, expected["Lx"]]
        if Path(actual["save_path"]).parts[-2:] != combined.parts[-2:]:
            raise ValueError("combined storage path is not the expected combined folder")
        # Suite2p preserves whether its caller supplied a relative or absolute root.
        expected["save_path"] = actual["save_path"]
        same(actual, expected, "combined db")
        report["all_metadata_and_tiled_images_match"] = True
    return report
