"""What a Suite2p output folder holds: every artifact's fields, and that combined/ adds nothing beyond the planes."""

from pathlib import Path

import numpy as np

from .suite2p import _plane_folders, _relative, _roots, _sources, read_plane, source_base


def artifact_inventory(root):
    """List every artifact and every dictionary field, without changing their values. `root` is one Suite2p folder or several."""

    def describe(value):
        if isinstance(value, dict):
            return {str(k): describe(v) for k, v in value.items()}
        if isinstance(value, np.ndarray):
            return dict(shape=list(value.shape), dtype=str(value.dtype))
        return type(value).__name__

    base = source_base(root)
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
        result[_relative(path, base)] = entry
    for path in sorted(movie for folder in _roots(root) for movie in folder.rglob("data.bin")):
        result[_relative(path, base)] = dict(
            bytes=path.stat().st_size,
            storage="excluded: registered movie stays external; sampled reconstruction evidence stored in NWB",
        )
    return result


def _tile_offsets(dbs):
    """(dy, dx) of each plane folder in combined, as Suite2p 1.1's compute_dydx places them: a near-square grid of planes,
    or, when the dbs carry mesoscope dy/dx, those offsets, with planes tiled in a grid of the ROIs' extent."""
    if any(db.get("dx") is None or db.get("dy") is None for db in dbs):
        ncols = int(np.ceil(np.sqrt(dbs[0]["Ly"] * dbs[0]["Lx"] * len(dbs)) / dbs[0]["Lx"]))
        return [((j // ncols) * dbs[0]["Ly"], (j % ncols) * dbs[0]["Lx"]) for j in range(len(dbs))]
    dy = np.array([int(db["dy"]) for db in dbs])
    dx = np.array([int(db["dx"]) for db in dbs])
    nrois = np.unique(np.vstack((dy, dx)), axis=1).shape[1]
    if nrois < len(dbs):
        nplanes = len(dbs) // nrois
        ymax = max(y + db["Ly"] for y, db in zip(dy, dbs))
        xmax = max(x + db["Lx"] for x, db in zip(dx, dbs))
        ncols = int(np.ceil(np.sqrt(ymax * xmax * nplanes) / xmax))
        for j in range(nplanes):
            for k in range(nrois):
                dx[j * nrois + k] += (j % ncols) * xmax
                dy[j * nrois + k] += (j // ncols) * ymax
    return [(int(y), int(x)) for y, x in zip(dy, dx)]


def verify_combined(root):
    """Check that combined adds only tiled coordinates/images and zero-padded traces.

    Every artifact is archived even if a new upstream metadata key is not recognized.
    """
    root = Path(root)
    combined = root / "combined"
    if not combined.exists():
        return {"present": False}
    folders = [folder for _, folder in sorted(_plane_folders(root).items())]
    dbs = []
    for folder in folders:
        meta = read_plane(folder)
        if meta["layout"] != "1.x":
            return {"present": True, "checked": False, "reason": f"{meta['layout']} Suite2p combined/ isn't checked; it's archived byte-exact and never supplies traces"}
        dbs.append({**meta["db"], **meta["reg_outputs"], **meta["detect_outputs"]})
    offsets = _tile_offsets(dbs)
    canvas = (max(dy + db["Ly"] for (dy, _), db in zip(offsets, dbs)), max(dx + db["Lx"] for (_, dx), db in zip(offsets, dbs)))
    cursor = 0
    stat = np.load(combined / "stat.npy", allow_pickle=True)
    cells = np.load(combined / "iscell.npy")
    matrices = {key: np.load(combined / f"{key}.npy", mmap_mode="r") for key in ("F", "Fneu", "spks")}
    image_keys = ("meanImg", "meanImgE", "Vcorr", "max_proj")
    tiled = {key: np.zeros(canvas) for key in image_keys if key in dbs[0]}

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

    for plane, (folder, db, (dy, dx)) in enumerate(zip(folders, dbs, offsets)):
        rows, cols = db["Ly"], db["Lx"]
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
