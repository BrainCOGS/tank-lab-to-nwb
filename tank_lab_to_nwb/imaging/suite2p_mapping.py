"""Independent NumPy reconstruction of Suite2p 1.1 rigid/nonrigid registration.

This reproduces the saved transform, not shift estimation. Numerical differences from
CUDA interpolation are measured, never treated as permission for trace round-trip loss.
"""

import numpy as np
from scipy.ndimage import map_coordinates


def reconstruct_frame(raw, registration, settings, index):
    """Apply saved shifts to a TIFF page; Suite2p uint16 divides by two, int16 does not."""
    if settings.get("do_bidiphase") or settings.get("bidiphase", 0) or settings.get("two_step_registration"):
        raise ValueError("Mapping reconstruction supports no bidiphase correction and one registration pass only")
    if raw.dtype == np.uint16:
        raw = (raw // 2).astype("int16")
    elif raw.dtype != np.int16:
        raise ValueError(f"Unsupported source dtype: {raw.dtype}")
    frame = np.roll(raw, (-int(registration["yoff"][index]), -int(registration["xoff"][index])), axis=(0, 1))
    if not settings.get("nonrigid", False):
        return frame
    rows, cols = raw.shape
    by, bx = settings["block_size"]
    ny = 1 if by >= rows else int(np.ceil(1.5 * rows / by))
    nx = 1 if bx >= cols else int(np.ceil(1.5 * cols / bx))
    ys = np.linspace(0, rows - min(rows, by), ny).astype(int) + min(rows, by) // 2
    xs = np.linspace(0, cols - min(cols, bx), nx).astype(int) + min(cols, bx) // 2
    if ny == 1 or nx == 1:
        raise ValueError("Mapping reconstruction needs at least two nonrigid blocks on each axis")
    # Suite2p interpolates to (last_center-first_center), then border-pads.
    gy = np.linspace(0, ny - 1, int(ys[-1] - ys[0]), dtype="float32")
    gx = np.linspace(0, nx - 1, int(xs[-1] - xs[0]), dtype="float32")
    grid = np.meshgrid(gy, gx, indexing="ij")
    shifts = []
    for key in ("yoff1", "xoff1"):
        block = registration[key][index].reshape(ny, nx)
        expanded = map_coordinates(block, grid, order=1, mode="nearest", prefilter=False)
        shifts.append(np.pad(expanded, ((ys[0], rows - ys[-1]), (xs[0], cols - xs[-1])), mode="edge"))
    y, x = np.meshgrid(np.arange(rows, dtype="float32"), np.arange(cols, dtype="float32"), indexing="ij")
    y += shifts[0]
    x += shifts[1]
    # Include grid normalization and inverse used by torch.grid_sample, in float32.
    y = ((y / np.float32(rows - 1) * 2 - 1) + 1) / 2 * np.float32(rows - 1)
    x = ((x / np.float32(cols - 1) * 2 - 1) + 1) / 2 * np.float32(cols - 1)
    interpolated = map_coordinates(frame.astype("float32"), [y, x], order=1, mode="nearest", prefilter=False)
    return interpolated.astype("int16")  # torch.short truncates toward zero


def prove_frame_mapping(acquisition, root, samples_per_file=3):
    """Check each plane's first/last pages of every file and interior samples, with ±1 controls.

    Returns per-sample evidence. Refuses mapping if reconstruction RMSE exceeds 2
    intensity units or either valid neighboring-page control is less than 10x worse.
    This is sampled evidence across every file, not an assertion that every pixel was checked.
    """
    from pathlib import Path
    from .scanimage import read_frames

    root = Path(root)
    evidence = []
    for plane in range(acquisition.layout.n_planes):
        folder = root / f"plane{plane}"
        reg = np.load(folder / "reg_outputs.npy", allow_pickle=True).item()
        settings = np.load(folder / "settings.npy", allow_pickle=True).item()["registration"]
        indices = acquisition.plane_frames(plane)
        movie = np.memmap(
            folder / "data.bin", mode="r", dtype="int16", shape=(len(indices), *acquisition.layout.frame_shape)
        )
        if (folder / "data.bin").stat().st_size != movie.nbytes:
            raise ValueError(f"plane {plane}: registered movie frame count differs")
        positions = []
        for file in range(len(acquisition.files)):
            inside = np.flatnonzero(acquisition.frames["file_index"][indices] == file)
            positions.extend(np.linspace(inside[0], inside[-1], max(3, samples_per_file), dtype=int))
        for i in sorted(set(positions)):
            errors = {}
            for delta in (-1, 0, 1):
                if not 0 <= i + delta < len(indices):
                    continue
                raw = read_frames(acquisition, [indices[i + delta]])[0]
                predicted = reconstruct_frame(raw, reg, settings, i)
                diff = predicted.astype("float64") - movie[i]
                errors[str(delta)] = float(np.sqrt(np.mean(diff**2)))
            if errors["0"] > 2 or any(
                value <= 10 * max(errors["0"], 0.01) for key, value in errors.items() if key != "0"
            ):
                raise ValueError(f"plane {plane}, frame {i}: frame mapping not demonstrated: {errors}")
            evidence.append(
                dict(
                    plane=plane,
                    frame=int(i),
                    tiff_frame=int(indices[i]),
                    file_index=int(acquisition.frames["file_index"][indices[i]]),
                    page_index=int(acquisition.frames["page_index"][indices[i]]),
                    rmse=errors,
                )
            )
    return evidence
