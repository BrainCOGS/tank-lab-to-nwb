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


def is_separated(errors, limit):
    """Whether the match in errors["0"] is within `limit` and every neighbor is more than ten times worse."""
    return errors["0"] <= limit and all(value > 10 * max(errors["0"], 0.01) for key, value in errors.items() if key != "0")


def predict_traces(frame, stat, allow_overlap):
    """What Suite2p's extraction gives for each ROI on one registered frame: normalized weights over its pixels."""
    traces = np.zeros(len(stat), dtype="float32")
    for index, roi in enumerate(stat):
        keep = np.ones(len(roi["xpix"]), dtype=bool) if allow_overlap or "overlap" not in roi else ~np.asarray(roi["overlap"], dtype=bool)
        weights = np.asarray(roi["lam"], dtype="float32")[keep]
        if weights.size:
            traces[index] = np.asarray(frame[roi["ypix"][keep], roi["xpix"][keep]], dtype="float32") @ (weights / weights.sum())
    return traces


def _trace_errors(movie, trace_at_frame, stat, allow_overlap, frame):
    """RMSE of the saved traces at this frame against those predicted from it and its neighbors, or None if none has a value."""
    finite = np.isfinite(trace_at_frame)
    if not finite.any():
        return None
    errors = {}
    for delta in (-1, 0, 1):
        if 0 <= frame + delta < len(movie):
            predicted = predict_traces(movie[frame + delta], stat, allow_overlap)
            errors[str(delta)] = float(np.sqrt(np.mean((predicted[finite].astype("float64") - trace_at_frame[finite]) ** 2)))
    return errors


def prove_frame_mapping(acquisition, root):
    """Check each plane's first, middle and last pages of every file, with ±1 controls, in two links.

    TIFF page to registered frame: reconstruction RMSE at most 2 intensity units. Registered frame to F.npy: traces
    recomputed from the frame within 0.01. In both, each neighboring frame must be more than ten times worse, or the
    conversion is refused. Returns per-sample evidence. Fneu and spks aren't recomputed; they share F's order and length.
    This is sampled evidence across every file, not an assertion that every pixel was checked.
    """
    from pathlib import Path
    from .scanimage import read_frames

    root = Path(root)
    evidence = []
    for plane in range(acquisition.layout.n_planes):
        folder = root / f"plane{plane}"
        reg = np.load(folder / "reg_outputs.npy", allow_pickle=True).item()
        all_settings = np.load(folder / "settings.npy", allow_pickle=True).item()
        settings = all_settings["registration"]
        allow_overlap = all_settings.get("extraction", {}).get("allow_overlap", False)
        stat = np.load(folder / "stat.npy", allow_pickle=True)
        traces = np.load(folder / "F.npy", mmap_mode="r")
        indices = acquisition.plane_frames(plane)
        movie = np.memmap(
            folder / "data.bin", mode="r", dtype="int16", shape=(len(indices), *acquisition.layout.frame_shape)
        )
        if (folder / "data.bin").stat().st_size != movie.nbytes:
            raise ValueError(f"plane {plane}: registered movie frame count differs")
        positions = []
        for file in range(len(acquisition.files)):
            inside = np.flatnonzero(acquisition.frames["file_index"][indices] == file)
            if inside.size:  # a recording can stop before reaching this plane in the last file
                positions.extend(np.linspace(inside[0], inside[-1], 3, dtype=int))
        for i in sorted(set(positions)):
            errors = {}
            for delta in (-1, 0, 1):
                if not 0 <= i + delta < len(indices):
                    continue
                raw = read_frames(acquisition, [indices[i + delta]])[0]
                predicted = reconstruct_frame(raw, reg, settings, i)
                diff = predicted.astype("float64") - movie[i]
                errors[str(delta)] = float(np.sqrt(np.mean(diff**2)))
            if not is_separated(errors, limit=2):
                raise ValueError(f"plane {plane}, frame {i}: frame mapping not demonstrated: {errors}")
            trace_errors = _trace_errors(movie, np.asarray(traces[:, i]), stat, allow_overlap, i) if len(stat) else None
            if trace_errors is not None and not is_separated(trace_errors, limit=0.01):
                raise ValueError(f"plane {plane}, frame {i}: traces don't match the registered movie: {trace_errors}")
            evidence.append(
                dict(
                    plane=plane,
                    frame=int(i),
                    tiff_frame=int(indices[i]),
                    file_index=int(acquisition.frames["file_index"][indices[i]]),
                    page_index=int(acquisition.frames["page_index"][indices[i]]),
                    rmse=errors,
                    trace_rmse=trace_errors,
                )
            )
    return evidence
