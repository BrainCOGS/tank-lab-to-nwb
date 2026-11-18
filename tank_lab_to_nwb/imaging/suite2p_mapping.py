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


def sample_positions(field):
    """The first, middle and last frame of this field in every file holding it, as positions in the field."""
    positions = []
    for file in sorted(set(field.file_index.tolist())):
        inside = np.flatnonzero(field.file_index == file)
        positions.extend(np.linspace(inside[0], inside[-1], 3, dtype=int))
    return sorted(set(int(position) for position in positions))


def suite2p_input(acquisition, plane, positions):
    """The pages Suite2p read for these positions of a field: its rows as stored, or as the lab's splitter copies them,
    with negative pixels set to 0 in uint16."""
    from .scanimage import read_field

    pages = read_field(acquisition, plane.field, positions)
    return np.clip(pages, 0, None).astype("uint16") if plane.split_copy else pages


def prove_frame_mapping(acquisition, roots):
    """Check each field's first, middle and last pages of every file, with ±1 controls, in two links.

    TIFF page to registered frame: reconstruction RMSE at most 2 intensity units. Registered frame to F.npy: traces
    recomputed from the frame within 0.01. In both, each neighboring frame must be more than ten times worse, or the
    conversion is refused. Returns per-sample evidence. Fneu and spks aren't recomputed; they share F's order and length.
    This is sampled evidence across every file, not an assertion that every pixel was checked.

    `roots` is one Suite2p output folder or several; `suite2p.match_planes` says which folder holds which field. A field's
    pages are its rows of the stack, or its split pages, or, when Suite2p read the lab's split copies of the given stack,
    those rows as the splitter copies them.
    """
    from .suite2p import match_planes, source_base

    base = source_base(roots)
    return [sample for plane in match_planes(roots, acquisition) for sample in prove_plane(acquisition, plane, base)]


def prove_plane(acquisition, plane, base):
    """prove_frame_mapping for one Suite2pPlane. `base` is what its folder is named relative to."""
    from .suite2p import _relative

    field, folder, meta = plane.field, plane.folder, plane.meta
    reg, settings, allow_overlap = meta["reg_outputs"], meta["registration"], meta["allow_overlap"]
    stat = np.load(folder / "stat.npy", allow_pickle=True)
    traces = np.load(folder / "F.npy", mmap_mode="r")
    count = field.frame_indices.size
    if not (folder / "data.bin").exists():
        raise ValueError(f"{field.label}: no registered movie, data.bin, in {folder}, so its frame order can't be proven. Suite2p keeps it unless delete_bin is set, sometimes on a separate fast disk; put it in the plane folder")
    expected_bytes = count * field.shape[0] * field.shape[1] * 2
    if (folder / "data.bin").stat().st_size != expected_bytes:
        raise ValueError(f"{field.label}: registered movie frame count differs")
    movie = np.memmap(folder / "data.bin", mode="r", dtype="int16", shape=(count, *field.shape))
    evidence = []
    for i in sample_positions(field):
        errors = {}
        for delta in (-1, 0, 1):
            if not 0 <= i + delta < count:
                continue
            raw = suite2p_input(acquisition, plane, [i + delta])[0]
            predicted = reconstruct_frame(raw, reg, settings, i)
            diff = predicted.astype("float64") - movie[i]
            errors[str(delta)] = float(np.sqrt(np.mean(diff**2)))
        if not is_separated(errors, limit=2):
            raise ValueError(f"{field.label}, frame {i}: frame mapping not demonstrated: {errors}")
        trace_errors = _trace_errors(movie, np.asarray(traces[:, i]), stat, allow_overlap, i) if len(stat) else None
        if trace_errors is not None and not is_separated(trace_errors, limit=0.01):
            raise ValueError(f"{field.label}, frame {i}: traces don't match the registered movie: {trace_errors}")
        file_index = int(field.file_index[i])
        evidence.append(
            dict(
                field=field.label,
                plane=int(field.plane),
                roi=int(field.number),
                folder=_relative(folder, base),
                input=plane.input,
                frame=int(i),
                tiff_frame=int(field.frame_indices[i]),
                file_index=file_index,
                file=acquisition.files[file_index].name,
                # Every ROI at a depth has the same pages in its split files, so the frame table's page is this one's too.
                page_index=int(acquisition.frames["page_index"][field.frame_indices[i]]),
                rmse=errors,
                trace_rmse=trace_errors,
            )
        )
    return evidence
