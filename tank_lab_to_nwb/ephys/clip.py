"""Cut one time window out of every stream of a SpikeGLX run, for tests and benchmarks. Reads only the window.

Each stream is cut at `round(start_s * rate)` of its own samples, so the clips start within a sample plus the streams'
rate-calibration difference of each other (about 10 ppm, so 6 ms at 600 s; measured on S3). That's only for looking at
them side by side: the exact relation between streams comes from the sync edges, and the clips keep the original sample
numbering to find them. Each clip's .meta is the original's with these changed:

- `fileSizeBytes`, `fileTimeSecs` and `fileSHA1` describe the clip;
- `firstSample` moves by the clip's start, as SpikeGLX writes it for a file that starts later in the same run;
- `clipStartSample` (where the clip starts in the source file) and `clipSourceFile` are added.

Usage: python -m tank_lab_to_nwb.ephys.clip <run folder> <output folder> [--start 600] [--seconds 60] [--probes 0 1] [--no-lf]
"""

import argparse
import hashlib
from pathlib import Path

from .spikeglx_layout import BYTES_PER_SAMPLE, CLIP_SOURCE_KEY, CLIP_START_KEY, find_recording, read_meta, write_meta

COPY_BYTES = 64 * 2**20


def clip_window(stream, start_s, seconds):
    """(first sample, sample count) of the window in `stream`'s own samples."""
    start = int(round(start_s * stream.sampling_rate))
    count = int(round(seconds * stream.sampling_rate))
    if start < 0 or count <= 0 or start + count > stream.n_samples:
        raise ValueError(f"{stream.bin_path.name}: window {start_s} s + {seconds} s is outside its {stream.n_samples / stream.sampling_rate:.3f} s")
    return start, count


def clip_stream(stream, out_path, start_s, seconds):
    """Copy the window of one stream to `out_path` (and its .meta beside it). Returns the clip's path."""
    start, count = clip_window(stream, start_s, seconds)
    row = BYTES_PER_SAMPLE * stream.n_channels
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sha1 = hashlib.sha1()
    remaining = count * row
    with open(stream.bin_path, "rb") as source, open(out_path, "wb") as target:
        source.seek(start * row)
        while remaining:
            block = source.read(min(COPY_BYTES, remaining))
            if not block:
                raise OSError(f"{stream.bin_path}: ended {remaining} bytes early")
            target.write(block)
            sha1.update(block)
            remaining -= len(block)
    meta = read_meta(stream.bin_path.with_suffix(".meta"))
    meta.update(
        fileSizeBytes=str(count * row),
        fileTimeSecs=repr(count / stream.sampling_rate),
        fileSHA1=sha1.hexdigest().upper(),
        firstSample=str(stream.first_sample + start),
    )
    meta[CLIP_START_KEY] = str(stream.clip_start + start)
    meta[CLIP_SOURCE_KEY] = str(stream.bin_path)
    # Keep the `~` keys last, where SpikeGLX writes them.
    write_meta(out_path.with_suffix(".meta"), dict(sorted(meta.items(), key=lambda item: item[0].startswith("~"))))
    return out_path


def clip_recording(folder, out_folder, start_s=600.0, seconds=60.0, probes=None, lf=True, ap=True, sync=True):
    """Clip every chosen stream of the run in `folder` into `out_folder/<run folder name>/`, keeping the layout."""
    recording = find_recording(folder)
    target = Path(out_folder) / recording.folder.name
    streams = []
    for probe in recording.probes:
        if probes is not None and int(probe.device[4:]) not in probes:
            continue
        streams += ([probe.ap] if ap else []) + ([probe.lf] if lf and probe.lf is not None else [])
    if sync and recording.sync is not None:
        streams.append(recording.sync)
    written = []
    for stream in streams:
        relative = stream.bin_path.relative_to(recording.folder)
        written.append(clip_stream(stream, target / relative, start_s, seconds))
    return target, written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("folder")
    parser.add_argument("out_folder")
    parser.add_argument("--start", type=float, default=600.0)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--probes", type=int, nargs="*")
    parser.add_argument("--no-lf", action="store_true")
    parser.add_argument("--no-ap", action="store_true")
    args = parser.parse_args(argv)
    target, written = clip_recording(args.folder, args.out_folder, args.start, args.seconds, args.probes, lf=not args.no_lf, ap=not args.no_ap)
    for path in written:
        print(path, path.stat().st_size)


if __name__ == "__main__":
    main()
