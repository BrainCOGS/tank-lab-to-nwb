"""Read what the clock needs from SpikeGLX files: meta values, digital lines and sync edges. See docs/ephys-timing.md.

Binary files are read through memory maps a slice at a time, so a 90 GB AP file is never loaded. Only the sync channel of
an AP file is read, and only in short windows.
"""

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# SpikeGLX puts the 1 Hz sync wave on bit 6 of an imec or OneBox SY word.
SY_SYNC_BIT = 6
_SLICE_SAMPLES = 2_000_000


class SpikeGLXError(ValueError):
    """A SpikeGLX file is missing, or doesn't hold what the conversion needs."""


def read_meta(path):
    """The key=value pairs of a .meta file, as text. Keys starting with ~ keep it."""
    path = Path(path)
    if path.suffix != ".meta":
        path = path.with_suffix(".meta")
    if not path.is_file():
        raise SpikeGLXError(f"{path}: no SpikeGLX meta file")
    meta = {}
    for line in path.read_text(errors="replace").splitlines():
        key, equals, value = line.partition("=")
        if equals:
            meta[key.strip()] = value.strip()
    return meta


@dataclass(frozen=True)
class Stream:
    """One SpikeGLX binary file and the meta values the clock uses."""

    bin_path: Path
    meta: dict
    kind: str  # "imec", "nidq" or "obx"
    rate: float  # calibrated samples per second from the meta file
    n_channels: int
    n_samples: int
    first_sample: int

    @property
    def start_seconds(self):
        """When the file starts, in seconds since the run started, on this stream's own clock."""
        return self.first_sample / self.rate


def open_stream(bin_path):
    """A SpikeGLX binary file and its meta, checked against each other."""
    bin_path = Path(bin_path)
    meta = read_meta(bin_path.with_suffix(".meta"))
    kind = meta.get("typeThis")
    rate_key = {"imec": "imSampRate", "nidq": "niSampRate", "obx": "obSampRate"}.get(kind)
    if rate_key is None or rate_key not in meta:
        raise SpikeGLXError(f"{bin_path}: meta has typeThis={kind!r} and no sampling rate this converter knows")
    n_channels = int(meta["nSavedChans"])
    size = int(meta["fileSizeBytes"])
    if bin_path.is_file() and bin_path.stat().st_size != size:
        raise SpikeGLXError(f"{bin_path}: {bin_path.stat().st_size} bytes on disk, but the meta says {size}. Is the file incomplete?")
    if size % (2 * n_channels):
        raise SpikeGLXError(f"{bin_path}: {size} bytes isn't a whole number of {n_channels}-channel int16 samples")
    return Stream(bin_path, meta, kind, float(meta[rate_key]), n_channels, size // (2 * n_channels), int(meta.get("firstSample", 0)))


def _counts(meta, key):
    return [int(value) for value in meta[key].split(",")]


def digital_channel(stream, word=0):
    """Index of the saved channel holding digital word `word` of a NIDQ or OneBox file."""
    if stream.kind == "nidq":
        mn, ma, xa, dw = _counts(stream.meta, "snsMnMaXaDw")
        if word >= dw:
            raise SpikeGLXError(f"{stream.bin_path}: no digital word {word}; the file saves {dw}")
        return mn + ma + xa + word
    if stream.kind == "obx":
        xa, dw, _ = _counts(stream.meta, "snsXaDwSy")
        if word >= dw:
            raise SpikeGLXError(f"{stream.bin_path}: no digital word {word}; the file saves {dw}")
        return xa + word
    raise SpikeGLXError(f"{stream.bin_path}: a {stream.kind} file has no digital word")


def sync_channel(stream):
    """Index of the saved SY channel of an imec or OneBox file."""
    if stream.kind == "imec":
        ap, lf, sy = _counts(stream.meta, "snsApLfSy")
        if not sy:
            raise SpikeGLXError(f"{stream.bin_path}: the SY channel wasn't saved, so this probe can't be put on the sync clock")
        return ap + lf
    if stream.kind == "obx":
        xa, dw, sy = _counts(stream.meta, "snsXaDwSy")
        if not sy:
            raise SpikeGLXError(f"{stream.bin_path}: the SY channel wasn't saved")
        return xa + dw
    raise SpikeGLXError(f"{stream.bin_path}: a {stream.kind} file has no SY channel")


def _memmap(stream):
    if not stream.bin_path.is_file():
        raise SpikeGLXError(f"{stream.bin_path}: binary file not found")
    return np.memmap(stream.bin_path, dtype="<i2", mode="r", shape=(stream.n_samples, stream.n_channels))


def read_channel(stream, channel, start=0, stop=None):
    """One saved channel as int16, read a slice at a time."""
    data = _memmap(stream)
    stop = stream.n_samples if stop is None else min(stop, stream.n_samples)
    out = np.empty(max(stop - start, 0), dtype="<i2")
    for position in range(start, stop, _SLICE_SAMPLES):
        end = min(position + _SLICE_SAMPLES, stop)
        out[position - start : end - start] = data[position:end, channel]
    return out


def bit(word, line):
    """One digital line of a 16-bit word, as uint8 0 or 1."""
    return ((np.asarray(word).view("<u2") >> line) & 1).astype(np.uint8)


def rising_edges(line):
    """Indices of the first high sample of each rising edge."""
    return np.flatnonzero(np.diff(line.astype(np.int8)) == 1) + 1


def sync_line_of_nidq(stream):
    """(channel, bit) carrying the sync wave in a NIDQ file, as its meta says."""
    if stream.meta.get("syncNiChanType", "0") != "0":
        raise SpikeGLXError(f"{stream.bin_path}: the sync wave is on an analog channel (syncNiChanType={stream.meta['syncNiChanType']}), which this converter doesn't read")
    line = int(stream.meta.get("syncNiChan", "0"))
    word, line = divmod(line, 16)
    return digital_channel(stream, word), line


def sync_edges(stream, windows=None):
    """Rising edges of the 1 Hz sync wave, as sample indices into the file.

    `windows` is a list of (start, stop) samples to read; None reads the whole channel. NIDQ files read the line their
    meta names; imec and OneBox files read bit 6 of the SY word.
    """
    if stream.kind == "nidq":
        channel, line = sync_line_of_nidq(stream)
    else:
        channel, line = sync_channel(stream), SY_SYNC_BIT
    if windows is None:
        windows = [(0, stream.n_samples)]
    edges = []
    for start, stop in windows:
        word = read_channel(stream, channel, start, stop)
        found = rising_edges(bit(word, line)) + start
        edges.append(found)
    edges = np.unique(np.concatenate(edges)) if edges else np.zeros(0, dtype=np.int64)
    return edges.astype(np.int64)


def sampling_windows(stream, every_seconds=60.0, length_seconds=1.25):
    """Windows long enough to hold one sync edge each, spread over the file, the first and last included."""
    length = int(np.ceil(length_seconds * stream.rate))
    if stream.n_samples <= length:
        return [(0, stream.n_samples)]
    starts = np.arange(0, stream.n_samples - length, every_seconds * stream.rate).astype(np.int64)
    starts = np.unique(np.append(starts, stream.n_samples - length))
    return [(int(start), int(start) + length) for start in starts]


_RUN = re.compile(r"^(?P<run>.+)_g(?P<gate>\d+)_t(?P<trigger>\d+|cat)\.(?P<stream>.+)\.bin$")


def find_sync_stream(session_folder):
    """The NIDQ file of a session folder, or its OneBox file when there's no NIDQ, as U19 chooses them."""
    session_folder = Path(session_folder)
    nidq = sorted(session_folder.glob("*nidq.bin"))
    obx = sorted(session_folder.glob("*obx.bin"))
    found = nidq or obx
    if not found:
        raise SpikeGLXError(f"{session_folder}: no *nidq.bin or *obx.bin, so nothing ties the probes to behavior")
    if len(found) > 1:
        raise SpikeGLXError(f"{session_folder}: several sync files ({', '.join(p.name for p in found)}); name the one to use")
    return found[0]


def find_ap_files(session_folder):
    """{probe number: AP .bin path} for the raw t0 AP files of a session folder, one per probe."""
    found = {}
    for path in sorted(Path(session_folder).glob("*_imec*/*.ap.bin")):
        match = re.search(r"\.imec(\d+)\.ap\.bin$", path.name)
        if match and "_tcat." not in path.name:
            probe = int(match[1])
            if probe in found:
                raise SpikeGLXError(f"{session_folder}: several raw AP files for imec{probe}: {found[probe].name}, {path.name}")
            found[probe] = path
    if not found:
        raise SpikeGLXError(f"{session_folder}: no *_imec*/*.ap.bin files")
    return found


def isolated_stream_folder(bin_path, root=None):
    """A folder holding links to just this stream's .bin and .meta, for libraries that scan a whole folder.

    neo 0.14.6 (under SpikeInterface 0.105.4 and neuroconv 0.10.2) parses every SpikeGLX file in a folder and fails on a
    session that mixes CatGT's `_tcat` files with the raw `_t0` ones (S3: `KeyError: (0, 'imec1.ap')`). The links are
    relative, because the lab's scratch share refuses absolute ones. Nothing is copied.
    """
    bin_path = Path(bin_path).resolve()
    root = Path(root or os.environ.get("TANK_EPHYS_LINK_ROOT") or tempfile.gettempdir()) / "tank-lab-to-nwb-streams"
    folder = root / hashlib.sha1(str(bin_path).encode()).hexdigest()[:16]
    folder.mkdir(parents=True, exist_ok=True)
    for source in (bin_path, bin_path.with_suffix(".meta")):
        if not source.is_file():
            raise SpikeGLXError(f"{source}: not found")
        link = folder / source.name
        target = os.path.relpath(source, folder)
        if link.is_symlink() and os.readlink(link) == target:
            continue
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
    return folder
