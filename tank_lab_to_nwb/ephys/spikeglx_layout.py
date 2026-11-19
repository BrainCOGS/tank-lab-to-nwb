"""Find and describe every SpikeGLX stream of one recording (for writing raw data), and refuse the layouts not handled.


`spikeglx.py` beside this reads what the clock needs (sync edges, digital lines); this module describes whole files for
the raw writer and the clip tool: band, probe, gains, sync channels, CatGT origin.
Only `.meta` files are parsed here; `.bin` data is opened as a memory map by whoever reads it. Gains and channel names come
from neo's own SpikeGLX parser, so they match what SpikeInterface reports for the same files.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BYTES_PER_SAMPLE = 2  # every SpikeGLX stream is int16
# Keys the clip tool adds to a clip's .meta. SpikeGLX and neo ignore keys they don't know.
CLIP_START_KEY = "clipStartSample"  # first sample of the clip, counted from the start of the source file
CLIP_SOURCE_KEY = "clipSourceFile"

_NAME = re.compile(r"^(?P<run>.+)_g(?P<gate>\d+)_(?P<trigger>t\d+|tcat)\.(?P<device>imec\d+|nidq|obx\d+)(?:\.(?P<band>ap|lf|obx))?\.(?P<ext>c?bin)$")


class SpikeGLXLayoutError(ValueError):
    """The recording's files are laid out in a way this converter doesn't handle; the message says which and why."""


def read_meta(path):
    """A .meta file as an ordered dict of strings. Keys keep their leading `~`, and values are left as written."""
    meta = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            meta[key] = value
    return meta


def write_meta(path, meta):
    Path(path).write_text("".join(f"{key}={value}\n" for key, value in meta.items()))


@dataclass(frozen=True)
class Stream:
    """One SpikeGLX .bin file and what its .meta says about it."""

    bin_path: Path
    meta: dict = field(repr=False, compare=False)
    device: str  # imec0, nidq, obx0
    band: str  # ap, lf, nidq or obx
    trigger: str  # t0 or tcat
    sampling_rate: float  # as calibrated by SpikeGLX, on the stream's own clock
    n_channels: int  # saved channels, including any sync channel
    n_samples: int
    first_sample: int  # the stream's own sample count at the first sample of this file
    clip_start: int  # 0 for an original file; for a clip, where it starts in its source file

    @property
    def kind(self):
        return self.meta.get("typeThis", "")

    @property
    def probe_part_number(self):
        return self.meta.get("imDatPrb_pn")

    @property
    def is_catgt(self):
        return self.trigger == "tcat" or "catGTCmdline0" in self.meta

    @property
    def sync_channel(self):
        """Index of the SY channel (imec, OneBox), or None."""
        counts = self.meta.get("snsApLfSy") or self.meta.get("snsXaDwSy")
        if counts is None or int(counts.split(",")[-1]) == 0:
            return None
        return self.n_channels - 1

    @property
    def neural_channels(self):
        """Indices of the neural channels of an imec stream: everything but the SY channel."""
        if self.kind != "imec":
            return np.arange(0)
        stop = self.n_channels - (1 if self.sync_channel is not None else 0)
        return np.arange(stop)

    @property
    def digital_word_channel(self):
        """Index of the first digital word of a nidq or OneBox stream, or None."""
        if self.kind == "nidq":
            mn, ma, xa, dw = (int(v) for v in self.meta["snsMnMaXaDw"].split(","))
            return mn + ma + xa if dw else None
        if self.kind == "obx":
            xa, dw, _ = (int(v) for v in self.meta["snsXaDwSy"].split(","))
            return xa if dw else None
        return None

    def memmap(self):
        """The whole file as (samples, channels) int16, read lazily."""
        return np.memmap(self.bin_path, dtype=np.int16, mode="r", shape=(self.n_samples, self.n_channels))

    def info(self):
        """neo's description of this stream: channel names and gains (microvolts per bit for imec, volts otherwise)."""
        from neo.rawio.spikeglxrawio import extract_stream_info, read_meta_file

        meta_path = self.bin_path.with_suffix(".meta")
        return extract_stream_info(meta_path, read_meta_file(meta_path))

    def volts_per_bit(self):
        """Per-channel gain of the neural channels, in volts per bit."""
        gains = np.asarray(self.info()["channel_gains"], dtype=float)[self.neural_channels]
        return gains * 1e-6


def stream_from_bin(bin_path):
    bin_path = Path(bin_path)
    match = _NAME.match(bin_path.name)
    if match is None:
        raise SpikeGLXLayoutError(f"{bin_path.name}: not a SpikeGLX file name (<run>_g<N>_t<N>.<device>[.<band>].bin)")
    if match["ext"] == "cbin":
        raise SpikeGLXLayoutError(f"{bin_path.name}: compressed with mtscomp (.cbin); decompress it first, this converter reads only .bin")
    meta_path = bin_path.with_suffix(".meta")
    if not meta_path.exists():
        raise SpikeGLXLayoutError(f"{bin_path.name}: no .meta file beside it, so its rate, channels and gains are unknown")
    meta = read_meta(meta_path)
    n_channels = int(meta["nSavedChans"])
    size = bin_path.stat().st_size
    declared = int(meta["fileSizeBytes"])
    if size != declared:
        raise SpikeGLXLayoutError(f"{bin_path.name}: {size} bytes on disk but the .meta says {declared}; the file is truncated or was changed")
    if size % (BYTES_PER_SAMPLE * n_channels):
        raise SpikeGLXLayoutError(f"{bin_path.name}: {size} bytes is not a whole number of {n_channels}-channel samples")
    rate_key = {"imec": "imSampRate", "nidq": "niSampRate", "obx": "obSampRate"}.get(meta.get("typeThis", ""))
    if rate_key is None or rate_key not in meta:
        raise SpikeGLXLayoutError(f"{bin_path.name}: .meta has no typeThis/sample rate; files older than SpikeGLX phase 3B aren't handled")
    device = match["device"]
    band = match["band"] if device.startswith("imec") else ("obx" if device.startswith("obx") else "nidq")
    if device.startswith("imec") and band is None:
        raise SpikeGLXLayoutError(f"{bin_path.name}: imec file without .ap or .lf band")
    return Stream(
        bin_path=bin_path,
        meta=meta,
        device=device,
        band=band,
        trigger=match["trigger"],
        sampling_rate=float(meta[rate_key]),
        n_channels=n_channels,
        n_samples=size // (BYTES_PER_SAMPLE * n_channels),
        first_sample=int(meta.get("firstSample", 0)),
        clip_start=int(meta.get(CLIP_START_KEY, 0)),
    )


@dataclass(frozen=True)
class Probe:
    device: str  # imec0
    ap: Stream
    lf: Stream | None
    lf_note: str  # where the LF band comes from, or why there is none


@dataclass(frozen=True)
class Recording:
    folder: Path
    probes: tuple
    sync: Stream | None  # the nidq or OneBox stream carrying the behavior pulses


def find_recording(folder):
    """Every stream of one SpikeGLX run folder (`<run>_g0/` with `<run>_g0_imecN/` inside), checked against the layouts handled.

    Handled: one gate and one trigger (t0) per run; per probe a raw `t0` AP file and an LF file that is either SpikeGLX's
    own `t0` (NP1 hardware band) or CatGT's `tcat` (NP2, derived from AP); a `nidq` or a OneBox `obx` stream at the top.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise SpikeGLXLayoutError(f"{folder}: not a directory")
    files = sorted(p for p in folder.glob("*") if p.suffix in (".bin", ".cbin")) + sorted(
        p for p in folder.glob("*_imec*/*") if p.suffix in (".bin", ".cbin")
    )
    streams = [stream_from_bin(path) for path in files]
    gates = {_NAME.match(s.bin_path.name)["gate"] for s in streams}
    if len(gates) > 1:
        raise SpikeGLXLayoutError(f"{folder}: several gates ({sorted(gates)}); convert one gate per file")
    triggers = {s.trigger for s in streams if s.trigger != "tcat"}
    if triggers - {"t0"}:
        raise SpikeGLXLayoutError(f"{folder}: triggers {sorted(triggers)}; only a single t0 per run is handled (concatenate with CatGT, then decide)")
    by_device = {}
    for stream in streams:
        by_device.setdefault(stream.device, []).append(stream)
    sync = [s for s in streams if s.kind in ("nidq", "obx")]
    if len(sync) > 1:
        raise SpikeGLXLayoutError(f"{folder}: more than one sync stream ({[s.bin_path.name for s in sync]})")
    probes = []
    for device in sorted((d for d in by_device if d.startswith("imec")), key=lambda d: int(d[4:])):
        group = by_device[device]
        ap = [s for s in group if s.band == "ap"]
        if any(s.trigger == "tcat" for s in ap):
            raise SpikeGLXLayoutError(f"{device}: AP file written by CatGT (tcat); it is filtered and isn't the raw recording")
        if len(ap) != 1:
            raise SpikeGLXLayoutError(f"{device}: expected one t0 AP file, found {len(ap)}")
        lf = [s for s in group if s.band == "lf"]
        if len(lf) > 1:
            raise SpikeGLXLayoutError(f"{device}: {len(lf)} LF files ({[s.bin_path.name for s in lf]}); keep one")
        lf = lf[0] if lf else None
        if lf is None:
            note = "No LF file: NP2 probes have no hardware LF band and CatGT wasn't run with -lf for this probe."
        elif lf.is_catgt:
            command = lf.meta.get("catGTCmdline0", "").strip("<>")
            note = f"LF written by CatGT from the AP band (not recorded by SpikeGLX): {command}"
        else:
            note = "LF band recorded by SpikeGLX (NP1 hardware band)."
        probes.append(Probe(device=device, ap=ap[0], lf=lf, lf_note=note))
    if not probes:
        raise SpikeGLXLayoutError(f"{folder}: no imec AP files")
    return Recording(folder=folder, probes=tuple(probes), sync=sync[0] if sync else None)
