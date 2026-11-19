"""Write raw SpikeGLX AP and LF bands, and the edges of the sync lines, to an NWB file on the behavior clock.

The clock and the electrode rows come from the caller (the units conversion owns both), so the raw series reference the
same electrodes as the units and share their clock:

- `clock(samples) -> seconds`: maps sample indices of one stream, counted from the start of its *original* file, to
  seconds on the behavior clock. A clip's samples are shifted by its `clipStartSample` before the clock sees them.
  `LinearClock` is the stand-in used until the real mapping is wired in.
- `electrodes`: a `DynamicTableRegion` of the file's electrode table, or the row indices, one per neural channel in the
  stream's channel order (the SY channel is never written as a neural channel).

Times are written as `starting_time` + `rate` when the clock is a straight line over the stream to within
`TIME_TOLERANCE_SAMPLES` of a sample, and as explicit timestamps otherwise. See docs/ephys-raw.md.

Data are read from the file in time blocks, so memory stays at about `buffer_bytes` whatever the file size.
"""

import os
from dataclasses import dataclass

import numpy as np
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.common import DynamicTable, DynamicTableRegion, ElementIdentifiers, VectorData
from hdmf.data_utils import GenericDataChunkIterator
from pynwb.ecephys import LFP, ElectricalSeries

# gzip is the one filter every HDF5 build has, so DANDI and any reader can open the file without plugins.
# Settings chosen from the measurements in MEASUREMENTS.md; see docs/ephys-raw.md.
COMPRESSION = dict(compression="gzip", compression_opts=1, shuffle=True)
# A near-linear float64 ramp: shuffle puts its constant high bytes together, 26x smaller than without (timestamp_cost.py).
TIMESTAMP_COMPRESSION = dict(compression="gzip", compression_opts=4, shuffle=True)
CHUNK_SECONDS = 1.0  # time extent of a chunk
CHUNK_CHANNELS = 64  # channel extent of a chunk: 1 s x 64 channels x 2 bytes = 3.8 MB at 30 kHz
BUFFER_BYTES = 256 * 2**20  # read and compress this much per step
TIME_TOLERANCE_SAMPLES = 0.1  # how far the clock may bend away from a straight line before timestamps are written
CLOCK_PROBE_SECONDS = 1.0  # spacing of the points at which the clock is checked for straightness
CLOCK_PROBE_MIN_POINTS = 1000  # ... but at least this many points


class RawEphysError(ValueError):
    pass


@dataclass(frozen=True)
class LinearClock:
    """Stand-in for the real mapping: seconds = start + samples / rate. Samples count from the start of the original file."""

    start: float
    rate: float

    def __call__(self, samples):
        return self.start + np.asarray(samples, dtype=np.float64) / self.rate


def series_timing(clock, stream, n_samples, tolerance_samples=TIME_TOLERANCE_SAMPLES):
    """`dict(starting_time=, rate=)` if the clock is a straight line over these samples, else `dict(timestamps=...)`.

    The line goes through the clock at the first and last sample; it is checked at points `CLOCK_PROBE_SECONDS` apart, or
    closer for short series.
    """
    first = stream.clip_start
    last = first + n_samples - 1
    step = max(1, min(int(stream.sampling_rate * CLOCK_PROBE_SECONDS), n_samples // CLOCK_PROBE_MIN_POINTS))
    probe = np.unique(np.r_[np.arange(first, last, step), last])
    times = clock(probe)
    if n_samples == 1:
        return dict(starting_time=float(times[0]), rate=float(stream.sampling_rate))
    slope = (times[-1] - times[0]) / (last - first)
    residual = np.max(np.abs(times - (times[0] + (probe - first) * slope)))
    if residual <= tolerance_samples * abs(slope):
        return dict(starting_time=float(times[0]), rate=float(1.0 / slope))
    return dict(timestamps=H5DataIO(_ClockTimes(clock, first, n_samples, stream.sampling_rate), **TIMESTAMP_COMPRESSION))


class _ClockTimes(GenericDataChunkIterator):
    """Timestamps from the clock, computed a block at a time."""

    def __init__(self, clock, first, n_samples, rate):
        self._clock, self._first, self._n = clock, first, n_samples
        chunk = min(n_samples, int(rate * 60))
        super().__init__(chunk_shape=(chunk,), buffer_shape=(min(n_samples, chunk * 4),), display_progress=False)

    def _get_data(self, selection):
        span = selection[0]
        return self._clock(np.arange(self._first + span.start, self._first + span.stop))

    def _get_maxshape(self):
        return (self._n,)

    def _get_dtype(self):
        return np.dtype(np.float64)


class SpikeGLXChunks(GenericDataChunkIterator):
    """The neural channels of one stream, (time, channels) int16, read from the file a buffer at a time."""

    def __init__(self, stream, n_samples=None, chunk_shape=None, buffer_bytes=BUFFER_BYTES):
        self._stream = stream
        self._channels = stream.neural_channels
        self._n = stream.n_samples if n_samples is None else min(n_samples, stream.n_samples)
        n_channels = self._channels.size
        if chunk_shape is None:
            chunk_shape = (int(round(stream.sampling_rate * CHUNK_SECONDS)), CHUNK_CHANNELS)
        chunk_shape = (max(1, min(chunk_shape[0], self._n)), max(1, min(chunk_shape[1], n_channels)))
        # The buffer spans all channels, so each read is a contiguous block of the file.
        rows = max(chunk_shape[0], buffer_bytes // (2 * stream.n_channels) // chunk_shape[0] * chunk_shape[0])
        super().__init__(chunk_shape=chunk_shape, buffer_shape=(min(self._n, rows), n_channels), display_progress=False)

    def _get_data(self, selection):
        rows, columns = selection
        channels = self._channels[columns]
        # A plain positioned read, not a memory map, and the pages dropped from the cache afterwards: pages read through a
        # map stay resident and count against the process, and cached pages count against the cgroup, so on a 92 GB file
        # memory would otherwise track the page cache instead of the buffer.
        row_bytes = 2 * self._stream.n_channels
        offset, size = rows.start * row_bytes, (rows.stop - rows.start) * row_bytes
        fd = os.open(self._stream.bin_path, os.O_RDONLY)
        try:
            raw = bytearray(size)
            view, done = memoryview(raw), 0
            while done < size:
                got = os.preadv(fd, [view[done:]], offset + done)
                if got == 0:
                    raise OSError(f"{self._stream.bin_path}: ended {size - done} bytes early")
                done += got
            os.posix_fadvise(fd, offset, size, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        block = np.frombuffer(raw, dtype=np.int16).reshape(-1, self._stream.n_channels)
        return block[:, channels[0] : channels[-1] + 1]

    def _get_maxshape(self):
        return (self._n, self._channels.size)

    def _get_dtype(self):
        return np.dtype(np.int16)


def _filtering(stream):
    pn = stream.probe_part_number or "unknown probe"
    if stream.is_catgt:
        return f"Low-passed and downsampled by CatGT from the AP band of {pn}: {stream.meta.get('catGTCmdline0', '').strip('<>')}"
    if stream.band == "lf":
        return f"{pn} LF band as recorded by SpikeGLX (hardware band, about 0.5 to 500 Hz)."
    full = stream.meta.get("imAnyChanFullBand") == "true"
    return f"{pn} AP band as recorded by SpikeGLX, {'full band (no hardware high-pass)' if full else 'hardware band, about 300 Hz to 10 kHz'}."


def _region(nwbfile, electrodes, n_channels, description):
    if isinstance(electrodes, DynamicTableRegion):
        region = electrodes
    else:
        region = nwbfile.create_electrode_table_region(region=[int(i) for i in electrodes], description=description)
    if len(region.data) != n_channels:
        raise RawEphysError(f"{len(region.data)} electrode rows for {n_channels} neural channels; give one row per channel, in channel order")
    return region


def add_electrical_series(
    nwbfile,
    stream,
    electrodes,
    clock,
    name=None,
    n_samples=None,
    chunk_shape=None,
    buffer_bytes=BUFFER_BYTES,
    compression=None,
):
    """Add one AP or LF stream as an ElectricalSeries (data stays int16; `conversion` turns it into volts).

    SpikeGLX's own LF (NP1) goes in acquisition beside AP; LF that CatGT computed from AP (NP2) is processed data, so it
    goes in processing/ecephys/LFP. `n_samples` limits how much is written (for stubs).
    """
    if stream.kind != "imec":
        raise RawEphysError(f"{stream.bin_path.name}: not an imec stream")
    gains = stream.volts_per_bit()
    n_channels = gains.size
    name = name or f"ElectricalSeries{stream.band.upper()}{stream.device.capitalize()}"
    data = SpikeGLXChunks(stream, n_samples=n_samples, chunk_shape=chunk_shape, buffer_bytes=buffer_bytes)
    n_written = data.maxshape[0]
    uniform = np.allclose(gains, gains[0], rtol=0, atol=0)
    series = ElectricalSeries(
        name=name,
        data=H5DataIO(data, **(COMPRESSION if compression is None else compression)),
        electrodes=_region(nwbfile, electrodes, n_channels, f"Channels of {stream.device} {stream.band.upper()}"),
        conversion=float(gains[0]) if uniform else 1.0,
        channel_conversion=None if uniform else gains,
        filtering=_filtering(stream),
        description=(
            f"Raw {stream.band.upper()} band of {stream.device} ({stream.probe_part_number}), int16 as saved by SpikeGLX, from "
            f"{stream.bin_path.name}{' samples ' + str(stream.clip_start) + ' on' if stream.clip_start else ''}. "
            f"SpikeGLX rate {stream.sampling_rate!r} Hz on the probe's clock; times are on the behavior clock."
        ),
        **series_timing(clock, stream, n_written),
    )
    if stream.is_catgt and stream.band == "lf":
        ecephys = nwbfile.processing.get("ecephys") or nwbfile.create_processing_module("ecephys", "Processed extracellular electrophysiology.")
        lfp = ecephys.data_interfaces.get("LFP") or ecephys.add(LFP(name="LFP"))
        lfp.add_electrical_series(series)
    else:
        nwbfile.add_acquisition(series)
    return series


def digital_edges(stream, channel, n_samples=None, block_samples=2**20):
    """Every change of each bit of one int16 channel: (sample, bit, new level), samples from the original file's start.

    The first sample's state is reported as an edge at the first sample, so the trace can be rebuilt exactly.
    """
    data = stream.memmap()
    n = stream.n_samples if n_samples is None else min(n_samples, stream.n_samples)
    samples, bits, levels = [], [], []
    previous = None
    for start in range(0, n, block_samples):
        word = np.asarray(data[start : min(n, start + block_samples), channel]).view(np.uint16)
        if previous is None:
            ref = np.r_[~word[0], word]  # a fake previous sample that differs in every bit
        else:
            ref = np.r_[previous, word]
        changed = np.bitwise_xor(ref[1:], ref[:-1])
        index = np.flatnonzero(changed)
        for bit in range(16):
            hit = index[(changed[index] >> bit) & 1 == 1]
            if hit.size:
                samples.append(hit + start)
                bits.append(np.full(hit.size, bit, dtype=np.uint8))
                levels.append(((word[hit] >> bit) & 1).astype(np.uint8))
        previous = word[-1:]
    if not samples:
        return np.zeros(0, np.int64), np.zeros(0, np.uint8), np.zeros(0, np.uint8)
    samples, bits, levels = np.concatenate(samples), np.concatenate(bits), np.concatenate(levels)
    order = np.lexsort((bits, samples))
    return samples[order] + stream.clip_start, bits[order], levels[order]


def sync_channels(stream):
    """{label: channel} of the digital channels of a nidq, OneBox or imec stream worth keeping as edges."""
    channels = {}
    if stream.digital_word_channel is not None:
        channels["XD0"] = stream.digital_word_channel
    if stream.sync_channel is not None:
        channels["SY0"] = stream.sync_channel
    return channels


def add_sync_edges(nwbfile, streams_and_clocks, name="spikeglx_sync_edges", n_samples=None, bits=None):
    """One table of every edge on the digital channels of the given streams: the evidence the clock was fitted to.

    Columns: stream (file name), channel (XD0/SY0), bit, level after the edge, sample (from the start of the original
    file, on the stream's clock) and time (on the behavior clock). Edges keep a digital trace exactly, at a few bytes
    per edge, where the trace itself costs 2 bytes per sample on every channel.
    """
    descriptions = dict(
        stream="SpikeGLX file the edge was read from.",
        channel="Channel name in the SpikeGLX channel map.",
        bit="Bit (line) of the channel.",
        level="Level of the bit from this sample on.",
        sample="Sample index in the original file, on that stream's clock.",
        time="Seconds on the behavior clock.",
    )
    columns = {key: [] for key in descriptions}
    for stream, clock in streams_and_clocks:
        for label, channel in sync_channels(stream).items():
            samples, edge_bits, levels = digital_edges(stream, channel, n_samples=n_samples)
            keep = np.ones(samples.size, bool) if bits is None else np.isin(edge_bits, bits)
            columns["stream"] += [stream.bin_path.name] * int(keep.sum())
            columns["channel"] += [label] * int(keep.sum())
            columns["bit"].append(edge_bits[keep])
            columns["level"].append(levels[keep])
            columns["sample"].append(samples[keep])
            columns["time"].append(clock(samples[keep]))
    dtypes = dict(bit=np.uint8, level=np.uint8, sample=np.int64, time=np.float64)
    for key, dtype in dtypes.items():
        columns[key] = np.concatenate(columns[key]).astype(dtype) if columns[key] else np.zeros(0, dtype)
    n_rows = len(columns["stream"])
    table = DynamicTable(
        name=name,
        description=(
            "Every change of each bit of the SpikeGLX digital channels (nidq or OneBox digital word XD0, sync channel SY0). "
            "The first row per bit is that bit's level at the first sample written. `sample` counts from the start of the "
            "original .bin on that stream's own clock; `time` is the same instant on the behavior clock."
        ),
        id=ElementIdentifiers(name="id", data=np.arange(n_rows)),
        columns=[
            VectorData(name=key, description=text, data=H5DataIO(columns[key], **COMPRESSION) if key in dtypes and n_rows else columns[key])
            for key, text in descriptions.items()
        ],
    )
    ecephys = nwbfile.processing.get("ecephys") or nwbfile.create_processing_module("ecephys", "Processed extracellular electrophysiology.")
    ecephys.add(table)
    return table
