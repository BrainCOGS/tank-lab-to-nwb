"""Convert one ephys session: a ViRMEn log, the raw SpikeGLX folder for the clock, and one sorter folder per probe. No DB.

Usage: python -m tank_lab_to_nwb.ephys.convert_session <log.mat> <raw run folder> <output folder> imec0=<sorter folder> [imec1=<folder> ...] [--date YYYY-MM-DD]
       python -m tank_lab_to_nwb.ephys.convert_session --units-only <output folder> imec0=<sorter folder> ... --subject ID --session-id ID [--binary imec0=<ap.bin>]

--units-only writes the units of a sorting nothing ties to a recording or to behavior, on the sorted binary's own clock,
and says so in the file (units_only.py). It's only done when asked for; the session converter never falls back to it.

A sorter folder is the one holding spike_times.npy (`kilosort_output`, `kilosort4_output`, ...); a job folder holding
exactly one such folder is accepted too. Only the raw folder's sync file and short windows of each AP file's SY channel
are read, never the AP samples. Behavior and spikes go on the recording's clock by the lab's rule; --probe-clock and
--iteration-time choose the two departures from it (docs/ephys-timing.md). Raw SpikeGLX bands go in the same file on the
same clock: LF by default, AP with --raw-bands ap lf, none with --raw-bands (empty), and --raw-folder takes them from a
folder of clips (docs/ephys-raw.md). The NWB file goes in <output> with a report beside it.

--dandiset <folder> also names the file for DANDI under that folder (dandi organize, description "units", or "ecephys"
with raw bands) and runs `dandi validate` on it. A file dandi rejects is an error: it stays out of the folder, and the report says why.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ..virmen.layout import DandiValidationError, organize_for_dandi
from .clock import ITERATION_TIME, ITERATION_TIMES, PROBE_CLOCK, PROBE_CLOCKS
from .convert import conversion_report, write_ephys_nwbfile
from .raw import RAW_BANDS, verify_raw_ephys
from .session import build_session_sync
from .sorting import SortingError, find_sorter_folders
from .units_only import CLOCK, TIED_TO, UnitsOnlyError, write_units_only
from .verify import verify_ephys

__all__ = ["DandiValidationError", "SyncTimestampsError", "UnitsOnlyError", "convert", "convert_to_file", "convert_units_only"]


def _write_report(path, report):
    path.with_suffix(".report.json").write_text(json.dumps(report, indent=1, default=str))


def organize_and_validate(path, dandiset, description, report):
    """Name the file under `dandiset` and run `dandi validate` on it, and record the outcome in the file's report.

    DandiValidationError is raised, after the report is written, when dandi rejects the file; it then stays out of `dandiset`.
    """
    try:
        (organized,) = organize_for_dandi(path, dandiset, description=description, validate=True)
        report["dandi"] = dict(organized=str(organized), validated=True)
    except DandiValidationError as error:
        report["dandi"] = dict(organized=None, validated=False, error=str(error))
        raise
    finally:
        _write_report(path, report)
    return organized


def resolve_sorter_folder(path):
    path = Path(path)
    if (path / "spike_times.npy").is_file():
        return path
    found = find_sorter_folders(path)
    if len(found) > 1:
        raise SortingError(f"{path}: holds several sorter folders ({', '.join(p.name for p in found)}); name one")
    return found[0]


def _kind(raw_bands):
    """What names the file, in its folder and for DANDI: "ecephys" with raw bands, "units" without."""
    return "ecephys" if tuple(raw_bands or ()) else "units"


def convert(log, raw_folder, output, sorter_folders, session_date=None, metadata=None, name=None, probe_clock=PROBE_CLOCK, iteration_time=ITERATION_TIME, allow_virmen_borrowed=False,
            raw_bands=RAW_BANDS, raw_from=None, sync_edges=True, raw_max_seconds=None, dandiset=None):
    """Convert, verify and report, and return the path of the file. See docs/ephys-timing.md for the two clock choices.

    `raw_bands` ("ap", "lf") are written from `raw_from` (default `raw_folder`) with the sync edges; see docs/ephys-raw.md.
    The file is `<output>/<name or the log's stem>_units.nwb` without raw bands, `..._ecephys.nwb` with them.

    With `dandiset`, the file is also named under that folder (`desc-units` or `desc-ecephys`) and checked by
    `dandi validate`; the report's "dandi" entry holds the organized path, and DandiValidationError is raised if dandi
    rejects the file. The file is copied there, so a file with raw bands takes its size twice.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name or Path(log).stem}_{_kind(raw_bands)}.nwb"
    return convert_to_file(log, raw_folder, path, sorter_folders, session_date=session_date, metadata=metadata, probe_clock=probe_clock, iteration_time=iteration_time, allow_virmen_borrowed=allow_virmen_borrowed,
                           raw_bands=raw_bands, raw_from=raw_from, sync_edges=sync_edges, raw_max_seconds=raw_max_seconds, dandiset=dandiset)


def convert_to_file(log, raw_folder, path, sorter_folders, session_date=None, metadata=None, probe_clock=PROBE_CLOCK, iteration_time=ITERATION_TIME, allow_virmen_borrowed=False,
                    raw_bands=RAW_BANDS, raw_from=None, sync_edges=True, raw_max_seconds=None, dandiset=None, sync_timestamps=None):
    """What `convert` does, writing exactly `path` (its folder must exist), with the report beside it as `<stem>.report.json`.

    `sync_timestamps`, per-iteration times computed elsewhere (such as the DB's), are checked against the iteration
    times recomputed from the raw sync file: SyncTimestampsError if they disagree, before anything is written; the
    report's "sync_timestamps" entry if they agree. They never set a time in the file.
    """
    path = Path(path)
    sorter_folders = {probe: resolve_sorter_folder(folder) for probe, folder in sorter_folders.items()}
    sync = build_session_sync(raw_folder, Path(log), probes=sorted(sorter_folders), session_date=session_date, probe_clock=probe_clock, iteration_time=iteration_time, allow_virmen_borrowed=allow_virmen_borrowed)
    agreement = None if sync_timestamps is None else check_sync_timestamps(sync, sync_timestamps)
    raw_bands = tuple(raw_bands or ())
    raw = dict(bands=raw_bands, raw_folder=raw_from, sync_edges=sync_edges, max_seconds=raw_max_seconds) if raw_bands or sync_edges else None
    raw_notes = {}
    prepared = write_ephys_nwbfile(sync, sorter_folders, path, metadata, raw=raw, raw_notes=raw_notes)
    checked = verify_ephys(sync, prepared, path)
    if raw is not None:
        checked["raw"] = verify_raw_ephys(path, sync, {probe.name: probe.electrodes.rows for probe, _ in prepared}, **raw)
    report = conversion_report(sync, prepared)
    if agreement is not None:
        report["sync_timestamps"] = agreement
    report["verified"] = checked
    report["raw"] = raw_notes
    _write_report(path, report)
    if dandiset is not None:
        organize_and_validate(path, dandiset, _kind(raw_bands), report)
    return path


class SyncTimestampsError(ValueError):
    """Per-iteration times given with a conversion disagree with the ones recomputed from the raw sync file."""


def check_sync_timestamps(sync, given):
    """Compare `given` per-iteration times with the session's, before the file's shift; one sync-stream sample is allowed.

    Returns what the report records; raises SyncTimestampsError, stating the size of the disagreement, otherwise.
    """
    given = np.asarray(given, dtype="float64").ravel()
    recomputed = np.concatenate(sync.iteration_seconds) - sync.clock.shift
    where = f"the iteration times recomputed from {sync.sync_stream.bin_path} (iteration_time {sync.iteration_time})"
    if given.size != recomputed.size:
        raise SyncTimestampsError(f"sync_timestamps has another length than {where}: {given.size} given, {recomputed.size} recomputed")
    difference = np.abs(given - recomputed)
    worst = int(np.argmax(difference)) if difference.size else 0
    samples = float(difference[worst] * sync.sync_stream.rate) if difference.size else 0.0
    if not samples <= 1.0:
        raise SyncTimestampsError(
            f"sync_timestamps differs from {where} by up to {difference[worst] * 1e3:.4f} ms ({samples:.1f} samples of the {sync.sync_stream.rate} Hz sync stream), "
            f"at iteration {worst}; {int(np.sum(difference * sync.sync_stream.rate > 1.0))} of {given.size} differ by more than one sample"
        )
    return dict(agrees=True, iterations=int(given.size), max_difference_seconds=float(difference.max(initial=0.0)), max_difference_samples=samples, tolerance_samples=1.0, compared_before_shift_seconds=float(sync.clock.shift))


def convert_units_only(output, sorter_folders, subject_id, session_id, binaries=None, metadata=None, name=None):
    """Write units alone, on the sorted binary's own clock, verify them, and return the path. See units_only.py."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sorter_folders = {probe: resolve_sorter_folder(folder) for probe, folder in sorter_folders.items()}
    path = output / f"{name or session_id}_unitsonly.nwb"
    units, checked = write_units_only(sorter_folders, path, subject_id, session_id, binaries, metadata)
    report = dict(units_only=True, clock=CLOCK, tied_to=TIED_TO, verified=checked, probes={probe.name: dict(sorter=probe.sorted.sorter, version=probe.sorted.sorter_version, evidence=probe.sorted.sorter_evidence, units=int(probe.sorted.unit_ids.size), spikes=int(probe.counts.sum()), files=probe.sorted.files, notes=probe.sorted.notes, ignored=probe.sorted.ignored, sample_origin_check=how) for probe, how in units})
    path.with_suffix(".report.json").write_text(json.dumps(report, indent=1, default=str))
    return path


def _probe_folder(text):
    probe, equals, folder = text.partition("=")
    if not equals or not probe.startswith("imec"):
        raise argparse.ArgumentTypeError(f"{text!r}: expected imec<N>=<sorter folder>")
    return probe, Path(folder)


def main_units_only(argv):
    parser = argparse.ArgumentParser(prog="convert_session --units-only", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--units-only", action="store_true", required=True)
    parser.add_argument("output", type=Path)
    parser.add_argument("sorted", type=_probe_folder, nargs="+", help="imec<N>=<sorter folder>, one per probe.")
    parser.add_argument("--subject", required=True, help="Subject id; a sorter folder doesn't say whose it is.")
    parser.add_argument("--session-id", required=True, help="Session id for the file.")
    parser.add_argument("--binary", type=_probe_folder, action="append", default=[], help="imec<N>=<ap.bin the sorter read>, when its files don't name one that exists.")
    parser.add_argument("--name", help="Stem of the output file; default the session id.")
    arguments = parser.parse_args(argv)
    print(convert_units_only(arguments.output, dict(arguments.sorted), arguments.subject, arguments.session_id, binaries=dict(arguments.binary), name=arguments.name))


def session_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path)
    parser.add_argument("raw_folder", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("sorted", type=_probe_folder, nargs="+", help="imec<N>=<sorter folder>, one per probe.")
    parser.add_argument("--date", help="Recording date, YYYY-MM-DD; sessions before 2021-06-01 are refused.")
    parser.add_argument("--name", help="Stem of the output file; default the log's.")
    parser.add_argument("--probe-clock", choices=PROBE_CLOCKS, default=PROBE_CLOCK, help=f"Spike times: sync_edges (drift corrected by the 1 Hz sync edges), meta_rates (sample / imSampRate, the lab's rule) or params_rate (sample / params.py rate, as the DB stores). Default {PROBE_CLOCK}.")
    parser.add_argument("--iteration-time", choices=ITERATION_TIMES, default=ITERATION_TIME, help=f"Iteration times: pipeline_index (the lab's rule) or edge (the line's edge). Default {ITERATION_TIME}.")
    parser.add_argument("--allow-virmen-borrowed", action="store_true", help="Convert a session whose pulse matching failed, with the pipeline's ViRMEn-borrowed vectors (lower accuracy; the file says so).")
    parser.add_argument("--raw-bands", nargs="*", choices=("ap", "lf"), default=list(RAW_BANDS), help=f"Raw SpikeGLX bands to write. Default {' '.join(RAW_BANDS)}; AP is 40-50 GB per probe-hour compressed.")
    # dest: the positional raw_folder is the recording the clock is built from, and this option must not replace it.
    parser.add_argument("--raw-folder", dest="raw_from", type=Path, help="Take the raw bands from this folder (for example clips from tank_lab_to_nwb.ephys.clip) instead of the raw run folder.")
    parser.add_argument("--raw-max-seconds", type=float, help="Write only the first seconds of each raw stream (a stub).")
    parser.add_argument("--no-sync-edges", action="store_true", help="Leave out the table of every sync-line edge.")
    parser.add_argument("--dandiset", type=Path, help="Also name the file for DANDI under this folder and run dandi validate on it; a file dandi rejects is an error.")
    return parser


def main():
    import sys

    if "--units-only" in sys.argv[1:]:
        return main_units_only(sys.argv[1:])
    arguments = session_parser().parse_args()
    print(convert(arguments.log, arguments.raw_folder, arguments.output, dict(arguments.sorted), session_date=arguments.date, name=arguments.name, probe_clock=arguments.probe_clock, iteration_time=arguments.iteration_time, allow_virmen_borrowed=arguments.allow_virmen_borrowed,
                  raw_bands=arguments.raw_bands, raw_from=arguments.raw_from, sync_edges=not arguments.no_sync_edges, raw_max_seconds=arguments.raw_max_seconds, dandiset=arguments.dandiset))


if __name__ == "__main__":
    main()
