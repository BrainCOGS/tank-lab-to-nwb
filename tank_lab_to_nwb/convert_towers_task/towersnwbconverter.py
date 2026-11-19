"""The neuroconv converter for a Towers-task session: ViRMEn behavior alone, or behavior with spike-sorted units.

    source_data = {"VirmenData": {"file_path": <log.mat>}}                    # behavior, always
    source_data["KilosortProbe0"] = {"folder_path": <sorter folder>}          # units of imec0; "Kilosort" is imec0 too
    source_data["SpikeGLX"] = {"folder_path": <raw SpikeGLX run folder>}      # the clock, required with units
    converter = TowersNWBConverter(source_data, sync_timestamps=None)
    converter.run_conversion(nwbfile_path=..., metadata=converter.get_metadata(), overwrite=True)

Behavior alone is a neuroconv conversion of `VirmenDataInterface`; `sync_timestamps`, one time per logged frame, moves
it with `set_aligned_timestamps`. With units, `run_conversion` writes the file through the ephys converter
(`tank_lab_to_nwb.ephys.convert_session.convert_to_file`, the code behind `python -m tank_lab_to_nwb.ephys.convert_session`):
behavior and spikes on the recording's clock, rebuilt from the run folder's sync file (nidq or obx) and each probe's AP
meta and SY channel (docs/ephys-timing.md), checked by reading the file back, with a report beside it. There
`sync_timestamps` sets nothing: it must agree with the recomputed iteration times to one sync sample, and the report says
it did. Every refusal of the ephys converter reaches the caller unchanged.

Raw bands (SpikeGLXAP, SpikeGLXLFP) and imaging aren't written through this class.
"""

from pathlib import Path

from neuroconv import NWBConverter

from ..ephys import convert_session
from ..ephys.clock import ITERATION_TIME, PROBE_CLOCK
from ..virmen.interface import VirmenDataInterface

CLOCK_DEFAULTS = dict(probe_clock=PROBE_CLOCK, iteration_time=ITERATION_TIME, allow_virmen_borrowed=False, session_date=None)
# NWBFile entries the units file takes from the caller's metadata, by the names virmen.convert.build_nwbfile uses.
NWBFILE_EDITS = ("session_description", "experimenter", "institution", "lab", "experiment_description", "keywords")
# Made again by neuroconv on every get_metadata(); nothing to carry.
NWBFILE_FRESH = ("identifier", "source_script", "source_script_file_name")
SPIKEGLX_NEEDED = (
    'units need source_data["SpikeGLX"] = {{"folder_path": <raw SpikeGLX run folder>}}, the clock of {keys}: the folder '
    "that holds the run's *.nidq.bin (or *.obx.bin) with its meta file and each probe's <run>_imec<N> folder with its "
    ".ap.bin and .ap.meta. Only the sync file and the AP files' SY channel are read (docs/ephys-timing.md)."
)


def _probe(key):
    """The probe a Kilosort key names: "Kilosort" is imec0, "KilosortProbe<N>" is imec<N>."""
    number = "0" if key == "Kilosort" else key.removeprefix("KilosortProbe")
    if not number.isdigit():
        raise ValueError(f"{key}: expected Kilosort or KilosortProbe<N>")
    return f"imec{int(number)}"


class TowersNWBConverter(NWBConverter):
    """A ViRMEn behavior session, with the sorted units of its Neuropixels probes when Kilosort keys are given.

    The subject and session come from the log, through the ndx-tank-metadata field registry. `probe_clock`,
    `iteration_time`, `allow_virmen_borrowed` and `session_date` are the ephys converter's clock options, with its
    defaults; they apply only with units. `ttl_source`, the old TTL alignment, is refused.
    """

    data_interface_classes = {"VirmenData": VirmenDataInterface}

    def __init__(self, source_data, ttl_source=None, sync_timestamps=None, verbose=False, *, probe_clock=PROBE_CLOCK, iteration_time=ITERATION_TIME, allow_virmen_borrowed=False, session_date=None):
        if ttl_source is not None:
            raise ValueError("ttl_source: the TTL alignment it chose is gone. With units, give the raw run folder as source_data['SpikeGLX']; the clock is built from it (docs/ephys-timing.md).")
        source_data = dict(source_data)
        raw_bands = sorted(key for key in source_data if key.startswith("SpikeGLX") and key != "SpikeGLX")  # SpikeGLXAP, SpikeGLXLFP, ...
        if raw_bands:
            raise ValueError(f"{', '.join(raw_bands)}: raw bands are not written through TowersNWBConverter yet; SpikeGLX alone gives the run folder for the clock of sorted units")
        imaging = sorted(key for key in source_data if key.startswith(("Suite2p", "Tiff")))
        if imaging:
            raise ValueError(f"{', '.join(imaging)}: imaging is converted by tank_lab_to_nwb.imaging, not by TowersNWBConverter")
        if "VirmenData" not in source_data:
            raise ValueError("VirmenData is required: the converter writes a ViRMEn behavior session")
        self.sorter_folders = {}
        for key in sorted(key for key in source_data if key.startswith("Kilosort")):
            probe = _probe(key)
            if probe in self.sorter_folders:
                raise ValueError(f"{key}: a second sorting of {probe}")
            self.sorter_folders[probe] = Path(source_data.pop(key)["folder_path"])
        spikeglx = source_data.pop("SpikeGLX", None)
        unknown = sorted(set(source_data) - set(self.data_interface_classes))
        if unknown:
            raise ValueError(f"{', '.join(unknown)}: TowersNWBConverter takes VirmenData, Kilosort or KilosortProbe<N>, and SpikeGLX")
        clock = dict(probe_clock=probe_clock, iteration_time=iteration_time, allow_virmen_borrowed=allow_virmen_borrowed, session_date=session_date)
        if self.sorter_folders and spikeglx is None:
            raise ValueError(SPIKEGLX_NEEDED.format(keys=", ".join(sorted(self.sorter_folders))))
        if not self.sorter_folders:
            if spikeglx is not None:
                raise ValueError("SpikeGLX: the run folder is the clock of sorted units, and no Kilosort key is given; raw bands are not written through TowersNWBConverter yet")
            changed = [name for name, value in clock.items() if value != CLOCK_DEFAULTS[name]]
            if changed:
                raise ValueError(f"{', '.join(changed)}: clock options of a conversion with units; no Kilosort key is given")
        self.raw_folder = None if spikeglx is None else Path(spikeglx["folder_path"])
        self.clock_options = clock
        self._sync_timestamps = sync_timestamps
        super().__init__(source_data=source_data, verbose=verbose)

    def temporally_align_data_interfaces(self, metadata=None, conversion_options=None):
        if self._sync_timestamps is not None:
            self.data_interface_objects["VirmenData"].set_aligned_timestamps(self._sync_timestamps)

    def run_conversion(self, nwbfile_path=None, nwbfile=None, metadata=None, overwrite=False, *, dandiset=None, **options):
        """Write the session. With units, `nwbfile_path` is written exactly, with `<stem>.report.json` beside it.

        `dandiset` (units only) also names the file for DANDI under that folder and runs `dandi validate` on it.
        """
        metadata = _with_birth_timezone(metadata or self.get_metadata())
        if not self.sorter_folders:
            if dandiset is not None:
                raise ValueError("dandiset: only a conversion with units is named for DANDI here")
            return super().run_conversion(nwbfile_path=nwbfile_path, nwbfile=nwbfile, metadata=metadata, overwrite=overwrite, **options)
        if nwbfile is not None or nwbfile_path is None or any(value not in (None, False) for value in options.values()):
            given = ["nwbfile"] * (nwbfile is not None) + sorted(name for name, value in options.items() if value not in (None, False))
            raise ValueError(f"{', '.join(given) or 'nwbfile_path'}: with units the file is written to nwbfile_path by the ephys converter, alone")
        path = Path(nwbfile_path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path}: exists; pass overwrite=True to replace it")
        path.parent.mkdir(parents=True, exist_ok=True)
        log = self.data_interface_objects["VirmenData"].source_data["file_path"]
        return convert_session.convert_to_file(
            log, self.raw_folder, path, self.sorter_folders, metadata=self._units_metadata(metadata), dandiset=dandiset, sync_timestamps=self._sync_timestamps, **self.clock_options
        )

    def _units_metadata(self, metadata):
        """The caller's edits of neuroconv's NWBFile and Subject metadata, in the form virmen.convert.build_nwbfile takes.

        Only edits are passed, so the ephys converter's own descriptions stay where the caller changed nothing. The
        session's start, id and subject id come from the recording and the log; an edit of them, or of anything else the
        file can't carry, is refused rather than dropped.
        """
        default = _with_birth_timezone(self.get_metadata())
        refused = sorted(set(metadata) - {"NWBFile", "Subject"})
        nwbfile, edits = metadata.get("NWBFile", {}), {}
        for key, value in nwbfile.items():
            if key in NWBFILE_FRESH or value == default["NWBFile"].get(key):
                continue
            if key in NWBFILE_EDITS:
                edits[key] = value
            else:
                refused.append(f"NWBFile.{key}")
        subject = dict(metadata.get("Subject", {}))
        if subject.pop("subject_id", default["Subject"]["subject_id"]) != default["Subject"]["subject_id"]:
            refused.append("Subject.subject_id")
        if refused:
            raise ValueError(f"{', '.join(refused)}: the units file takes these from the recording and the log, or has no place for them")
        return {**edits, "subject": subject}


def _with_birth_timezone(metadata):
    """A naive date of birth (the U19 DB's) gets the session's timezone, not the converting machine's."""
    subject, start = metadata.get("Subject", {}), metadata.get("NWBFile", {}).get("session_start_time")
    birth = subject.get("date_of_birth")
    if birth is not None and getattr(birth, "tzinfo", True) is None and getattr(start, "tzinfo", None) is not None:
        metadata = {**metadata, "Subject": {**subject, "date_of_birth": birth.replace(tzinfo=start.tzinfo)}}
    return metadata
