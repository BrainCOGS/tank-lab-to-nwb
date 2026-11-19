"""The neuroconv converter for a Towers-task behavior session.

It used to combine SpikeGLX and per-probe Kilosort interfaces too, with a single `sync_timestamps` array that skipped
Kilosort and no drift correction. Spike-sorted units now have one supported path, `tank_lab_to_nwb.ephys.convert_session`,
which puts them and the behavior on the recording's clock as docs/ephys-timing.md describes. This converter refuses
ephys interfaces and says where to go, rather than writing spikes on an unknown clock.
"""

from neuroconv import NWBConverter

from ..virmen.interface import VirmenDataInterface

EPHYS_MOVED = (
    "{keys}: ephys no longer goes through TowersNWBConverter, whose single sync_timestamps array left Kilosort units off "
    "the clock of the behavior. Convert units with `python -m tank_lab_to_nwb.ephys.convert_session <log> <raw run folder> <output> "
    "imec0=<sorter folder> ...` (see docs/ephys-timing.md)."
)


class TowersNWBConverter(NWBConverter):
    """A ViRMEn behavior session. The subject and session come from the log, through the ndx-tank-metadata field registry."""

    data_interface_classes = {"VirmenData": VirmenDataInterface}

    def __init__(self, source_data, verbose=False):
        other = sorted(set(source_data) - set(self.data_interface_classes))
        if other:
            raise ValueError(EPHYS_MOVED.format(keys=", ".join(other)))
        if "VirmenData" not in source_data:
            raise ValueError("VirmenData is required: the converter writes a ViRMEn behavior session")
        super().__init__(source_data=source_data, verbose=verbose)
