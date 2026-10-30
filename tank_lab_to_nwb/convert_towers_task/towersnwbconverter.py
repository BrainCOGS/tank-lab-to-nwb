"""Authors: Cody Baker and Ben Dichter."""

import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import spikeinterface.extractors as se

# from neuroconv import NWBConverter, SpikeGLXRecordingInterface, SpikeGLXLFPInterface
from neuroconv import NWBConverter
from neuroconv.datainterfaces import (
    ScanImageImagingInterface,
    SpikeGLXRecordingInterface,
    Suite2pSegmentationInterface,
    TiffImagingInterface,
)

from ..utils import convert_mat_file_to_dict
from .virmenbehaviordatainterface import VirmenDataInterface
from .kilosortinterface import KiloSortWithProbeInterface

OptionalArrayType = Optional[Union[list, np.ndarray]]
PathType = Union[Path, str]


class TowersNWBConverter(NWBConverter):
    """Primary conversion class for the Tank lab Towers task processing pipeline.

    This converter supports multiple optional data modalities:
    - VirmenData (required): Behavioral data from Virmen
    - SpikeGLXAP/SpikeGLXLFP (optional): Electrophysiology recordings
    - Kilosort (optional): Spike sorting data
    - Suite2pSegmentation (optional): Calcium imaging segmentation
    - ScanImageImaging (optional): Raw ScanImage two-photon data. For multiple
      fields of view use "ScanImageImagingFOV0", "ScanImageImagingFOV1", etc.
    - TiffImaging (optional): Raw imaging data from a generic TIFF
    """

    data_interface_classes = {
        "SpikeGLXAP": SpikeGLXRecordingInterface,
        "SpikeGLXLFP": SpikeGLXRecordingInterface,
        "VirmenData": VirmenDataInterface,
        "Kilosort": KiloSortWithProbeInterface,  # Use custom interface for electrode group support
        "Suite2pSegmentation": Suite2pSegmentationInterface,
        "ScanImageImaging": ScanImageImagingInterface,
        # Kept for backwards compatibility with existing source_data dicts.
        # ScanImage BigTIFFs belong on ScanImageImagingInterface above; the
        # generic TiffImagingInterface cannot read their volumetric fastZ
        # layout or their per-frame headers.
        "TiffImaging": TiffImagingInterface,
    }

    def __init__(
        self,
        source_data,
        ttl_source: Optional[PathType] = None,
        sync_timestamps: Optional[np.ndarray] = None,
        aligned_timestamps: Optional[dict] = None,
        sample_ranges: Optional[dict] = None,
    ):
        """
        Initialize the NWBConverter object.

        Parameters
        ----------
        source_data : dict
            Dictionary mapping interface names to their source data dictionaries.
            VirmenData is required; all other interfaces are optional.
            For multiple Kilosort probes, use names like "KilosortProbe0", "KilosortProbe1", etc.
        ttl_source : PathType, optional
            Path to data file containing the TTL signals to use for basic synchronization.
            Used for simple TTL-based alignment. If sync_timestamps is provided, this is ignored.
        sync_timestamps : np.ndarray, optional
            Pre-computed synchronized timestamps (e.g., from DataJoint BehaviorSync table).
            Applied to every temporal interface that does not have its own entry in
            aligned_timestamps. Suitable only for interfaces that share a sample count.
        aligned_timestamps : dict, optional
            Per-interface timestamp arrays, keyed by interface name, e.g.
            {"ScanImageImaging": np.ndarray}. Takes precedence over
            sync_timestamps for the interfaces it names.

            This exists because interfaces do not share a sample count: an imaging
            interface has one timestamp per frame (or per volume, for fastZ stacks)
            while behavior has one per ViRMEn iteration. Applying a single array to
            both silently mistimes whichever one it does not describe.

            All arrays must be on the same clock. For U19 that is the ViRMEn behavior
            clock, zeroed at log.session.start — note that vr.timeElapsed
            quantities are zeroed at *block* start and need the block-vs-session
            offset added first. See docs/imaging_behavior_sync.md section 6 in
            U19-pipeline-python.
        sample_ranges : dict, optional
            Per-interface half-open (start, stop) sample ranges, keyed by
            interface name, e.g. {"ScanImageImagingFOV0Plane0": (548, 17_800)}.
            Each named imaging interface is cut to that range before anything is
            written, so frames outside it (imaging recorded before or after the
            behavior) are left out of the file. Any aligned_timestamps given
            for the same interface must already cover exactly that range.
        """
        # Copy onto the instance before registering anything dynamic:
        # data_interface_classes is a class attribute, so mutating it in place
        # leaked every session's probes and fields of view into the next
        # converter built in the same process.
        self.data_interface_classes = dict(self.data_interface_classes)

        # Dynamically add Kilosort interfaces for multiple probes
        # Check for any keys starting with "Kilosort" (e.g., "KilosortProbe0", "KilosortProbe1")
        for key in source_data.keys():
            if key.startswith("Kilosort") and key not in self.data_interface_classes:
                self.data_interface_classes[key] = KiloSortWithProbeInterface
                
                # Add electrode_group_name to source_data for this interface
                if key == "Kilosort":
                    electrode_group_name = "IMEC"
                else:
                    probe_id = key.replace("KilosortProbe", "")
                    electrode_group_name = f"IMEC{probe_id}"
                
                source_data[key]["electrode_group_name"] = electrode_group_name

        # Same for imaging: a mesoscope session has one field of view per
        # TiffSplit ("ScanImageImagingFOV0", "ScanImageImagingFOV1", ...). Each
        # is a distinct region and gets its own interface and TwoPhotonSeries --
        # concatenating them into one interface would present unrelated fields
        # of view as a single continuous recording.
        for key in source_data.keys():
            if key.startswith("ScanImageImaging") and key not in self.data_interface_classes:
                self.data_interface_classes[key] = ScanImageImagingInterface

        # Every ScanImage interface needs its own metadata_key. neuroconv's
        # default depends only on channel and plane, so two fields of view with
        # the same plane share one key and write the same TwoPhotonSeries name,
        # which pynwb rejects. Default the key to the interface name, which is
        # unique by construction. Copy rather than mutate the caller's dict.
        source_data = {
            key: (
                {**value, "metadata_key": value.get("metadata_key") or key}
                if key.startswith("ScanImageImaging")
                else value
            )
            for key, value in source_data.items()
        }

        super().__init__(source_data=source_data)

        for name, (start, stop) in (sample_ranges or {}).items():
            if name not in self.data_interface_objects:
                raise ValueError(
                    f"sample_ranges names an interface that is not in source_data: {name!r}."
                )
            interface = self.data_interface_objects[name]
            n_samples = interface.imaging_extractor.get_num_samples()
            if not 0 <= start < stop <= n_samples:
                raise ValueError(
                    f"sample_ranges[{name!r}] = ({start}, {stop}) is outside the "
                    f"interface's {n_samples} samples."
                )
            parent = interface.imaging_extractor
            sliced = parent.slice_samples(start_sample=start, end_sample=stop)
            # ScanImageImagingInterface reads these straight off its extractor
            # (session start time from file_path, header metadata from
            # _general_metadata); the sliced wrapper does not forward them.
            for attr in ("file_path", "_general_metadata"):
                if hasattr(parent, attr):
                    setattr(sliced, attr, getattr(parent, attr))
            interface.imaging_extractor = sliced

        # Check that VirmenData interface is present
        if "VirmenData" not in self.data_interface_objects:
            raise ValueError("VirmenData interface is required but not found in source_data.")

        # Store sync timestamps for later use in temporally_align_data_interfaces
        self._sync_timestamps = sync_timestamps
        self._aligned_timestamps = dict(aligned_timestamps) if aligned_timestamps else {}

        unknown = set(self._aligned_timestamps) - set(self.data_interface_objects)
        if unknown:
            raise ValueError(
                f"aligned_timestamps names interfaces that are not in source_data: "
                f"{sorted(unknown)}. Available: {sorted(self.data_interface_objects)}."
            )

        # Only perform TTL-based alignment if no pre-computed sync timestamps provided
        if sync_timestamps is None and ttl_source is not None:
            recording = se.SpikeGLXRecordingExtractor(ttl_source)
            ttl, states = recording.get_ttl_events()
            rising_times = ttl[states == 1]

            assert len(rising_times) > 0, f"No TTL events found in ttl_source file ({ttl_source})."
            start_time = recording.frame_to_time(rising_times[0])

            for interface_name in ["SpikeGLXRecording", "SpikeGLXLFP"]:
                if interface_name in self.data_interface_objects:  # specified in source_data
                    interface_extractor = self.data_interface_objects[interface_name].recording_extractor
                    re_start_frame = int(interface_extractor.time_to_frame(start_time))
                    self.data_interface_objects[interface_name].recording_extractor = se.SubRecordingExtractor(
                        parent_recording=interface_extractor, start_frame=re_start_frame
                    )
        elif sync_timestamps is None and ttl_source is None and not self._aligned_timestamps:
            warnings.warn(
                "No sync_timestamps or ttl_source provided. "
                "Timestamps will not be aligned across interfaces. "
                "For proper synchronization, provide either sync_timestamps from DataJoint or ttl_source."
            )

    def temporally_align_data_interfaces(
        self, metadata: Optional[dict] = None, conversion_options: Optional[dict] = None
    ):
        """
        Apply aligned timestamps to the temporal interfaces.

        Resolution order per interface:
          1. an explicit array from aligned_timestamps[interface_name]
          2. otherwise sync_timestamps, but only for interfaces whose sample
             count matches it

        The sample-count guard is the point of this method. sync_timestamps
        describes the behavior stream; an imaging interface reports a different
        number of samples (one per frame, or one per volume for fastZ stacks).
        Applying the behavior array to it does not raise — it just writes an
        imaging series with the wrong times. So we check, skip, and say why.

        Parameters
        ----------
        metadata : dict, optional
            Metadata dictionary (required by the base class signature; unused here)
        conversion_options : dict, optional
            Conversion options (required by the base class signature; unused here)
        """
        if self._sync_timestamps is None and not self._aligned_timestamps:
            if self.verbose:
                print("No synchronized timestamps available. Skipping temporal alignment.")
            return

        # Every interface is a candidate except Kilosort, which needs a
        # registered recording before its timestamps mean anything. Considering
        # them all is deliberate: an interface left out of this list would take
        # no alignment and emit no warning, which is how imaging would silently
        # keep its raw acquisition-clock timestamps in a behavior-clock file.
        candidates = {
            name for name in self.data_interface_objects if not name.startswith("Kilosort")
        } | set(self._aligned_timestamps)

        for interface_name in sorted(candidates):
            if interface_name not in self.data_interface_objects:
                if self.verbose:
                    print(f"Optional interface {interface_name} not provided, skipping temporal alignment for it.")
                continue

            interface = self.data_interface_objects[interface_name]
            if not hasattr(interface, "set_aligned_timestamps"):
                continue

            timestamps = self._aligned_timestamps.get(interface_name)
            explicit = timestamps is not None
            if not explicit:
                timestamps = self._sync_timestamps
            if timestamps is None:
                continue
            timestamps = np.asarray(timestamps)

            if not explicit:
                # Only broadcast the shared array where the shape actually fits.
                try:
                    n_expected = np.size(interface.get_original_timestamps())
                except Exception as e:  # interface cannot report its own length
                    warnings.warn(
                        f"Could not determine the sample count for {interface_name} "
                        f"({e}); applying sync_timestamps unchecked."
                    )
                    n_expected = timestamps.size
                if n_expected != timestamps.size:
                    warnings.warn(
                        f"Skipping temporal alignment for {interface_name}: it has "
                        f"{n_expected} samples but sync_timestamps has {timestamps.size}. "
                        f"Pass an explicit array via aligned_timestamps['{interface_name}'] "
                        f"instead of relying on the shared sync_timestamps array."
                    )
                    continue

            try:
                interface.set_aligned_timestamps(timestamps)
                if self.verbose:
                    source = "aligned_timestamps" if explicit else "sync_timestamps"
                    print(f"Applied {timestamps.size} timestamps to {interface_name} (from {source})")
            except Exception as e:
                warnings.warn(f"Failed to apply synchronized timestamps to {interface_name}: {e}")

    def get_metadata(self):
        vermin_file_path = Path(self.data_interface_objects["VirmenData"].source_data["file_path"])
        session_id = vermin_file_path.stem

        metadata = super().get_metadata()
        metadata["NWBFile"].update(session_id=session_id, institution="Princeton", lab="Tank")

        # Name each imaging interface's objects after its key, so
        # "ScanImageImagingFOV1Plane2" writes TwoPhotonSeriesFOV1Plane2 on
        # ImagingPlaneFOV1Plane2. neuroconv names them by plane only
        # (TwoPhotonSeriesPlane2), which collides across fields of view.
        ophys = metadata.get("Ophys", {})
        for name, interface in self.data_interface_objects.items():
            suffix = name[len("ScanImageImaging"):] if name.startswith("ScanImageImaging") else ""
            if not suffix:
                continue
            key = interface.metadata_key
            if key in ophys.get("ImagingPlanes", {}):
                ophys["ImagingPlanes"][key]["name"] = f"ImagingPlane{suffix}"
            if key in ophys.get("MicroscopySeries", {}):
                ophys["MicroscopySeries"][key]["name"] = f"TwoPhotonSeries{suffix}"

        if vermin_file_path.is_file():
            session_data = convert_mat_file_to_dict(mat_file_name=vermin_file_path)
            subject_data = session_data["log"]["animal"]
            metadata.update(Subject=dict(subject_id=subject_data["name"]))
        else:
            print(f"Warning: no subject file detected for session {session_id}!")

        # Configure electrode groups for multi-probe Kilosort data
        # This ensures each probe has a distinct electrode group in the NWB file
        kilosort_interfaces = [name for name in self.data_interface_objects.keys() if name.startswith("Kilosort")]
        
        if kilosort_interfaces:
            if "Ecephys" not in metadata:
                metadata["Ecephys"] = {}
            
            # Add devices and electrode groups for each Kilosort probe
            for interface_name in kilosort_interfaces:
                # Extract probe ID from interface name (e.g., "KilosortProbe0" -> "0")
                if interface_name == "Kilosort":
                    probe_id = ""  # Single probe case
                    group_name = "IMEC"
                    device_name = "Neuropixels"
                else:
                    probe_id = interface_name.replace("KilosortProbe", "")
                    group_name = f"IMEC{probe_id}"
                    device_name = f"Neuropixels-IMEC{probe_id}"
                
                # Add device metadata for this probe
                if "Device" not in metadata["Ecephys"]:
                    metadata["Ecephys"]["Device"] = []
                
                if not any(d.get("name") == device_name for d in metadata["Ecephys"]["Device"]):
                    metadata["Ecephys"]["Device"].append({
                        "name": device_name,
                        "description": f"Neuropixels probe{' IMEC' + probe_id if probe_id else ''}",
                        "manufacturer": "IMEC"
                    })
                
                # Add electrode group metadata for this probe
                if "ElectrodeGroup" not in metadata["Ecephys"]:
                    metadata["Ecephys"]["ElectrodeGroup"] = []
                
                if not any(eg.get("name") == group_name for eg in metadata["Ecephys"]["ElectrodeGroup"]):
                    metadata["Ecephys"]["ElectrodeGroup"].append({
                        "name": group_name,
                        "description": f"Electrodes from Neuropixels probe{' IMEC' + probe_id if probe_id else ''}",
                        "location": "brain",
                        "device": device_name
                    })
                
                # Set the electrode group name in the interface metadata
                if interface_name not in metadata:
                    metadata[interface_name] = {}
                metadata[interface_name]["electrode_group_name"] = group_name

        return metadata

    def add_to_nwbfile(self, nwbfile, metadata, conversion_options=None):
        """
        Override add_to_nwbfile to properly set up electrode groups for multi-probe Kilosort.
        
        This method ensures that:
        1. Devices are created for each probe
        2. Electrode groups are created and linked to devices
        3. Ecephys processing module is created before Kilosort interfaces add units
        4. Each probe gets its own units table in processing/ecephys
        """
        # First, add devices and electrode groups from metadata
        from pynwb.device import Device
        from pynwb.ecephys import ElectrodeGroup
        
        if "Ecephys" in metadata:
            # Add devices
            if "Device" in metadata["Ecephys"]:
                for device_meta in metadata["Ecephys"]["Device"]:
                    if device_meta["name"] not in nwbfile.devices:
                        device = Device(
                            name=device_meta["name"],
                            description=device_meta.get("description", ""),
                            manufacturer=device_meta.get("manufacturer", "")
                        )
                        nwbfile.add_device(device)
            
            # Add electrode groups
            if "ElectrodeGroup" in metadata["Ecephys"]:
                for eg_meta in metadata["Ecephys"]["ElectrodeGroup"]:
                    if eg_meta["name"] not in nwbfile.electrode_groups:
                        device = nwbfile.devices[eg_meta["device"]]
                        electrode_group = ElectrodeGroup(
                            name=eg_meta["name"],
                            description=eg_meta.get("description", ""),
                            location=eg_meta.get("location", "unknown"),
                            device=device
                        )
                        nwbfile.add_electrode_group(electrode_group)
        
        # Create ecephys processing module for Kilosort units tables
        # This must be done before calling super() so interfaces can add to it
        kilosort_interfaces = [name for name in self.data_interface_objects.keys() if name.startswith("Kilosort")]
        if kilosort_interfaces and 'ecephys' not in nwbfile.processing:
            nwbfile.create_processing_module(
                name='ecephys',
                description='Electrophysiology processing module containing spike sorting results from multiple probes'
            )
        
        # Call parent add_to_nwbfile to add all interface data
        super().add_to_nwbfile(nwbfile=nwbfile, metadata=metadata, conversion_options=conversion_options)

        # Persist behavioral_start_offset for traceability when provided via metadata.
        # The LabMetaData extension has no field for it, so it is stored in scratch.
        # This value is the delay (seconds) of the first behavioral frame relative to
        # IMEC recording start (t=0); all timestamps in the file are already IMEC-relative.
        behavioral_start_offset = (metadata or {}).get("LabMetaData", {}).get("behavioral_start_offset")
        if behavioral_start_offset is not None and "behavioral_start_offset" not in nwbfile.scratch:
            nwbfile.add_scratch(
                np.float64(behavioral_start_offset),
                name="behavioral_start_offset",
                description=(
                    "Delay in seconds of the first behavioral frame relative to IMEC "
                    "recording start (t=0). All timestamps in this file are IMEC-relative; "
                    "this is informational only and equals the first behavioral timestamp."
                ),
            )

