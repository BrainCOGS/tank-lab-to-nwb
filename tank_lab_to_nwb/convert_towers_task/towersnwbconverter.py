"""Authors: Cody Baker and Ben Dichter."""

import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import spikeinterface.extractors as se

# from neuroconv import NWBConverter, SpikeGLXRecordingInterface, SpikeGLXLFPInterface
from neuroconv import NWBConverter
from neuroconv.datainterfaces import (
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
    - TiffImaging (optional): Raw imaging data
    """

    data_interface_classes = {
        "SpikeGLXAP": SpikeGLXRecordingInterface,
        "SpikeGLXLFP": SpikeGLXRecordingInterface,
        "VirmenData": VirmenDataInterface,
        "Kilosort": KiloSortWithProbeInterface,  # Use custom interface for electrode group support
        "Suite2pSegmentation": Suite2pSegmentationInterface,
        "TiffImagaging": TiffImagingInterface,
    }

    def __init__(
        self, source_data, ttl_source: Optional[PathType] = None, sync_timestamps: Optional[np.ndarray] = None
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
            If provided, these will be applied to all temporal interfaces via set_aligned_timestamps.
            This is the preferred method for U19 pipeline integration.
        """
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

        super().__init__(source_data=source_data)

        # Check that VirmenData interface is present
        if "VirmenData" not in self.data_interface_objects:
            raise ValueError("VirmenData interface is required but not found in source_data.")

        # Store sync timestamps for later use in temporally_align_data_interfaces
        self._sync_timestamps = sync_timestamps

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
        elif sync_timestamps is None and ttl_source is None:
            warnings.warn(
                "No sync_timestamps or ttl_source provided. "
                "Timestamps will not be aligned across interfaces. "
                "For proper synchronization, provide either sync_timestamps from DataJoint or ttl_source."
            )

    def temporally_align_data_interfaces(
        self, metadata: Optional[dict] = None, conversion_options: Optional[dict] = None
    ):
        """
        Apply synchronized timestamps to all temporal interfaces.

        If sync_timestamps were provided during initialization, apply them to all
        interfaces that support temporal alignment (those with set_aligned_timestamps method).
        Skips missing optional interfaces gracefully.

        Parameters
        ----------
        metadata : dict, optional
            Metadata dictionary (required by base class but not used here)
        conversion_options : dict, optional
            Conversion options (required by base class but not used here)
        """
        if self._sync_timestamps is None:
            if self.verbose:
                print("No synchronized timestamps available. Skipping temporal alignment.")
            return

        # List of interfaces that support temporal alignment
        # Note: Kilosort requires a recording to be registered first, so it's excluded from automatic alignment
        temporal_interfaces = ["VirmenData", "SpikeGLXAP", "SpikeGLXLFP", "Suite2pSegmentation"]

        for interface_name in temporal_interfaces:
            if interface_name in self.data_interface_objects:
                interface = self.data_interface_objects[interface_name]
                if hasattr(interface, "set_aligned_timestamps"):
                    try:
                        interface.set_aligned_timestamps(self._sync_timestamps)
                        if self.verbose:
                            print(f"Applied synchronized timestamps to {interface_name}")
                    except Exception as e:
                        warnings.warn(f"Failed to apply synchronized timestamps to {interface_name}: {e}")
            else:
                if self.verbose:
                    print(f"Optional interface {interface_name} not provided, skipping temporal alignment for it.")

    def get_metadata(self):
        vermin_file_path = Path(self.data_interface_objects["VirmenData"].source_data["file_path"])
        session_id = vermin_file_path.stem

        metadata = super().get_metadata()
        metadata["NWBFile"].update(session_id=session_id, institution="Princeton", lab="Tank")

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

