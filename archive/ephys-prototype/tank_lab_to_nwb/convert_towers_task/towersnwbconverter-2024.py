"""Authors: Christian Tabedzki"""

from pathlib import Path

# import spikeinterface as si
# import spikeinterface
from neuroconv import NWBConverter
from neuroconv.converters import SpikeGLXConverterPipe
from neuroconv.datainterfaces import Suite2pSegmentationInterface, TiffImagingInterface

from tank_lab_to_nwb.convert_towers_task.virmenbehaviordatainterface import VirmenDataInterface
from tank_lab_to_nwb.utils import convert_mat_file_to_dict


class TowersNWBConverter(NWBConverter):
    """Main conversion class for the Brody/Tank Lab Towers Task data"""

    data_interface_classes = {
        "SpikeGLXConverter": SpikeGLXConverterPipe,
        "VirmenData": VirmenDataInterface,
        "Suite2pSegmentation": Suite2pSegmentationInterface,
        "TiffImagaging": TiffImagingInterface,
    }

    def __init__(self, source_data, ttl_source: Path | str):
        """
        Doc string here
        """
        super().__init__(source_data=source_data)

        interface_name = ["SpikeGLXConverter", "SpikeGLXRecording"]
        # if self.data_interface_objects.get(interface_name, None)
        if interface_name in self.data_interface_objects:
            interface_extractor = self.data_interface_objects[interface_name]
        else:
            raise KeyError(
                "SpikeGLXConverter or SpikeGLXRecording must be provided in order to properly sync the data."
            )

        vdi = "VirmenDataInterface"
        if vdi not in self.data_interface_objects:
            raise KeyError(f"'{vdi}' must be included in 'source_data'")

    def get_metadata(self):
        vermin_file_path = Path(self.data_interface_objects["VirmenData"].source_data["file_path"])
        session_id = vermin_file_path.stem

        # This will automatically trace through the defined interfaces
        # and collate the metadata from each individual interface.
        # No need to manually call interfaces
        metadata = super().get_metadata()
        metadata["NWBFile"].update(session_id=session_id, institution="Princeton Neuroscience Institute", lab="Tank")

        # TODO: Implement calls to DataJoint that fetches some key data based off the current metadata

        # Get the metadata from VirmenDataInterface class
        if VirmenDataInterface in self.data_interface_objects:
            virmen_obj = self.data_interface_objects[VirmenDataInterface]

        self.data_interface_objects[VirmenDataInterface]

        if vermin_file_path.is_file():
            session_data = convert_mat_file_to_dict(mat_file_name=vermin_file_path)
            subject_data = session_data["log"]["animal"]
            metadata.update(Subject={"subject_id": subject_data["name"]})
        else:
            print(f"Warning: no subject file detected for session {session_id}!")

        return metadata

    def temporally_align_data_interfaces(self):
        """Funciton to align and synchronize the various data classes."""
        # "SpikeGLXConverter": SpikeGLXConverterPipe,
        # "VirmenData": VirmenDataInterface,
        # "Suite2pSegmentation": Suite2pSegmentationInterface,
        # "TiffImagaging": TiffImagingInterface,

        behavior_interface = self.data_interface_objects["VirmenData"]
        spikeglx = self.data_interface_objects["SpikeGLXConverter"]

        # From here, you will want to query the datajoint table and get the nidq times for this given session

        nidq_times

        # behavior_annotations_interface.align_by_interpolation(
        #     unaligned_timestamps=camera_ttl_sent_times,
        #     aligned_timestamps=acquisition_system_ttl_received_times,
        # )
        pass

    # def temporally_align_data_interfaces(self):
    #     spikeglx_interface: SpikeGLXConverterPipe = self.data_interface_objects["SpikeGLXConverter"]
    #     nidq_interface: SpikeGLXNIDQInterface = spikeglx_interface.data_interface_objects["nidq"]

    #     nidq_bin_path = nidq_interface.source_data["file_path"]

    #     # Using spikeinterface.extractors break the data into the constituent bits
    #     # NeuroConv might update this code to extract and write the bits for us down the line.

    #     event = spikeinterface.extractors.read_spikeglx_event(nidq_bin_path)

    #     nidq_start_trial_pulses = event.get_event_times(1)
    #     nidq_start_frame_pulses = event.get_event_times(2)

    #     pulses = [nidq_start_trial_pulses, nidq_start_frame_pulses]
    #     for pulse_type in pulses:
    #         if not len(pulse_type):
    #             raise ValueError(f"{len(pulse_type)=}. Please verify the {nidq_bin_path} contains valid data.")

    #     # Synchronize the Virmen data
    #     virmen_interface: VirmenDataInterface = self.data_interface_objects["VirmenData"]

    #     virmen_interface.set_aligned_timestamps(nidq_start_trial_pulses)

    #     pass
