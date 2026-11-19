"""Custom KiloSort interface that adds electrode group information for multi-probe support."""

from neuroconv.datainterfaces import KiloSortSortingInterface


class KiloSortWithProbeInterface(KiloSortSortingInterface):
    """
    Custom KiloSort interface that properly handles electrode group assignment.

    This subclass extends KiloSortSortingInterface to add electrode_group information
    to units, enabling identification of which probe each unit came from in multi-probe
    recordings.

    Parameters
    ----------
    folder_path : str
        Path to the Kilosort output folder
    electrode_group_name : str, optional
        Name of the electrode group (probe) these units belong to
    **kwargs : dict
        Additional keyword arguments passed to KiloSortSortingInterface
    """

    # Override the ExtractorName to use the parent's extractor, not a custom one
    ExtractorName = "KiloSortSortingExtractor"

    def __init__(self, folder_path: str, electrode_group_name: str = None, **kwargs):
        # Store electrode_group_name before passing to parent
        self.electrode_group_name = electrode_group_name

        # Remove electrode_group_name from kwargs if it exists (shouldn't be passed to parent)
        kwargs.pop("electrode_group_name", None)

        # Call parent class __init__
        super().__init__(folder_path=folder_path, **kwargs)

    def add_to_nwbfile(self, nwbfile, metadata, **kwargs):
        """
        Add units to the main nwbfile.units table with probe identification.

        All units from all probes are added to nwbfile.units with a probe_id column
        identifying which probe they came from. This is the standard NWB approach
        for multi-probe recordings.
        """
        import numpy as np
        
        # Get all units from the sorting extractor first (we need this info to set up columns)
        sorting = self.sorting_extractor
        unit_ids = sorting.get_unit_ids()
        property_keys = sorting.get_property_keys()
        
        # Determine probe identifier for this interface
        if self.electrode_group_name is None:
            probe_identifier = "probe0"
        else:
            # Extract probe number from electrode_group_name (e.g., "IMEC0" -> "0")
            probe_identifier = self.electrode_group_name.replace("IMEC", "")
        
        # Get electrode group object if available
        electrode_group_obj = None
        if self.electrode_group_name is not None and self.electrode_group_name in nwbfile.electrode_groups:
            electrode_group_obj = nwbfile.electrode_groups[self.electrode_group_name]
        
        # Check if units table exists and has units already
        units_table = nwbfile.units
        table_has_units = units_table is not None and len(units_table) > 0
        
        # Add columns if they don't exist (only safe to do before adding units or if table exists)
        if not table_has_units:
            # Safe to add columns - no units yet
            if units_table is None or 'probe_id' not in units_table.colnames:
                nwbfile.add_unit_column(
                    name='probe_id',
                    description='Identifier for the probe/electrode group this unit was recorded from'
                )
            
            # Add columns for Kilosort properties
            for prop_name in property_keys:
                if units_table is None or prop_name not in units_table.colnames:
                    # Determine description based on common Kilosort properties
                    if prop_name == 'group':
                        description = 'Electrode group (shank) ID from Kilosort'
                    elif prop_name == 'quality':
                        description = 'Unit quality label from Kilosort'
                    elif prop_name == 'ch':
                        description = 'Peak channel ID from Kilosort'
                    else:
                        description = f'{prop_name} property from Kilosort'
                    
                    nwbfile.add_unit_column(
                        name=prop_name,
                        description=description
                    )
            
            # Add electrode_group column if we have an electrode group
            if electrode_group_obj is not None:
                if units_table is None or 'electrode_group' not in units_table.colnames:
                    nwbfile.add_unit_column(
                        name='electrode_group',
                        description='Electrode group (probe) this unit belongs to'
                    )
        
        # Add all units (use nwbfile.add_unit, not units_table.add_unit)
        for unit_idx, unit_id in enumerate(unit_ids):
            # Get spike times for this unit
            spike_times = sorting.get_unit_spike_train(unit_id=unit_id)
            
            # Convert spike indices to timestamps if needed
            if hasattr(sorting, 'get_sampling_frequency'):
                sampling_frequency = sorting.get_sampling_frequency()
                spike_times_sec = spike_times / sampling_frequency
            else:
                spike_times_sec = spike_times
            
            # Get all properties for this unit, starting with probe_id
            unit_properties = {'probe_id': probe_identifier}
            
            for prop_name in property_keys:
                prop_value = sorting.get_property(prop_name)[unit_idx]
                # Convert numpy types to Python native types
                if hasattr(prop_value, 'item'):
                    prop_value = prop_value.item()
                unit_properties[prop_name] = prop_value
            
            # Add electrode group if available
            if electrode_group_obj is not None:
                unit_properties['electrode_group'] = electrode_group_obj
            
            # Add the unit to the table using nwbfile.add_unit
            nwbfile.add_unit(
                spike_times=spike_times_sec,
                id=int(unit_id),
                **unit_properties
            )
        
        print(f"Added {len(unit_ids)} units from {self.electrode_group_name or 'probe'} to nwbfile.units (probe_id={probe_identifier})")
