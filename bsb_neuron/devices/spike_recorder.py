import numpy as np
from bsb import LocationTargetting, config
from patch import p
from patch.objects import Vector

from ..device import NeuronDevice


@config.node
class SpikeRecorder(NeuronDevice, classmap_entry="spike_recorder"):
    """
    Device to record the spike events in selected locations.

    :param location: The LocationTargetting chosen to select location on cells, default selects "soma".
    :type LocatioTargetting: ~bsb.simulation.targetting.LocationTargetting

    :param join_population: If set to True, a SpikeTrain object will be created for each cell population; if set to False,
     a SpikeTrain will be stored for each individual location. The default value is False.
    :type bool
    """

    locations = config.attr(type=LocationTargetting, default={"strategy": "soma"})
    join_population = config.attr(type=bool, default=False)

    def implement(self, adapter, simulation, simdata):
        for model, pop in self.targetting.get_targets(
            adapter, simulation, simdata
        ).items():
            if self.join_population:
                spike_times = p.parallel.Vector
                neuron_gids = p.parallel.Vector
                gids_to_cell = {}
                gids_to_labels = {}
                gids_to_locs = {}
                for target in pop:
                    for location in self.locations.get_locations(target):

                        gid = target.check_netcon(location, adapter)
                        # Call record_spike() method on selected gid using common spike_times and neuron_gids Vector for
                        # cells in the same population
                        gids_to_cell[gid] = target.id
                        gids_to_labels[gid] = location.section.labels
                        gids_to_locs[gid] = location._loc
                        spike_times, neuron_gids = p.parallel.spike_record(
                            gid, spike_times, neuron_gids
                        )
                # If join_population is selected Record a SpikeTrain obj for every model
                self._add_spike_recorder(
                    simdata.result,
                    spike_times,
                    neuron_gids,
                    gids_to_cell,
                    gids_to_labels,
                    gids_to_locs,
                    device=self.name,
                    t_stop=simulation.duration,
                    cell_type=target.cell_model.name,
                    pop_size=len(pop),
                )
            else:  # We are splitting the outputs
                for target in pop:
                    for location in self.locations.get_locations(target):
                        gid = target.check_netcon(location, adapter)

                        # Call record_spike() method on selected gid, it will build a SpikeTrain obj for every location
                        spike_times, neuron_gids = p.parallel.spike_record(gid)
                        self._add_spike_recorder(
                            simdata.result,
                            spike_times,
                            neuron_gids,
                            device=self.name,
                            t_stop=simulation.duration,
                            cell_type=target.cell_model.name,
                            cell_id=target.id,
                            labels=location.section.labels,
                            loc=location._loc,
                            pop_size=len(pop),
                        )

    def _add_spike_recorder(
        self,
        results,
        spike_times,
        gids,
        cell_dict=None,
        labels_dict=None,
        locs_dict=None,
        **annotations
    ):
        results.record_spike(
            spike_times, gids, cell_dict, labels_dict, locs_dict, **annotations
        )
