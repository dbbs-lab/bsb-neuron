import numpy as np
from bsb import LocationTargetting, config
from patch import p

from ..device import NeuronDevice


@config.node
class SpikeRecorder(NeuronDevice, classmap_entry="spike_recorder"):
    locations = config.attr(type=LocationTargetting, default={"strategy": "soma"})

    def implement(self, adapter, simulation, simdata):
        for model, pop in self.targetting.get_targets(
            adapter, simulation, simdata
        ).items():
            spike_times = p.parallel._interpreter.Vector()
            neuron_gids = p.parallel._interpreter.Vector()
            gids_to_cell = {}
            for target in pop:
                locations = [
                    location._loc for location in self.locations.get_locations(target)
                ]
                for location in locations:
                    # Insert a NetCon (if not already present) and retrieve its gid
                    la = target.get_location(location)
                    if hasattr(la.section, "_transmitter"):
                        gid = la.section._transmitter.gid
                    else:
                        gid = target.insert_transmitter(adapter.next_gid, location).gid
                        adapter.next_gid += 1
                    gids_to_cell[gid] = target.id

                    # Call record_spike() method on selected gid using common spike_times and neuron_gids Vector for
                    # cells in the same population
                    spike_times, neuron_gids = p.parallel.spike_record(
                        gid, spike_times, neuron_gids
                    )
            # Record a SpikeTrain obj for every model
            self._add_spike_recorder(
                simdata.result,
                spike_times,
                neuron_gids,
                gids_to_cell,
                device=self.name,
                t_stop=simulation.duration,
                cell_type=target.cell_model.name,
                cell_id=target.id,
                pop_size=len(pop),
            )

    def _add_spike_recorder(self, results, spike_times, gids, cell_dict, **annotations):
        results.record_spike(spike_times, gids, cell_dict, **annotations)
