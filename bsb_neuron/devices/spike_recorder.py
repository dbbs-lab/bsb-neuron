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
            for target in pop:
                for location in self.locations.get_locations(target):
                    self._add_spike_recorder(
                        simdata.result,
                        target,
                        location,
                        adapter.next_gid,
                        name=self.name,
                        cell_type=target.cell_model.name,
                        cell_id=target.id,
                    )
                    adapter.next_gid += 1

    def _add_spike_recorder(self, results, cell, location, gid, **annotations):
        gid = cell.insert_transmitter(gid, location).gid
        results.record_spike(p.parallel.spike_record(gids=gid), **annotations)
