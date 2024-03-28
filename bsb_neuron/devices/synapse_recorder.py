from bsb import LocationTargetting, config

from ..device import NeuronDevice


@config.node
class SynapseRecorder(NeuronDevice, classmap_entry="synapse_recorder"):
    locations = config.attr(type=LocationTargetting, required=True)
    synapse_types = config.list()

    def implement(self, adapter, simulation, simdata):
        for model, pop in self.targetting.get_targets(
            adapter, simulation, simdata
        ).items():
            for target in pop:
                for location in self.locations.get_locations(target):
                    for synapse in location.section.synapses:
                        if (
                            not self.synapse_types
                            or synapse.synapse_name in self.synapse_types
                        ):
                            _record_synaptic_current(simdata.result, synapse)


def _record_synaptic_current(result, synapse):
    result.record(synapse._pp._ref_i)
