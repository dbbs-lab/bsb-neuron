from bsb import LocationTargetting, config

from ..device import NeuronDevice


@config.node
class MembraneCurrentRecorder(NeuronDevice, classmap_entry="membrane_current_recorder"):
    locations = config.attr(type=LocationTargetting, default={"strategy": "everywhere"})

    def implement(self, adapter, simulation, simdata):
        for model, pop in self.targetting.get_targets(
            adapter, simulation, simdata
        ).items():
            for target in pop:
                for location in self.locations.get_locations(target):
                    self._add_imem_recorder(
                        simdata.result,
                        location,
                        name=self.name,
                        cell_type=target.cell_model.name,
                        cell_id=target.id,
                        loc=location._loc,
                    )

    def _get_imem(self, location):
        from patch import p

        section = location.section
        x = location.arc(0)
        return p.record(section(x).__record_imem__())

    def _add_imem_recorder(self, results, location, **annotations):
        section = location.section
        x = location.arc(0)
        results.record(section(x).__record_imem__(), **annotations, units="nA")
