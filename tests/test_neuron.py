import importlib
import itertools
import traceback
import unittest
from copy import copy

from arborize import define_model
from bsb.core import Scaffold
from bsb.services import MPI
from bsb.simulation import get_simulation_adapter
from bsb_test import (
    ConfigFixture,
    MorphologiesFixture,
    NetworkFixture,
    RandomStorageFixture,
)

from bsb_neuron.cell import ArborizedModel
from bsb_neuron.connection import TransceiverModel


def neuron_installed():
    return importlib.util.find_spec("neuron")


@unittest.skipIf(not neuron_installed(), "NEURON is not installed")
class TestNeuronMinimal(
    RandomStorageFixture,
    ConfigFixture,
    NetworkFixture,
    MorphologiesFixture,
    unittest.TestCase,
    config="neuron_minimal",
    morpho_filters=["2comp"],
    engine_name="hdf5",
):
    def setUp(self):
        super().setUp()
        self.network.compile()

    def test_minimal(self):
        from neuron import h

        sim = self.network.simulations.test
        self.network.run_simulation("test")
        self.assertAlmostEqual(h.t, sim.duration, msg="sim duration incorrect")

    def test_double_sim_minimal(self):
        from neuron import h

        scaffold_copy = Scaffold(copy(self.cfg), self.storage)
        sim = self.network.simulations.test
        sim2 = scaffold_copy.simulations.test
        sim2.duration *= 2
        adapter = get_simulation_adapter(sim.simulator)
        adapter.simulate(sim, sim2)

        self.assertAlmostEqual(h.t, sim2.duration, msg="sim duration incorrect")


@unittest.skipIf(not neuron_installed(), "NEURON is not installed")
class TestNeuronMultichunk(
    RandomStorageFixture,
    ConfigFixture,
    NetworkFixture,
    MorphologiesFixture,
    unittest.TestCase,
    config="chunked",
    morpho_filters=["2comp"],
    engine_name="hdf5",
):
    def setUp(self):
        super().setUp()
        for ct in self.network.cell_types.values():
            ct.spatial.morphologies = ["2comp"]
        hh_soma = {
            "cable_types": {
                "soma": {
                    "cable": {"Ra": 10, "cm": 1},
                    "mechanisms": {"pas": {}, "hh": {}},
                }
            },
            "synapse_types": {"ExpSyn": {}},
        }
        self.network.simulations.add(
            "test",
            simulator="neuron",
            duration=1000,
            resolution=0.1,
            temperature=32,
            cell_models=dict(
                A=ArborizedModel(model=hh_soma),
                B=ArborizedModel(model=hh_soma),
                C=ArborizedModel(model=hh_soma),
            ),
            connection_models=dict(
                A_to_B=TransceiverModel(synapses=[dict(synapse="ExpSyn")]),
                B_to_C=TransceiverModel(synapses=[dict(synapse="ExpSyn")]),
                C_to_A=TransceiverModel(synapses=[dict(synapse="ExpSyn")]),
            ),
            devices=dict(),
        )
        self.network.compile()

    def test_4ch_all_to_all(self):
        """
        Tests runnability of the NEURON adapter with 4 chunks filled with 100 single
        compartment HH cells and ExpSyn synapses connected all to all
        """
        sim = self.network.simulations.test
        adapter = get_simulation_adapter(sim.simulator)
        simdata = adapter.prepare(sim)
        transmitting_cells = sorted(
            itertools.chain.from_iterable(
                MPI.allgather(
                    [
                        (model.name, cell.id, transmitter.gid)
                        for model, pop in simdata.populations.items()
                        for cell in pop
                        if (
                            transmitter := getattr(cell.sections[0], "_transmitter", None)
                        )
                    ]
                )
            )
        )
        receiving_cells = sorted(
            itertools.chain.from_iterable(
                MPI.allgather(
                    [
                        (model.name, cell.id, synapse.gid)
                        for model, pop in simdata.populations.items()
                        for cell in pop
                        for synapse in getattr(cell.sections[0], "synapses", [])
                    ]
                )
            )
        )
        self.assertEqual(
            [
                # A to B
                ("A", 0, 0),
                ("A", 1, 1),
                ("A", 3, 2),
                ("A", 5, 3),
                # B to C
                ("B", 5, 4),
                # C to A
                ("C", 1, 5),
                ("C", 5, 6),
            ],
            transmitting_cells,
        )
        self.assertEqual(
            [
                # C to A
                ("A", 1, 6),
                ("A", 5, 5),
                ("A", 11, 6),
                # A to B
                ("B", 0, 0),
                ("B", 0, 1),
                ("B", 2, 3),
                ("B", 3, 1),
                ("B", 8, 2),
                # B to C
                ("C", 9, 4),
                ("C", 10, 4),
                ("C", 11, 4),
            ],
            receiving_cells,
        )
