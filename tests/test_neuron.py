import importlib
from copy import copy

import unittest

from arborize import define_model
from bsb.simulation import get_simulation_adapter
from bsb_test import (
    ConfigFixture,
    MorphologiesFixture,
    NetworkFixture,
    RandomStorageFixture,
)

from bsb.core import Scaffold
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
            }
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
                A_to_B=TransceiverModel(synapses=[dict(synapse="ExpSyn")])
            ),
            devices=dict(),
        )
        self.network.compile()

    def test_4ch_all_to_all(self):
        """
        Tests runnability of the NEURON adapter with 4 chunks filled with 100 single
        compartment HH cells and ExpSyn synapses connected all to all
        """
        self.network.run_simulation("test")
