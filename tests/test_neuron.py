import importlib
import unittest

from arborize import Schematic, define_model
from bsb_test import (
    ConfigFixture,
    MorphologiesFixture,
    NetworkFixture,
    RandomStorageFixture,
)

from bsb_neuron.cell import ArborizedModel


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


@unittest.skipIf(not neuron_installed(), "NEURON is not installed")
class TestNeuronMultichunk(
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
        self.cfg.cell_types.add("A", {"spatial": {"count": 10}})
        self.network.compile()

    def test_minimal(self):
        from neuron import h

        sim = self.network.simulations.test
        self.network.run_simulation("test")
        self.assertAlmostEqual(h.t, sim.duration, msg="sim duration incorrect")

    def test_4ch_all_to_all(self):
        """
        Tests runnability of the NEURON adapter with 4 chunks filled with 100 single
        compartment HH cells and ExpSyn synapses connected all to all
        """
        hh_soma = define_model(
            {
                "cable_types": {
                    "soma": {
                        "cable": {"Ra": 10, "cm": 1},
                        "mechanisms": {"pas": {}, "hh": {}},
                    }
                }
            }
        )
        schematic = Schematic("single_comp")
        schematic.create_location((0, 0), (0, 0, 0), 1, ["soma"])
        schematic.create_location((0, 1), (100, 0, 0), 1, ["soma"])
        schematic.definition = hh_soma
        ArborizedModel(model=hh_soma)
