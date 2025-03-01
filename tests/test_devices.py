import importlib
import unittest
from copy import copy

import numpy as np
from bsb.services import MPI
from bsb.simulation import get_simulation_adapter
from bsb_test import (
    ConfigFixture,
    MorphologiesFixture,
    NetworkFixture,
    NumpyTestCase,
    RandomStorageFixture,
)
from patch import p

from bsb_neuron.cell import ArborizedModel
from bsb_neuron.connection import TransceiverModel


def neuron_installed():
    return importlib.util.find_spec("neuron")


@unittest.skipIf(not neuron_installed(), "NEURON is not installed")
class TestSpikeRecorder(
    RandomStorageFixture,
    ConfigFixture,
    NetworkFixture,
    MorphologiesFixture,
    NumpyTestCase,
    unittest.TestCase,
    config="complete",
    morpho_filters=["3branch"],
    engine_name="hdf5",
):

    def setUp(self):
        super().setUp()
        p.parallel.gid_clear()
        self.network.network.chunk_size = [10, 10, 10]
        for ct in self.network.cell_types.values():
            ct.spatial.morphologies = ["3branch"]
        hh_soma = {
            "cable_types": {
                "soma": {
                    "cable": {"Ra": 10, "cm": 1},
                    "mechanisms": {"pas": {}, "hh": {}},
                },
                "dendrites": {
                    "cable": {"Ra": 2, "cm": 5},
                    "mechanisms": {"pas": {}, "hh": {}},
                },
            },
            "synapse_types": {"ExpSyn": {}},
        }
        self.network.simulations.add(
            "test",
            simulator="neuron",
            duration=50,
            resolution=0.1,
            temperature=32,
            cell_models=dict(
                A=ArborizedModel(model=hh_soma),
                B=ArborizedModel(model=hh_soma),
                C=ArborizedModel(model=hh_soma),
            ),
            connection_models=dict(
                B_to_C=TransceiverModel(
                    synapses=[dict(synapse="ExpSyn", weight=0.001, delay=1)]
                ),
            ),
            devices=dict(
                spike_detector=dict(
                    device="spike_recorder",
                    targetting={
                        "strategy": "cell_model",
                        "cell_models": ["A", "B", "C"],
                    },
                ),
                first_current=dict(
                    device="current_clamp",
                    targetting={
                        "strategy": "cell_model",
                        "cell_models": ["A", "C"],
                    },
                    locations={"strategy": "soma"},
                    before=5,
                    amplitude=50,
                    duration=1,
                ),
                second_current=dict(
                    device="current_clamp",
                    targetting={
                        "strategy": "cell_model",
                        "cell_models": ["C"],
                    },
                    locations={"strategy": "soma"},
                    before=35,
                    amplitude=50,
                    duration=1,
                ),
            ),
        )
        self.network.compile()

    def test_simple_stimulus(self):
        "Test that spike_recorder correctly records stimulus"

        # result = self.network.run_simulation("test")
        sim = self.network.simulations.test
        adapter = get_simulation_adapter(sim.simulator)
        simdata = adapter.prepare(sim)
        results = adapter.run(sim)
        result = adapter.collect(sim, simdata, results[0])
        self.assertEqual(
            len(result.spiketrains),
            3,
            "No event should be recorded for B cells but a SpikeTrain should still be allocated",
        )
        # control_data = [
        #     ["A", [0], np.array([5.1], dtype=np.float64)],
        #     ["B", [], np.array([], dtype=np.float64)],
        #     ["C", [1, 1, 3], np.array([5.1, 35.1, 35.1], dtype=np.float64)],
        # ]
        control_data = []
        for cm in sim.cell_models:
            appo = []
            pop = [cell.id for cell in simdata.populations[sim.cell_models[cm]]]
            pop_len = len(pop)
            appo.append(cm)
            if cm == "A":
                appo.append(list(pop))
                appo.append(np.full(pop_len, [5.1], dtype=np.float64))
            if cm == "B":
                appo.append([])
                appo.append(np.array([], dtype=np.float64))
            if cm == "C":
                appo.append([*pop, *pop])
                tot_times = np.concatenate(
                    (
                        np.full(pop_len, [5.1], dtype=np.float64),
                        np.full(pop_len, [35.1], dtype=np.float64),
                    )
                )
                appo.append(tot_times)
            control_data.append(appo)

        for elem, spike_train in enumerate(result.spiketrains):
            self.assertEqual(control_data[elem][0], spike_train.annotations["cell_type"])
            self.assertEqual(
                control_data[elem][1], list(spike_train.annotations["senders"])
            )
            self.assertClose(control_data[elem][2], spike_train.magnitude)
