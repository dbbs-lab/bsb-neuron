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

from bsb_neuron.cell import ArborizedModel, ArborizeModelTypeHandler


class TestArborizedModel(unittest.TestCase):

    def setUp(self):
        hh_soma = {
            "cable_types": {
                "soma": {
                    "cable": {"Ra": 10, "cm": 1},
                    "mechanisms": {"pas": {}, "hh": {}},
                }
            },
            "synapse_types": {"ExpSyn": {}},
        }
        self.model = ArborizedModel(model=hh_soma)

    def test_typehandler_inv(self):

        self.model._cfg_inv = "model_definition"
        model_handler = ArborizeModelTypeHandler()
        reverse_model = model_handler.__inv__(self.model)
        assert isinstance(reverse_model, str)
        self.assertEqual(reverse_model, "model_definition")
        # Now i check if a ModelDefinition is correctly converted to a string
        self.model._cfg_inv = define_model(
            {
                "cable_types": {"soma": {}},
                "synapse_types": {"ExpSyn": {}},
            }
        )
        reverse_model = model_handler.__inv__(self.model)
        assert isinstance(reverse_model, str)
