"""
NEURON simulator adapter for the BSB framework
"""

from bsb.simulation import SimulationBackendPlugin

from . import devices
from .adapter import NeuronAdapter
from .simulation import NeuronSimulation

__version__ = "0.0.0b3"
__plugin__ = SimulationBackendPlugin(Simulation=NeuronSimulation, Adapter=NeuronAdapter)
