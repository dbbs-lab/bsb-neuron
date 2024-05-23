"""
NEURON simulator adapter for the BSB framework
"""

from bsb import SimulationBackendPlugin

from . import devices
from .adapter import NeuronAdapter
from .simulation import NeuronSimulation

__version__ = "4.0.1"
__plugin__ = SimulationBackendPlugin(Simulation=NeuronSimulation, Adapter=NeuronAdapter)
