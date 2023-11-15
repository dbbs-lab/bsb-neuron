"""
NEURON simulator adapter for the BSB framework
"""

from bsb.simulation import SimulationBackendPlugin
from .adapter import NeuronAdapter
from .simulation import NeuronSimulation
from . import devices

__version__ = "0.0.0b1"
__plugin__ = SimulationBackendPlugin(Simulation=NeuronSimulation, Adapter=NeuronAdapter)
