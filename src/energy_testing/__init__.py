"""
Energy Testing Framework

A framework for testing energy distribution systems with realistic mocks.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Core simulator
from .simulators.grid import EnergyGridSimulator

# Mock components
from .simulators.mocks import (
    MockEnergyProducer,
    MockEnergyConsumer,
    MockGridOperator,
    MockEnergyMarket,
)

# Models
from .models.energy import (
    EnergySource,
    EnergyReading,
    EnergyGridState,
    SimulationResult,
    ProducerConfig,
    ConsumerConfig,
)

__all__ = [
    # Simulator
    "EnergyGridSimulator",
    # Mocks
    "MockEnergyProducer",
    "MockEnergyConsumer",
    "MockGridOperator",
    "MockEnergyMarket",
    # Models
    "EnergySource",
    "EnergyReading",
    "EnergyGridState",
    "SimulationResult",
    "ProducerConfig",
    "ConsumerConfig",
]
