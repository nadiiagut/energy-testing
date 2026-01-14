"""Grid stress scenario definitions for Penalty & Revenue Evaluation testing.

This module provides scenario generation for testing grid frequency response,
penalty calculations, and revenue implications under various stress conditions.

Scenarios mirror real NESO events but are safely testable offline.
"""

from .scenario_loader import (
    Scenario,
    ScenarioLoader,
    FrequencyEvent,
    FrequencyTrajectory,
    load_scenario,
    list_scenarios,
)

__all__ = [
    "Scenario",
    "ScenarioLoader", 
    "FrequencyEvent",
    "FrequencyTrajectory",
    "load_scenario",
    "list_scenarios",
]
