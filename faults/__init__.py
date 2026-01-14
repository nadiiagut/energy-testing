"""Degradation & Fault Injection for realistic "almost-working" states.

These are NOT outages. They are penalty generators.

Fault types:
- inverter_loss: Partial inverter capacity reduction
- telemetry_drop: Intermittent telemetry failures  
- control_lag: Degraded control response timing
- soe_bias: State of energy measurement drift
- mode_misclassification: Service mode mismatch (DC/DR/DM)

Each fault module provides:
- Configurable degradation parameters
- Composable fault injection
- Penalty calculation helpers
"""

from .inverter_loss import InverterLossFault, InverterLossMode, InverterLossConfig
from .telemetry_drop import TelemetryDropFault, DropPattern, TelemetryDropConfig, SettlementMode
from .control_lag import ControlLagFault, LagProfile, ControlLagConfig
from .soe_bias import SoEBiasFault, BiasType, SoEBiasConfig
from .mode_misclassification import ModeMisclassificationFault, apply_mode_faults

__all__ = [
    "InverterLossFault",
    "InverterLossMode",
    "InverterLossConfig",
    "TelemetryDropFault", 
    "DropPattern",
    "TelemetryDropConfig",
    "SettlementMode",
    "ControlLagFault",
    "LagProfile",
    "ControlLagConfig",
    "SoEBiasFault",
    "BiasType",
    "SoEBiasConfig",
    "ModeMisclassificationFault",
    "apply_mode_faults",
]
