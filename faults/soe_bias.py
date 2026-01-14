"""State of Energy Bias Fault Injection.

Simulates SoE measurement drift - NOT a BMS failure.
The system operates, but with inaccurate energy state.

This is a penalty generator because:
- Declared availability doesn't match actual
- Unexpected SoE limits during events
- Over/under-delivery due to capacity miscalculation
- Compliance failures on energy throughput

Real-world causes:
- Cell degradation not calibrated
- Temperature effects on capacity
- Coulomb counting drift
- BMS calibration errors
- Aging effects
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
import random
import math


class BiasType(Enum):
    """Types of SoE bias."""
    FIXED_OFFSET = "fixed_offset"       # Constant MWh offset
    PERCENTAGE_BIAS = "percentage_bias"  # Percentage of true SoE
    DRIFT = "drift"                      # Accumulating drift over time
    TEMPERATURE = "temperature"          # Temperature-dependent bias
    HYSTERESIS = "hysteresis"            # Direction-dependent bias


@dataclass
class SoEBiasConfig:
    """Configuration for SoE bias fault."""
    bias_type: BiasType = BiasType.FIXED_OFFSET
    
    # Fixed offset
    offset_mwh: float = 1.0  # Measured shows 1MWh more than actual
    
    # Percentage bias
    bias_percent: float = 5.0  # Measured is 5% higher than actual
    
    # Drift
    drift_rate_mwh_per_hour: float = 0.1  # Drift accumulation rate
    max_drift_mwh: float = 2.0
    
    # Temperature
    temp_coefficient_percent_per_c: float = 0.5
    reference_temp_c: float = 25.0
    current_temp_c: float = 25.0
    
    # Hysteresis
    charge_bias_percent: float = 3.0   # Overestimate when charging
    discharge_bias_percent: float = -3.0  # Underestimate when discharging
    
    # Common
    seed: Optional[int] = None


@dataclass
class SoEBiasState:
    """Current state of SoE bias fault."""
    active: bool = False
    true_soe_mwh: float = 0.0
    reported_soe_mwh: float = 0.0
    cumulative_drift_mwh: float = 0.0
    
    # Tracking
    total_cycles: int = 0
    total_energy_error_mwh: float = 0.0
    max_error_mwh: float = 0.0
    
    # Last direction for hysteresis
    last_direction: str = "idle"  # "charging", "discharging", "idle"
    
    # History
    error_history: List[Tuple[float, float, float]] = field(default_factory=list)


class SoEBiasFault:
    """Injects State of Energy measurement bias.
    
    Usage:
        fault = SoEBiasFault(SoEBiasConfig(
            bias_type=BiasType.FIXED_OFFSET,
            offset_mwh=1.0
        ))
        fault.activate(true_soe_mwh=10.0)
        
        # Get biased reading
        reported_soe = fault.apply(true_soe_mwh, power_mw, dt)
    """

    def __init__(self, config: Optional[SoEBiasConfig] = None):
        self.config = config or SoEBiasConfig()
        self.state = SoEBiasState()
        self._rng = random.Random(self.config.seed)

    def activate(self, true_soe_mwh: float = 10.0) -> None:
        """Activate the fault with initial true SoE."""
        self.state.active = True
        self.state.true_soe_mwh = true_soe_mwh
        self.state.reported_soe_mwh = self._calculate_reported(true_soe_mwh, 0)

    def deactivate(self) -> None:
        """Deactivate the fault."""
        self.state.active = False

    def apply(
        self,
        true_soe_mwh: float,
        power_mw: float = 0.0,
        dt_seconds: float = 1.0,
    ) -> Tuple[float, Dict]:
        """Apply SoE bias and return reported value.
        
        Args:
            true_soe_mwh: Actual state of energy
            power_mw: Current power (for hysteresis direction)
            dt_seconds: Time step
            
        Returns:
            Tuple of (reported_soe_mwh, metadata)
        """
        self.state.total_cycles += 1
        self.state.true_soe_mwh = true_soe_mwh
        
        if not self.state.active:
            self.state.reported_soe_mwh = true_soe_mwh
            return true_soe_mwh, {"fault_active": False, "error_mwh": 0.0}

        # Update direction for hysteresis
        if power_mw > 0.1:
            self.state.last_direction = "discharging"
        elif power_mw < -0.1:
            self.state.last_direction = "charging"
        else:
            self.state.last_direction = "idle"

        # Calculate reported SoE with bias
        reported = self._calculate_reported(true_soe_mwh, dt_seconds)
        self.state.reported_soe_mwh = reported
        
        # Track error
        error = reported - true_soe_mwh
        self.state.total_energy_error_mwh += abs(error) * (dt_seconds / 3600)
        self.state.max_error_mwh = max(self.state.max_error_mwh, abs(error))
        
        # Record history
        self.state.error_history.append((
            self.state.total_cycles,
            true_soe_mwh,
            error
        ))
        
        metadata = {
            "fault_active": True,
            "true_soe_mwh": true_soe_mwh,
            "reported_soe_mwh": reported,
            "error_mwh": error,
            "error_percent": (error / true_soe_mwh * 100) if true_soe_mwh > 0 else 0,
            "direction": self.state.last_direction,
        }
        
        return reported, metadata

    def _calculate_reported(self, true_soe: float, dt_seconds: float) -> float:
        """Calculate reported SoE based on bias type."""
        bias_type = self.config.bias_type
        
        if bias_type == BiasType.FIXED_OFFSET:
            return true_soe + self.config.offset_mwh
        
        elif bias_type == BiasType.PERCENTAGE_BIAS:
            return true_soe * (1 + self.config.bias_percent / 100)
        
        elif bias_type == BiasType.DRIFT:
            # Accumulate drift over time
            dt_hours = dt_seconds / 3600
            drift_increment = self.config.drift_rate_mwh_per_hour * dt_hours
            self.state.cumulative_drift_mwh += drift_increment
            
            # Cap drift
            if abs(self.state.cumulative_drift_mwh) > self.config.max_drift_mwh:
                self.state.cumulative_drift_mwh = math.copysign(
                    self.config.max_drift_mwh,
                    self.state.cumulative_drift_mwh
                )
            
            return true_soe + self.state.cumulative_drift_mwh
        
        elif bias_type == BiasType.TEMPERATURE:
            temp_delta = self.config.current_temp_c - self.config.reference_temp_c
            temp_factor = 1 + (temp_delta * self.config.temp_coefficient_percent_per_c / 100)
            return true_soe * temp_factor
        
        elif bias_type == BiasType.HYSTERESIS:
            direction = self.state.last_direction
            if direction == "charging":
                return true_soe * (1 + self.config.charge_bias_percent / 100)
            elif direction == "discharging":
                return true_soe * (1 + self.config.discharge_bias_percent / 100)
            else:
                return true_soe
        
        return true_soe

    def set_temperature(self, temp_c: float) -> None:
        """Update current temperature for temperature-dependent bias."""
        self.config.current_temp_c = temp_c

    def get_true_soe(self) -> float:
        """Get actual (true) state of energy."""
        return self.state.true_soe_mwh

    def get_reported_soe(self) -> float:
        """Get reported (biased) state of energy."""
        return self.state.reported_soe_mwh

    def get_current_error(self) -> float:
        """Get current measurement error in MWh."""
        return self.state.reported_soe_mwh - self.state.true_soe_mwh

    def effective_capacity_mwh(self) -> float:
        """Get effective usable capacity accounting for bias.
        
        When SoE is biased, the system believes it has more/less
        energy than it actually does. This returns the actual
        usable capacity from the system's perspective.
        
        Example:
            True capacity: 20 MWh
            Bias: +2 MWh (system thinks it has more)
            Effective: System will exhaust 2 MWh early
            
        Returns:
            Effective usable capacity in MWh
        """
        error = self.get_current_error()
        true_soe = self.state.true_soe_mwh
        
        # If we overestimate SoE (positive bias), effective capacity is lower
        # because we'll hit actual empty before reported empty
        return max(0.0, true_soe - error)

    def get_penalty_metrics(self) -> Dict:
        """Get metrics for penalty calculation."""
        return {
            "total_cycles": self.state.total_cycles,
            "cumulative_error_mwh": self.state.total_energy_error_mwh,
            "max_error_mwh": self.state.max_error_mwh,
            "current_error_mwh": self.get_current_error(),
            "cumulative_drift_mwh": self.state.cumulative_drift_mwh,
            "error_rate_percent": (
                self.state.max_error_mwh / self.state.true_soe_mwh * 100
                if self.state.true_soe_mwh > 0 else 0
            ),
        }

    def reset(self, true_soe_mwh: float = 10.0) -> None:
        """Reset fault state."""
        self.state = SoEBiasState()
        self.state.true_soe_mwh = true_soe_mwh
        self._rng = random.Random(self.config.seed)
