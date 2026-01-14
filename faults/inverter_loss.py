"""Inverter Loss Fault Injection.

Simulates partial inverter capacity loss - NOT a full outage.
The plant still responds, but at reduced capacity.

This is a penalty generator because:
- Declared capacity != delivered capacity
- Response is degraded but not absent
- Availability declarations become inaccurate

Real-world causes:
- Single inverter trip in multi-inverter plant
- Thermal derating
- DC bus voltage issues
- Communication loss to inverter subset
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
import random


class InverterLossMode(Enum):
    """Types of inverter capacity loss."""
    FIXED_PERCENT = "fixed_percent"       # Constant capacity reduction
    THERMAL_DERATING = "thermal_derating"  # Temperature-dependent
    INTERMITTENT = "intermittent"          # Randomly drops in/out
    PROGRESSIVE = "progressive"            # Degrades over time
    ASYMMETRIC = "asymmetric"              # Different for charge/discharge


@dataclass
class InverterLossConfig:
    """Configuration for inverter loss fault."""
    mode: InverterLossMode = InverterLossMode.FIXED_PERCENT
    
    # Fixed percent mode
    capacity_loss_percent: float = 20.0  # 20% capacity lost
    
    # Thermal derating mode
    ambient_temp_c: float = 25.0
    derating_threshold_c: float = 35.0
    derating_slope_percent_per_c: float = 2.0  # 2% per degree above threshold
    
    # Intermittent mode
    availability_probability: float = 0.8  # 80% of time at full capacity
    intermittent_seed: Optional[int] = None
    
    # Progressive mode
    initial_loss_percent: float = 5.0
    degradation_rate_percent_per_hour: float = 2.0
    max_loss_percent: float = 50.0
    
    # Asymmetric mode
    charge_loss_percent: float = 10.0
    discharge_loss_percent: float = 30.0


@dataclass
class InverterLossState:
    """Current state of inverter loss fault."""
    active: bool = False
    current_capacity_factor: float = 1.0  # 1.0 = full capacity
    cumulative_lost_energy_mwh: float = 0.0
    time_degraded_seconds: float = 0.0
    fault_start_time: Optional[datetime] = None
    
    # For intermittent mode
    currently_available: bool = True
    
    # History
    capacity_history: List[Tuple[float, float]] = field(default_factory=list)


class InverterLossFault:
    """Injects partial inverter capacity loss.
    
    Usage:
        fault = InverterLossFault(InverterLossConfig(
            mode=InverterLossMode.FIXED_PERCENT,
            capacity_loss_percent=20.0
        ))
        fault.activate()
        
        # Apply to power setpoint
        actual_power = fault.apply(commanded_power, dt)
    """

    def __init__(self, config: Optional[InverterLossConfig] = None):
        self.config = config or InverterLossConfig()
        self.state = InverterLossState()
        self._rng = random.Random(self.config.intermittent_seed)

    def activate(self, now: Optional[datetime] = None) -> None:
        """Activate the fault."""
        self.state.active = True
        self.state.fault_start_time = now or datetime.utcnow()
        self.state.current_capacity_factor = self._calculate_capacity_factor(0)

    def deactivate(self) -> None:
        """Deactivate the fault."""
        self.state.active = False
        self.state.current_capacity_factor = 1.0

    def apply(
        self,
        commanded_power_mw: float,
        dt_seconds: float,
        is_charging: bool = False,
    ) -> Tuple[float, Dict]:
        """Apply fault to commanded power.
        
        Returns:
            Tuple of (actual_power, metadata)
        """
        if not self.state.active:
            return commanded_power_mw, {"fault_active": False}

        self.state.time_degraded_seconds += dt_seconds
        
        # Calculate capacity factor based on mode
        capacity_factor = self._calculate_capacity_factor(
            self.state.time_degraded_seconds,
            is_charging=is_charging,
        )
        self.state.current_capacity_factor = capacity_factor
        
        # Apply capacity reduction
        actual_power = commanded_power_mw * capacity_factor
        
        # Track lost energy
        lost_power = abs(commanded_power_mw - actual_power)
        lost_energy = lost_power * (dt_seconds / 3600)
        self.state.cumulative_lost_energy_mwh += lost_energy
        
        # Record history
        t = self.state.time_degraded_seconds
        self.state.capacity_history.append((t, capacity_factor))
        
        metadata = {
            "fault_active": True,
            "capacity_factor": capacity_factor,
            "commanded_mw": commanded_power_mw,
            "actual_mw": actual_power,
            "lost_mw": lost_power,
            "cumulative_lost_mwh": self.state.cumulative_lost_energy_mwh,
        }
        
        return actual_power, metadata

    def _calculate_capacity_factor(
        self,
        elapsed_seconds: float,
        is_charging: bool = False,
    ) -> float:
        """Calculate current capacity factor based on mode."""
        mode = self.config.mode
        
        if mode == InverterLossMode.FIXED_PERCENT:
            return 1.0 - (self.config.capacity_loss_percent / 100)
        
        elif mode == InverterLossMode.THERMAL_DERATING:
            temp = self.config.ambient_temp_c
            threshold = self.config.derating_threshold_c
            if temp <= threshold:
                return 1.0
            excess_temp = temp - threshold
            derating = excess_temp * self.config.derating_slope_percent_per_c
            return max(0.1, 1.0 - (derating / 100))
        
        elif mode == InverterLossMode.INTERMITTENT:
            # Randomly available or not
            if self._rng.random() < self.config.availability_probability:
                self.state.currently_available = True
                return 1.0
            else:
                self.state.currently_available = False
                return 1.0 - (self.config.capacity_loss_percent / 100)
        
        elif mode == InverterLossMode.PROGRESSIVE:
            elapsed_hours = elapsed_seconds / 3600
            loss = self.config.initial_loss_percent + (
                elapsed_hours * self.config.degradation_rate_percent_per_hour
            )
            loss = min(loss, self.config.max_loss_percent)
            return 1.0 - (loss / 100)
        
        elif mode == InverterLossMode.ASYMMETRIC:
            if is_charging:
                return 1.0 - (self.config.charge_loss_percent / 100)
            else:
                return 1.0 - (self.config.discharge_loss_percent / 100)
        
        return 1.0

    def get_penalty_metrics(self) -> Dict:
        """Get metrics for penalty calculation."""
        return {
            "total_degraded_time_seconds": self.state.time_degraded_seconds,
            "cumulative_lost_energy_mwh": self.state.cumulative_lost_energy_mwh,
            "average_capacity_factor": (
                sum(cf for _, cf in self.state.capacity_history) / 
                len(self.state.capacity_history)
                if self.state.capacity_history else 1.0
            ),
            "min_capacity_factor": (
                min(cf for _, cf in self.state.capacity_history)
                if self.state.capacity_history else 1.0
            ),
        }

    def reset(self) -> None:
        """Reset fault state."""
        self.state = InverterLossState()
        self._rng = random.Random(self.config.intermittent_seed)
