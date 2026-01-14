"""Control Loop Emulator for realistic plant behavior simulation.

This module simulates plant behavior, NOT ideal response. It models the
real-world constraints and behaviors where software bugs live:

- Response delay (phase lag, communication latency)
- Ramp rate limits (physical constraints on power change)
- Saturation (power output limits)
- State of Energy coupling (battery SoC constraints)

This is NOT a PPC implementation - it's a test harness that models
how real plants behave imperfectly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
import math
import random

from models.energy import ObservedSignal, ServiceMode


class AssetType(Enum):
    """Types of controllable assets."""
    BATTERY = "battery"
    DEMAND_RESPONSE = "demand_response"
    GENERATOR = "generator"
    INVERTER = "inverter"


class ControlMode(Enum):
    """Control modes for frequency response."""
    DROOP = "droop"           # Proportional to frequency deviation
    DEADBAND = "deadband"     # No response within band
    DYNAMIC = "dynamic"       # Dynamic containment (fast response)
    FFR = "ffr"               # Firm Frequency Response


@dataclass
class PlantLimits:
    """Physical limits of the plant."""
    max_power_mw: float = 10.0
    min_power_mw: float = -10.0  # Negative = import/charge
    max_ramp_rate_mw_per_sec: float = 5.0
    min_ramp_rate_mw_per_sec: float = -5.0
    
    # Battery-specific
    max_soe_mwh: Optional[float] = None
    min_soe_mwh: Optional[float] = None
    
    # Efficiency
    charge_efficiency: float = 0.92
    discharge_efficiency: float = 0.92


@dataclass
class ControllerConfig:
    """Controller configuration."""
    mode: ControlMode = ControlMode.DROOP
    
    # Droop settings
    droop_percent: float = 4.0  # 4% droop = 1MW per 0.02Hz at 50MW capacity
    deadband_hz: float = 0.015  # ±15mHz deadband
    
    # Response timing
    response_delay_seconds: float = 0.5  # Communication + processing lag
    measurement_delay_seconds: float = 0.1  # Frequency measurement lag
    
    # Dynamic response
    time_constant_seconds: float = 1.0  # First-order lag time constant
    
    # Frequency thresholds
    nominal_frequency_hz: float = 50.0
    low_frequency_threshold_hz: float = 49.5
    high_frequency_threshold_hz: float = 50.5
    
    # Service mode-specific parameters (slope in MW per Hz, deadband in Hz)
    # DC: Dynamic Containment - aggressive, near-instant response
    dc_slope_mw_per_hz: float = 20.0   # Steep slope for fast response
    dc_deadband_hz: float = 0.015      # Tight deadband
    dc_time_constant_seconds: float = 0.5  # Fast time constant
    
    # DR: Dynamic Regulation - gentler, tracking behavior
    dr_slope_mw_per_hz: float = 10.0   # Moderate slope
    dr_deadband_hz: float = 0.03       # Wider deadband
    dr_time_constant_seconds: float = 2.0  # Slower time constant
    
    # DM: Dynamic Moderation - moderate response
    dm_slope_mw_per_hz: float = 15.0   # Middle ground
    dm_deadband_hz: float = 0.02       # Medium deadband
    dm_time_constant_seconds: float = 1.5  # Medium time constant


@dataclass
class ControlState:
    """Current state of the control loop."""
    current_power_mw: float = 0.0
    target_power_mw: float = 0.0
    commanded_power_mw: float = 0.0
    
    # State of Energy (for batteries)
    soe_mwh: float = 0.0
    soe_percent: float = 50.0
    
    # Timing
    last_update_time: Optional[datetime] = None
    last_frequency_measurement: float = 50.0
    measurement_age_seconds: float = 0.0
    
    # Saturations and limits hit
    power_saturated: bool = False
    ramp_limited: bool = False
    soe_limited: bool = False
    in_deadband: bool = False
    
    # History for analysis
    power_history: List[Tuple[float, float]] = field(default_factory=list)
    frequency_history: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class ControlEvent:
    """Record of a control action for audit.
    
    Includes explicit timing metadata for latency analysis:
    - response_latency_ms: Total time from frequency change to power output
    - measurement_age_ms: How stale the frequency measurement was
    """
    timestamp: datetime
    frequency_hz: float
    target_power_mw: float
    actual_power_mw: float
    soe_mwh: Optional[float]
    flags: List[str]
    
    # Timing metadata (explicit for penalty analysis)
    response_latency_ms: float = 0.0      # Total response delay
    measurement_age_ms: float = 0.0       # Staleness of frequency data
    command_delay_ms: float = 0.0         # Communication/processing delay
    ramp_time_ms: float = 0.0             # Time spent ramping
    
    @property
    def total_latency_ms(self) -> float:
        """Total latency from frequency event to actual response."""
        return self.response_latency_ms + self.measurement_age_ms


class ControlLoop:
    """Emulates realistic plant control behavior.
    
    This is where software bugs live:
    - Stale frequency measurements
    - Ramp rate violations
    - SoE exhaustion
    - Saturation windup
    - Deadband hunting
    """

    def __init__(
        self,
        asset_type: AssetType = AssetType.BATTERY,
        limits: Optional[PlantLimits] = None,
        config: Optional[ControllerConfig] = None,
    ):
        self.asset_type = asset_type
        self.limits = limits or PlantLimits()
        self.config = config or ControllerConfig()
        self.state = ControlState()
        self.events: List[ControlEvent] = []
        
        # Initialize SoE for batteries
        if asset_type == AssetType.BATTERY and self.limits.max_soe_mwh:
            self.state.soe_mwh = self.limits.max_soe_mwh * 0.5
            self.state.soe_percent = 50.0

        # Internal state for lag simulation
        self._frequency_buffer: List[Tuple[float, float]] = []  # (time, freq)
        self._command_buffer: List[Tuple[float, float]] = []    # (time, power)

    def _get_mode_params(self, service_mode: Optional[ServiceMode] = None) -> Tuple[float, float, float]:
        """Get mode-specific control parameters.
        
        Args:
            service_mode: The service mode (DC, DR, DM). If None, uses default config.
            
        Returns:
            Tuple of (slope_mw_per_hz, deadband_hz, time_constant_seconds)
        """
        if service_mode is None:
            # Use default config values
            return (
                self.limits.max_power_mw / 0.5,  # Default slope
                self.config.deadband_hz,
                self.config.time_constant_seconds,
            )
        
        if service_mode == ServiceMode.DC:
            return (
                self.config.dc_slope_mw_per_hz,
                self.config.dc_deadband_hz,
                self.config.dc_time_constant_seconds,
            )
        elif service_mode == ServiceMode.DR:
            return (
                self.config.dr_slope_mw_per_hz,
                self.config.dr_deadband_hz,
                self.config.dr_time_constant_seconds,
            )
        else:  # DM
            return (
                self.config.dm_slope_mw_per_hz,
                self.config.dm_deadband_hz,
                self.config.dm_time_constant_seconds,
            )

    def respond(
        self,
        frequency_hz: float,
        dt_seconds: float,
        now: Optional[datetime] = None,
        service_mode: Optional[ServiceMode] = None,
    ) -> float:
        """Calculate and apply response to frequency deviation.
        
        This is the main entry point. It models:
        1. Measurement delay (stale frequency)
        2. Phase lag (response delay)
        3. Ramp rate limits
        4. Power saturation
        5. SoE coupling
        
        Args:
            frequency_hz: Current grid frequency
            dt_seconds: Time step in seconds
            now: Current timestamp (optional)
            service_mode: Service mode (DC, DR, DM) for mode-specific response
        
        Returns:
            Actual power output after all constraints applied
        """
        now = now or datetime.utcnow()
        
        # Reset flags
        self.state.power_saturated = False
        self.state.ramp_limited = False
        self.state.soe_limited = False
        self.state.in_deadband = False

        # Step 1: Apply measurement delay (we see old frequency)
        measured_frequency = self._apply_measurement_delay(frequency_hz, dt_seconds)
        self.state.last_frequency_measurement = measured_frequency
        
        # Step 2: Calculate target power based on control mode and service mode
        target_power = self._calculate_target_power(measured_frequency, service_mode)
        self.state.target_power_mw = target_power
        
        # Step 3: Apply response delay (phase lag)
        commanded_power = self._apply_response_delay(target_power, dt_seconds)
        self.state.commanded_power_mw = commanded_power
        
        # Step 4: Apply ramp rate limits
        ramped_power = self._apply_ramp_limits(commanded_power, dt_seconds)
        
        # Step 5: Apply power saturation
        saturated_power = self._apply_saturation(ramped_power)
        
        # Step 6: Apply SoE constraints (for batteries)
        final_power = self._apply_soe_constraints(saturated_power, dt_seconds)
        
        # Update state (capture previous power before updating for timing metrics)
        prev_power = self.state.current_power_mw
        self._update_soe(final_power, dt_seconds)
        self.state.current_power_mw = final_power
        self.state.last_update_time = now
        
        # Record history
        t = len(self.state.power_history) * dt_seconds
        self.state.power_history.append((t, final_power))
        self.state.frequency_history.append((t, frequency_hz))
        
        # Record event
        flags = []
        if self.state.power_saturated:
            flags.append("SATURATED")
        if self.state.ramp_limited:
            flags.append("RAMP_LIMITED")
        if self.state.soe_limited:
            flags.append("SOE_LIMITED")
        if self.state.in_deadband:
            flags.append("IN_DEADBAND")
        
        # Calculate timing metadata
        measurement_age_ms = self.state.measurement_age_seconds * 1000
        command_delay_ms = self.config.response_delay_seconds * 1000
        response_latency_ms = command_delay_ms + measurement_age_ms
        ramp_time_ms = 0.0
        if self.state.ramp_limited:
            # Estimate ramp time based on power delta (use prev_power captured before state update)
            power_delta = abs(final_power - prev_power)
            max_ramp = self.limits.max_ramp_rate_mw_per_sec
            if max_ramp > 0:
                ramp_time_ms = (power_delta / max_ramp) * 1000
            
        self.events.append(ControlEvent(
            timestamp=now,
            frequency_hz=frequency_hz,
            target_power_mw=target_power,
            actual_power_mw=final_power,
            soe_mwh=self.state.soe_mwh if self.asset_type == AssetType.BATTERY else None,
            flags=flags,
            response_latency_ms=response_latency_ms,
            measurement_age_ms=measurement_age_ms,
            command_delay_ms=command_delay_ms,
            ramp_time_ms=ramp_time_ms,
        ))
        
        return final_power

    def _apply_measurement_delay(self, actual_frequency: float, dt: float) -> float:
        """Simulate frequency measurement delay (stale data)."""
        delay = self.config.measurement_delay_seconds
        
        # Add current measurement to buffer
        t = len(self._frequency_buffer) * dt
        self._frequency_buffer.append((t, actual_frequency))
        
        # Find measurement from delay ago
        target_time = t - delay
        if target_time <= 0 or len(self._frequency_buffer) < 2:
            return actual_frequency
        
        # Linear interpolation from buffer
        for i in range(len(self._frequency_buffer) - 1, 0, -1):
            t1, f1 = self._frequency_buffer[i - 1]
            t2, f2 = self._frequency_buffer[i]
            if t1 <= target_time <= t2:
                ratio = (target_time - t1) / (t2 - t1) if t2 != t1 else 0
                measured = f1 + ratio * (f2 - f1)
                self.state.measurement_age_seconds = t - target_time
                return measured
        
        return actual_frequency

    def _calculate_target_power(
        self,
        frequency_hz: float,
        service_mode: Optional[ServiceMode] = None,
    ) -> float:
        """Calculate target power based on control mode and service mode.
        
        Args:
            frequency_hz: The measured frequency
            service_mode: Service mode (DC, DR, DM) for mode-specific parameters
        """
        nominal = self.config.nominal_frequency_hz
        deviation = frequency_hz - nominal
        
        # Get mode-specific parameters if service_mode is provided
        if service_mode is not None:
            slope, deadband, _ = self._get_mode_params(service_mode)
            
            # Check deadband
            if abs(deviation) <= deadband:
                self.state.in_deadband = True
                return 0.0
            
            # Apply deadband offset
            if deviation > 0:
                effective_deviation = deviation - deadband
            else:
                effective_deviation = deviation + deadband
            
            # Calculate target using mode-specific slope
            target = -slope * effective_deviation
            capacity = self.limits.max_power_mw
            return max(min(target, capacity), -capacity)
        
        # Fall back to original control mode logic
        # Check deadband
        if abs(deviation) <= self.config.deadband_hz:
            self.state.in_deadband = True
            return 0.0
        
        if self.config.mode == ControlMode.DROOP:
            # Droop control: P = -K * (f - f0)
            # K = capacity / (droop% * nominal_freq)
            capacity = self.limits.max_power_mw
            droop_gain = capacity / (self.config.droop_percent / 100 * nominal)
            
            # Apply deadband offset
            if deviation > 0:
                effective_deviation = deviation - self.config.deadband_hz
            else:
                effective_deviation = deviation + self.config.deadband_hz
            
            target = -droop_gain * effective_deviation
            return target
            
        elif self.config.mode == ControlMode.DYNAMIC:
            # Dynamic containment: faster, steeper response
            capacity = self.limits.max_power_mw
            # Full response at ±0.5Hz deviation
            slope = capacity / 0.5
            
            if deviation > self.config.deadband_hz:
                effective_deviation = deviation - self.config.deadband_hz
            elif deviation < -self.config.deadband_hz:
                effective_deviation = deviation + self.config.deadband_hz
            else:
                effective_deviation = 0
            
            target = -slope * effective_deviation
            return max(min(target, capacity), -capacity)
            
        elif self.config.mode == ControlMode.FFR:
            # Firm Frequency Response: threshold-based
            if frequency_hz < self.config.low_frequency_threshold_hz:
                return self.limits.max_power_mw  # Full export
            elif frequency_hz > self.config.high_frequency_threshold_hz:
                return self.limits.min_power_mw  # Full import
            else:
                return 0.0
        
        return 0.0

    def _apply_response_delay(self, target_power: float, dt: float) -> float:
        """Apply first-order lag to simulate response delay."""
        # First-order lag: dP/dt = (target - P) / tau
        tau = self.config.time_constant_seconds
        if tau <= 0:
            return target_power
        
        current = self.state.commanded_power_mw
        alpha = dt / (tau + dt)  # Discrete approximation
        return current + alpha * (target_power - current)

    def _apply_ramp_limits(self, target_power: float, dt: float) -> float:
        """Apply ramp rate limits (physical constraint)."""
        current = self.state.current_power_mw
        delta = target_power - current
        max_delta = self.limits.max_ramp_rate_mw_per_sec * dt
        min_delta = self.limits.min_ramp_rate_mw_per_sec * dt
        
        if delta > max_delta:
            self.state.ramp_limited = True
            return current + max_delta
        elif delta < min_delta:
            self.state.ramp_limited = True
            return current + min_delta
        
        return target_power

    def _apply_saturation(self, power: float) -> float:
        """Apply power saturation limits."""
        if power > self.limits.max_power_mw:
            self.state.power_saturated = True
            return self.limits.max_power_mw
        elif power < self.limits.min_power_mw:
            self.state.power_saturated = True
            return self.limits.min_power_mw
        return power

    def _apply_soe_constraints(self, power: float, dt: float) -> float:
        """Apply State of Energy constraints (battery specific)."""
        if self.asset_type != AssetType.BATTERY:
            return power
        
        if self.limits.max_soe_mwh is None:
            return power
        
        current_soe = self.state.soe_mwh
        
        # Calculate energy change
        if power > 0:  # Discharging
            energy_change = power * (dt / 3600) / self.limits.discharge_efficiency
            projected_soe = current_soe - energy_change
            
            min_soe = self.limits.min_soe_mwh or 0
            if projected_soe < min_soe:
                self.state.soe_limited = True
                # Limit power to what SoE allows
                available_energy = (current_soe - min_soe) * self.limits.discharge_efficiency
                max_power = available_energy / (dt / 3600)
                return min(power, max(0, max_power))
                
        else:  # Charging (power < 0)
            energy_change = abs(power) * (dt / 3600) * self.limits.charge_efficiency
            projected_soe = current_soe + energy_change
            
            max_soe = self.limits.max_soe_mwh
            if projected_soe > max_soe:
                self.state.soe_limited = True
                # Limit power to what SoE allows
                available_headroom = (max_soe - current_soe) / self.limits.charge_efficiency
                max_charge = available_headroom / (dt / 3600)
                return max(power, -max_charge)
        
        return power

    def _update_soe(self, power: float, dt: float) -> None:
        """Update State of Energy based on power output."""
        if self.asset_type != AssetType.BATTERY:
            return
        
        if self.limits.max_soe_mwh is None:
            return
        
        hours = dt / 3600
        
        if power > 0:  # Discharging
            energy_out = power * hours / self.limits.discharge_efficiency
            self.state.soe_mwh -= energy_out
        else:  # Charging
            energy_in = abs(power) * hours * self.limits.charge_efficiency
            self.state.soe_mwh += energy_in
        
        # Clamp to limits
        min_soe = self.limits.min_soe_mwh or 0
        max_soe = self.limits.max_soe_mwh
        self.state.soe_mwh = max(min_soe, min(max_soe, self.state.soe_mwh))
        
        # Update percentage
        self.state.soe_percent = (self.state.soe_mwh / max_soe) * 100

    def reset(self, soe_percent: float = 50.0) -> None:
        """Reset the control loop state."""
        self.state = ControlState()
        self.events.clear()
        self._frequency_buffer.clear()
        self._command_buffer.clear()
        
        if self.asset_type == AssetType.BATTERY and self.limits.max_soe_mwh:
            self.state.soe_mwh = self.limits.max_soe_mwh * (soe_percent / 100)
            self.state.soe_percent = soe_percent

    def get_response_metrics(self) -> Dict:
        """Get metrics about the control response."""
        if not self.events:
            return {}
        
        saturated_count = sum(1 for e in self.events if "SATURATED" in e.flags)
        ramp_limited_count = sum(1 for e in self.events if "RAMP_LIMITED" in e.flags)
        soe_limited_count = sum(1 for e in self.events if "SOE_LIMITED" in e.flags)
        deadband_count = sum(1 for e in self.events if "IN_DEADBAND" in e.flags)
        
        powers = [e.actual_power_mw for e in self.events]
        
        return {
            "total_events": len(self.events),
            "saturated_events": saturated_count,
            "ramp_limited_events": ramp_limited_count,
            "soe_limited_events": soe_limited_count,
            "deadband_events": deadband_count,
            "max_power_mw": max(powers),
            "min_power_mw": min(powers),
            "final_soe_percent": self.state.soe_percent if self.asset_type == AssetType.BATTERY else None,
        }


class ControlLoopFactory:
    """Factory for creating pre-configured control loops."""

    @staticmethod
    def battery_10mw_20mwh(config: Optional[ControllerConfig] = None) -> ControlLoop:
        """Create a 10MW/20MWh battery control loop."""
        limits = PlantLimits(
            max_power_mw=10.0,
            min_power_mw=-10.0,
            max_ramp_rate_mw_per_sec=10.0,  # Fast ramp
            min_ramp_rate_mw_per_sec=-10.0,
            max_soe_mwh=20.0,
            min_soe_mwh=2.0,  # 10% minimum SoE
            charge_efficiency=0.92,
            discharge_efficiency=0.92,
        )
        return ControlLoop(AssetType.BATTERY, limits, config)

    @staticmethod
    def demand_response_5mw(config: Optional[ControllerConfig] = None) -> ControlLoop:
        """Create a 5MW demand response control loop."""
        limits = PlantLimits(
            max_power_mw=5.0,
            min_power_mw=0.0,  # DR typically can only reduce demand
            max_ramp_rate_mw_per_sec=1.0,  # Slower ramp
            min_ramp_rate_mw_per_sec=-1.0,
        )
        cfg = config or ControllerConfig(
            response_delay_seconds=2.0,  # DR is slower
            time_constant_seconds=5.0,
        )
        return ControlLoop(AssetType.DEMAND_RESPONSE, limits, cfg)

    @staticmethod
    def generator_50mw(config: Optional[ControllerConfig] = None) -> ControlLoop:
        """Create a 50MW generator control loop."""
        limits = PlantLimits(
            max_power_mw=50.0,
            min_power_mw=10.0,  # Minimum stable generation
            max_ramp_rate_mw_per_sec=2.0,  # Thermal ramp limits
            min_ramp_rate_mw_per_sec=-2.0,
        )
        cfg = config or ControllerConfig(
            response_delay_seconds=1.0,
            time_constant_seconds=3.0,
        )
        return ControlLoop(AssetType.GENERATOR, limits, cfg)


# -----------------------------------------------------------------------------
# Observation Boundary Functions
# -----------------------------------------------------------------------------

def emit_metered_power(
    power_mw: float,
    signal: ObservedSignal,
    rng: Optional[random.Random] = None,
) -> Tuple[Optional[float], ObservedSignal]:
    """Apply observation effects at the plant boundary.
    
    Control logic produces ideal internal signals. This function applies
    observation effects only when signals leave the plant boundary.
    
    This mirrors reality:
    - Plant behaves one way internally
    - Observation distorts what is seen externally
    
    Args:
        power_mw: The actual power output from the plant
        signal: The observation signal properties
        rng: Optional random number generator for determinism
        
    Returns:
        Tuple of (observed_power_mw, applied_signal)
        observed_power_mw may be None if signal was dropped
    """
    rng = rng or random.Random()
    
    # Apply drop probability
    if signal.drop_probability > 0 and rng.random() < signal.drop_probability:
        return None, signal
    
    # Apply jitter (timing uncertainty affects value slightly)
    observed_power = power_mw
    if signal.jitter_ms > 0:
        # Jitter introduces small measurement noise proportional to jitter
        jitter_factor = signal.jitter_ms / 1000.0  # Convert to seconds
        noise = rng.gauss(0, jitter_factor * 0.01)  # ~1% noise per 100ms jitter
        observed_power = power_mw * (1.0 + noise)
    
    return observed_power, signal


def apply_signal_latency(
    signal: ObservedSignal,
    additional_latency_ms: float = 0.0,
) -> ObservedSignal:
    """Apply additional latency to an observation signal.
    
    Args:
        signal: The base observation signal
        additional_latency_ms: Additional latency to add
        
    Returns:
        New ObservedSignal with combined latency
    """
    return replace(
        signal,
        latency_ms=signal.latency_ms + additional_latency_ms,
    )


# Default observation signals for common scenarios
SIMULATED_SIGNAL = ObservedSignal(
    source="simulated",
    sample_rate_hz=2.0,
    jitter_ms=0.0,
    latency_ms=0.0,
    drop_probability=0.0,
)

SCADA_SIGNAL = ObservedSignal(
    source="scada",
    sample_rate_hz=1.0,
    jitter_ms=50.0,
    latency_ms=200.0,
    drop_probability=0.01,
)

NIMBUS_SIGNAL = ObservedSignal(
    source="nimbus",
    sample_rate_hz=1.0,
    jitter_ms=100.0,
    latency_ms=500.0,
    drop_probability=0.02,
)
