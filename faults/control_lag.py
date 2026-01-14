"""Control Lag Fault Injection.

Simulates degraded control response timing - NOT a control failure.
The plant responds, but slower than contracted.

This is a penalty generator because:
- Response time exceeds contracted limits
- Energy delivery during events is reduced
- Phase margin violations in oscillatory scenarios

Real-world causes:
- PLC scan time increases
- Network latency spikes
- Computation overload
- Sensor processing delays
- Actuator response degradation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
from collections import deque
import random
import math


class LagProfile(Enum):
    """Types of control lag degradation."""
    FIXED_DELAY = "fixed_delay"           # Constant additional delay
    VARIABLE_DELAY = "variable_delay"     # Random delay within range
    LOAD_DEPENDENT = "load_dependent"     # Delay increases with load
    JITTER = "jitter"                     # Random timing variations
    STEP_RESPONSE = "step_response"       # Degraded step response time constant


@dataclass
class ControlLagConfig:
    """Configuration for control lag fault."""
    profile: LagProfile = LagProfile.FIXED_DELAY
    
    # Fixed delay
    additional_delay_seconds: float = 0.5
    
    # Variable delay
    min_delay_seconds: float = 0.1
    max_delay_seconds: float = 2.0
    
    # Load dependent
    base_delay_seconds: float = 0.1
    delay_per_mw: float = 0.02  # Additional delay per MW of output
    
    # Jitter
    jitter_std_seconds: float = 0.1  # Standard deviation of timing jitter
    
    # Step response degradation
    nominal_time_constant_seconds: float = 1.0
    degraded_time_constant_seconds: float = 3.0
    
    # Common settings
    seed: Optional[int] = None


@dataclass
class ControlLagState:
    """Current state of control lag fault."""
    active: bool = False
    total_commands: int = 0
    total_excess_delay_seconds: float = 0.0
    max_delay_seconds: float = 0.0
    
    # Command buffer for delay simulation
    command_buffer: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Step response tracking
    current_time_constant: float = 1.0
    
    # History
    delay_history: List[Tuple[float, float]] = field(default_factory=list)


class ControlLagFault:
    """Injects control response lag.
    
    Usage:
        fault = ControlLagFault(ControlLagConfig(
            profile=LagProfile.FIXED_DELAY,
            additional_delay_seconds=0.5
        ))
        fault.activate()
        
        # Apply to command
        delayed_cmd, delay = fault.apply(command, current_output, timestamp)
    """

    def __init__(self, config: Optional[ControlLagConfig] = None):
        self.config = config or ControlLagConfig()
        self.state = ControlLagState()
        self._rng = random.Random(self.config.seed)

    def activate(self) -> None:
        """Activate the fault."""
        self.state.active = True
        if self.config.profile == LagProfile.STEP_RESPONSE:
            self.state.current_time_constant = self.config.degraded_time_constant_seconds
        else:
            self.state.current_time_constant = self.config.nominal_time_constant_seconds

    def deactivate(self) -> None:
        """Deactivate the fault."""
        self.state.active = False
        self.state.current_time_constant = self.config.nominal_time_constant_seconds

    def apply(
        self,
        command: float,
        current_output: float,
        timestamp: float,  # Simulation time in seconds
        dt_seconds: float = 1.0,
    ) -> Tuple[float, Dict]:
        """Apply control lag to command.
        
        For delay-based profiles: buffers commands and returns delayed version.
        For step response: applies degraded time constant.
        
        Returns:
            Tuple of (effective_command, metadata)
        """
        self.state.total_commands += 1
        
        if not self.state.active:
            return command, {"fault_active": False, "delay_seconds": 0.0}

        profile = self.config.profile
        
        if profile == LagProfile.STEP_RESPONSE:
            # Apply degraded time constant to step response
            return self._apply_degraded_step_response(
                command, current_output, dt_seconds
            )
        else:
            # Apply delay-based lag
            return self._apply_delay(command, timestamp, current_output)

    def _apply_delay(
        self,
        command: float,
        timestamp: float,
        current_output: float,
    ) -> Tuple[float, Dict]:
        """Apply delay-based lag."""
        # Calculate delay for this command
        delay = self._calculate_delay(current_output)
        
        # Add to buffer
        self.state.command_buffer.append((timestamp, command, delay))
        
        # Track statistics
        self.state.total_excess_delay_seconds += delay
        self.state.max_delay_seconds = max(self.state.max_delay_seconds, delay)
        self.state.delay_history.append((timestamp, delay))
        
        # Find command that should be active now (accounting for delay)
        effective_command = command  # Default to current if no delayed available
        for t, cmd, d in reversed(self.state.command_buffer):
            if timestamp >= t + d:
                effective_command = cmd
                break
        
        metadata = {
            "fault_active": True,
            "delay_seconds": delay,
            "effective_command": effective_command,
            "original_command": command,
        }
        
        return effective_command, metadata

    def _apply_degraded_step_response(
        self,
        target: float,
        current: float,
        dt_seconds: float,
    ) -> Tuple[float, Dict]:
        """Apply degraded step response (slower time constant)."""
        tau = self.state.current_time_constant
        nominal_tau = self.config.nominal_time_constant_seconds
        
        # First-order response with degraded time constant
        alpha = dt_seconds / (tau + dt_seconds)
        effective = current + alpha * (target - current)
        
        # Calculate equivalent delay
        # Degraded response at time t equals nominal response at time t - delay
        delay_equivalent = tau - nominal_tau
        
        self.state.total_excess_delay_seconds += delay_equivalent * dt_seconds
        self.state.delay_history.append((self.state.total_commands, delay_equivalent))
        
        metadata = {
            "fault_active": True,
            "time_constant_seconds": tau,
            "nominal_time_constant_seconds": nominal_tau,
            "delay_equivalent_seconds": delay_equivalent,
            "target": target,
            "effective": effective,
        }
        
        return effective, metadata

    def _calculate_delay(self, current_output: float) -> float:
        """Calculate delay based on profile."""
        profile = self.config.profile
        
        if profile == LagProfile.FIXED_DELAY:
            return self.config.additional_delay_seconds
        
        elif profile == LagProfile.VARIABLE_DELAY:
            return self._rng.uniform(
                self.config.min_delay_seconds,
                self.config.max_delay_seconds
            )
        
        elif profile == LagProfile.LOAD_DEPENDENT:
            return (
                self.config.base_delay_seconds + 
                abs(current_output) * self.config.delay_per_mw
            )
        
        elif profile == LagProfile.JITTER:
            # Gaussian jitter around base delay
            base = self.config.additional_delay_seconds
            jitter = self._rng.gauss(0, self.config.jitter_std_seconds)
            return max(0, base + jitter)
        
        return 0.0

    def get_time_constant(self) -> float:
        """Get current effective time constant."""
        return self.state.current_time_constant

    def get_penalty_metrics(self) -> Dict:
        """Get metrics for penalty calculation."""
        total = self.state.total_commands
        avg_delay = (
            self.state.total_excess_delay_seconds / total 
            if total > 0 else 0.0
        )
        
        return {
            "total_commands": total,
            "total_excess_delay_seconds": self.state.total_excess_delay_seconds,
            "average_delay_seconds": avg_delay,
            "max_delay_seconds": self.state.max_delay_seconds,
            "current_time_constant_seconds": self.state.current_time_constant,
        }

    def reset(self) -> None:
        """Reset fault state."""
        self.state = ControlLagState()
        self._rng = random.Random(self.config.seed)
