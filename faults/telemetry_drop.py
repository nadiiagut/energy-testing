"""Telemetry Drop Fault Injection.

Simulates intermittent telemetry failures - NOT a full comms outage.
Data arrives, but with gaps or stale values.

This is a penalty generator because:
- Operator can't verify delivery
- Settlement disputes arise
- Control decisions use stale data
- Availability signals become unreliable

Real-world causes:
- Network congestion
- SCADA polling failures
- RTU buffer overflows
- Cellular connectivity issues
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import random

from models.energy import ObservedSignal


class DropPattern(Enum):
    """Patterns of telemetry drops."""
    RANDOM = "random"           # Random drops with probability
    BURST = "burst"             # Clustered drops
    PERIODIC = "periodic"       # Regular drop intervals
    DEGRADING = "degrading"     # Increasing drop rate over time
    THRESHOLD = "threshold"     # Drops when value exceeds threshold


class SettlementMode(Enum):
    """How missing telemetry is treated for settlement/penalty calculation."""
    ZERO_ON_MISSING = "zero_on_missing"   # Treat missing as zero delivery (worst case)
    HOLD_LAST = "hold_last"               # Use last known good value
    IGNORE_WINDOW = "ignore_window"       # Exclude window from calculation entirely


@dataclass
class TelemetryDropConfig:
    """Configuration for telemetry drop fault."""
    pattern: DropPattern = DropPattern.RANDOM
    
    # Random pattern
    drop_probability: float = 0.1  # 10% drop rate
    
    # Burst pattern
    burst_probability: float = 0.05  # Probability of burst starting
    burst_length_samples: int = 5    # How many samples in a burst
    
    # Periodic pattern
    drop_every_n_samples: int = 10   # Drop every Nth sample
    
    # Degrading pattern
    initial_drop_rate: float = 0.01
    degradation_rate_per_hour: float = 0.05
    max_drop_rate: float = 0.5
    
    # Threshold pattern
    threshold_value: float = 5.0     # Drop when abs(value) > threshold
    threshold_drop_prob: float = 0.3  # Probability of drop when exceeded
    
    # Common settings
    seed: Optional[int] = None
    hold_last_value: bool = True     # Return last good value on drop (legacy, use settlement_mode)
    
    # Settlement policy for penalty calculation
    settlement_mode: SettlementMode = SettlementMode.ZERO_ON_MISSING


@dataclass
class TelemetryDropState:
    """Current state of telemetry drop fault."""
    active: bool = False
    total_samples: int = 0
    dropped_samples: int = 0
    current_drop_rate: float = 0.0
    last_good_value: Optional[float] = None
    last_good_time: Optional[datetime] = None
    
    # Burst tracking
    in_burst: bool = False
    burst_remaining: int = 0
    
    # Staleness tracking
    current_staleness_seconds: float = 0.0
    max_staleness_seconds: float = 0.0
    
    # History
    drop_history: List[Tuple[float, bool]] = field(default_factory=list)


class TelemetryDropFault:
    """Injects intermittent telemetry drops.
    
    Usage:
        fault = TelemetryDropFault(TelemetryDropConfig(
            pattern=DropPattern.RANDOM,
            drop_probability=0.1
        ))
        fault.activate()
        
        # Apply to telemetry value
        value, dropped = fault.apply(measured_value, timestamp)
    """

    def __init__(self, config: Optional[TelemetryDropConfig] = None):
        self.config = config or TelemetryDropConfig()
        self.state = TelemetryDropState()
        self._rng = random.Random(self.config.seed)

    def activate(self) -> None:
        """Activate the fault."""
        self.state.active = True
        self.state.current_drop_rate = (
            self.config.initial_drop_rate 
            if self.config.pattern == DropPattern.DEGRADING 
            else self.config.drop_probability
        )

    def deactivate(self) -> None:
        """Deactivate the fault."""
        self.state.active = False

    def modify_signal(self, signal: ObservedSignal) -> ObservedSignal:
        """Modify an ObservedSignal to reflect telemetry fault effects.
        
        This transforms telemetry faults into observation faults, which
        maps exactly to real-world failure modes where:
        - Telemetry issues are observation issues
        - Not power delivery issues
        
        Args:
            signal: The base observation signal to modify
            
        Returns:
            New ObservedSignal with fault effects applied
        """
        if not self.state.active:
            return signal
        
        # Calculate effective drop probability
        effective_drop_prob = max(
            signal.drop_probability,
            self.state.current_drop_rate,
        )
        
        # Calculate effective jitter (burst patterns add jitter)
        effective_jitter = signal.jitter_ms
        if self.config.pattern == DropPattern.BURST and self.state.in_burst:
            effective_jitter = max(signal.jitter_ms, 200.0)  # Bursts add timing uncertainty
        
        # Calculate effective latency (degrading patterns add latency)
        effective_latency = signal.latency_ms
        if self.config.pattern == DropPattern.DEGRADING:
            # Degradation adds latency proportional to drop rate
            effective_latency = signal.latency_ms + (self.state.current_drop_rate * 500)
        
        return replace(
            signal,
            drop_probability=effective_drop_prob,
            jitter_ms=effective_jitter,
            latency_ms=effective_latency,
        )

    def apply(
        self,
        value: float,
        timestamp: Optional[datetime] = None,
        dt_seconds: float = 1.0,
    ) -> Tuple[Optional[float], Dict]:
        """Apply telemetry drop fault.
        
        Returns:
            Tuple of (returned_value, metadata)
            returned_value may be None (dropped) or last good value (held)
        """
        timestamp = timestamp or datetime.utcnow()
        self.state.total_samples += 1
        
        if not self.state.active:
            self._update_good_value(value, timestamp)
            return value, {"dropped": False, "fault_active": False}

        # Determine if this sample is dropped
        should_drop = self._should_drop(value, dt_seconds)
        
        # Record history
        t = self.state.total_samples
        self.state.drop_history.append((t, should_drop))
        
        if should_drop:
            self.state.dropped_samples += 1
            self.state.current_staleness_seconds += dt_seconds
            self.state.max_staleness_seconds = max(
                self.state.max_staleness_seconds,
                self.state.current_staleness_seconds
            )
            
            # Apply settlement policy
            settlement_mode = self.config.settlement_mode
            if settlement_mode == SettlementMode.ZERO_ON_MISSING:
                returned_value = 0.0
                settlement_value = 0.0
            elif settlement_mode == SettlementMode.HOLD_LAST:
                returned_value = self.state.last_good_value
                settlement_value = self.state.last_good_value
            elif settlement_mode == SettlementMode.IGNORE_WINDOW:
                returned_value = None  # Caller should exclude this window
                settlement_value = None
            else:
                # Legacy fallback
                returned_value = (
                    self.state.last_good_value 
                    if self.config.hold_last_value 
                    else None
                )
                settlement_value = returned_value
            
            metadata = {
                "dropped": True,
                "fault_active": True,
                "settlement_mode": settlement_mode.value,
                "settlement_value": settlement_value,
                "staleness_seconds": self.state.current_staleness_seconds,
                "drop_rate": self.state.dropped_samples / self.state.total_samples,
                "exclude_from_settlement": settlement_mode == SettlementMode.IGNORE_WINDOW,
            }
            return returned_value, metadata
        
        else:
            self._update_good_value(value, timestamp)
            self.state.current_staleness_seconds = 0.0
            
            metadata = {
                "dropped": False,
                "fault_active": True,
                "staleness_seconds": 0.0,
                "drop_rate": self.state.dropped_samples / self.state.total_samples,
            }
            return value, metadata

    def _should_drop(self, value: float, dt_seconds: float) -> bool:
        """Determine if current sample should be dropped."""
        pattern = self.config.pattern
        
        if pattern == DropPattern.RANDOM:
            return self._rng.random() < self.config.drop_probability
        
        elif pattern == DropPattern.BURST:
            # Check if we're in a burst
            if self.state.in_burst:
                self.state.burst_remaining -= 1
                if self.state.burst_remaining <= 0:
                    self.state.in_burst = False
                return True
            
            # Check if burst starts
            if self._rng.random() < self.config.burst_probability:
                self.state.in_burst = True
                self.state.burst_remaining = self.config.burst_length_samples - 1
                return True
            
            return False
        
        elif pattern == DropPattern.PERIODIC:
            return self.state.total_samples % self.config.drop_every_n_samples == 0
        
        elif pattern == DropPattern.DEGRADING:
            # Increase drop rate over time
            elapsed_hours = (self.state.total_samples * dt_seconds) / 3600
            self.state.current_drop_rate = min(
                self.config.initial_drop_rate + 
                elapsed_hours * self.config.degradation_rate_per_hour,
                self.config.max_drop_rate
            )
            return self._rng.random() < self.state.current_drop_rate
        
        elif pattern == DropPattern.THRESHOLD:
            if abs(value) > self.config.threshold_value:
                return self._rng.random() < self.config.threshold_drop_prob
            return self._rng.random() < self.config.drop_probability
        
        return False

    def _update_good_value(self, value: float, timestamp: datetime) -> None:
        """Update last known good value."""
        self.state.last_good_value = value
        self.state.last_good_time = timestamp

    def get_penalty_metrics(self) -> Dict:
        """Get metrics for penalty calculation."""
        total = self.state.total_samples
        dropped = self.state.dropped_samples
        
        return {
            "total_samples": total,
            "dropped_samples": dropped,
            "drop_rate": dropped / total if total > 0 else 0.0,
            "max_staleness_seconds": self.state.max_staleness_seconds,
            "availability_percent": ((total - dropped) / total * 100) if total > 0 else 100.0,
        }

    def reset(self) -> None:
        """Reset fault state."""
        self.state = TelemetryDropState()
        self._rng = random.Random(self.config.seed)
