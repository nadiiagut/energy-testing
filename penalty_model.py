"""Lightweight Penalty Model for configuration comparison.

NOT exact NESO formulas. This provides:
- Relative penalty estimation
- K-factor-like deviation metric
- SoE breach detection
- Availability derating

Purpose: Compare configurations A vs B (pre-penalty validation), not calculate exact £ penalties.
Complements production observability platforms (e.g., Nimbus) by making failure modes reproducible offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple
import math

from models.energy import ObservedSignal, ServiceMode


class ServiceType(Enum):
    """Frequency response service types."""
    DCL = "dcl"   # Dynamic Containment Low
    DCH = "dch"   # Dynamic Containment High
    DRL = "drl"   # Dynamic Regulation Low
    DRH = "drh"   # Dynamic Regulation High
    DMH = "dmh"   # Dynamic Moderation High
    DML = "dml"   # Dynamic Moderation Low
    FFR = "ffr"   # Firm Frequency Response


@dataclass
class DeliveryWindow:
    """A single delivery measurement window."""
    start_time: float  # seconds from event start
    end_time: float
    expected_mw: float
    delivered_mw: float
    frequency_hz: float
    soe_percent: float
    
    # Per-window penalty attribution (populated by PenaltyModel)
    penalty_contribution: float = 0.0      # This window's contribution to total penalty
    k_factor_contribution: float = 0.0     # K-factor component
    soe_breach_contribution: float = 0.0   # SoE breach component
    availability_contribution: float = 0.0  # Availability component
    
    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def deviation_mw(self) -> float:
        return self.delivered_mw - self.expected_mw
    
    @property
    def deviation_percent(self) -> float:
        if self.expected_mw == 0:
            return 0.0
        return (self.deviation_mw / abs(self.expected_mw)) * 100
    
    @property
    def k_factor(self) -> float:
        """Per-window K-factor (requires contracted_mw to be set externally)."""
        # This is a simplified calculation; full K needs contracted_mw
        if self.expected_mw == 0:
            return 0.0
        return abs(self.deviation_mw) / abs(self.expected_mw)


@dataclass
class EventRecord:
    """Record of a single frequency event response."""
    event_id: str
    service_type: ServiceType
    contracted_mw: float
    windows: List[DeliveryWindow] = field(default_factory=list)
    
    # Calculated metrics (populated by PenaltyModel)
    k_factor: float = 0.0
    penalty_score: float = 0.0
    availability_factor: float = 1.0
    soe_breaches: int = 0


@dataclass 
class PenaltyConfig:
    """Configuration for penalty calculation."""
    # K-factor thresholds (relative, not exact NESO)
    k_factor_tolerance: float = 0.03  # 3% tolerance before penalties
    k_factor_cap: float = 1.0         # Maximum K-factor (100% penalty)
    
    # SoE thresholds
    soe_min_percent: float = 10.0     # Below this = breach
    soe_max_percent: float = 90.0     # Above this = breach (for charging)
    
    # Availability thresholds
    availability_threshold: float = 0.95  # 95% availability expected
    
    # Weighting for composite score
    weight_k_factor: float = 0.5
    weight_soe: float = 0.3
    weight_availability: float = 0.2
    
    # Response time requirements (seconds)
    response_time_limit: float = 1.0  # Must respond within this time
    
    # Service mode-specific tracking tolerances (MW deviation allowed)
    # DC: Dynamic Containment - tightest tolerance (fast, accurate response required)
    dc_tracking_tolerance_mw: float = 0.5   # Very tight
    dc_response_time_limit: float = 0.5     # Must respond in 500ms
    
    # DR: Dynamic Regulation - moderate tolerance (tracking behavior)
    dr_tracking_tolerance_mw: float = 1.5   # More relaxed
    dr_response_time_limit: float = 2.0     # 2 second response ok
    
    # DM: Dynamic Moderation - widest tolerance
    dm_tracking_tolerance_mw: float = 1.0   # Medium tolerance
    dm_response_time_limit: float = 1.5     # 1.5 second response


class KFactor:
    """K-factor-like deviation metric.
    
    K = |delivered - expected| / contracted
    
    Lower is better. 0 = perfect delivery.
    """
    
    @staticmethod
    def calculate(
        delivered_mw: float,
        expected_mw: float,
        contracted_mw: float,
    ) -> float:
        """Calculate K-factor for a single measurement."""
        if contracted_mw == 0:
            return 0.0
        deviation = abs(delivered_mw - expected_mw)
        return deviation / contracted_mw
    
    @staticmethod
    def calculate_weighted(
        windows: List[DeliveryWindow],
        contracted_mw: float,
    ) -> float:
        """Calculate time-weighted K-factor across windows."""
        if not windows or contracted_mw == 0:
            return 0.0
        
        total_weighted_k = 0.0
        total_duration = 0.0
        
        for w in windows:
            k = KFactor.calculate(w.delivered_mw, w.expected_mw, contracted_mw)
            total_weighted_k += k * w.duration_seconds
            total_duration += w.duration_seconds
        
        if total_duration == 0:
            return 0.0
        
        return total_weighted_k / total_duration
    
    @staticmethod
    def to_penalty_score(k_factor: float, tolerance: float = 0.03, cap: float = 1.0) -> float:
        """Convert K-factor to penalty score (0-1).
        
        - K below tolerance = 0 penalty
        - K above cap = 1.0 penalty (maximum)
        - Linear interpolation between
        """
        if k_factor <= tolerance:
            return 0.0
        if k_factor >= cap:
            return 1.0
        return (k_factor - tolerance) / (cap - tolerance)


class SoEBreachDetector:
    """Detects State of Energy breaches during events."""
    
    def __init__(self, min_percent: float = 10.0, max_percent: float = 90.0):
        self.min_percent = min_percent
        self.max_percent = max_percent
        self.breaches: List[Dict] = []
    
    def check(self, soe_percent: float, timestamp: float, direction: str = "discharge") -> bool:
        """Check for SoE breach.
        
        Args:
            soe_percent: Current SoE percentage
            timestamp: Time of measurement
            direction: "discharge" or "charge"
            
        Returns:
            True if breach detected
        """
        breach = False
        breach_type = None
        
        if direction == "discharge" and soe_percent < self.min_percent:
            breach = True
            breach_type = "low_soe"
        elif direction == "charge" and soe_percent > self.max_percent:
            breach = True
            breach_type = "high_soe"
        
        if breach:
            self.breaches.append({
                "timestamp": timestamp,
                "soe_percent": soe_percent,
                "breach_type": breach_type,
                "limit": self.min_percent if breach_type == "low_soe" else self.max_percent,
            })
        
        return breach
    
    def check_windows(self, windows: List[DeliveryWindow]) -> int:
        """Check all windows for SoE breaches."""
        self.breaches.clear()
        
        for w in windows:
            direction = "discharge" if w.delivered_mw > 0 else "charge"
            self.check(w.soe_percent, w.start_time, direction)
        
        return len(self.breaches)
    
    def get_breach_severity(self) -> float:
        """Get severity score (0-1) based on breaches."""
        if not self.breaches:
            return 0.0
        
        # Severity based on how far below/above threshold
        max_severity = 0.0
        for b in self.breaches:
            if b["breach_type"] == "low_soe":
                severity = (self.min_percent - b["soe_percent"]) / self.min_percent
            else:
                severity = (b["soe_percent"] - self.max_percent) / (100 - self.max_percent)
            max_severity = max(max_severity, min(1.0, severity))
        
        return max_severity


class AvailabilityCalculator:
    """Calculates availability derating."""
    
    @staticmethod
    def calculate(
        windows: List[DeliveryWindow],
        contracted_mw: float,
        threshold: float = 0.95,
    ) -> float:
        """Calculate availability factor.
        
        Availability = time_delivering / total_time
        where "delivering" means output >= threshold * expected
        
        Returns:
            Availability factor (0-1)
        """
        if not windows or contracted_mw == 0:
            return 1.0
        
        total_duration = sum(w.duration_seconds for w in windows)
        if total_duration == 0:
            return 1.0
        
        delivering_duration = 0.0
        for w in windows:
            if w.expected_mw == 0:
                delivering_duration += w.duration_seconds
            elif abs(w.delivered_mw) >= threshold * abs(w.expected_mw):
                delivering_duration += w.duration_seconds
        
        return delivering_duration / total_duration
    
    @staticmethod
    def to_derating(availability: float, threshold: float = 0.95) -> float:
        """Convert availability to derating factor.
        
        If availability >= threshold: no derating (1.0)
        If availability < threshold: proportional derating
        """
        if availability >= threshold:
            return 1.0
        return availability / threshold


class PenaltyModel:
    """Lightweight penalty model for configuration comparison."""
    
    def __init__(self, config: Optional[PenaltyConfig] = None):
        self.config = config or PenaltyConfig()
        self.soe_detector = SoEBreachDetector(
            min_percent=self.config.soe_min_percent,
            max_percent=self.config.soe_max_percent,
        )
        self.events: List[EventRecord] = []
    
    def evaluate_event(self, event: EventRecord) -> Dict:
        """Evaluate a single event and calculate penalty metrics.
        
        Returns:
            Dictionary with all penalty metrics
        """
        # K-factor calculation
        k_factor = KFactor.calculate_weighted(event.windows, event.contracted_mw)
        k_penalty = KFactor.to_penalty_score(
            k_factor,
            tolerance=self.config.k_factor_tolerance,
            cap=self.config.k_factor_cap,
        )
        
        # SoE breach detection
        soe_breaches = self.soe_detector.check_windows(event.windows)
        soe_severity = self.soe_detector.get_breach_severity()
        
        # Availability calculation
        availability = AvailabilityCalculator.calculate(
            event.windows,
            event.contracted_mw,
            threshold=self.config.availability_threshold,
        )
        availability_derating = AvailabilityCalculator.to_derating(
            availability,
            threshold=self.config.availability_threshold,
        )
        
        # Composite penalty score (0-1, higher = worse)
        composite_score = (
            self.config.weight_k_factor * k_penalty +
            self.config.weight_soe * soe_severity +
            self.config.weight_availability * (1 - availability_derating)
        )
        
        # Calculate per-window penalty attribution
        self._attribute_penalties_to_windows(event, k_penalty, soe_severity, availability_derating)
        
        # Update event record
        event.k_factor = k_factor
        event.penalty_score = composite_score
        event.availability_factor = availability
        event.soe_breaches = soe_breaches
        
        self.events.append(event)
        
        return {
            "event_id": event.event_id,
            "k_factor": k_factor,
            "k_penalty_score": k_penalty,
            "soe_breaches": soe_breaches,
            "soe_severity": soe_severity,
            "availability": availability,
            "availability_derating": availability_derating,
            "composite_penalty_score": composite_score,
            "rating": self._score_to_rating(composite_score),
        }
    
    def _score_to_rating(self, score: float) -> str:
        """Convert penalty score to human-readable rating."""
        if score < 0.05:
            return "EXCELLENT"
        elif score < 0.15:
            return "GOOD"
        elif score < 0.30:
            return "ACCEPTABLE"
        elif score < 0.50:
            return "MARGINAL"
        else:
            return "POOR"
    
    def _attribute_penalties_to_windows(
        self,
        event: EventRecord,
        k_penalty: float,
        soe_severity: float,
        availability_derating: float,
    ) -> None:
        """Calculate per-window penalty attribution.
        
        Distributes the total penalty across windows based on their
        individual contribution to each penalty component.
        """
        if not event.windows:
            return
        
        total_duration = sum(w.duration_seconds for w in event.windows)
        if total_duration == 0:
            return
        
        for window in event.windows:
            # K-factor contribution: proportional to this window's K-factor
            window_k = KFactor.calculate(
                window.delivered_mw,
                window.expected_mw,
                event.contracted_mw,
            )
            window_k_penalty = KFactor.to_penalty_score(
                window_k,
                tolerance=self.config.k_factor_tolerance,
                cap=self.config.k_factor_cap,
            )
            window.k_factor_contribution = (
                self.config.weight_k_factor * window_k_penalty * 
                (window.duration_seconds / total_duration)
            )
            
            # SoE breach contribution: check if this window breached
            direction = "discharge" if window.delivered_mw > 0 else "charge"
            soe_breach = (
                (direction == "discharge" and window.soe_percent < self.config.soe_min_percent) or
                (direction == "charge" and window.soe_percent > self.config.soe_max_percent)
            )
            if soe_breach:
                # Attribute breach severity to this window
                if direction == "discharge":
                    severity = (self.config.soe_min_percent - window.soe_percent) / self.config.soe_min_percent
                else:
                    severity = (window.soe_percent - self.config.soe_max_percent) / (100 - self.config.soe_max_percent)
                window.soe_breach_contribution = (
                    self.config.weight_soe * min(1.0, max(0.0, severity)) *
                    (window.duration_seconds / total_duration)
                )
            else:
                window.soe_breach_contribution = 0.0
            
            # Availability contribution: did this window deliver?
            delivered_ok = (
                window.expected_mw == 0 or 
                abs(window.delivered_mw) >= self.config.availability_threshold * abs(window.expected_mw)
            )
            if not delivered_ok:
                window.availability_contribution = (
                    self.config.weight_availability * (1 - availability_derating) *
                    (window.duration_seconds / total_duration)
                )
            else:
                window.availability_contribution = 0.0
            
            # Total contribution
            window.penalty_contribution = (
                window.k_factor_contribution +
                window.soe_breach_contribution +
                window.availability_contribution
            )
    
    def get_summary(self) -> Dict:
        """Get summary of all evaluated events."""
        if not self.events:
            return {"total_events": 0}
        
        k_factors = [e.k_factor for e in self.events]
        penalties = [e.penalty_score for e in self.events]
        availabilities = [e.availability_factor for e in self.events]
        
        return {
            "total_events": len(self.events),
            "avg_k_factor": sum(k_factors) / len(k_factors),
            "max_k_factor": max(k_factors),
            "avg_penalty_score": sum(penalties) / len(penalties),
            "max_penalty_score": max(penalties),
            "avg_availability": sum(availabilities) / len(availabilities),
            "total_soe_breaches": sum(e.soe_breaches for e in self.events),
            "overall_rating": self._score_to_rating(sum(penalties) / len(penalties)),
        }
    
    def reset(self) -> None:
        """Reset model state."""
        self.events.clear()
        self.soe_detector.breaches.clear()


def compare_configurations(
    config_a_events: List[EventRecord],
    config_b_events: List[EventRecord],
    penalty_config: Optional[PenaltyConfig] = None,
) -> Dict:
    """Compare two configurations based on penalty metrics.
    
    Returns:
        Dictionary with comparison results and winner determination
    """
    model_a = PenaltyModel(penalty_config)
    model_b = PenaltyModel(penalty_config)
    
    # Evaluate all events
    for event in config_a_events:
        model_a.evaluate_event(event)
    
    for event in config_b_events:
        model_b.evaluate_event(event)
    
    summary_a = model_a.get_summary()
    summary_b = model_b.get_summary()
    
    # Determine winner (lower penalty = better)
    score_a = summary_a.get("avg_penalty_score", 1.0)
    score_b = summary_b.get("avg_penalty_score", 1.0)
    
    if abs(score_a - score_b) < 0.01:
        winner = "TIE"
        margin = 0.0
    elif score_a < score_b:
        winner = "A"
        margin = (score_b - score_a) / score_b * 100 if score_b > 0 else 100
    else:
        winner = "B"
        margin = (score_a - score_b) / score_a * 100 if score_a > 0 else 100
    
    return {
        "config_a": summary_a,
        "config_b": summary_b,
        "winner": winner,
        "margin_percent": margin,
        "comparison": {
            "k_factor_better": "A" if summary_a.get("avg_k_factor", 1) < summary_b.get("avg_k_factor", 1) else "B",
            "availability_better": "A" if summary_a.get("avg_availability", 0) > summary_b.get("avg_availability", 0) else "B",
            "soe_breaches_better": "A" if summary_a.get("total_soe_breaches", 999) < summary_b.get("total_soe_breaches", 999) else "B",
        },
    }


def quick_penalty_score(
    delivered_mw: float,
    expected_mw: float,
    contracted_mw: float,
    soe_percent: float = 50.0,
) -> Dict:
    """Quick single-point penalty calculation for simple comparisons.
    
    Returns:
        Dictionary with k_factor, soe_ok, and penalty_score
    """
    k = KFactor.calculate(delivered_mw, expected_mw, contracted_mw)
    k_penalty = KFactor.to_penalty_score(k)
    soe_ok = 10.0 <= soe_percent <= 90.0
    
    return {
        "k_factor": k,
        "k_penalty": k_penalty,
        "soe_ok": soe_ok,
        "delivery_percent": (delivered_mw / expected_mw * 100) if expected_mw != 0 else 100,
    }


def score_delivery(
    requested_mw: float,
    metered_mw: Optional[float],
    signal: ObservedSignal,
    contracted_mw: float = 10.0,
) -> Dict:
    """Signal-aware delivery scoring.
    
    Instead of assuming "missing telemetry = zero delivery", this function
    considers the observation signal source to determine appropriate scoring.
    
    Args:
        requested_mw: The power requested/expected
        metered_mw: The metered power (None if dropped)
        signal: The observation signal properties
        contracted_mw: The contracted capacity
        
    Returns:
        Dictionary with:
        - penalty_score: The calculated penalty (0-1)
        - penalty_reason: Human-readable explanation
        - signal_quality: Assessment of signal reliability
        - metered_value_used: The value used for calculation
    """
    # Determine signal quality
    signal_quality = "good"
    if signal.drop_probability > 0.1:
        signal_quality = "poor"
    elif signal.drop_probability > 0.02:
        signal_quality = "degraded"
    
    # Handle missing metered value based on signal source
    if metered_mw is None:
        if signal.source == "nimbus":
            # Nimbus is authoritative - missing data is severe
            return {
                "penalty_score": 1.0,
                "penalty_reason": "Missing Nimbus data treated as non-delivery",
                "signal_quality": signal_quality,
                "metered_value_used": 0.0,
            }
        elif signal.source == "scada":
            # SCADA drops may be temporary - moderate penalty
            return {
                "penalty_score": 0.5,
                "penalty_reason": "SCADA drop - partial penalty applied",
                "signal_quality": signal_quality,
                "metered_value_used": None,
            }
        else:  # simulated
            # Simulated should not drop - indicates test issue
            return {
                "penalty_score": 0.0,
                "penalty_reason": "Simulated signal drop - no penalty (test artifact)",
                "signal_quality": signal_quality,
                "metered_value_used": None,
            }
    
    # Calculate normal K-factor penalty
    k = KFactor.calculate(metered_mw, requested_mw, contracted_mw)
    k_penalty = KFactor.to_penalty_score(k)
    
    # Adjust for signal quality
    if signal_quality == "poor":
        # Poor signal quality adds uncertainty penalty
        k_penalty = min(1.0, k_penalty + 0.1)
        reason = f"K-factor {k:.3f} + poor signal quality penalty"
    elif signal_quality == "degraded":
        k_penalty = min(1.0, k_penalty + 0.05)
        reason = f"K-factor {k:.3f} + degraded signal penalty"
    else:
        reason = f"K-factor {k:.3f}"
    
    return {
        "penalty_score": k_penalty,
        "penalty_reason": reason,
        "signal_quality": signal_quality,
        "metered_value_used": metered_mw,
    }


def score_delivery_by_mode(
    requested_mw: float,
    delivered_mw: float,
    service_mode: ServiceMode,
    contracted_mw: float = 10.0,
    config: Optional[PenaltyConfig] = None,
) -> Dict:
    """Score delivery using mode-specific tolerances.
    
    Different service modes have different performance requirements:
    - DC (Dynamic Containment): Tightest tolerance, fastest response required
    - DR (Dynamic Regulation): More relaxed, tracking behavior
    - DM (Dynamic Moderation): Middle ground
    
    This is the key to detecting mode misclassification bugs:
    - Controller may deliver "reasonable" power
    - But if evaluated against wrong mode envelope, penalties apply
    
    Args:
        requested_mw: The power requested/expected
        delivered_mw: The actual delivered power
        service_mode: The service mode for evaluation (contract expectation)
        contracted_mw: The contracted capacity
        config: Penalty configuration (uses defaults if None)
        
    Returns:
        Dictionary with:
        - penalty_score: The calculated penalty (0-1)
        - deviation_mw: Absolute deviation from requested
        - tolerance_mw: Mode-specific tolerance used
        - within_tolerance: Whether delivery was within tolerance
        - penalty_reason: Human-readable explanation
    """
    config = config or PenaltyConfig()
    
    # Get mode-specific tolerance
    if service_mode == ServiceMode.DC:
        tolerance_mw = config.dc_tracking_tolerance_mw
        mode_name = "Dynamic Containment"
    elif service_mode == ServiceMode.DR:
        tolerance_mw = config.dr_tracking_tolerance_mw
        mode_name = "Dynamic Regulation"
    else:  # DM
        tolerance_mw = config.dm_tracking_tolerance_mw
        mode_name = "Dynamic Moderation"
    
    # Calculate deviation
    deviation_mw = abs(delivered_mw - requested_mw)
    within_tolerance = deviation_mw <= tolerance_mw
    
    # Calculate penalty based on how far outside tolerance
    if within_tolerance:
        penalty_score = 0.0
        reason = f"{mode_name}: Within {tolerance_mw:.1f}MW tolerance"
    else:
        # Penalty scales with how much over tolerance
        excess_deviation = deviation_mw - tolerance_mw
        # Normalize to contracted capacity
        penalty_score = min(1.0, excess_deviation / contracted_mw)
        reason = f"{mode_name}: {deviation_mw:.2f}MW deviation exceeds {tolerance_mw:.1f}MW tolerance"
    
    return {
        "penalty_score": penalty_score,
        "deviation_mw": deviation_mw,
        "tolerance_mw": tolerance_mw,
        "within_tolerance": within_tolerance,
        "service_mode": service_mode.value,
        "penalty_reason": reason,
    }


def get_mode_tolerance(service_mode: ServiceMode, config: Optional[PenaltyConfig] = None) -> float:
    """Get the tracking tolerance for a service mode.
    
    Args:
        service_mode: The service mode
        config: Penalty configuration (uses defaults if None)
        
    Returns:
        Tolerance in MW
    """
    config = config or PenaltyConfig()
    
    if service_mode == ServiceMode.DC:
        return config.dc_tracking_tolerance_mw
    elif service_mode == ServiceMode.DR:
        return config.dr_tracking_tolerance_mw
    else:  # DM
        return config.dm_tracking_tolerance_mw
