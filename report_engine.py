"""Report / Diff Engine for configuration comparison.

Answers the primary operator question: "Which change reduces penalty exposure / saves money?"

Output:
- Penalty delta (how much worse/better)
- Revenue at risk (£ exposure)
- Time-to-penalty (when failures start costing)

Designed for pre-deployment validation and scenario replay; complements (not replaces) production monitoring/observability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import json

from penalty_model import (
    PenaltyModel, PenaltyConfig, EventRecord, DeliveryWindow,
    ServiceType, KFactor, compare_configurations,
)


@dataclass
class RevenueConfig:
    """Revenue and pricing configuration."""
    # Contracted revenue per MW per hour (£/MW/h)
    contracted_rate_per_mw_hour: float = 10.0
    
    # Penalty multipliers (relative to contracted rate)
    underdelivery_penalty_multiplier: float = 1.5
    availability_penalty_multiplier: float = 2.0
    
    # Contract parameters
    contracted_mw: float = 10.0
    contract_hours_per_day: float = 24.0
    
    # Risk thresholds
    warning_penalty_percent: float = 5.0   # Warn if penalties > 5% of revenue
    critical_penalty_percent: float = 15.0  # Critical if > 15%


@dataclass
class TimeToEvent:
    """Time-based metrics for penalty exposure."""
    first_deviation_seconds: Optional[float] = None
    first_penalty_seconds: Optional[float] = None
    time_in_penalty_seconds: float = 0.0
    time_at_risk_seconds: float = 0.0
    
    @property
    def time_to_penalty_seconds(self) -> Optional[float]:
        """Time from start until first penalty-generating event."""
        return self.first_penalty_seconds


@dataclass
class PenaltyDelta:
    """Delta between two configurations."""
    baseline_penalty_score: float
    comparison_penalty_score: float
    absolute_delta: float  # Positive = comparison is worse
    relative_delta_percent: float
    direction: str  # "BETTER", "WORSE", "SAME"
    
    @property
    def saves_money(self) -> bool:
        return self.direction == "BETTER"


@dataclass
class RevenueAtRisk:
    """Revenue exposure calculation."""
    gross_contracted_revenue: float  # Total contracted £
    expected_penalties: float        # Expected penalty £
    net_revenue: float              # Gross - penalties
    revenue_at_risk_percent: float  # Penalties as % of gross
    risk_level: str                 # "LOW", "MEDIUM", "HIGH", "CRITICAL"


@dataclass
class ComparisonReport:
    """Complete comparison report between configurations."""
    timestamp: datetime
    baseline_name: str
    comparison_name: str
    
    # Core metrics
    penalty_delta: PenaltyDelta
    baseline_revenue: RevenueAtRisk
    comparison_revenue: RevenueAtRisk
    
    # Time metrics
    baseline_time: TimeToEvent
    comparison_time: TimeToEvent
    
    # Summary
    recommendation: str
    savings_estimate: float
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    confidence_reason: str = ""  # Explanation of confidence level
    
    # Details
    details: Dict = field(default_factory=dict)


class ReportEngine:
    """Generates comparison reports answering 'Which saves money?'"""
    
    def __init__(
        self,
        revenue_config: Optional[RevenueConfig] = None,
        penalty_config: Optional[PenaltyConfig] = None,
    ):
        self.revenue_config = revenue_config or RevenueConfig()
        self.penalty_config = penalty_config or PenaltyConfig()
    
    def compare(
        self,
        baseline_events: List[EventRecord],
        comparison_events: List[EventRecord],
        baseline_name: str = "Baseline",
        comparison_name: str = "Proposed",
        simulation_hours: float = 1.0,
    ) -> ComparisonReport:
        """Generate full comparison report.
        
        Args:
            baseline_events: Events from baseline configuration
            comparison_events: Events from comparison configuration
            baseline_name: Name for baseline config
            comparison_name: Name for comparison config
            simulation_hours: Duration of simulation in hours
            
        Returns:
            ComparisonReport with all metrics
        """
        # Evaluate both configurations
        baseline_model = PenaltyModel(self.penalty_config)
        comparison_model = PenaltyModel(self.penalty_config)
        
        baseline_results = [baseline_model.evaluate_event(e) for e in baseline_events]
        comparison_results = [comparison_model.evaluate_event(e) for e in comparison_events]
        
        baseline_summary = baseline_model.get_summary()
        comparison_summary = comparison_model.get_summary()
        
        # Calculate penalty delta
        penalty_delta = self._calculate_penalty_delta(
            baseline_summary, comparison_summary
        )
        
        # Calculate revenue at risk
        baseline_revenue = self._calculate_revenue_at_risk(
            baseline_summary, simulation_hours
        )
        comparison_revenue = self._calculate_revenue_at_risk(
            comparison_summary, simulation_hours
        )
        
        # Calculate time metrics
        baseline_time = self._calculate_time_metrics(baseline_events)
        comparison_time = self._calculate_time_metrics(comparison_events)
        
        # Generate recommendation
        recommendation, savings, confidence, confidence_reason = self._generate_recommendation(
            penalty_delta, baseline_revenue, comparison_revenue
        )
        
        return ComparisonReport(
            timestamp=datetime.utcnow(),
            baseline_name=baseline_name,
            comparison_name=comparison_name,
            penalty_delta=penalty_delta,
            baseline_revenue=baseline_revenue,
            comparison_revenue=comparison_revenue,
            baseline_time=baseline_time,
            comparison_time=comparison_time,
            recommendation=recommendation,
            savings_estimate=savings,
            confidence=confidence,
            confidence_reason=confidence_reason,
            details={
                "baseline_summary": baseline_summary,
                "comparison_summary": comparison_summary,
                "simulation_hours": simulation_hours,
            },
        )
    
    def _calculate_penalty_delta(
        self,
        baseline: Dict,
        comparison: Dict,
    ) -> PenaltyDelta:
        """Calculate penalty delta between configurations."""
        baseline_score = baseline.get("avg_penalty_score", 0.0)
        comparison_score = comparison.get("avg_penalty_score", 0.0)
        
        absolute_delta = comparison_score - baseline_score
        
        if baseline_score > 0:
            relative_delta = (absolute_delta / baseline_score) * 100
        else:
            relative_delta = 100.0 if comparison_score > 0 else 0.0
        
        if abs(absolute_delta) < 0.01:
            direction = "SAME"
        elif absolute_delta < 0:
            direction = "BETTER"  # Comparison has lower penalty
        else:
            direction = "WORSE"
        
        return PenaltyDelta(
            baseline_penalty_score=baseline_score,
            comparison_penalty_score=comparison_score,
            absolute_delta=absolute_delta,
            relative_delta_percent=relative_delta,
            direction=direction,
        )
    
    def _calculate_revenue_at_risk(
        self,
        summary: Dict,
        simulation_hours: float,
    ) -> RevenueAtRisk:
        """Calculate revenue at risk for a configuration."""
        cfg = self.revenue_config
        
        # Gross contracted revenue for simulation period
        gross = cfg.contracted_rate_per_mw_hour * cfg.contracted_mw * simulation_hours
        
        # Expected penalties based on K-factor and availability
        avg_k = summary.get("avg_k_factor", 0.0)
        avg_availability = summary.get("avg_availability", 1.0)
        soe_breaches = summary.get("total_soe_breaches", 0)
        
        # Penalty from underdelivery (K-factor)
        k_penalty = gross * avg_k * cfg.underdelivery_penalty_multiplier
        
        # Penalty from availability issues
        availability_gap = max(0, 1.0 - avg_availability)
        availability_penalty = gross * availability_gap * cfg.availability_penalty_multiplier
        
        # Penalty from SoE breaches (fixed penalty per breach)
        soe_penalty = soe_breaches * (gross * 0.01)  # 1% of gross per breach
        
        total_penalties = k_penalty + availability_penalty + soe_penalty
        net_revenue = gross - total_penalties
        
        risk_percent = (total_penalties / gross * 100) if gross > 0 else 0
        
        # Determine risk level
        if risk_percent < cfg.warning_penalty_percent:
            risk_level = "LOW"
        elif risk_percent < cfg.critical_penalty_percent:
            risk_level = "MEDIUM"
        elif risk_percent < 30:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        return RevenueAtRisk(
            gross_contracted_revenue=gross,
            expected_penalties=total_penalties,
            net_revenue=net_revenue,
            revenue_at_risk_percent=risk_percent,
            risk_level=risk_level,
        )
    
    def _calculate_time_metrics(
        self,
        events: List[EventRecord],
    ) -> TimeToEvent:
        """Calculate time-to-penalty metrics."""
        metrics = TimeToEvent()
        
        tolerance = self.penalty_config.k_factor_tolerance
        
        for event in events:
            for window in event.windows:
                k = KFactor.calculate(
                    window.delivered_mw,
                    window.expected_mw,
                    event.contracted_mw,
                )
                
                # First deviation (any non-zero K)
                if k > 0 and metrics.first_deviation_seconds is None:
                    metrics.first_deviation_seconds = window.start_time
                
                # First penalty (K exceeds tolerance)
                if k > tolerance:
                    if metrics.first_penalty_seconds is None:
                        metrics.first_penalty_seconds = window.start_time
                    metrics.time_in_penalty_seconds += window.duration_seconds
                
                # Time at risk (approaching penalty threshold)
                if k > tolerance * 0.5:
                    metrics.time_at_risk_seconds += window.duration_seconds
        
        return metrics
    
    def _generate_recommendation(
        self,
        delta: PenaltyDelta,
        baseline_rev: RevenueAtRisk,
        comparison_rev: RevenueAtRisk,
    ) -> Tuple[str, float, str, str]:
        """Generate recommendation, savings estimate, confidence, and reason."""
        savings = baseline_rev.expected_penalties - comparison_rev.expected_penalties
        
        # Determine confidence and reason based on magnitude and source of difference
        reasons = []
        
        if abs(delta.relative_delta_percent) > 20:
            confidence = "HIGH"
            reasons.append(f"Large penalty delta ({delta.relative_delta_percent:.1f}%)")
        elif abs(delta.relative_delta_percent) > 5:
            confidence = "MEDIUM"
            reasons.append(f"Moderate penalty delta ({delta.relative_delta_percent:.1f}%)")
        else:
            confidence = "LOW"
            reasons.append(f"Small penalty delta ({delta.relative_delta_percent:.1f}%)")
        
        # Add context about what drove the difference
        if abs(delta.absolute_delta) > 0.1:
            reasons.append(f"Absolute penalty delta of {delta.absolute_delta:.3f}")
        
        # Check penalty scores
        if delta.baseline_penalty_score > 0.3 or delta.comparison_penalty_score > 0.3:
            worse_config = "comparison" if delta.comparison_penalty_score > delta.baseline_penalty_score else "baseline"
            reasons.append(f"High penalty score in {worse_config}")
        
        # Direction-specific context
        if delta.direction == "WORSE" and delta.comparison_penalty_score > 0.2:
            reasons.append("Comparison has elevated penalty risk")
        elif delta.direction == "BETTER" and delta.baseline_penalty_score > 0.2:
            reasons.append("Baseline had elevated penalty risk")
        
        confidence_reason = "; ".join(reasons)
        
        # Generate recommendation
        if delta.direction == "BETTER":
            if savings > 0:
                recommendation = f"ADOPT: Saves £{savings:.2f} in penalties"
            else:
                recommendation = "ADOPT: Lower penalty risk"
        elif delta.direction == "WORSE":
            if savings < 0:
                recommendation = f"REJECT: Costs £{abs(savings):.2f} more in penalties"
            else:
                recommendation = "REJECT: Higher penalty risk"
        else:
            recommendation = "NEUTRAL: No significant difference"
        
        return recommendation, savings, confidence, confidence_reason
    
    def format_report(self, report: ComparisonReport, format: str = "text") -> str:
        """Format report for output."""
        if format == "json":
            return self._format_json(report)
        else:
            return self._format_text(report)
    
    def _format_text(self, r: ComparisonReport) -> str:
        """Format report as text."""
        lines = [
            "=" * 60,
            "CONFIGURATION COMPARISON REPORT",
            "=" * 60,
            f"Generated: {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Baseline:   {r.baseline_name}",
            f"Comparison: {r.comparison_name}",
            "",
            "-" * 60,
            "PENALTY DELTA",
            "-" * 60,
            f"  Baseline score:   {r.penalty_delta.baseline_penalty_score:.4f}",
            f"  Comparison score: {r.penalty_delta.comparison_penalty_score:.4f}",
            f"  Delta:            {r.penalty_delta.absolute_delta:+.4f} ({r.penalty_delta.relative_delta_percent:+.1f}%)",
            f"  Direction:        {r.penalty_delta.direction}",
            "",
            "-" * 60,
            "REVENUE AT RISK",
            "-" * 60,
            f"  {'Metric':<25} {'Baseline':>12} {'Comparison':>12}",
            f"  {'-'*25} {'-'*12} {'-'*12}",
            f"  {'Gross Revenue':<25} £{r.baseline_revenue.gross_contracted_revenue:>10.2f} £{r.comparison_revenue.gross_contracted_revenue:>10.2f}",
            f"  {'Expected Penalties':<25} £{r.baseline_revenue.expected_penalties:>10.2f} £{r.comparison_revenue.expected_penalties:>10.2f}",
            f"  {'Net Revenue':<25} £{r.baseline_revenue.net_revenue:>10.2f} £{r.comparison_revenue.net_revenue:>10.2f}",
            f"  {'Risk %':<25} {r.baseline_revenue.revenue_at_risk_percent:>11.1f}% {r.comparison_revenue.revenue_at_risk_percent:>11.1f}%",
            f"  {'Risk Level':<25} {r.baseline_revenue.risk_level:>12} {r.comparison_revenue.risk_level:>12}",
            "",
            "-" * 60,
            "TIME TO PENALTY",
            "-" * 60,
            f"  {'Metric':<25} {'Baseline':>12} {'Comparison':>12}",
            f"  {'-'*25} {'-'*12} {'-'*12}",
            f"  {'First Deviation (s)':<25} {self._fmt_time(r.baseline_time.first_deviation_seconds):>12} {self._fmt_time(r.comparison_time.first_deviation_seconds):>12}",
            f"  {'First Penalty (s)':<25} {self._fmt_time(r.baseline_time.first_penalty_seconds):>12} {self._fmt_time(r.comparison_time.first_penalty_seconds):>12}",
            f"  {'Time in Penalty (s)':<25} {r.baseline_time.time_in_penalty_seconds:>12.1f} {r.comparison_time.time_in_penalty_seconds:>12.1f}",
            "",
            "=" * 60,
            "RECOMMENDATION",
            "=" * 60,
            f"  {r.recommendation}",
            f"  Savings Estimate: £{r.savings_estimate:.2f}",
            f"  Confidence: {r.confidence}",
            f"  Reason: {r.confidence_reason}",
            "=" * 60,
        ]
        return "\n".join(lines)
    
    def _fmt_time(self, t: Optional[float]) -> str:
        """Format time value."""
        return f"{t:.1f}" if t is not None else "N/A"
    
    def _format_json(self, r: ComparisonReport) -> str:
        """Format report as JSON."""
        data = {
            "timestamp": r.timestamp.isoformat(),
            "baseline_name": r.baseline_name,
            "comparison_name": r.comparison_name,
            "penalty_delta": {
                "baseline_score": r.penalty_delta.baseline_penalty_score,
                "comparison_score": r.penalty_delta.comparison_penalty_score,
                "absolute_delta": r.penalty_delta.absolute_delta,
                "relative_delta_percent": r.penalty_delta.relative_delta_percent,
                "direction": r.penalty_delta.direction,
                "saves_money": r.penalty_delta.saves_money,
            },
            "revenue_at_risk": {
                "baseline": {
                    "gross": r.baseline_revenue.gross_contracted_revenue,
                    "penalties": r.baseline_revenue.expected_penalties,
                    "net": r.baseline_revenue.net_revenue,
                    "risk_percent": r.baseline_revenue.revenue_at_risk_percent,
                    "risk_level": r.baseline_revenue.risk_level,
                },
                "comparison": {
                    "gross": r.comparison_revenue.gross_contracted_revenue,
                    "penalties": r.comparison_revenue.expected_penalties,
                    "net": r.comparison_revenue.net_revenue,
                    "risk_percent": r.comparison_revenue.revenue_at_risk_percent,
                    "risk_level": r.comparison_revenue.risk_level,
                },
            },
            "time_to_penalty": {
                "baseline": {
                    "first_deviation_s": r.baseline_time.first_deviation_seconds,
                    "first_penalty_s": r.baseline_time.first_penalty_seconds,
                    "time_in_penalty_s": r.baseline_time.time_in_penalty_seconds,
                },
                "comparison": {
                    "first_deviation_s": r.comparison_time.first_deviation_seconds,
                    "first_penalty_s": r.comparison_time.first_penalty_seconds,
                    "time_in_penalty_s": r.comparison_time.time_in_penalty_seconds,
                },
            },
            "recommendation": r.recommendation,
            "savings_estimate": r.savings_estimate,
            "confidence": r.confidence,
            "confidence_reason": r.confidence_reason,
        }
        return json.dumps(data, indent=2)


def quick_diff(
    baseline_k_factor: float,
    comparison_k_factor: float,
    contracted_mw: float = 10.0,
    rate_per_mw_hour: float = 10.0,
    hours: float = 1.0,
) -> Dict:
    """Quick diff for simple A vs B comparison.
    
    Returns:
        Dictionary with saves_money, delta, and estimated savings
    """
    gross = rate_per_mw_hour * contracted_mw * hours
    
    baseline_penalty = gross * baseline_k_factor * 1.5
    comparison_penalty = gross * comparison_k_factor * 1.5
    
    savings = baseline_penalty - comparison_penalty
    
    return {
        "saves_money": comparison_k_factor < baseline_k_factor,
        "k_factor_delta": comparison_k_factor - baseline_k_factor,
        "baseline_penalty": baseline_penalty,
        "comparison_penalty": comparison_penalty,
        "savings": savings,
        "recommendation": "ADOPT" if savings > 0 else "REJECT" if savings < 0 else "NEUTRAL",
    }
