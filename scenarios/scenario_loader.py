"""Scenario loader for grid stress pattern testing.

Loads and applies frequency trajectory scenarios for Penalty & Revenue
Evaluation testing. Scenarios define:
- Frequency trajectory (time series of frequency values)
- Ramp rates (df/dt characteristics)
- Duration and recovery behavior
- Penalty implications and test assertions
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple


@dataclass
class FrequencyEvent:
    """A single point in a frequency trajectory."""
    time_offset_seconds: float
    frequency_hz: float
    state: str


@dataclass
class FrequencyTrajectory:
    """Frequency trajectory over time."""
    nominal_hz: float
    events: List[FrequencyEvent]
    oscillation_pattern: Optional[Dict] = None

    def frequency_at(self, t_seconds: float) -> float:
        """Get interpolated frequency at time t."""
        if not self.events:
            return self.nominal_hz

        # Find surrounding events
        prev_event = self.events[0]
        for event in self.events:
            if event.time_offset_seconds > t_seconds:
                break
            prev_event = event

        next_event = prev_event
        for event in self.events:
            if event.time_offset_seconds >= t_seconds:
                next_event = event
                break

        if prev_event == next_event:
            return prev_event.frequency_hz

        # Linear interpolation
        dt = next_event.time_offset_seconds - prev_event.time_offset_seconds
        if dt <= 0:
            return prev_event.frequency_hz

        ratio = (t_seconds - prev_event.time_offset_seconds) / dt
        return prev_event.frequency_hz + ratio * (next_event.frequency_hz - prev_event.frequency_hz)

    def df_dt_at(self, t_seconds: float, delta: float = 0.1) -> float:
        """Get rate of change of frequency (RoCoF) at time t."""
        f1 = self.frequency_at(t_seconds)
        f2 = self.frequency_at(t_seconds + delta)
        return (f2 - f1) / delta

    def iterate(self, step_seconds: float = 1.0) -> Iterator[Tuple[float, float]]:
        """Iterate through trajectory at given time step."""
        if not self.events:
            return
        
        max_time = max(e.time_offset_seconds for e in self.events)
        t = 0.0
        while t <= max_time:
            yield t, self.frequency_at(t)
            t += step_seconds


@dataclass
class RampRate:
    """Ramp rate characteristics."""
    max_df_dt_hz_per_sec: float
    threshold_trigger_hz_per_sec: float = -0.125
    bidirectional: bool = False
    description: str = ""


@dataclass 
class Duration:
    """Event duration characteristics."""
    event_duration_seconds: float
    critical_window_seconds: float
    recovery_window_seconds: float
    oscillation_period_seconds: Optional[float] = None
    settling_time_seconds: Optional[float] = None


@dataclass
class RecoveryBehavior:
    """Recovery behavior specification."""
    type: str  # exponential_decay, linear_ramp, damped_oscillation, constrained_linear
    target_frequency_hz: float = 50.0
    time_constant_seconds: Optional[float] = None
    ramp_rate_hz_per_min: Optional[float] = None
    damping_ratio: Optional[float] = None
    natural_frequency_hz: Optional[float] = None
    overshoot_allowed: bool = False
    max_overshoot_hz: Optional[float] = None
    capacity_limited: bool = False


@dataclass
class PenaltyImplications:
    """Penalty and revenue implications."""
    revenue_impact: str  # high, medium, low, variable, reduced
    dfr_activation_expected: bool = False
    lfdd_risk: bool = False
    sustained_response_required: bool = False
    hunting_risk: bool = False
    partial_delivery_expected: bool = False
    notes: str = ""


@dataclass
class TestAssertions:
    """Test assertions for the scenario."""
    frequency_nadir_above_hz: Optional[float] = None
    recovery_within_seconds: Optional[float] = None
    dfr_response_within_seconds: Optional[float] = None
    sustained_delivery_minutes: Optional[float] = None
    no_hunting_behavior: bool = False
    max_response_reversals: Optional[int] = None
    settling_within_seconds: Optional[float] = None
    deadband_respected: bool = False
    delivery_matches_declaration: bool = False
    no_overcommitment: bool = False
    availability_signal_accurate: bool = False
    penalty_correctly_calculated: bool = False


@dataclass
class CapacityConstraints:
    """Capacity constraint specification for partial capacity scenarios."""
    available_capacity_percent: float
    constraint_reasons: List[str] = field(default_factory=list)
    affected_assets: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class Scenario:
    """A complete grid stress scenario."""
    scenario_id: str
    name: str
    description: str
    category: str
    severity: str
    
    frequency_trajectory: FrequencyTrajectory
    ramp_rate: RampRate
    duration: Duration
    recovery_behavior: RecoveryBehavior
    penalty_implications: PenaltyImplications
    test_assertions: TestAssertions
    capacity_constraints: Optional[CapacityConstraints] = None

    def frequency_at(self, t_seconds: float) -> float:
        """Get frequency at time t."""
        return self.frequency_trajectory.frequency_at(t_seconds)

    def df_dt_at(self, t_seconds: float) -> float:
        """Get df/dt at time t."""
        return self.frequency_trajectory.df_dt_at(t_seconds)

    def is_within_limits(self, t_seconds: float) -> bool:
        """Check if frequency is within acceptable limits at time t."""
        f = self.frequency_at(t_seconds)
        return 49.5 <= f <= 50.5

    def get_event_state(self, t_seconds: float) -> str:
        """Get the event state at time t."""
        for event in reversed(self.frequency_trajectory.events):
            if event.time_offset_seconds <= t_seconds:
                return event.state
        return "normal"


class ScenarioLoader:
    """Loads and manages grid stress scenarios."""

    def __init__(self, scenarios_dir: Optional[Path] = None):
        if scenarios_dir is None:
            scenarios_dir = Path(__file__).parent
        self.scenarios_dir = scenarios_dir
        self._cache: Dict[str, Scenario] = {}

    def list_available(self) -> List[str]:
        """List available scenario files."""
        return [f.stem for f in self.scenarios_dir.glob("*.json")]

    def load(self, scenario_name: str) -> Scenario:
        """Load a scenario by name."""
        if scenario_name in self._cache:
            return self._cache[scenario_name]

        path = self.scenarios_dir / f"{scenario_name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Scenario not found: {scenario_name}")

        with open(path) as f:
            data = json.load(f)

        scenario = self._parse_scenario(data)
        self._cache[scenario_name] = scenario
        return scenario

    def _parse_scenario(self, data: Dict) -> Scenario:
        """Parse scenario from JSON data."""
        # Parse frequency trajectory
        traj_data = data["frequency_trajectory"]
        events = [
            FrequencyEvent(
                time_offset_seconds=e["time_offset_seconds"],
                frequency_hz=e["frequency_hz"],
                state=e["state"],
            )
            for e in traj_data.get("events", [])
        ]
        trajectory = FrequencyTrajectory(
            nominal_hz=traj_data["nominal_hz"],
            events=events,
            oscillation_pattern=traj_data.get("oscillation_pattern"),
        )

        # Parse ramp rate
        rr_data = data["ramp_rate"]
        ramp_rate = RampRate(
            max_df_dt_hz_per_sec=rr_data["max_df_dt_hz_per_sec"],
            threshold_trigger_hz_per_sec=rr_data.get("threshold_trigger_hz_per_sec", -0.125),
            bidirectional=rr_data.get("bidirectional", False),
            description=rr_data.get("description", ""),
        )

        # Parse duration
        dur_data = data["duration"]
        duration = Duration(
            event_duration_seconds=dur_data["event_duration_seconds"],
            critical_window_seconds=dur_data["critical_window_seconds"],
            recovery_window_seconds=dur_data["recovery_window_seconds"],
            oscillation_period_seconds=dur_data.get("oscillation_period_seconds"),
            settling_time_seconds=dur_data.get("settling_time_seconds"),
        )

        # Parse recovery behavior
        rec_data = data["recovery_behavior"]
        recovery = RecoveryBehavior(
            type=rec_data["type"],
            target_frequency_hz=rec_data.get("target_frequency_hz", 50.0),
            time_constant_seconds=rec_data.get("time_constant_seconds"),
            ramp_rate_hz_per_min=rec_data.get("ramp_rate_hz_per_min"),
            damping_ratio=rec_data.get("damping_ratio"),
            natural_frequency_hz=rec_data.get("natural_frequency_hz"),
            overshoot_allowed=rec_data.get("overshoot_allowed", False),
            max_overshoot_hz=rec_data.get("max_overshoot_hz"),
            capacity_limited=rec_data.get("capacity_limited", False),
        )

        # Parse penalty implications
        pen_data = data["penalty_implications"]
        penalty = PenaltyImplications(
            revenue_impact=pen_data["revenue_impact"],
            dfr_activation_expected=pen_data.get("dfr_activation_expected", False),
            lfdd_risk=pen_data.get("lfdd_risk", False),
            sustained_response_required=pen_data.get("sustained_response_required", False),
            hunting_risk=pen_data.get("hunting_risk", False),
            partial_delivery_expected=pen_data.get("partial_delivery_expected", False),
            notes=pen_data.get("notes", ""),
        )

        # Parse test assertions
        assert_data = data["test_assertions"]
        assertions = TestAssertions(
            frequency_nadir_above_hz=assert_data.get("frequency_nadir_above_hz"),
            recovery_within_seconds=assert_data.get("recovery_within_seconds"),
            dfr_response_within_seconds=assert_data.get("dfr_response_within_seconds"),
            sustained_delivery_minutes=assert_data.get("sustained_delivery_minutes"),
            no_hunting_behavior=assert_data.get("no_hunting_behavior", False),
            max_response_reversals=assert_data.get("max_response_reversals"),
            settling_within_seconds=assert_data.get("settling_within_seconds"),
            deadband_respected=assert_data.get("deadband_respected", False),
            delivery_matches_declaration=assert_data.get("delivery_matches_declaration", False),
            no_overcommitment=assert_data.get("no_overcommitment", False),
            availability_signal_accurate=assert_data.get("availability_signal_accurate", False),
            penalty_correctly_calculated=assert_data.get("penalty_correctly_calculated", False),
        )

        # Parse capacity constraints if present
        capacity = None
        if "capacity_constraints" in data:
            cap_data = data["capacity_constraints"]
            capacity = CapacityConstraints(
                available_capacity_percent=cap_data["available_capacity_percent"],
                constraint_reasons=cap_data.get("constraint_reasons", []),
                affected_assets=cap_data.get("affected_assets", {}),
            )

        return Scenario(
            scenario_id=data["scenario_id"],
            name=data["name"],
            description=data["description"],
            category=data["category"],
            severity=data["severity"],
            frequency_trajectory=trajectory,
            ramp_rate=ramp_rate,
            duration=duration,
            recovery_behavior=recovery,
            penalty_implications=penalty,
            test_assertions=assertions,
            capacity_constraints=capacity,
        )


# Module-level convenience functions
_default_loader: Optional[ScenarioLoader] = None


def _get_loader() -> ScenarioLoader:
    global _default_loader
    if _default_loader is None:
        _default_loader = ScenarioLoader()
    return _default_loader


def load_scenario(name: str) -> Scenario:
    """Load a scenario by name."""
    return _get_loader().load(name)


def list_scenarios() -> List[str]:
    """List available scenarios."""
    return _get_loader().list_available()
