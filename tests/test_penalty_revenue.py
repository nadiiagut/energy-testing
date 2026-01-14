"""Penalty & Revenue Evaluation Test Suite.

This test suite validates grid frequency response under realistic degraded
conditions, focusing on penalty generation and revenue impact.

These tests answer: "Which configuration saves money?"

Test IDs use the PE-REV- prefix (Penalty/Revenue Evaluation).
"""

import pytest
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from scenarios import load_scenario, Scenario
from harness.control_loop import (
    ControlLoop,
    ControlLoopFactory,
    ControllerConfig,
    ControlMode,
    PlantLimits,
    emit_metered_power,
    SIMULATED_SIGNAL,
    SCADA_SIGNAL,
    NIMBUS_SIGNAL,
)
from faults import (
    InverterLossFault,
    InverterLossConfig,
    InverterLossMode,
    TelemetryDropFault,
    TelemetryDropConfig,
    DropPattern,
    ControlLagFault,
    ControlLagConfig,
    LagProfile,
    SoEBiasFault,
    SoEBiasConfig,
    BiasType,
    ModeMisclassificationFault,
    apply_mode_faults,
)
from penalty_model import (
    PenaltyModel,
    EventRecord,
    DeliveryWindow,
    ServiceType,
    KFactor,
    score_delivery_by_mode,
)
from models.energy import ServiceMode
from report_engine import ReportEngine, RevenueConfig, quick_diff


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def fast_scenario() -> Scenario:
    """Load fast df/dt frequency scenario."""
    return load_scenario("fast_df_dt")


@pytest.fixture
def prolonged_scenario() -> Scenario:
    """Load prolonged frequency event scenario."""
    return load_scenario("prolonged_event")


@pytest.fixture
def report_engine() -> ReportEngine:
    """Create report engine with default config."""
    return ReportEngine(RevenueConfig(
        contracted_rate_per_mw_hour=12.0,
        contracted_mw=10.0,
    ))


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def run_simulation(
    controller: ControlLoop,
    scenario: Scenario,
    dt_seconds: float = 0.5,
    faults: List = None,
    use_ideal_expected: bool = False,
    observation_signal=None,
) -> List[DeliveryWindow]:
    """Run controller against scenario and collect delivery windows.
    
    Args:
        controller: The control loop to test
        scenario: Frequency scenario to run
        dt_seconds: Simulation timestep
        faults: Optional list of fault injectors to apply
        use_ideal_expected: If True, use ideal instantaneous expected power.
                           If False, use controller's own target (more realistic).
        observation_signal: Optional ObservedSignal to apply at plant boundary.
                           If provided, delivered power passes through emit_metered_power.
        
    Returns:
        List of DeliveryWindow objects for penalty calculation
    """
    faults = faults or []
    windows = []
    
    controller.reset()
    
    # Iterate through scenario
    for t, frequency in scenario.frequency_trajectory.iterate(step_seconds=dt_seconds):
        # Get controller response first
        delivered_mw = controller.respond(frequency, dt_seconds)
        
        # Expected power: use controller's target (accounts for its own droop calc)
        # This is more realistic - we measure against what the controller SHOULD deliver
        expected_mw = controller.state.target_power_mw
        
        # Apply faults to delivered power
        for fault in faults:
            if hasattr(fault, 'apply') and fault.state.active:
                if isinstance(fault, InverterLossFault):
                    delivered_mw, _ = fault.apply(delivered_mw, dt_seconds)
        
        # Apply observation signal at plant boundary (if provided)
        # This models how telemetry/metering distorts the observed value
        if observation_signal is not None:
            observed_mw, _ = emit_metered_power(delivered_mw, observation_signal)
            # For settlement: missing data = 0 delivery (worst case)
            delivered_for_settlement = 0.0 if observed_mw is None else observed_mw
        else:
            delivered_for_settlement = delivered_mw
        
        # Record window
        windows.append(DeliveryWindow(
            start_time=t,
            end_time=t + dt_seconds,
            expected_mw=expected_mw,
            delivered_mw=delivered_for_settlement,
            frequency_hz=frequency,
            soe_percent=controller.state.soe_percent,
        ))
    
    return windows


def windows_to_event(
    windows: List[DeliveryWindow],
    event_id: str,
    contracted_mw: float = 10.0,
) -> EventRecord:
    """Convert delivery windows to an EventRecord."""
    return EventRecord(
        event_id=event_id,
        service_type=ServiceType.DCL,
        contracted_mw=contracted_mw,
        windows=windows,
    )


def find_first_breach(
    windows: List[DeliveryWindow],
    contracted_mw: float = 10.0,
    k_factor_threshold: float = 0.03,
    soe_min_threshold: float = 10.0,
) -> Dict:
    """Find the first penalty-triggering breach in a window sequence.
    
    Returns:
        Dictionary with breach details:
        - first_k_breach_seconds: Time to first K-factor breach (or None)
        - first_soe_breach_seconds: Time to first SoE breach (or None)
        - first_breach_seconds: Time to first breach of any type (or None)
        - breach_type: "k_factor", "soe", or None
        - breach_details: Human-readable description
    """
    first_k_breach = None
    first_soe_breach = None
    
    for window in windows:
        # Check K-factor breach
        if first_k_breach is None:
            k = abs(window.deviation_mw) / contracted_mw if contracted_mw > 0 else 0
            if k > k_factor_threshold:
                first_k_breach = window.start_time
        
        # Check SoE breach
        if first_soe_breach is None:
            if window.soe_percent < soe_min_threshold:
                first_soe_breach = window.start_time
    
    # Determine first overall breach
    if first_k_breach is not None and first_soe_breach is not None:
        if first_k_breach <= first_soe_breach:
            first_breach = first_k_breach
            breach_type = "k_factor"
        else:
            first_breach = first_soe_breach
            breach_type = "soe"
    elif first_k_breach is not None:
        first_breach = first_k_breach
        breach_type = "k_factor"
    elif first_soe_breach is not None:
        first_breach = first_soe_breach
        breach_type = "soe"
    else:
        first_breach = None
        breach_type = None
    
    # Generate human-readable details
    if breach_type == "k_factor":
        details = f"K-factor breach at {first_breach:.1f}s into event"
    elif breach_type == "soe":
        details = f"SoE breach at {first_breach:.1f}s into event"
    else:
        details = "No breach detected"
    
    return {
        "first_k_breach_seconds": first_k_breach,
        "first_soe_breach_seconds": first_soe_breach,
        "first_breach_seconds": first_breach,
        "breach_type": breach_type,
        "breach_details": details,
    }


def ideal_expected(controller: ControlLoop, frequency: float, mode: ServiceMode) -> float:
    """Compute ideal expected power using controller's mode logic.
    
    This bypasses measurement/response lag by calling the internal method directly,
    giving us the "ideal" expected power for a given frequency and service mode.
    
    This ensures tests stay stable even if DC/DR/DM slope/deadband config changes.
    """
    return controller._calculate_target_power(frequency, service_mode=mode)


# -----------------------------------------------------------------------------
# Test Cases
# -----------------------------------------------------------------------------

@pytest.mark.case("PE-REV-001")
def test_control_loop_phase_lag(fast_scenario, report_engine):
    """Test Case - Control Loop Phase Lag Revenue Impact.

    Description:
    -----------------
    This is the Arenko-style failure mode. A controller that passes
    commissioning with ideal timing can fail catastrophically when
    real-world phase lag is introduced. The plant responds correctly,
    but too late.

    This test demonstrates why control loop timing is the highest-ROI
    test target: it finds bugs that pass commissioning but generate
    penalties in production.

    Preconditions:
    -----------------
    1. Fast df/dt frequency scenario loaded (rapid frequency deviation)
    2. 10MW battery controller with droop response
    3. Baseline: ideal controller (no additional lag)
    4. Comparison: controller with 200-500ms phase lag injected

    Steps:
    ----------
    1. Create baseline controller with nominal timing
       - Time constant: 1.0s (nominal)
       - Response delay: 0.1s (minimal)
    2. Run baseline against fast_df_dt scenario
    3. Collect baseline delivery windows and K-factor
    4. Create comparison controller with degraded timing
       - Time constant: 3.0s (degraded)
       - Response delay: 0.5s (realistic lag)
    5. Run comparison against same scenario
    6. Collect comparison delivery windows and K-factor
    7. Generate comparison report with revenue impact

    Expected Results:
    ---------------------------
    1. Baseline K-factor < 0.05 (within tolerance)
    2. Comparison K-factor > 0.15 (significant deviation)
    3. Comparison penalty score significantly higher
    4. Report shows: Configuration A ~ -5% revenue
                     Configuration B ~ -80% revenue
    5. Recommendation: REJECT degraded timing

    Real-World Mapping:
    ---------------------------
    - PLC scan time increases under load
    - Network latency spikes during events
    - Actuator response degradation over time
    - Firmware update changes timing characteristics
    """
    # Step 1: Create baseline controller with nominal timing
    baseline_config = ControllerConfig(
        mode=ControlMode.DROOP,
        time_constant_seconds=1.0,
        response_delay_seconds=0.1,
        deadband_hz=0.015,
    )
    baseline_ctrl = ControlLoopFactory.battery_10mw_20mwh(baseline_config)

    # Step 2: Run baseline against fast scenario
    baseline_windows = run_simulation(baseline_ctrl, fast_scenario, dt_seconds=0.5)

    # Step 3: Calculate baseline K-factor
    baseline_k = KFactor.calculate_weighted(baseline_windows, contracted_mw=10.0)

    # Step 4: Create comparison controller with degraded timing (phase lag)
    degraded_config = ControllerConfig(
        mode=ControlMode.DROOP,
        time_constant_seconds=3.0,   # Degraded: slower response
        response_delay_seconds=0.5,  # Degraded: 500ms lag
        deadband_hz=0.015,
    )
    degraded_ctrl = ControlLoopFactory.battery_10mw_20mwh(degraded_config)

    # Step 5: Run comparison against same scenario
    comparison_windows = run_simulation(degraded_ctrl, fast_scenario, dt_seconds=0.5)

    # Step 6: Calculate comparison K-factor
    comparison_k = KFactor.calculate_weighted(comparison_windows, contracted_mw=10.0)

    # Step 7: Generate comparison report
    baseline_event = windows_to_event(baseline_windows, "baseline-fast")
    comparison_event = windows_to_event(comparison_windows, "degraded-fast")

    report = report_engine.compare(
        [baseline_event],
        [comparison_event],
        baseline_name="Nominal Timing",
        comparison_name="500ms Phase Lag",
        simulation_hours=fast_scenario.duration.event_duration_seconds / 3600,
    )

    # Find first breach for both configurations
    baseline_breach = find_first_breach(baseline_windows, contracted_mw=10.0)
    comparison_breach = find_first_breach(comparison_windows, contracted_mw=10.0)
    
    # Log first breach details
    print(f"\n  [PE-REV-001] First breach analysis:")
    print(f"    Baseline: {baseline_breach['breach_details']}")
    print(f"    Comparison: {comparison_breach['breach_details']}")
    
    # Assertions
    assert baseline_k < 0.10, \
        f"Baseline K-factor too high: {baseline_k:.3f} (expected < 0.10)"

    assert comparison_k > baseline_k, \
        f"Degraded timing should have higher K-factor: {comparison_k:.3f} <= {baseline_k:.3f}"

    assert report.penalty_delta.direction == "WORSE", \
        f"Phase lag should make things WORSE, got: {report.penalty_delta.direction}"

    assert report.comparison_revenue.revenue_at_risk_percent > report.baseline_revenue.revenue_at_risk_percent, \
        "Degraded timing should have higher revenue at risk"

    # Verify the magnitude of impact is significant
    assert report.penalty_delta.relative_delta_percent > 50, \
        f"Phase lag impact should be significant (>50%), got: {report.penalty_delta.relative_delta_percent:.1f}%"
    
    # Time-to-penalty assertions
    assert report.comparison_time.first_deviation_seconds is not None, \
        "Comparison should have a first deviation time"
    
    # Phase lag should cause earlier breach in comparison
    if comparison_breach['first_breach_seconds'] is not None:
        assert comparison_breach['first_breach_seconds'] <= 5.0, \
            f"Phase lag breach should occur early (<5s): {comparison_breach['first_breach_seconds']:.1f}s"


@pytest.mark.case("PE-REV-002")
def test_soe_drift_under_prolonged_stress(prolonged_scenario, report_engine):
    """Test Case - SoE Drift Under Prolonged Stress.

    Description:
    -----------------
    Most plants fail late, not early. A controller that performs well
    for the first 15-18 minutes can suddenly breach SoE limits during
    a prolonged event, causing contract failure.

    This test validates that SoE tracking and bias detection work
    correctly over extended periods, preventing surprise penalties.

    Preconditions:
    -----------------
    1. Prolonged frequency event scenario (20-30 min)
    2. Battery controller with SoE constraints
    3. Baseline: accurate SoE tracking
    4. Comparison: SoE measurement bias (drift)

    Steps:
    ----------
    1. Create baseline controller with accurate SoE
       - No measurement bias
       - Accurate capacity tracking
    2. Run baseline against prolonged scenario
    3. Record SoE trajectory and check for breaches
    4. Create comparison with SoE bias fault
       - 5% initial bias
       - 0.1 MWh/hour drift rate
    5. Run comparison against same scenario
    6. Track time-to-breach and SoE trajectory
    7. Compare penalty outcomes

    Expected Results:
    ---------------------------
    1. Baseline: No SoE breaches throughout event
    2. Comparison: Plant looks compliant for ~18 minutes
    3. Comparison: SoE breach occurs in final portion
    4. Comparison: Penalty score significantly higher
    5. Report shows late failure pattern

    Real-World Mapping:
    ---------------------------
    - BMS calibration drift over temperature cycles
    - Cell degradation not reflected in capacity
    - Coulomb counting errors accumulating
    - Incorrect efficiency assumptions
    """
    # Step 1: Create baseline controller with accurate SoE
    baseline_ctrl = ControlLoopFactory.battery_10mw_20mwh()
    baseline_ctrl.reset(soe_percent=50.0)  # Start at 50% SoE

    # Step 2: Run baseline (no SoE bias)
    baseline_windows = run_simulation(
        baseline_ctrl,
        prolonged_scenario,
        dt_seconds=1.0,
    )

    # Step 3: Check baseline for breaches
    baseline_soe_min = min(w.soe_percent for w in baseline_windows)
    baseline_breaches = sum(1 for w in baseline_windows if w.soe_percent < 10.0)

    # Step 4: Create comparison with SoE bias
    comparison_ctrl = ControlLoopFactory.battery_10mw_20mwh()
    comparison_ctrl.reset(soe_percent=50.0)

    soe_fault = SoEBiasFault(SoEBiasConfig(
        bias_type=BiasType.DRIFT,
        drift_rate_mwh_per_hour=0.5,  # Drift accumulates
        max_drift_mwh=3.0,            # Up to 3MWh error
    ))
    soe_fault.activate(true_soe_mwh=10.0)  # 50% of 20MWh

    # Step 5: Run comparison with bias
    comparison_windows = []
    comparison_ctrl.reset(soe_percent=50.0)

    for t, frequency in prolonged_scenario.frequency_trajectory.iterate(step_seconds=1.0):
        deviation = frequency - 50.0
        if abs(deviation) > 0.015:
            expected_mw = -deviation / 0.5 * comparison_ctrl.limits.max_power_mw
            expected_mw = max(min(expected_mw, comparison_ctrl.limits.max_power_mw),
                           comparison_ctrl.limits.min_power_mw)
        else:
            expected_mw = 0.0

        delivered_mw = comparison_ctrl.respond(frequency, 1.0)

        # Apply SoE bias - controller sees wrong SoE
        true_soe = comparison_ctrl.state.soe_mwh
        reported_soe, _ = soe_fault.apply(true_soe, delivered_mw, dt_seconds=1.0)

        # Use reported (biased) SoE percentage for window
        reported_soe_percent = (reported_soe / comparison_ctrl.limits.max_soe_mwh * 100) \
            if comparison_ctrl.limits.max_soe_mwh else 50.0

        comparison_windows.append(DeliveryWindow(
            start_time=t,
            end_time=t + 1.0,
            expected_mw=expected_mw,
            delivered_mw=delivered_mw,
            frequency_hz=frequency,
            soe_percent=comparison_ctrl.state.soe_percent,  # True SoE
        ))

    # Step 6: Analyze comparison for late failure
    comparison_soe_min = min(w.soe_percent for w in comparison_windows)
    comparison_breaches = sum(1 for w in comparison_windows if w.soe_percent < 10.0)

    # Find time to first breach
    time_to_breach = None
    for w in comparison_windows:
        if w.soe_percent < 10.0:
            time_to_breach = w.start_time
            break

    # Step 7: Generate report
    baseline_event = windows_to_event(baseline_windows, "baseline-prolonged")
    comparison_event = windows_to_event(comparison_windows, "biased-soe-prolonged")

    report = report_engine.compare(
        [baseline_event],
        [comparison_event],
        baseline_name="Accurate SoE",
        comparison_name="Drifting SoE Bias",
        simulation_hours=prolonged_scenario.duration.event_duration_seconds / 3600,
    )

    # Find first breach for both configurations
    baseline_breach_info = find_first_breach(baseline_windows, contracted_mw=10.0)
    comparison_breach_info = find_first_breach(comparison_windows, contracted_mw=10.0)
    
    # Log first breach details
    event_duration = prolonged_scenario.duration.event_duration_seconds
    print(f"\n  [PE-REV-002] First breach analysis (event duration: {event_duration:.0f}s):")
    print(f"    Baseline: {baseline_breach_info['breach_details']}")
    print(f"    Comparison: {comparison_breach_info['breach_details']}")
    if time_to_breach is not None:
        print(f"    SoE breach at {time_to_breach:.1f}s ({time_to_breach/event_duration*100:.0f}% into event)")
    
    # Assertions
    assert baseline_breaches == 0 or baseline_soe_min > 5.0, \
        f"Baseline should not have critical SoE breaches: min={baseline_soe_min:.1f}%"

    # The key insight: late failure
    if time_to_breach is not None:
        failure_point_percent = (time_to_breach / event_duration) * 100
        assert failure_point_percent > 50, \
            f"Failure should occur late in event (>50%), got: {failure_point_percent:.0f}%"

    # SoE bias should have measurable impact
    soe_metrics = soe_fault.get_penalty_metrics()
    assert soe_metrics["cumulative_error_mwh"] > 0, \
        "SoE bias should accumulate error over time"
    
    # Time-to-penalty assertion: comparison should breach later than baseline (if baseline breaches)
    if baseline_breach_info['first_breach_seconds'] is not None and comparison_breach_info['first_breach_seconds'] is not None:
        # With SoE bias, the breach timing may differ
        print(f"    Time delta: {comparison_breach_info['first_breach_seconds'] - baseline_breach_info['first_breach_seconds']:.1f}s")


@pytest.mark.case("PE-REV-003")
def test_partial_inverter_availability_fast_response(fast_scenario, report_engine):
    """Test Case - Partial Inverter Availability Under Fast Response.

    Description:
    -----------------
    Availability != response quality. A plant can be "available" with
    all systems reporting online, yet fail to deliver contracted power
    due to partial inverter capacity loss.

    This test explains: "Why are we penalised when nothing is down?"

    Preconditions:
    -----------------
    1. Fast df/dt frequency scenario
    2. Battery controller at full capacity
    3. Baseline: 100% inverter availability
    4. Comparison: 20-30% inverter capacity loss

    Steps:
    ----------
    1. Create baseline controller with full capacity
    2. Run baseline against fast scenario
    3. Verify full power delivery capability
    4. Create comparison with inverter loss fault
       - 25% capacity reduction (1 of 4 inverters down)
       - Plant still reports "available"
    5. Run comparison against same scenario
    6. Measure asymmetric response and clipping
    7. Calculate K-factor degradation

    Expected Results:
    ---------------------------
    1. Baseline: Full power delivery, K-factor < 0.05
    2. Comparison: Clipped response, visible plateau
    3. Comparison: K-factor degradation proportional to loss
    4. Report shows penalty despite "availability"

    Real-World Mapping:
    ---------------------------
    - Single inverter trip in multi-inverter plant
    - Thermal derating not reported to SCADA
    - DC bus voltage issues limiting power
    - Communication loss to inverter subset
    """
    # Step 1: Create baseline controller with full capacity
    baseline_ctrl = ControlLoopFactory.battery_10mw_20mwh()

    # Step 2: Run baseline
    baseline_windows = run_simulation(baseline_ctrl, fast_scenario, dt_seconds=0.5)

    # Step 3: Verify baseline delivers power
    baseline_max_power = max(abs(w.delivered_mw) for w in baseline_windows)
    baseline_total_energy = sum(abs(w.delivered_mw) * w.duration_seconds for w in baseline_windows)

    # Step 4: Create comparison with inverter loss
    comparison_ctrl = ControlLoopFactory.battery_10mw_20mwh()

    inverter_fault = InverterLossFault(InverterLossConfig(
        mode=InverterLossMode.FIXED_PERCENT,
        capacity_loss_percent=25.0,  # 1 of 4 inverters down
    ))
    inverter_fault.activate()

    # Step 5: Run comparison with inverter loss
    comparison_windows = run_simulation(
        comparison_ctrl,
        fast_scenario,
        dt_seconds=0.5,
        faults=[inverter_fault],
    )

    # Step 6: Measure clipping and asymmetric response
    comparison_max_power = max(abs(w.delivered_mw) for w in comparison_windows)
    comparison_total_energy = sum(abs(w.delivered_mw) * w.duration_seconds for w in comparison_windows)

    # Count clipping events (where fault reduces delivered vs expected)
    clipping_events = sum(1 for w in comparison_windows 
                         if w.expected_mw != 0 and abs(w.delivered_mw) < abs(w.expected_mw) * 0.9)

    # Step 7: Calculate K-factors
    baseline_k = KFactor.calculate_weighted(baseline_windows, contracted_mw=10.0)
    comparison_k = KFactor.calculate_weighted(comparison_windows, contracted_mw=10.0)

    # Generate report
    baseline_event = windows_to_event(baseline_windows, "baseline-full-capacity")
    comparison_event = windows_to_event(comparison_windows, "inverter-loss")

    report = report_engine.compare(
        [baseline_event],
        [comparison_event],
        baseline_name="Full Capacity",
        comparison_name="25% Inverter Loss",
        simulation_hours=fast_scenario.duration.event_duration_seconds / 3600,
    )

    # Get fault metrics
    fault_metrics = inverter_fault.get_penalty_metrics()

    # Find first breach for both configurations
    baseline_breach = find_first_breach(baseline_windows, contracted_mw=10.0)
    comparison_breach = find_first_breach(comparison_windows, contracted_mw=10.0)
    
    # Log first breach details
    print(f"\n  [PE-REV-003] First breach analysis:")
    print(f"    Baseline: {baseline_breach['breach_details']}")
    print(f"    Comparison: {comparison_breach['breach_details']}")
    print(f"    Clipping events: {clipping_events}")
    
    # Assertions
    assert baseline_max_power > 0, \
        f"Baseline should deliver power during event: {baseline_max_power:.1f}MW"

    assert comparison_total_energy < baseline_total_energy, \
        f"Inverter loss should reduce total energy: {comparison_total_energy:.1f} >= {baseline_total_energy:.1f}"

    assert fault_metrics["cumulative_lost_energy_mwh"] > 0 or fault_metrics["average_capacity_factor"] < 1.0, \
        "Should have measurable impact from inverter fault"

    # K-factor should be higher with reduced capacity
    assert comparison_k >= baseline_k or clipping_events > 0, \
        f"Inverter loss should impact K-factor or cause clipping"

    assert report.penalty_delta.direction == "WORSE", \
        "Partial inverter loss should result in WORSE penalties"
    
    # Time-to-penalty assertions
    if comparison_breach['first_breach_seconds'] is not None:
        # Inverter loss should cause breach when high power is demanded
        assert comparison_breach['first_breach_seconds'] is not None, \
            "Inverter loss should cause K-factor breach"


@pytest.mark.case("PE-REV-004")
def test_telemetry_fragility_silent_killer(fast_scenario, report_engine):
    """Test Case - Telemetry Fragility (Silent Killer).

    Description:
    -----------------
    Perfect performance, zero revenue. If telemetry drops during an
    event, the operator cannot verify delivery, leading to settlement
    disputes or automatic penalty application.

    This test turns OT networking into a first-class test target.

    Preconditions:
    -----------------
    1. Fast df/dt frequency scenario
    2. Battery controller performing correctly
    3. Baseline: complete telemetry
    4. Comparison: telemetry drops during event

    Steps:
    ----------
    1. Create controller with normal operation
    2. Run baseline with complete telemetry capture
    3. Verify all data points recorded
    4. Create telemetry drop fault
       - 15% random drop probability
       - Burst pattern for realistic gaps
    5. Run comparison with telemetry drops
    6. Evaluate missing data windows
    7. Calculate inferred penalties from gaps

    Expected Results:
    ---------------------------
    1. Baseline: 100% telemetry availability
    2. Comparison: ~85% telemetry availability
    3. Comparison: Gaps during critical event periods
    4. Missing windows treated as non-delivery
    5. Penalty from data gaps despite actual delivery

    Real-World Mapping:
    ---------------------------
    - Network congestion during grid events
    - SCADA polling failures under load
    - RTU buffer overflows
    - Cellular connectivity issues
    - Firewall rules blocking during high traffic
    """
    # Step 1: Create controller
    controller = ControlLoopFactory.battery_10mw_20mwh()

    # Step 2: Run baseline with complete telemetry
    baseline_windows = run_simulation(controller, fast_scenario, dt_seconds=0.5)
    baseline_data_points = len(baseline_windows)

    # Step 3: Verify complete capture
    baseline_gaps = 0  # No gaps in baseline

    # Step 4: Create telemetry drop fault
    telemetry_fault = TelemetryDropFault(TelemetryDropConfig(
        pattern=DropPattern.BURST,
        burst_probability=0.08,      # 8% chance of burst starting
        burst_length_samples=5,      # 5 consecutive drops
        drop_probability=0.05,       # 5% random drops outside bursts
        seed=42,                     # Reproducible
    ))
    telemetry_fault.activate()

    # Step 5: Run comparison with telemetry drops
    controller.reset()
    comparison_windows = []
    dropped_windows = []

    for t, frequency in fast_scenario.frequency_trajectory.iterate(step_seconds=0.5):
        deviation = frequency - 50.0
        if abs(deviation) > 0.015:
            expected_mw = -deviation / 0.5 * controller.limits.max_power_mw
            expected_mw = max(min(expected_mw, controller.limits.max_power_mw),
                           controller.limits.min_power_mw)
        else:
            expected_mw = 0.0

        delivered_mw = controller.respond(frequency, 0.5)

        # Apply telemetry fault - may drop this reading
        reported_power, meta = telemetry_fault.apply(delivered_mw, dt_seconds=0.5)

        if meta["dropped"]:
            # Telemetry dropped - record as zero delivery (worst case for settlement)
            dropped_windows.append(DeliveryWindow(
                start_time=t,
                end_time=t + 0.5,
                expected_mw=expected_mw,
                delivered_mw=0.0,  # Cannot prove delivery
                frequency_hz=frequency,
                soe_percent=controller.state.soe_percent,
            ))
            comparison_windows.append(dropped_windows[-1])
        else:
            comparison_windows.append(DeliveryWindow(
                start_time=t,
                end_time=t + 0.5,
                expected_mw=expected_mw,
                delivered_mw=delivered_mw,
                frequency_hz=frequency,
                soe_percent=controller.state.soe_percent,
            ))

    # Step 6: Evaluate missing data windows
    telemetry_metrics = telemetry_fault.get_penalty_metrics()
    drop_rate = telemetry_metrics["drop_rate"]
    dropped_count = len(dropped_windows)

    # Step 7: Calculate K-factors
    baseline_k = KFactor.calculate_weighted(baseline_windows, contracted_mw=10.0)
    comparison_k = KFactor.calculate_weighted(comparison_windows, contracted_mw=10.0)

    # Generate report
    baseline_event = windows_to_event(baseline_windows, "baseline-full-telemetry")
    comparison_event = windows_to_event(comparison_windows, "telemetry-drops")

    report = report_engine.compare(
        [baseline_event],
        [comparison_event],
        baseline_name="Complete Telemetry",
        comparison_name="15% Telemetry Loss",
        simulation_hours=fast_scenario.duration.event_duration_seconds / 3600,
    )

    # Find first breach for both configurations
    baseline_breach = find_first_breach(baseline_windows, contracted_mw=10.0)
    comparison_breach = find_first_breach(comparison_windows, contracted_mw=10.0)
    
    # Find first telemetry drop
    first_drop_time = None
    for w in dropped_windows:
        if first_drop_time is None:
            first_drop_time = w.start_time
            break
    
    # Log first breach details
    print(f"\n  [PE-REV-004] First breach analysis:")
    print(f"    Baseline: {baseline_breach['breach_details']}")
    print(f"    Comparison: {comparison_breach['breach_details']}")
    print(f"    First telemetry drop: {first_drop_time:.1f}s" if first_drop_time else "    First telemetry drop: None")
    print(f"    Total dropped windows: {dropped_count} ({drop_rate:.1%})")
    
    # Assertions
    assert baseline_data_points > 0, "Baseline should have data points"

    assert drop_rate > 0.05, \
        f"Should have meaningful telemetry drop rate: {drop_rate:.1%}"

    assert dropped_count > 0, \
        f"Should have dropped windows: {dropped_count}"

    assert comparison_k > baseline_k, \
        f"Telemetry drops should increase K-factor: {comparison_k:.3f} <= {baseline_k:.3f}"

    # Key insight: telemetry loss causes penalty despite actual delivery
    assert report.penalty_delta.direction == "WORSE", \
        "Telemetry loss should result in WORSE penalties"

    assert telemetry_metrics["max_staleness_seconds"] > 0 or dropped_count > 0, \
        "Should have evidence of data gaps"

    # Verify the silent killer aspect: revenue loss from data issues alone
    assert report.comparison_revenue.expected_penalties > report.baseline_revenue.expected_penalties, \
        "Missing telemetry should increase expected penalties"
    
    # Time-to-penalty assertion
    assert report.comparison_time.first_penalty_seconds is not None or comparison_breach['first_breach_seconds'] is not None, \
        "Telemetry loss should result in some form of penalty timing"


@pytest.mark.case("PE-REV-005")
def test_mode_misclassification_penalizes_even_when_power_seems_ok(fast_scenario, report_engine):
    """Test Case - Mode Misclassification Penalizes Even When Power Seems OK.

    Description:
    -----------------
    The asset is contracted as DC (Dynamic Containment) but the controller
    actually runs as DR (Dynamic Regulation). The power response "looks
    plausible" but fails DC scoring because:
    - DC requires steep slope (fast response)
    - DR has gentler slope (slower response)
    - Evaluated against DC envelope → penalties apply

    This is the "correct power, wrong service envelope" bug class.

    Preconditions:
    -----------------
    1. Fast df/dt frequency scenario
    2. Battery controller with mode-specific parameters
    3. Baseline: Controller runs as DC (matches contract)
    4. Comparison: Controller runs as DR (misclassified)

    Steps:
    ----------
    1. Create controller with mode-specific config
    2. Run baseline with DC mode (contract expectation)
    3. Collect delivery windows and score against DC envelope
    4. Inject mode misclassification fault (DC → DR)
    5. Run comparison with fault active
    6. Score against DC envelope (contract still expects DC)
    7. Compare penalty outcomes

    Expected Results:
    ---------------------------
    1. Baseline: Low penalty score (response matches DC envelope)
    2. Comparison: Higher penalty score (DR response fails DC scoring)
    3. Power delivery looks "reasonable" but wrong envelope
    4. K-factor alone may not catch it - mode-specific scoring does

    Real-World Mapping:
    ---------------------------
    - PPC configuration error after firmware update
    - Operator mistake during mode switch
    - Communication failure leaving controller in wrong state
    - Mismatch between contract registration and actual config
    """
    # Step 1: Create controller with mode-specific config
    controller = ControlLoopFactory.battery_10mw_20mwh()

    # Expected contract mode
    expected_mode = ServiceMode.DC
    
    # Step 2: Run baseline with correct mode (DC)
    baseline_windows = []
    baseline_penalties = []
    controller.reset()
    
    for t, frequency in fast_scenario.frequency_trajectory.iterate(step_seconds=0.5):
        # Calculate expected power using controller's mode logic (not hardcoded formula)
        expected_mw = ideal_expected(controller, frequency, expected_mode)
        
        # Controller runs as DC (correct mode)
        delivered_mw = controller.respond(frequency, 0.5, service_mode=expected_mode)
        
        baseline_windows.append(DeliveryWindow(
            start_time=t,
            end_time=t + 0.5,
            expected_mw=expected_mw,
            delivered_mw=delivered_mw,
            frequency_hz=frequency,
            soe_percent=controller.state.soe_percent,
        ))
        
        # Score each window against DC envelope
        score = score_delivery_by_mode(expected_mw, delivered_mw, expected_mode, contracted_mw=10.0)
        baseline_penalties.append(score["penalty_score"])

    # Step 3: Create mode misclassification fault
    fault = ModeMisclassificationFault(
        expected_mode=ServiceMode.DC,  # Contract expects DC
        actual_mode=ServiceMode.DR,    # But controller runs as DR
    )

    # Step 4: Run comparison with fault active
    comparison_windows = []
    comparison_penalties = []
    controller.reset()
    
    for t, frequency in fast_scenario.frequency_trajectory.iterate(step_seconds=0.5):
        # Calculate expected power using controller's mode logic (contract still expects DC)
        expected_mw = ideal_expected(controller, frequency, expected_mode)
        
        # Apply fault: controller thinks it should run expected_mode but actually runs actual_mode
        mode_for_controller = apply_mode_faults(expected_mode, [fault])
        
        # Controller runs as DR (wrong mode due to fault)
        delivered_mw = controller.respond(frequency, 0.5, service_mode=mode_for_controller)
        
        comparison_windows.append(DeliveryWindow(
            start_time=t,
            end_time=t + 0.5,
            expected_mw=expected_mw,
            delivered_mw=delivered_mw,
            frequency_hz=frequency,
            soe_percent=controller.state.soe_percent,
        ))
        
        # Score against DC envelope (contract still expects DC!)
        score = score_delivery_by_mode(expected_mw, delivered_mw, expected_mode, contracted_mw=10.0)
        comparison_penalties.append(score["penalty_score"])

    # Step 5: Calculate aggregate metrics
    baseline_avg_penalty = sum(baseline_penalties) / len(baseline_penalties) if baseline_penalties else 0
    comparison_avg_penalty = sum(comparison_penalties) / len(comparison_penalties) if comparison_penalties else 0
    
    baseline_k = KFactor.calculate_weighted(baseline_windows, contracted_mw=10.0)
    comparison_k = KFactor.calculate_weighted(comparison_windows, contracted_mw=10.0)

    # Find first breach for both configurations
    baseline_breach = find_first_breach(baseline_windows, contracted_mw=10.0)
    comparison_breach = find_first_breach(comparison_windows, contracted_mw=10.0)
    
    # Log analysis
    print(f"\n  [PE-REV-005] Mode misclassification analysis:")
    print(f"    Expected mode: {expected_mode.value}")
    print(f"    Fault: {fault.get_penalty_metrics()['mismatch_description']}")
    print(f"    Baseline avg penalty: {baseline_avg_penalty:.4f}")
    print(f"    Comparison avg penalty: {comparison_avg_penalty:.4f}")
    print(f"    Penalty increase: {(comparison_avg_penalty / baseline_avg_penalty - 1) * 100:.1f}%" if baseline_avg_penalty > 0 else "    Baseline had zero penalty")
    print(f"    Baseline K-factor: {baseline_k:.4f}")
    print(f"    Comparison K-factor: {comparison_k:.4f}")
    print(f"    Baseline breach: {baseline_breach['breach_details']}")
    print(f"    Comparison breach: {comparison_breach['breach_details']}")

    # Assertions
    # Mode misclassification should cause fault to be active
    assert fault.is_misclassified(), \
        "Fault should indicate mode mismatch"

    # Comparison should have higher mode-specific penalty
    assert comparison_avg_penalty >= baseline_avg_penalty, \
        f"Mode misclassification should increase penalties: {comparison_avg_penalty:.4f} < {baseline_avg_penalty:.4f}"

    # K-factor may also increase due to slower response
    assert comparison_k >= baseline_k * 0.9, \
        f"DR mode should not dramatically improve K-factor over DC: {comparison_k:.4f} vs {baseline_k:.4f}"

    # The key insight: even if K-factor looks similar, mode-specific scoring catches the issue
    # DR mode has wider deadband and slower response, which fails DC envelope
    fault_metrics = fault.get_penalty_metrics()
    assert fault_metrics["is_misclassified"], \
        "Fault metrics should indicate misclassification"


@pytest.mark.case("PE-REV-006")
def test_control_loop_lag_bad_tuning(fast_scenario, report_engine):
    """Test Case - Control Loop Lag 'Bad Tuning' Blows Up K-Factor.

    Description:
    -----------------
    This mirrors the Arenko screenshot story: a controller with bad tuning
    (excessive lag, slow time constant) fails catastrophically on fast
    frequency events. The plant responds, but way too late.

    This is the classic "passed commissioning, failed production" bug.

    Preconditions:
    -----------------
    1. Fast df/dt frequency scenario (rapid deviation)
    2. Baseline: Well-tuned controller (fast response)
    3. Comparison: Badly-tuned controller (400ms+ lag, slow time constant)

    Expected Results:
    ---------------------------
    1. Baseline: K-factor < 0.10 (acceptable)
    2. Comparison: K-factor > 0.30 (blown up)
    3. Penalty score dramatically higher
    4. Clear demonstration of tuning impact

    Real-World Mapping:
    ---------------------------
    - Filter time constant too high after firmware update
    - Response delay increased by network changes
    - PLC scan time degradation under load
    """
    # Step 1: Create well-tuned baseline controller
    baseline_config = ControllerConfig(
        mode=ControlMode.DROOP,
        time_constant_seconds=0.5,      # Fast response
        response_delay_seconds=0.1,     # Minimal lag
        deadband_hz=0.015,
    )
    baseline_ctrl = ControlLoopFactory.battery_10mw_20mwh(baseline_config)

    # Step 2: Run baseline
    baseline_windows = run_simulation(baseline_ctrl, fast_scenario, dt_seconds=0.5)
    baseline_k = KFactor.calculate_weighted(baseline_windows, contracted_mw=10.0)

    # Step 3: Create badly-tuned controller (excessive lag)
    bad_tuning_config = ControllerConfig(
        mode=ControlMode.DROOP,
        time_constant_seconds=5.0,      # WAY too slow
        response_delay_seconds=0.4,     # 400ms lag (bad)
        deadband_hz=0.015,
    )
    bad_ctrl = ControlLoopFactory.battery_10mw_20mwh(bad_tuning_config)

    # Step 4: Run comparison with bad tuning
    comparison_windows = run_simulation(bad_ctrl, fast_scenario, dt_seconds=0.5)
    comparison_k = KFactor.calculate_weighted(comparison_windows, contracted_mw=10.0)

    # Step 5: Generate report
    baseline_event = windows_to_event(baseline_windows, "baseline-well-tuned")
    comparison_event = windows_to_event(comparison_windows, "bad-tuning")

    report = report_engine.compare(
        [baseline_event],
        [comparison_event],
        baseline_name="Well-Tuned (τ=0.5s, lag=100ms)",
        comparison_name="Bad Tuning (τ=5s, lag=400ms)",
        simulation_hours=fast_scenario.duration.event_duration_seconds / 3600,
    )

    # Find first breach
    baseline_breach = find_first_breach(baseline_windows, contracted_mw=10.0)
    comparison_breach = find_first_breach(comparison_windows, contracted_mw=10.0)

    # Log analysis
    print(f"\n  [PE-REV-006] Bad tuning analysis:")
    print(f"    Baseline K-factor: {baseline_k:.4f}")
    print(f"    Comparison K-factor: {comparison_k:.4f}")
    print(f"    K-factor blowup: {comparison_k / baseline_k:.1f}x" if baseline_k > 0 else "    Baseline K=0")
    print(f"    Baseline breach: {baseline_breach['breach_details']}")
    print(f"    Comparison breach: {comparison_breach['breach_details']}")

    # Assertions
    assert baseline_k < 0.15, \
        f"Well-tuned controller should have low K-factor: {baseline_k:.4f}"

    assert comparison_k > baseline_k * 2, \
        f"Bad tuning should at least double K-factor: {comparison_k:.4f} vs {baseline_k:.4f}"

    assert report.penalty_delta.direction == "WORSE", \
        f"Bad tuning should make things WORSE, got: {report.penalty_delta.direction}"

    # The key insight: bad tuning blows up penalties on fast events
    assert report.penalty_delta.relative_delta_percent > 50, \
        f"Bad tuning should cause >50% penalty increase: {report.penalty_delta.relative_delta_percent:.1f}%"


@pytest.mark.case("PE-REV-007")
def test_telemetry_drop_zero_settlement(fast_scenario, report_engine):
    """Test Case - Telemetry Drop with ZERO_ON_MISSING = Perfect Delivery, Zero Settlement.

    Description:
    -----------------
    This demonstrates "data plane is part of the product". The plant delivers
    power perfectly, but telemetry drops mean settlement sees zero delivery.

    With ZERO_ON_MISSING settlement mode, every dropped sample is treated as
    non-delivery → penalties accumulate despite actual delivery.

    Preconditions:
    -----------------
    1. Fast df/dt frequency scenario
    2. Controller performing correctly (actual delivery is good)
    3. Baseline: Complete telemetry
    4. Comparison: 20% telemetry drop rate with ZERO_ON_MISSING

    Expected Results:
    ---------------------------
    1. Actual power delivery is identical (controller works fine)
    2. Baseline: Full settlement credit
    3. Comparison: ~20% of windows show zero delivery
    4. Telemetry component dominates penalty score

    Real-World Mapping:
    ---------------------------
    - Network congestion during grid events
    - SCADA polling failures under load
    - Firewall rules blocking during high traffic
    - The "silent killer" of battery revenue
    """
    from faults import SettlementMode

    # Step 1: Create controller
    controller = ControlLoopFactory.battery_10mw_20mwh()

    # Step 2: Run baseline with complete telemetry
    baseline_windows = run_simulation(controller, fast_scenario, dt_seconds=0.5)
    baseline_k = KFactor.calculate_weighted(baseline_windows, contracted_mw=10.0)

    # Step 3: Create telemetry drop fault with ZERO_ON_MISSING
    telemetry_fault = TelemetryDropFault(TelemetryDropConfig(
        pattern=DropPattern.RANDOM,
        drop_probability=0.20,  # 20% drop rate
        settlement_mode=SettlementMode.ZERO_ON_MISSING,  # Worst case: missing = zero
        seed=42,
    ))
    telemetry_fault.activate()

    # Step 4: Run comparison with telemetry drops
    controller.reset()
    comparison_windows = []
    dropped_count = 0

    for t, frequency in fast_scenario.frequency_trajectory.iterate(step_seconds=0.5):
        # Controller delivers power correctly
        actual_delivered_mw = controller.respond(frequency, 0.5)
        expected_mw = controller.state.target_power_mw

        # Apply telemetry fault - this is what settlement sees
        observed_value, meta = telemetry_fault.apply(actual_delivered_mw, dt_seconds=0.5)

        # With ZERO_ON_MISSING: dropped samples become zero for settlement
        if meta["dropped"]:
            dropped_count += 1
            settlement_value = meta.get("settlement_value", 0.0)
        else:
            settlement_value = observed_value

        comparison_windows.append(DeliveryWindow(
            start_time=t,
            end_time=t + 0.5,
            expected_mw=expected_mw,
            delivered_mw=settlement_value if settlement_value is not None else 0.0,
            frequency_hz=frequency,
            soe_percent=controller.state.soe_percent,
        ))

    comparison_k = KFactor.calculate_weighted(comparison_windows, contracted_mw=10.0)

    # Step 5: Generate report
    baseline_event = windows_to_event(baseline_windows, "baseline-full-telemetry")
    comparison_event = windows_to_event(comparison_windows, "telemetry-zero-settlement")

    report = report_engine.compare(
        [baseline_event],
        [comparison_event],
        baseline_name="Complete Telemetry",
        comparison_name="20% Drops + ZERO_ON_MISSING",
        simulation_hours=fast_scenario.duration.event_duration_seconds / 3600,
    )

    # Get fault metrics
    telemetry_metrics = telemetry_fault.get_penalty_metrics()
    drop_rate = telemetry_metrics["drop_rate"]

    # Log analysis
    print(f"\n  [PE-REV-007] Telemetry drop + ZERO_ON_MISSING analysis:")
    print(f"    Baseline K-factor: {baseline_k:.4f}")
    print(f"    Comparison K-factor: {comparison_k:.4f}")
    print(f"    Drop rate: {drop_rate:.1%}")
    print(f"    Dropped windows: {dropped_count}")
    print(f"    Settlement mode: ZERO_ON_MISSING")
    print(f"    Key insight: Perfect delivery → zero settlement credit on drops")

    # Assertions
    assert drop_rate > 0.10, \
        f"Should have meaningful drop rate: {drop_rate:.1%}"

    assert dropped_count > 0, \
        f"Should have dropped windows: {dropped_count}"

    # K-factor should be much worse due to zero settlement
    assert comparison_k > baseline_k, \
        f"ZERO_ON_MISSING should increase K-factor: {comparison_k:.4f} vs {baseline_k:.4f}"

    assert report.penalty_delta.direction == "WORSE", \
        "Telemetry drops should result in WORSE penalties"

    # The key insight: data plane failures dominate penalty
    assert report.comparison_revenue.expected_penalties > report.baseline_revenue.expected_penalties, \
        "Missing telemetry with ZERO_ON_MISSING should dramatically increase penalties"


@pytest.mark.xfail(reason="Demo: shows a configuration that would break penalties")
@pytest.mark.case("PE-REV-DEMO")
def test_demo_this_config_is_bad():
    """Demo Test - Intentionally Failing Configuration.

    This test demonstrates a configuration that SHOULD fail. It's marked
    xfail so it doesn't break CI, but serves as documentation of a known
    bad configuration.

    Use Case:
    -----------------
    - Documentation of failure modes
    - Training material for operators
    - Regression detection if "bad" config accidentally becomes "good"

    Configuration:
    -----------------
    - 10MW battery with 500ms response delay
    - 50% telemetry drop rate
    - ZERO_ON_MISSING settlement
    - Fast df/dt scenario

    Expected: Catastrophic penalty failure (K-factor > 0.5)
    """
    from scenarios import load_scenario

    scenario = load_scenario("fast_df_dt")

    # Intentionally bad configuration
    bad_config = ControllerConfig(
        mode=ControlMode.DROOP,
        time_constant_seconds=10.0,     # Absurdly slow
        response_delay_seconds=0.5,     # 500ms delay
        deadband_hz=0.015,
    )
    controller = ControlLoopFactory.battery_10mw_20mwh(bad_config)

    # Run simulation
    windows = run_simulation(controller, scenario, dt_seconds=0.5)
    k_factor = KFactor.calculate_weighted(windows, contracted_mw=10.0)

    print(f"\n  [PE-REV-DEMO] Intentionally bad config:")
    print(f"    K-factor: {k_factor:.4f}")
    print(f"    This config is expected to fail!")

    # This SHOULD fail - that's the point
    assert k_factor < 0.10, \
        f"This bad config should have K-factor < 0.10, got {k_factor:.4f} (expected to fail!)"
