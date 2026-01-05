"""Test Suite - Arrears Harness and Invariant Enforcement.

Tests arrears decision logic, notification suppression, and core invariants (INV-1..4).
Uses conftest fixtures: time_ctl, ledger, stubs, engine, make_good_standing_account.
"""

import pytest
from datetime import datetime, timedelta, timezone

from energy_testing.harness import (
    DeterministicBus,
    LedgerState,
    NotificationType,
    evaluate_invariants,
)
from energy_testing.test_constants import make_good_standing_account, SUMMER_NOON, CHRISTMAS


# =============================================================================
# Helpers
# =============================================================================

def sent_templates(sink, account_id: str) -> list:
    """Return list of template names that were actually sent."""
    return [n.template for n in sink.list_by_account(account_id) if n.status == "sent"]


def all_templates(sink, account_id: str) -> list:
    """Return list of all template names (any status)."""
    return [n.template for n in sink.list_by_account(account_id)]


@pytest.mark.case("SM-AR-001")
def test_stale_reading_suppresses_arrears(time_ctl, ledger, stubs, engine):
    """Test Case - Stale Reading Suppresses Arrears Notifications.

    Description:
    -----------------
    Verify that stale but valid meter readings do not trigger arrears
    notifications. Readings older than 24 hours are considered stale
    and indicate incomplete data for decision-making.

    Preconditions:
    -----------------
    1. Account exists with initial reading at T0
    2. Reading becomes stale after 48+ hours without update
    3. Account has outstanding balance (debt)

    Steps:
    ----------
    1. Create account with reading at T0
    2. Advance time by 48+ hours (reading becomes stale)
    3. Add debt to account balance
    4. Trigger arrears evaluation via maybe_notify()
    5. Check decision result and reason codes

    Expected Results:
    ---------------------------
    1. Decision ledger_status is "INCOMPLETE"
    2. Reason codes include "STALE_TELEMETRY"
    3. No ARREARS_ templates are sent
    4. Input data freshness exceeds 24 hours
    5. All invariants pass
    """
    T0 = datetime(2024, 1, 10, 10, 0, 0, tzinfo=timezone.utc)
    time_ctl.freeze_at(T0)
    acct = "test-account"

    # Create account with reading at T0
    ledger[acct] = LedgerState(account_id=acct, balance_cents=0, reconciliation_status="PENDING")
    stubs["ingestion"].send_reading(acct, time_ctl.now(), value_kwh=10.0)

    # Advance +48h (reading becomes stale), add debt
    time_ctl.advance(timedelta(hours=48, minutes=5))
    ledger[acct].balance_cents = 1500
    stubs["payments"].send_payment_success(acct, txn_id="txn-1", timestamp=time_ctl.now())

    dr, _ = engine.maybe_notify(stubs["sink"], ledger[acct], now=time_ctl.now())

    assert dr.ledger_status == "INCOMPLETE"
    assert "STALE_TELEMETRY" in dr.reason_codes
    assert all(not t.startswith("ARREARS_") for t in sent_templates(stubs["sink"], acct))
    assert dr.input_data_freshness_seconds > 24 * 3600
    assert evaluate_invariants(ledger[acct], dr, sent_templates(stubs["sink"], acct), now=time_ctl.now()) == []


@pytest.mark.case("SM-AR-002")
def test_estimated_bill_blocks_hard_arrears(time_ctl, ledger, stubs, engine):
    """Test Case - Estimated Bill Blocks Hard Arrears.

    Description:
    -----------------
    Verify that accounts with estimated bills (no actual meter reading)
    cannot receive ARREARS_HARD notifications. Estimated data has
    insufficient confidence for aggressive collection actions.

    Preconditions:
    -----------------
    1. Account has outstanding balance
    2. Bill is marked as estimated (no actual reading)
    3. Reconciliation status is PENDING

    Steps:
    ----------
    1. Create account with estimated_bill=True
    2. Set outstanding balance
    3. Trigger arrears evaluation
    4. Check decision type and reason codes

    Expected Results:
    ---------------------------
    1. Decision is DATA_ISSUE, ARREARS_SOFT, or NONE (not HARD)
    2. Reason codes include ESTIMATED_BILL or MISSING_READING
    3. No ARREARS_HARD template in notifications
    """
    T0 = datetime(2024, 2, 1, 9, 0, 0, tzinfo=timezone.utc)
    time_ctl.freeze_at(T0)
    time_ctl.advance(timedelta(hours=72))
    acct = "test-account"

    ledger[acct] = LedgerState(
        account_id=acct,
        balance_cents=2000,
        estimated_bill=True,
        reconciliation_status="PENDING",
    )

    dr, _ = engine.maybe_notify(stubs["sink"], ledger[acct], now=time_ctl.now())

    assert dr.decision in (NotificationType.DATA_ISSUE, NotificationType.ARREARS_SOFT, NotificationType.NONE)
    assert "ESTIMATED_BILL" in dr.reason_codes or "MISSING_READING" in dr.reason_codes
    assert "ARREARS_HARD" not in all_templates(stubs["sink"], acct)


@pytest.mark.case("SM-AR-003")
def test_payment_after_evaluation_cancels_pending_arrears(time_ctl, ledger, stubs, engine):
    """Test Case - Payment After Evaluation Cancels Pending Arrears.

    Description:
    -----------------
    Verify that a payment arriving after arrears evaluation but before
    notification delivery cancels the pending notification. This handles
    the race condition between payment processing and notification dispatch.

    Preconditions:
    -----------------
    1. Account in good standing with outstanding balance
    2. Arrears evaluation creates held notification
    3. Payment arrives within notification hold window

    Steps:
    ----------
    1. Create account with debt triggering arrears
    2. Run arrears evaluation (creates held ARREARS_HARD)
    3. Advance time by 2 minutes
    4. Send payment success event
    5. Cancel pending templates with PAYMENT_RECEIVED reason

    Expected Results:
    ---------------------------
    1. ARREARS_HARD notification initially held
    2. At least 1 notification cancelled
    3. No ARREARS_HARD in sent templates
    """
    T0 = datetime(2024, 3, 1, 8, 0, 0, tzinfo=timezone.utc)
    time_ctl.freeze_at(T0)
    acct = "test-account"

    ledger[acct] = make_good_standing_account(acct, time_ctl.now(), balance_cents=5000)

    # Arrears evaluation creates held notification
    dr, notif = engine.maybe_notify(stubs["sink"], ledger[acct], now=time_ctl.now())
    assert any(n.template == "ARREARS_HARD" and n.status == "held" for n in stubs["sink"].list_by_account(acct))

    # Payment arrives 2 minutes later → cancel pending
    time_ctl.advance(timedelta(minutes=2))
    stubs["payments"].send_payment_success(acct, txn_id="txn-123", timestamp=time_ctl.now())
    cancelled = stubs["sink"].cancel_pending_templates(acct, ["ARREARS_HARD"], reason="PAYMENT_RECEIVED_BEFORE_SEND")

    assert cancelled >= 1
    assert "ARREARS_HARD" not in sent_templates(stubs["sink"], acct)


@pytest.mark.case("SM-AR-004")
def test_duplicate_payment_idempotency(time_ctl, ledger, stubs):
    """Test Case - Duplicate Payment Event Idempotency.

    Description:
    -----------------
    Verify that duplicate PAYMENT_SUCCESS events with the same
    transaction ID are properly ignored. This ensures idempotent
    payment processing and prevents double-crediting accounts.

    Preconditions:
    -----------------
    1. Account exists with outstanding balance
    2. Payment system can send duplicate events

    Steps:
    ----------
    1. Create account with balance
    2. Send PAYMENT_SUCCESS with txn_id "txn-dup"
    3. Send duplicate PAYMENT_SUCCESS with same txn_id
    4. Check account reason codes and processed transactions

    Expected Results:
    ---------------------------
    1. "duplicate_txn_ignored" in reason_codes
    2. Only 1 transaction ID in processed_txn_ids
    """
    time_ctl.freeze_at(datetime(2024, 4, 1, 10, 0, 0, tzinfo=timezone.utc))
    acct = "test-account"

    ledger[acct] = LedgerState(account_id=acct, balance_cents=1000)
    stubs["payments"].send_payment_success(acct, txn_id="txn-dup", timestamp=time_ctl.now())
    stubs["payments"].send_payment_success(acct, txn_id="txn-dup", timestamp=time_ctl.now())

    assert "duplicate_txn_ignored" in ledger[acct].reason_codes
    assert len(ledger[acct].processed_txn_ids) == 1


@pytest.mark.case("SM-AR-005")
def test_ingestion_backlog_blocks_hard_arrears(time_ctl, ledger, stubs, engine):
    """Test Case - Ingestion Backlog Blocks Hard Arrears.

    Description:
    -----------------
    Verify that significant meter data ingestion backlog prevents
    ARREARS_HARD notifications. Backlog indicates potential missing
    payments or readings that haven't been processed yet.

    Preconditions:
    -----------------
    1. Account in good standing with outstanding balance
    2. Ingestion backlog exceeds threshold (24 hours)

    Steps:
    ----------
    1. Create account with debt
    2. Set ingestion_backlog_seconds to 24 hours
    3. Trigger arrears evaluation
    4. Check decision and reason codes

    Expected Results:
    ---------------------------
    1. "INGESTION_DELAY" in reason codes
    2. Decision is DATA_ISSUE or ARREARS_SOFT (not HARD)
    3. All invariants pass
    """
    T0 = datetime(2024, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    time_ctl.freeze_at(T0)
    acct = "test-account"

    ledger[acct] = make_good_standing_account(acct, time_ctl.now(), balance_cents=2500)
    ledger[acct].ingestion_backlog_seconds = 24 * 3600  # 24h backlog

    dr, _ = engine.maybe_notify(stubs["sink"], ledger[acct], now=time_ctl.now())

    assert "INGESTION_DELAY" in dr.reason_codes
    assert dr.decision in (NotificationType.DATA_ISSUE, NotificationType.ARREARS_SOFT)
    assert evaluate_invariants(ledger[acct], dr, [], now=time_ctl.now()) == []


@pytest.mark.case("SM-AR-009")
def test_holiday_blocks_arrears(time_ctl, ledger, stubs, engine):
    """Test Case - Holiday Calendar Blocks Arrears Notifications.

    Description:
    -----------------
    Verify that arrears notifications are suppressed on configured
    holidays. Customer communications should respect holiday periods
    for both legal and customer experience reasons.

    Preconditions:
    -----------------
    1. Current time is a configured holiday (Christmas)
    2. Account has outstanding balance triggering arrears
    3. calendar_is_holiday flag is set

    Steps:
    ----------
    1. Set time to Christmas day
    2. Create account with debt and holiday flag
    3. Trigger arrears evaluation
    4. Check notifications and reason codes

    Expected Results:
    ---------------------------
    1. No ARREARS_HARD template sent
    2. "CALENDAR_POLICY_BLOCK" in reason codes
    """
    time_ctl.freeze_at(CHRISTMAS)
    acct = "test-account"

    ledger[acct] = make_good_standing_account(acct, time_ctl.now(), balance_cents=3000)
    ledger[acct].calendar_is_holiday = True

    dr, _ = engine.maybe_notify(stubs["sink"], ledger[acct], now=time_ctl.now())

    assert "ARREARS_HARD" not in all_templates(stubs["sink"], acct)
    assert "CALENDAR_POLICY_BLOCK" in dr.reason_codes


@pytest.mark.case("SM-AR-010")
@pytest.mark.parametrize("uncertainty_field,uncertainty_value,expected_reason", [
    ("reconciliation_status", "PENDING", "RECONCILIATION_INCOMPLETE"),
    ("interval_completeness_ratio", 0.8, "INTERVAL_INCOMPLETE"),
    ("meter_connectivity_degraded", True, "CONNECTIVITY_DEGRADED"),
    ("ledger_divergence", 0.1, "LEDGER_DIVERGENCE"),
], ids=["recon-pending", "interval-incomplete", "connectivity-degraded", "ledger-divergence"])
def test_uncertainty_downgrades_to_soft(time_ctl, ledger, stubs, engine, uncertainty_field, uncertainty_value, expected_reason):
    """Test Case - Uncertainty Conditions Downgrade to Soft Arrears.

    Description:
    -----------------
    Verify that any data uncertainty condition causes a downgrade from
    ARREARS_HARD to a softer action. Multiple uncertainty types are
    tested via parametrization to ensure comprehensive coverage.

    Preconditions:
    -----------------
    1. Account in good standing with debt
    2. One uncertainty condition is present (varies by parameter)

    Steps:
    ----------
    1. Create account with debt
    2. Set the specified uncertainty field to its test value
    3. Trigger arrears evaluation
    4. Check decision type and reason codes

    Expected Results:
    ---------------------------
    1. Decision is NOT ARREARS_HARD
    2. Expected reason code is present in decision
    """
    T0 = datetime(2024, 7, 1, 10, 0, 0, tzinfo=timezone.utc)
    time_ctl.freeze_at(T0)
    acct = f"test-{uncertainty_field}"

    ledger[acct] = make_good_standing_account(acct, time_ctl.now(), balance_cents=4000)
    setattr(ledger[acct], uncertainty_field, uncertainty_value)

    dr, _ = engine.maybe_notify(stubs["sink"], ledger[acct], now=time_ctl.now())

    assert dr.decision != NotificationType.ARREARS_HARD
    assert expected_reason in dr.reason_codes


@pytest.mark.case("SM-BUS-001")
def test_message_bus_delay_and_duplicate(time_ctl):
    """Test Case - Message Bus Delay and Duplicate Delivery.

    Description:
    -----------------
    Verify that the deterministic message bus supports configurable
    delays and duplicate message injection for testing race conditions
    and idempotency handling.

    Preconditions:
    -----------------
    1. DeterministicBus initialized with time controller
    2. Time frozen at known point

    Steps:
    ----------
    1. Publish message with 1000ms delay and duplicate=1
    2. Verify 2 message IDs returned (original + duplicate)
    3. Consume immediately - verify no messages available
    4. Advance time by 1 second
    5. Consume again - verify both messages delivered

    Expected Results:
    ---------------------------
    1. publish() returns 2 message IDs
    2. No messages before delay expires
    3. Both messages delivered after delay
    4. All messages have correct topic
    """
    bus = DeterministicBus(time_ctl)
    time_ctl.freeze_at(datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc))

    msg_ids = bus.publish("meter.reading", {"x": 1}, delay_ms=1000, duplicate=1)
    assert len(msg_ids) == 2
    assert bus.consume_available() == []

    time_ctl.advance(timedelta(seconds=1))
    msgs = bus.consume_available()
    assert len(msgs) == 2
    assert all(m.topic == "meter.reading" for m in msgs)


@pytest.mark.case("SM-BUS-002")
def test_message_bus_reorder_and_drop(time_ctl):
    """Test Case - Message Bus Reorder and Drop Predicates.

    Description:
    -----------------
    Verify that the deterministic message bus supports message
    reordering and selective dropping for testing out-of-order
    delivery and message loss scenarios.

    Preconditions:
    -----------------
    1. DeterministicBus initialized with time controller
    2. Multiple messages published

    Steps:
    ----------
    1. Publish two messages with payloads x=2 and x=3
    2. Set reorder mode to "reverse"
    3. Set drop predicate to drop messages where x=2
    4. Consume available messages
    5. Check trace for event types

    Expected Results:
    ---------------------------
    1. Only message with x=3 is delivered
    2. Trace contains published, delivered, and dropped events
    """
    bus = DeterministicBus(time_ctl)
    time_ctl.freeze_at(datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc))

    bus.publish("pay.success", {"x": 2})
    bus.publish("pay.success", {"x": 3})
    bus.set_reorder_mode("reverse")
    bus.set_drop_predicate(lambda m: m.payload.get("x") == 2)

    delivered = bus.consume_available()
    assert [m.payload["x"] for m in delivered] == [3]

    events = {t["event"] for t in bus.trace}
    assert events >= {"published", "delivered", "dropped"}
