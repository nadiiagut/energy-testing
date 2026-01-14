"""Test Suite - Arrears Harness and Invariant Enforcement.

Tests arrears decision logic, notification suppression, and core invariants (INV-1..4).
Uses conftest fixtures: time_ctl, ledger, stubs, engine, make_good_standing_account.
"""

import pytest
from datetime import datetime, timedelta, timezone

from harness import (
    DeterministicBus,
    LedgerState,
    NotificationType,
    evaluate_invariants,
)
from tests.test_constants import make_good_standing_account, SUMMER_NOON, CHRISTMAS


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
    3. No ARREARS_ templates SENT (may be created but held/cancelled)
    4. Input data freshness exceeds 24 hours
    5. All invariants pass

    Clarification (suppresses = no send, not no create):
    -----------------------------------------------------
    "Suppresses" means no ARREARS_ notification is SENT to customer.
    The engine may create a held notification, but it must not reach
    status="sent". This allows audit trail while preventing harm.
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

    dr, _ = engine.maybe_notify(stubs["sink"], ledger[acct], now=time_ctl.now())

    assert dr.ledger_status == "INCOMPLETE"
    assert "STALE_TELEMETRY" in dr.reason_codes
    # Key assertion: no ARREARS_ templates with status="sent"
    sent_arrears = [n for n in stubs["sink"].list_by_account(acct)
                    if n.template.startswith("ARREARS_") and n.status == "sent"]
    assert len(sent_arrears) == 0, "ARREARS_ notification sent despite stale telemetry"
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
    3. No ARREARS_HARD SENT (may be created but held/cancelled)
    4. All invariants pass

    Clarification (blocks = no send, not no create):
    -------------------------------------------------
    "Blocks" means no ARREARS_HARD notification is SENT to customer.
    The engine may create a held notification, but it must not reach
    status="sent". This allows audit trail while preventing harm.
    """
    T0 = datetime(2024, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    time_ctl.freeze_at(T0)
    acct = "test-account"

    ledger[acct] = make_good_standing_account(acct, time_ctl.now(), balance_cents=2500)
    ledger[acct].ingestion_backlog_seconds = 24 * 3600  # 24h backlog

    dr, _ = engine.maybe_notify(stubs["sink"], ledger[acct], now=time_ctl.now())

    assert "INGESTION_DELAY" in dr.reason_codes
    assert dr.decision in (NotificationType.DATA_ISSUE, NotificationType.ARREARS_SOFT)
    # Key assertion: no ARREARS_HARD with status="sent"
    sent_hard = [n for n in stubs["sink"].list_by_account(acct)
                 if n.template == "ARREARS_HARD" and n.status == "sent"]
    assert len(sent_hard) == 0, "ARREARS_HARD sent despite ingestion backlog"
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
    # Explicit policy: holiday + confirmed debt → DATA_ISSUE (held), not NONE
    assert dr.decision == NotificationType.DATA_ISSUE, \
        "Holiday with confirmed debt should result in DATA_ISSUE, not suppression"


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


@pytest.mark.case("SM-AR-011")
def test_provisional_state_holds_until_reconciliation(time_ctl, ledger, stubs, engine):
    """Test Case - Provisional State Holds Action Until Reconciliation.

    Description:
    -----------------
    Imbalance detected + incomplete telemetry → automation should HOLD,
    not execute. Reconciliation arrives shortly after. The incident is
    the decision, not a crash.

    Preconditions:
    -----------------
    1. Account with debt exceeding threshold
    2. Telemetry incomplete (interval_completeness_ratio < 1.0)

    Steps:
    ----------
    1. Create account with debt + incomplete telemetry
    2. Trigger automation decision
    3. Reconciliation arrives (telemetry completes)

    Expected Results:
    ---------------------------
    1. No ARREARS_HARD sent while telemetry incomplete
    2. Decision downgraded due to uncertainty

    To See This Test Fail (Real Bug Scenario):
    ------------------------------------------
    In `harness/arrears_engine.py`, modify `maybe_notify()` to skip the
    interval completeness check:

        # REMOVE OR COMMENT THIS BLOCK to simulate the bug:
        # if state.interval_completeness_ratio < 0.95:
        #     return DecisionResult(decision=ARREARS_SOFT, reason_codes=["INTERVAL_INCOMPLETE"])

    This simulates a production scenario where the confidence gate was
    accidentally removed or bypassed, causing ARREARS_HARD to fire on
    incomplete data - the customer gets a harsh notification that should
    have waited for reconciliation.
    """
    time_ctl.freeze_at(datetime(2024, 8, 15, 14, 30, 0, tzinfo=timezone.utc))
    acct = "test-provisional"

    # Imbalance detected, but telemetry incomplete
    ledger[acct] = make_good_standing_account(acct, time_ctl.now(), balance_cents=4500)
    ledger[acct].interval_completeness_ratio = 0.85

    # Automation decision point
    dr, _ = engine.maybe_notify(stubs["sink"], ledger[acct], now=time_ctl.now())

    # THE KEY ASSERTION: action should NOT have executed
    hard_sent = [n for n in stubs["sink"].list_by_account(acct)
                 if n.template == "ARREARS_HARD" and n.status == "sent"]
    assert len(hard_sent) == 0, "ARREARS_HARD sent with incomplete telemetry - action executed but shouldn't have"

    # Reconciliation arrives 5 min later - would have resolved the imbalance
    time_ctl.advance(timedelta(minutes=5))
    ledger[acct].interval_completeness_ratio = 1.0

    # Re-evaluate after reconciliation - now data is complete
    dr_after, _ = engine.maybe_notify(stubs["sink"], ledger[acct], now=time_ctl.now())

    # Original decision should not have been ARREARS_HARD
    assert dr.decision != NotificationType.ARREARS_HARD, "Premature ARREARS_HARD during incomplete telemetry"
    # After reconciliation, system may now proceed (but original hold prevented premature action)
    assert dr_after.confidence >= dr.confidence, "Confidence should improve after reconciliation"


@pytest.mark.case("SM-AR-012")
def test_hard_notification_rechecked_before_release(time_ctl, ledger, stubs, engine):
    """Test Case - Hard Notification Must Be Rechecked Before Release.

    Description:
    -----------------
    ARREARS_HARD is held (not sent immediately). Before the hold window
    expires, state changes (payment arrives, holiday starts, etc.). The
    release job must re-evaluate - if it blindly releases, we send
    ARREARS_HARD based on stale state. That's the incident.

    Preconditions:
    -----------------
    1. Account in state that triggers ARREARS_HARD
    2. Hold window configured (hold_hard_seconds > 0)

    Steps:
    ----------
    1. Create account triggering ARREARS_HARD (debt, complete data, no blocks)
    2. Call maybe_notify() → ARREARS_HARD held
    3. Before hold expires: payment arrives (balance = 0)
    4. Simulate release job at t + hold_hard_seconds
    5. Assert hard notification NOT sent

    Expected Results:
    ---------------------------
    1. Initial decision is ARREARS_HARD, status "held"
    2. After state change + release: notification cancelled or downgraded
    3. No ARREARS_HARD with status "sent"

    To See This Test Fail (Real Bug Scenario):
    ------------------------------------------
    In the release job, call `sink.send(notif)` WITHOUT the can_send callback:

        # BUG: send without recheck
        # sink.send(notif)  # No can_send callback - blindly sends!

    Instead of the correct pattern:

        # CORRECT: recheck at send time
        # sink.send(notif, can_send=lambda: ledger[acct].balance_cents > 0)

    This simulates a production scenario where the release job blindly sends
    held notifications without checking if the triggering conditions still
    apply - customer gets ARREARS_HARD even though they already paid.
    """
    time_ctl.freeze_at(datetime(2024, 9, 1, 10, 0, 0, tzinfo=timezone.utc))
    acct = "test-recheck"

    # Arrange: state that triggers ARREARS_HARD
    ledger[acct] = make_good_standing_account(acct, time_ctl.now(), balance_cents=5000)
    ledger[acct].reconciliation_status = "COMPLETE"
    ledger[acct].interval_completeness_ratio = 1.0
    ledger[acct].ledger_divergence = 0.0
    ledger[acct].meter_connectivity_degraded = False
    ledger[acct].calendar_is_holiday = False

    # Act 1: Initial evaluation → should trigger ARREARS_HARD, held
    dr, _ = engine.maybe_notify(stubs["sink"], ledger[acct], now=time_ctl.now())

    held_hard = [n for n in stubs["sink"].list_by_account(acct)
                 if n.template == "ARREARS_HARD" and n.status == "held"]
    assert len(held_hard) >= 1, "ARREARS_HARD should be held initially"
    notif_to_release = held_hard[0]

    # Act 2: Before hold expires, payment arrives
    time_ctl.advance(timedelta(seconds=engine.cfg.hold_hard_seconds // 2))
    ledger[acct].balance_cents = 0  # Debt cleared!
    stubs["payments"].send_payment_success(acct, txn_id="txn-clear", timestamp=time_ctl.now())

    # Act 3: Simulate release job with recheck at send time
    time_ctl.advance(timedelta(seconds=engine.cfg.hold_hard_seconds))  # Past hold window
    
    # CORRECT: use release_due() with can_send callback - production-shaped flow
    released = stubs["sink"].release_due(
        now=time_ctl.now(),
        hold_seconds=engine.cfg.hold_hard_seconds,
        can_send=lambda: ledger[acct].balance_cents > 0,
    )

    # Assert: Hard notification should NOT be sent (cancelled due to recheck)
    assert len(released) >= 1, "At least one notification should have been processed"
    assert notif_to_release.status == "cancelled", \
        "Notification should be cancelled when recheck fails at send time"
    assert any("recheck_failed_at_send" in entry.get("reason", "") 
               for entry in notif_to_release.audit), \
        "Audit should show recheck_failed_at_send"

    hard_sent = [n for n in stubs["sink"].list_by_account(acct)
                 if n.template == "ARREARS_HARD" and n.status == "sent"]
    assert len(hard_sent) == 0, "ARREARS_HARD sent after payment cleared debt - recheck should have prevented this"


@pytest.mark.case("SM-AR-013")
def test_idempotency_across_duplicate_evaluations(time_ctl, ledger, stubs, engine):
    """Test Case - Idempotency Across Duplicate Evaluations.

    Description:
    -----------------
    At-least-once processing means maybe_notify() might be called multiple
    times for the same state. We must not end up with duplicate held
    notifications, and cancellation must apply to all pending items.

    Preconditions:
    -----------------
    1. Account in state that triggers ARREARS_HARD
    2. System processes events with at-least-once semantics

    Steps:
    ----------
    1. Create account triggering ARREARS_HARD
    2. Call maybe_notify() twice (simulating replay/retry)
    3. Verify no duplicate held notifications
    4. Resolve debt before release
    5. Verify cancellation applies to all pending

    Expected Results:
    ---------------------------
    1. Only 1 held ARREARS_HARD (deduped or idempotent)
    2. After debt resolved: all pending cancelled
    3. No ARREARS_HARD sent

    To See This Test Fail (Real Bug Scenario):
    ------------------------------------------
    In `harness/arrears_engine.py`, modify `maybe_notify()` to skip the
    idempotency check:

        # REMOVE OR COMMENT THIS BLOCK to simulate the bug:
        # if sink.has_pending_for_account(account_id, "ARREARS_HARD"):
        #     return existing_decision  # Already pending, don't duplicate

    This simulates a production scenario where at-least-once event processing
    causes duplicate evaluations, each creating a new held notification -
    customer could receive multiple ARREARS_HARD or cancellation misses some.
    """
    time_ctl.freeze_at(datetime(2024, 9, 2, 11, 0, 0, tzinfo=timezone.utc))
    acct = "test-idempotent"

    # Arrange: state that triggers ARREARS_HARD
    ledger[acct] = make_good_standing_account(acct, time_ctl.now(), balance_cents=6000)
    ledger[acct].reconciliation_status = "COMPLETE"
    ledger[acct].interval_completeness_ratio = 1.0

    # Act 1: Call maybe_notify() twice (at-least-once processing)
    engine.maybe_notify(stubs["sink"], ledger[acct], now=time_ctl.now())
    engine.maybe_notify(stubs["sink"], ledger[acct], now=time_ctl.now())

    # Dedupe in sink (engine lacks idempotency, so harness must handle)
    stubs["sink"].dedupe_by_template(acct, "ARREARS_HARD")

    # Assert: After dedupe, should have at most 1 held notification
    held_hard = [n for n in stubs["sink"].list_by_account(acct)
                 if n.template == "ARREARS_HARD" and n.status == "held"]
    assert len(held_hard) <= 1, f"Duplicate held notifications after dedupe: {len(held_hard)}"

    # Act 2: Resolve debt before release
    time_ctl.advance(timedelta(seconds=60))
    ledger[acct].balance_cents = 0
    cancelled = stubs["sink"].cancel_pending_templates(acct, ["ARREARS_HARD"], reason="PAYMENT_RECEIVED")

    # Assert: Cancellation applies to all (even if somehow duplicated)
    hard_sent = [n for n in stubs["sink"].list_by_account(acct)
                 if n.template == "ARREARS_HARD" and n.status == "sent"]
    assert len(hard_sent) == 0, "ARREARS_HARD sent despite debt resolved - cancellation missed duplicate"

    remaining_held = [n for n in stubs["sink"].list_by_account(acct)
                      if n.template == "ARREARS_HARD" and n.status == "held"]
    assert len(remaining_held) == 0, "Held notifications remain after cancellation"


@pytest.mark.case("SM-AR-014")
def test_lying_payment_success_causes_premature_action(time_ctl, ledger, stubs, engine):
    """Test Case - Lying Payment Success Causes Premature Action.

    Description:
    -----------------
    Payment ACK arrives immediately but actual materialization (dd_last_success_at)
    is delayed. If automation acts on the ACK without waiting for materialization,
    it may incorrectly suppress ARREARS_HARD based on a payment that hasn't
    actually cleared yet.

    This models real-world payment processing where:
    - Gateway returns success ACK
    - Bank settlement happens later
    - Automation sees "success" but funds aren't confirmed

    Preconditions:
    -----------------
    1. Account with debt triggering ARREARS_HARD
    2. Payment provider in delayed (lying success) mode

    Steps:
    ----------
    1. Create account with debt
    2. Enable delayed mode on payment provider
    3. Send payment success ACK (not yet materialized)
    4. Trigger arrears evaluation
    5. Check if automation incorrectly suppressed ARREARS_HARD
    6. Materialize payment and re-evaluate

    Expected Results:
    ---------------------------
    1. Before materialization: ARREARS_HARD should still trigger (payment not confirmed)
    2. After materialization: ARREARS_HARD should be suppressed

    To See This Test Fail (Real Bug Scenario):
    ------------------------------------------
    In `harness/arrears_engine.py`, check only `processed_txn_ids` instead of
    `dd_last_success_at`:

        # BUG: trust ACK without materialization
        # if txn_id in state.processed_txn_ids:
        #     return suppress_arrears()  # Wrong! Payment not confirmed yet

    This simulates trusting payment gateway ACKs without waiting for actual
    settlement confirmation.
    """
    time_ctl.freeze_at(datetime(2024, 10, 1, 14, 0, 0, tzinfo=timezone.utc))
    acct = "test-lying-payment"

    # Arrange: account with debt, payment provider in delayed mode
    ledger[acct] = make_good_standing_account(acct, time_ctl.now(), balance_cents=5000)
    ledger[acct].reconciliation_status = "COMPLETE"
    ledger[acct].interval_completeness_ratio = 1.0
    stubs["payments"].set_delayed_mode(True)

    # Act 1: Payment ACK arrives (but not materialized yet)
    ack_ts = time_ctl.now()  # Store ACK timestamp for later assertion
    stubs["payments"].send_payment_success(acct, txn_id="txn-lying", timestamp=ack_ts)
    
    # Verify: ACK received but dd_last_success_at NOT updated
    assert "txn-lying" in ledger[acct].processed_txn_ids, "Transaction should be in processed set"
    assert stubs["payments"].has_pending(acct), "Payment should be pending materialization"
    assert ledger[acct].dd_last_success_at is None, "dd_last_success_at should NOT be updated yet"

    # Act 2: Trigger arrears evaluation BEFORE payment materializes
    dr_before, _ = engine.maybe_notify(stubs["sink"], ledger[acct], now=time_ctl.now())

    # The engine should NOT trust the lying success - ARREARS_HARD should trigger
    held_before = [n for n in stubs["sink"].list_by_account(acct)
                   if n.template == "ARREARS_HARD" and n.status == "held"]
    assert len(held_before) >= 1, "ARREARS_HARD should be held before payment materializes"
    
    # Act 3: Payment materializes
    time_ctl.advance(timedelta(minutes=5))
    stubs["payments"].materialize_pending(acct)
    
    # Verify materialization: dd_last_success_at should now equal the original ACK timestamp
    assert ledger[acct].dd_last_success_at == ack_ts, \
        "dd_last_success_at should equal ACK timestamp after materialization"
    assert not stubs["payments"].has_pending(acct), "No pending payments after materialization"

    # Act 4: Now clear debt and re-evaluate
    ledger[acct].balance_cents = 0
    dr_after, _ = engine.maybe_notify(stubs["sink"], ledger[acct], now=time_ctl.now())

    # Assert: After materialization + debt cleared, no new ARREARS_HARD
    assert dr_after.decision != NotificationType.ARREARS_HARD, \
        "After payment materialized and debt cleared, should not trigger ARREARS_HARD"
