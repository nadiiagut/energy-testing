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


# =============================================================================
# SM-AR-001: Stale reading misinterpreted as current
# =============================================================================

@pytest.mark.case("SM-AR-001")
def test_stale_reading_suppresses_arrears(time_ctl, ledger, stubs, engine):
    """Stale but valid readings must not trigger arrears notifications."""
    T0 = datetime(2024, 1, 10, 10, 0, 0, tzinfo=timezone.utc)
    time_ctl.freeze_at(T0)
    acct = "SM-AR-001"

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


# =============================================================================
# SM-AR-002: Missing reading → estimation fallback
# =============================================================================

@pytest.mark.case("SM-AR-002")
def test_estimated_bill_blocks_hard_arrears(time_ctl, ledger, stubs, engine):
    """Estimated bills must not generate ARREARS_HARD notifications."""
    T0 = datetime(2024, 2, 1, 9, 0, 0, tzinfo=timezone.utc)
    time_ctl.freeze_at(T0)
    time_ctl.advance(timedelta(hours=72))
    acct = "SM-AR-002"

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


# =============================================================================
# SM-AR-003: Out-of-order payment after arrears evaluation
# =============================================================================

@pytest.mark.case("SM-AR-003")
def test_payment_after_evaluation_cancels_pending_arrears(time_ctl, ledger, stubs, engine):
    """Payment arriving after arrears evaluation must cancel pending notifications."""
    T0 = datetime(2024, 3, 1, 8, 0, 0, tzinfo=timezone.utc)
    time_ctl.freeze_at(T0)
    acct = "SM-AR-003"

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


# =============================================================================
# SM-AR-004: Duplicate payment events (idempotency)
# =============================================================================

@pytest.mark.case("SM-AR-004")
def test_duplicate_payment_idempotency(time_ctl, ledger, stubs):
    """Duplicate PAYMENT_SUCCESS with same txn_id must be ignored."""
    time_ctl.freeze_at(datetime(2024, 4, 1, 10, 0, 0, tzinfo=timezone.utc))
    acct = "SM-AR-004"

    ledger[acct] = LedgerState(account_id=acct, balance_cents=1000)
    stubs["payments"].send_payment_success(acct, txn_id="txn-dup", timestamp=time_ctl.now())
    stubs["payments"].send_payment_success(acct, txn_id="txn-dup", timestamp=time_ctl.now())

    assert "duplicate_txn_ignored" in ledger[acct].reason_codes
    assert len(ledger[acct].processed_txn_ids) == 1


# =============================================================================
# SM-AR-005: Ingestion backlog blocks arrears
# =============================================================================

@pytest.mark.case("SM-AR-005")
def test_ingestion_backlog_blocks_hard_arrears(time_ctl, ledger, stubs, engine):
    """Ingestion backlog above threshold must block ARREARS_HARD."""
    T0 = datetime(2024, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    time_ctl.freeze_at(T0)
    acct = "SM-AR-005"

    ledger[acct] = make_good_standing_account(acct, time_ctl.now(), balance_cents=2500)
    ledger[acct].ingestion_backlog_seconds = 24 * 3600  # 24h backlog

    dr, _ = engine.maybe_notify(stubs["sink"], ledger[acct], now=time_ctl.now())

    assert "INGESTION_DELAY" in dr.reason_codes
    assert dr.decision in (NotificationType.DATA_ISSUE, NotificationType.ARREARS_SOFT)
    assert evaluate_invariants(ledger[acct], dr, [], now=time_ctl.now()) == []


# =============================================================================
# SM-AR-009: Holiday calendar policy
# =============================================================================

@pytest.mark.case("SM-AR-009")
def test_holiday_blocks_arrears(time_ctl, ledger, stubs, engine):
    """Arrears notifications must be suppressed on configured holidays."""
    time_ctl.freeze_at(CHRISTMAS)
    acct = "SM-AR-009"

    ledger[acct] = make_good_standing_account(acct, time_ctl.now(), balance_cents=3000)
    ledger[acct].calendar_is_holiday = True

    dr, _ = engine.maybe_notify(stubs["sink"], ledger[acct], now=time_ctl.now())

    assert "ARREARS_HARD" not in all_templates(stubs["sink"], acct)
    assert "CALENDAR_POLICY_BLOCK" in dr.reason_codes


# =============================================================================
# SM-AR-010: Confidence gating downgrades
# =============================================================================

@pytest.mark.case("SM-AR-010")
@pytest.mark.parametrize("uncertainty_field,uncertainty_value,expected_reason", [
    ("reconciliation_status", "PENDING", "RECONCILIATION_INCOMPLETE"),
    ("interval_completeness_ratio", 0.8, "INTERVAL_INCOMPLETE"),
    ("meter_connectivity_degraded", True, "CONNECTIVITY_DEGRADED"),
    ("ledger_divergence", 0.1, "LEDGER_DIVERGENCE"),
], ids=["recon-pending", "interval-incomplete", "connectivity-degraded", "ledger-divergence"])
def test_uncertainty_downgrades_to_soft(time_ctl, ledger, stubs, engine, uncertainty_field, uncertainty_value, expected_reason):
    """Any uncertainty condition must downgrade decision from ARREARS_HARD."""
    T0 = datetime(2024, 7, 1, 10, 0, 0, tzinfo=timezone.utc)
    time_ctl.freeze_at(T0)
    acct = f"SM-AR-010-{uncertainty_field}"

    ledger[acct] = make_good_standing_account(acct, time_ctl.now(), balance_cents=4000)
    setattr(ledger[acct], uncertainty_field, uncertainty_value)

    dr, _ = engine.maybe_notify(stubs["sink"], ledger[acct], now=time_ctl.now())

    assert dr.decision != NotificationType.ARREARS_HARD
    assert expected_reason in dr.reason_codes


# =============================================================================
# Message bus controls
# =============================================================================

def test_message_bus_delay_and_duplicate(time_ctl):
    """Bus must support delay and duplicate message delivery."""
    bus = DeterministicBus(time_ctl)
    time_ctl.freeze_at(datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc))

    msg_ids = bus.publish("meter.reading", {"x": 1}, delay_ms=1000, duplicate=1)
    assert len(msg_ids) == 2
    assert bus.consume_available() == []

    time_ctl.advance(timedelta(seconds=1))
    msgs = bus.consume_available()
    assert len(msgs) == 2
    assert all(m.topic == "meter.reading" for m in msgs)


def test_message_bus_reorder_and_drop(time_ctl):
    """Bus must support reorder and drop predicates."""
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
