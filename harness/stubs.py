from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional


@dataclass
class LedgerState:
    account_id: str
    balance_cents: int = 0
    last_reading_at: Optional[datetime] = None
    ingestion_backlog_seconds: int = 0
    reconciliation_status: str = "COMPLETE"  # COMPLETE|PENDING|FAILED
    meter_connectivity_degraded: bool = False
    dd_last_success_at: Optional[datetime] = None
    estimated_bill: bool = False
    interval_completeness_ratio: float = 1.0
    ledger_divergence: float = 0.0
    calendar_is_holiday: bool = False
    reason_codes: List[str] = field(default_factory=list)
    processed_txn_ids: set = field(default_factory=set)


class MeterIngestionStub:
    def __init__(self, ledger: Dict[str, LedgerState]):
        self._ledger = ledger

    def send_reading(self, account_id: str, timestamp: datetime, value_kwh: float) -> None:
        st = self._ledger.setdefault(account_id, LedgerState(account_id=account_id))
        st.last_reading_at = timestamp


class PaymentProviderStub:
    """Stub for payment provider with optional delayed materialization.
    
    Supports "lying success" mode where ACK arrives immediately but
    dd_last_success_at is only applied after materialize_pending() is called.
    This models real-world payment processing delays.
    """
    def __init__(self, ledger: Dict[str, LedgerState]):
        self._ledger = ledger
        self._pending_success: List[tuple] = []  # [(account_id, txn_id, timestamp), ...]
        self._delayed_mode = False

    def set_delayed_mode(self, enabled: bool) -> None:
        """Enable/disable delayed materialization (lying success mode)."""
        self._delayed_mode = enabled

    def send_payment_success(self, account_id: str, txn_id: str, timestamp: datetime) -> None:
        st = self._ledger.setdefault(account_id, LedgerState(account_id=account_id))
        if txn_id in st.processed_txn_ids:
            if "duplicate_txn_ignored" not in st.reason_codes:
                st.reason_codes.append("duplicate_txn_ignored")
            return
        st.processed_txn_ids.add(txn_id)
        
        if self._delayed_mode:
            # ACK received but not yet materialized - lying success
            self._pending_success.append((account_id, txn_id, timestamp))
        else:
            st.dd_last_success_at = timestamp

    def materialize_pending(self, account_id: Optional[str] = None) -> int:
        """Apply pending payment success timestamps.
        
        In delayed mode, this simulates the actual payment materialization
        that happens after the initial ACK.
        
        Returns:
            Number of payments materialized
        """
        count = 0
        remaining = []
        for acct, txn_id, ts in self._pending_success:
            if account_id is None or acct == account_id:
                st = self._ledger.get(acct)
                if st:
                    st.dd_last_success_at = ts
                    count += 1
            else:
                remaining.append((acct, txn_id, ts))
        self._pending_success = remaining
        return count

    def has_pending(self, account_id: str) -> bool:
        """Check if account has pending (not yet materialized) payments."""
        return any(acct == account_id for acct, _, _ in self._pending_success)


@dataclass
class Notification:
    id: int
    account_id: str
    template: str
    created_at: datetime
    status: str
    reasons: List[str]
    audit: List[Dict[str, str]] = field(default_factory=list)


class NotificationSink:
    def __init__(self):
        self._seq = 0
        self._items: List[Notification] = []

    def create(
        self,
        account_id: str,
        template: str,
        reasons: List[str],
        now: Optional[datetime] = None,
    ) -> Notification:
        self._seq += 1
        ts = now or datetime.now(timezone.utc)
        n = Notification(
            id=self._seq,
            account_id=account_id,
            template=template,
            created_at=ts,
            status="created",
            reasons=list(reasons),
            audit=[{"status": "created", "at": ts.isoformat()}],
        )
        self._items.append(n)
        return n

    def hold(self, notif: Notification, reason: str) -> None:
        notif.status = "held"
        notif.audit.append({"status": "held", "reason": reason, "at": datetime.now(timezone.utc).isoformat()})

    def cancel(self, notif: Notification, reason: str) -> None:
        notif.status = "cancelled"
        notif.audit.append({"status": "cancelled", "reason": reason, "at": datetime.now(timezone.utc).isoformat()})

    def send(self, notif: Notification, can_send: Callable[[], bool] = lambda: True) -> None:
        """Send a notification, optionally re-validating before send.
        
        Args:
            notif: The notification to send
            can_send: Optional callback to re-validate at send time.
                      If returns False, notification is cancelled instead of sent.
                      This allows proving: "the incident happens if you don't
                      re-check at execution time"
        """
        if not can_send():
            self.cancel(notif, "recheck_failed_at_send")
            return
        notif.status = "sent"
        notif.audit.append({"status": "sent", "at": datetime.now(timezone.utc).isoformat()})

    def list_by_account(self, account_id: str) -> List[Notification]:
        return [n for n in self._items if n.account_id == account_id]

    def list_all(self) -> List[Notification]:
        return list(self._items)

    def list_pending_by_account(self, account_id: str) -> List[Notification]:
        return [n for n in self._items if n.account_id == account_id and n.status in ("created", "held")]

    def cancel_pending_templates(self, account_id: str, templates: List[str], reason: str) -> int:
        count = 0
        for n in self._items:
            if n.account_id == account_id and n.template in set(templates) and n.status in ("created", "held"):
                self.cancel(n, reason)
                count += 1
        return count

    def dedupe_by_template(self, account_id: str, template: str, reason: str = "dedupe") -> int:
        items = [n for n in self._items if n.account_id == account_id and n.template == template]
        if len(items) <= 1:
            return 0
        # Keep the earliest created (lowest id), cancel others
        items_sorted = sorted(items, key=lambda n: n.id)
        keep = items_sorted[0]
        cancelled = 0
        for n in items_sorted[1:]:
            if n.status in ("created", "held"):
                self.cancel(n, reason)
                cancelled += 1
        return cancelled

    def release_due(
        self,
        now: datetime,
        hold_seconds: int,
        can_send: Callable[[], bool] = lambda: True,
    ) -> List[Notification]:
        """Release held notifications that have exceeded their hold window.
        
        Args:
            now: Current time for comparison
            hold_seconds: How long notifications must be held before release
            can_send: Optional callback to re-validate each notification
                      before sending. If False, notification is cancelled.
        
        Returns:
            List of notifications that were released (sent or cancelled)
        """
        released = []
        for n in self._items:
            if n.status == "held":
                held_duration = (now - n.created_at).total_seconds()
                if held_duration >= hold_seconds:
                    self.send(n, can_send)
                    released.append(n)
        return released
