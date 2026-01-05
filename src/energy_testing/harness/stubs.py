from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


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
    def __init__(self, ledger: Dict[str, LedgerState]):
        self._ledger = ledger

    def send_payment_success(self, account_id: str, txn_id: str, timestamp: datetime) -> None:
        st = self._ledger.setdefault(account_id, LedgerState(account_id=account_id))
        if txn_id in st.processed_txn_ids:
            if "duplicate_txn_ignored" not in st.reason_codes:
                st.reason_codes.append("duplicate_txn_ignored")
            return
        st.processed_txn_ids.add(txn_id)
        st.dd_last_success_at = timestamp


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

    def create(self, account_id: str, template: str, reasons: List[str]) -> Notification:
        self._seq += 1
        n = Notification(
            id=self._seq,
            account_id=account_id,
            template=template,
            created_at=datetime.utcnow(),
            status="created",
            reasons=list(reasons),
            audit=[{"status": "created", "at": datetime.utcnow().isoformat()}],
        )
        self._items.append(n)
        return n

    def hold(self, notif: Notification, reason: str) -> None:
        notif.status = "held"
        notif.audit.append({"status": "held", "reason": reason, "at": datetime.utcnow().isoformat()})

    def cancel(self, notif: Notification, reason: str) -> None:
        notif.status = "cancelled"
        notif.audit.append({"status": "cancelled", "reason": reason, "at": datetime.utcnow().isoformat()})

    def send(self, notif: Notification) -> None:
        notif.status = "sent"
        notif.audit.append({"status": "sent", "at": datetime.utcnow().isoformat()})

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
