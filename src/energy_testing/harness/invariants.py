from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

from .stubs import LedgerState
from .decision_trace import DecisionRecord


@dataclass
class InvariantViolation(Exception):
    code: str
    message: str


@dataclass
class InvariantConfig:
    reading_freshness_threshold: timedelta = timedelta(hours=24)
    backlog_threshold: timedelta = timedelta(hours=2)
    dd_protective_window: timedelta = timedelta(hours=48)


def _is_uncertain(st: LedgerState, now: datetime, cfg: InvariantConfig) -> bool:
    freshness_fail = False
    if st.last_reading_at is None:
        freshness_fail = True
    else:
        freshness_fail = (now - st.last_reading_at) > cfg.reading_freshness_threshold
    recon_incomplete = st.reconciliation_status != "COMPLETE"
    backlog = timedelta(seconds=st.ingestion_backlog_seconds) > cfg.backlog_threshold
    degraded = st.meter_connectivity_degraded
    return freshness_fail or recon_incomplete or backlog or degraded


def evaluate_invariants(
    st: LedgerState,
    decision: DecisionRecord,
    notifications_templates_sent: List[str],
    now: datetime,
    cfg: Optional[InvariantConfig] = None,
) -> List[InvariantViolation]:
    cfg = cfg or InvariantConfig()
    violations: List[InvariantViolation] = []

    uncertain = _is_uncertain(st, now, cfg)
    if uncertain:
        if any(t.startswith("ARREARS_") for t in notifications_templates_sent):
            violations.append(InvariantViolation(
                code="INV-1",
                message="Arrears notification sent while state uncertain",
            ))

    if st.dd_last_success_at is not None:
        if (now - st.dd_last_success_at) <= cfg.dd_protective_window:
            if any(t == "ARREARS_HARD" for t in notifications_templates_sent):
                violations.append(InvariantViolation(
                    code="INV-2",
                    message="DD success should protect from hard arrears",
                ))

    requires_confidence = [
        st.balance_cents > 0,
        st.reconciliation_status == "COMPLETE",
        (not _is_uncertain(st, now, cfg)),
        st.interval_completeness_ratio >= 1.0,
        st.ledger_divergence == 0.0,
        not st.estimated_bill,
    ]
    if not all(requires_confidence):
        if any(t == "ARREARS_HARD" for t in notifications_templates_sent):
            violations.append(InvariantViolation(
                code="INV-3",
                message="ARREARS_HARD sent without confidence gates",
            ))

    if st.calendar_is_holiday:
        if any(t.startswith("ARREARS_") and t != "DATA_ISSUE" for t in notifications_templates_sent):
            violations.append(InvariantViolation(
                code="INV-4",
                message="Arrears communications sent on holiday",
            ))

    return violations
