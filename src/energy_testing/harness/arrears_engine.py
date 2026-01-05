from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from .stubs import LedgerState, NotificationSink
from .decision_trace import DecisionRecord


class NotificationType:
    NONE = "NONE"
    ARREARS_HARD = "ARREARS_HARD"
    ARREARS_SOFT = "ARREARS_SOFT"
    DATA_ISSUE = "DATA_ISSUE"


@dataclass
class ArrearsConfig:
    reading_freshness_threshold: timedelta = timedelta(hours=24)
    backlog_threshold: timedelta = timedelta(hours=2)
    dd_protective_window: timedelta = timedelta(hours=48)
    hold_hard_seconds: int = 300


class ArrearsEngine:
    def __init__(self, cfg: Optional[ArrearsConfig] = None):
        self.cfg = cfg or ArrearsConfig()

    def evaluate(self, st: LedgerState, now: Optional[datetime] = None) -> Tuple[str, DecisionRecord]:
        now = now or datetime.utcnow()
        reasons: List[str] = []
        confidence = 1.0

        freshness_seconds = None
        if st.last_reading_at is None:
            reasons.append("MISSING_READING")
            confidence *= 0.4
        else:
            freshness_seconds = int((now - st.last_reading_at).total_seconds())
            if (now - st.last_reading_at) > self.cfg.reading_freshness_threshold:
                reasons.append("STALE_TELEMETRY")
                confidence *= 0.6

        if st.reconciliation_status != "COMPLETE":
            reasons.append("RECONCILIATION_INCOMPLETE")
            confidence *= 0.6

        if st.ingestion_backlog_seconds > int(self.cfg.backlog_threshold.total_seconds()):
            reasons.append("INGESTION_DELAY")
            confidence *= 0.6

        if st.meter_connectivity_degraded:
            reasons.append("CONNECTIVITY_DEGRADED")
            confidence *= 0.6

        if st.estimated_bill:
            reasons.append("ESTIMATED_BILL")
            confidence *= 0.5

        if st.interval_completeness_ratio < 1.0:
            reasons.append("INTERVAL_INCOMPLETE")
            confidence *= 0.6

        if st.ledger_divergence > 0.0:
            reasons.append("LEDGER_DIVERGENCE")
            confidence *= 0.5

        protected = False
        if st.dd_last_success_at is not None:
            if (now - st.dd_last_success_at) <= self.cfg.dd_protective_window:
                reasons.append("DD_RECENT")
                protected = True
                confidence *= 0.6

        decision_type = NotificationType.NONE

        confirmed_debt = st.balance_cents > 0 and not st.estimated_bill
        confidence_gates_ok = (
            (st.reconciliation_status == "COMPLETE") and
            (st.interval_completeness_ratio >= 1.0) and
            (st.ledger_divergence == 0.0) and
            (freshness_seconds is not None and freshness_seconds <= int(self.cfg.reading_freshness_threshold.total_seconds())) and
            (st.ingestion_backlog_seconds <= int(self.cfg.backlog_threshold.total_seconds())) and
            (not st.meter_connectivity_degraded)
        )

        if st.calendar_is_holiday:
            reasons.append("CALENDAR_POLICY_BLOCK")
            decision_type = NotificationType.DATA_ISSUE if confirmed_debt else NotificationType.NONE
        else:
            if confirmed_debt and confidence_gates_ok and not protected:
                decision_type = NotificationType.ARREARS_HARD
            elif confirmed_debt and (protected or not confidence_gates_ok):
                decision_type = NotificationType.ARREARS_SOFT
            else:
                if reasons:
                    decision_type = NotificationType.DATA_ISSUE
                else:
                    decision_type = NotificationType.NONE

        dr = DecisionRecord(
            account_id=st.account_id,
            decision=decision_type,
            reason_codes=reasons,
            confidence=max(0.0, min(1.0, confidence)),
            input_data_freshness_seconds=freshness_seconds,
            ledger_status=(
                "INCOMPLETE" if any(r in reasons for r in ["STALE_TELEMETRY","MISSING_READING","RECONCILIATION_INCOMPLETE","INGESTION_DELAY","CONNECTIVITY_DEGRADED","INTERVAL_INCOMPLETE"]) else "OK"
            ),
            payment_status=("PROTECTED" if protected else "NONE"),
            calendar_policy=("HOLIDAY" if st.calendar_is_holiday else None),
            telemetry={
                "ingestion_lag_seconds": float(st.ingestion_backlog_seconds),
                "interval_completeness_ratio": float(st.interval_completeness_ratio),
                "ledger_divergence": float(st.ledger_divergence),
            },
        )
        return decision_type, dr

    def maybe_notify(self, sink: NotificationSink, st: LedgerState, now: Optional[datetime] = None):
        notif_type, dr = self.evaluate(st, now=now)
        notif = None
        if notif_type == NotificationType.NONE:
            return dr, None
        if notif_type == NotificationType.DATA_ISSUE:
            notif = sink.create(st.account_id, "DATA_ISSUE", dr.reason_codes)
            sink.hold(notif, "uncertainty_present")
        elif notif_type == NotificationType.ARREARS_SOFT:
            notif = sink.create(st.account_id, "ARREARS_SOFT", dr.reason_codes)
            sink.hold(notif, "soft_pending_recheck")
        elif notif_type == NotificationType.ARREARS_HARD:
            notif = sink.create(st.account_id, "ARREARS_HARD", dr.reason_codes)
            if self.cfg.hold_hard_seconds > 0:
                sink.hold(notif, "hard_hold_window")
            else:
                sink.send(notif)
        return dr, notif
