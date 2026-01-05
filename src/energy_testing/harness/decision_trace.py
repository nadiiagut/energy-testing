from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class DecisionRecord:
    account_id: str
    decision: str
    reason_codes: List[str]
    confidence: float
    input_data_freshness_seconds: Optional[int] = None
    ledger_status: str = "UNKNOWN"
    payment_status: str = "UNKNOWN"
    calendar_policy: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    telemetry: Dict[str, float] = field(default_factory=dict)
    timeline: List[Dict[str, str]] = field(default_factory=list)

    def add_timeline(self, what: str, why: Optional[str] = None) -> None:
        self.timeline.append({
            "event": what,
            "reason": why or "",
            "at": self.timestamp.isoformat(),
        })
