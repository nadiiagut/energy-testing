"""Shared test constants and helpers for energy-testing framework."""

import sys
import pytest
from datetime import datetime, timezone

from .harness import LedgerState


# =============================================================================
# Common timestamps (all use timezone.utc)
# =============================================================================

SUMMER_NOON = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
SUMMER_MORNING = datetime(2024, 6, 15, 8, 0, tzinfo=timezone.utc)
SUMMER_EVENING = datetime(2024, 6, 15, 18, 0, tzinfo=timezone.utc)
SUMMER_NIGHT = datetime(2024, 6, 15, 2, 0, tzinfo=timezone.utc)
CHRISTMAS = datetime(2024, 12, 25, 9, 30, tzinfo=timezone.utc)


# =============================================================================
# Account helpers
# =============================================================================

def make_good_standing_account(
    account_id: str,
    now: datetime,
    balance_cents: int = 0,
) -> LedgerState:
    """
    Helper: Create an account in good standing with fresh data.
    
    Good standing means:
    - Fresh reading (just now)
    - Reconciliation complete
    - No backlog, no connectivity issues
    - Full interval completeness
    - No ledger divergence
    - Not estimated
    - Not a holiday
    """
    return LedgerState(
        account_id=account_id,
        balance_cents=balance_cents,
        last_reading_at=now,
        ingestion_backlog_seconds=0,
        reconciliation_status="COMPLETE",
        meter_connectivity_degraded=False,
        dd_last_success_at=None,
        estimated_bill=False,
        interval_completeness_ratio=1.0,
        ledger_divergence=0.0,
        calendar_is_holiday=False,
    )


# =============================================================================
# Platform helpers
# =============================================================================

def is_linux():
    """Check if running on Linux (for LD_PRELOAD tests)."""
    return sys.platform.startswith("linux")


skip_unless_linux = pytest.mark.skipif(
    not is_linux(),
    reason="LD_PRELOAD fault injection requires Linux"
)
