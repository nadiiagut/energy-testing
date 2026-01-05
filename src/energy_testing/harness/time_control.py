from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional


@dataclass
class TimeController:
    tz: timezone = timezone.utc
    _frozen_now: Optional[datetime] = None

    def now(self) -> datetime:
        if self._frozen_now is not None:
            return self._frozen_now
        return datetime.now(self.tz)

    def freeze_at(self, dt: datetime) -> None:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.tz)
        self._frozen_now = dt.astimezone(self.tz)

    def jump_to(self, dt: datetime) -> None:
        self.freeze_at(dt)

    def advance(self, delta: timedelta) -> None:
        if self._frozen_now is None:
            # If not frozen, freeze at current real time first
            self._frozen_now = datetime.now(self.tz)
        self._frozen_now = self._frozen_now + delta

    def set_timezone_offset(self, hours: int = 0, minutes: int = 0) -> None:
        self.tz = timezone(timedelta(hours=hours, minutes=minutes))
        # Keep frozen now aligned to new tz if already frozen
        if self._frozen_now is not None:
            self._frozen_now = self._frozen_now.astimezone(self.tz)
