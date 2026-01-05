from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from datetime import timedelta

from .time_control import TimeController


@dataclass
class Message:
    id: int
    topic: str
    payload: Dict[str, Any]
    key: Optional[str]
    published_at: float
    deliver_at: float
    attributes: Dict[str, Any] = field(default_factory=dict)


class DeterministicBus:
    def __init__(self, time: TimeController):
        self._time = time
        self._seq = itertools.count(1)
        self._queue: List[Message] = []
        self._trace: List[Dict[str, Any]] = []
        self._drop_predicate: Optional[Callable[[Message], bool]] = None
        self._reorder_mode: Optional[str] = None  # None|reverse|fifo|lifo
        self._rand_seed: Optional[int] = None

    def set_drop_predicate(self, fn: Optional[Callable[[Message], bool]]) -> None:
        self._drop_predicate = fn

    def set_reorder_mode(self, mode: Optional[str] = None) -> None:
        self._reorder_mode = mode

    def now_epoch(self) -> float:
        return self._time.now().timestamp()

    def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        *,
        key: Optional[str] = None,
        delay_ms: int = 0,
        duplicate: int = 0,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        ids: List[int] = []
        count = 1 + max(0, duplicate)
        for dup_idx in range(count):
            msg_id = next(self._seq)
            deliver_at = self.now_epoch() + (delay_ms / 1000.0)
            attrs = dict(attributes or {})
            if dup_idx > 0:
                attrs["duplicate_of"] = ids[0]
            msg = Message(
                id=msg_id,
                topic=topic,
                payload=payload,
                key=key,
                published_at=self.now_epoch(),
                deliver_at=deliver_at,
                attributes=attrs,
            )
            self._queue.append(msg)
            ids.append(msg_id)
            self._trace.append({
                "event": "published",
                "id": msg_id,
                "topic": topic,
                "time": self.now_epoch(),
                "delay_ms": delay_ms,
                "duplicate_of": attrs.get("duplicate_of"),
            })
        return ids

    def _eligible(self) -> List[Message]:
        now = self.now_epoch()
        ready = [m for m in self._queue if m.deliver_at <= now]
        return ready

    def _apply_reorder(self, ready: List[Message]) -> List[Message]:
        if not self._reorder_mode or len(ready) <= 1:
            return sorted(ready, key=lambda m: m.id)
        if self._reorder_mode == "reverse":
            return sorted(ready, key=lambda m: m.id, reverse=True)
        if self._reorder_mode == "lifo":
            return list(sorted(ready, key=lambda m: m.id))[-1::-1]
        return sorted(ready, key=lambda m: m.id)

    def consume_available(self, topic: Optional[str] = None, limit: Optional[int] = None) -> List[Message]:
        ready = self._eligible()
        if topic is not None:
            ready = [m for m in ready if m.topic == topic]
        ordered = self._apply_reorder(ready)
        out: List[Message] = []
        for m in ordered:
            if limit is not None and len(out) >= limit:
                break
            dropped = self._drop_predicate(m) if self._drop_predicate else False
            self._queue.remove(m)
            if dropped:
                self._trace.append({"event": "dropped", "id": m.id, "topic": m.topic, "time": self.now_epoch()})
                continue
            self._trace.append({"event": "delivered", "id": m.id, "topic": m.topic, "time": self.now_epoch()})
            out.append(m)
        return out

    def drain(self) -> List[Message]:
        out: List[Message] = []
        while True:
            batch = self.consume_available()
            if not batch:
                break
            out.extend(batch)
        return out

    def queue_depth(self) -> int:
        return len(self._queue)

    @property
    def trace(self) -> List[Dict[str, Any]]:
        return list(self._trace)
