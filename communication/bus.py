"""Pub/sub abstraction every coordination layer (CBBA, Karma, telemetry)
talks to. Key expressions follow Zenoh convention (`fleet/robot3/pose`,
wildcards `*` for one segment / `**` for any number) so `InProcessBus`
(single-process sim, synchronous, zero latency) and `ZenohBus` (real
sessions, real network) are drop-in replacements for each other -- swapping
the bus is the only change needed to go from one process to a real
decentralized fleet.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

Handler = Callable[[str, Any], None]


def key_matches(pattern: str, topic: str) -> bool:
    """Zenoh-style key-expression match: '*' matches exactly one segment,
    '**' matches zero or more segments."""
    p_parts = pattern.split("/")
    t_parts = topic.split("/")

    def match(pi: int, ti: int) -> bool:
        if pi == len(p_parts):
            return ti == len(t_parts)
        seg = p_parts[pi]
        if seg == "**":
            for k in range(ti, len(t_parts) + 1):
                if match(pi + 1, k):
                    return True
            return False
        if ti == len(t_parts):
            return False
        if seg == "*" or seg == t_parts[ti]:
            return match(pi + 1, ti + 1)
        return False

    return match(0, 0)


class MessageBus(ABC):
    @abstractmethod
    def publish(self, topic: str, payload: Any) -> None: ...

    @abstractmethod
    def subscribe(self, pattern: str, handler: Handler) -> None: ...

    @abstractmethod
    def message_count(self) -> int:
        """Cumulative number of publish() calls, for instrumentation."""
        ...
