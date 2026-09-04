"""Synchronous in-process pub/sub: every publish() calls matching
subscribers immediately, in-process, with zero latency. Used for the
single-window integrated sim where all robots share one process. Same
`MessageBus` interface as `ZenohBus`, so the coordination code (CBBA,
Karma) never knows which transport it's running over.
"""
from __future__ import annotations

from typing import Any

from core.comms.bus import Handler, MessageBus, key_matches


class InProcessBus(MessageBus):
    def __init__(self) -> None:
        self._subscribers: list[tuple[str, Handler]] = []
        self._message_count = 0

    def publish(self, topic: str, payload: Any) -> None:
        self._message_count += 1
        for pattern, handler in list(self._subscribers):
            if key_matches(pattern, topic):
                handler(topic, payload)

    def subscribe(self, pattern: str, handler: Handler) -> None:
        self._subscribers.append((pattern, handler))

    def message_count(self) -> int:
        return self._message_count
