"""Real Zenoh-backed MessageBus: same publish/subscribe interface as
InProcessBus, but backed by actual zenoh sessions and key expressions
(`fleet/robot3/pose`, wildcards native to Zenoh) so swapping this in for
InProcessBus is the only change needed to go from one shared-memory process
to genuinely decentralized peers -- each with its own session, discovering
each other over the network with no broker or central server.

Payloads are pickled: this is a local validation/demo tool talking to its
own trusted robot processes, not a wire protocol exposed to untrusted
peers, so pickle's simplicity is an acceptable trade-off here.
"""
from __future__ import annotations

import pickle
import random
import threading
from typing import Any, Optional

import zenoh

from core.comms.bus import Handler, MessageBus


class ZenohBus(MessageBus):
    def __init__(
        self,
        config: Optional[zenoh.Config] = None,
        inject_delay_s: float = 0.0,
        inject_loss_rate: float = 0.0,
    ) -> None:
        """`inject_delay_s`/`inject_loss_rate` are an application-level fault
        injector for environments without root (kernel `tc netem` needs
        CAP_NET_ADMIN) -- see zenoh_netem_manual.sh for the real kernel-level
        equivalent. This is a deliberately weaker substitute: it delays/drops
        at the publish call in this process, not on the actual network path,
        so it demonstrates the coordination logic's tolerance to delay/loss,
        not true link-level impairment."""
        self._session = zenoh.open(config or zenoh.Config())
        self._subscribers: list[zenoh.Subscriber] = []
        self._message_count = 0
        self._closed = False
        self._inject_delay_s = inject_delay_s
        self._inject_loss_rate = inject_loss_rate

    def publish(self, topic: str, payload: Any) -> None:
        if self._closed:
            return
        self._message_count += 1
        if self._inject_loss_rate and random.random() < self._inject_loss_rate:
            return
        data = pickle.dumps(payload)
        if self._inject_delay_s:
            threading.Timer(self._inject_delay_s, self._deliver, args=(topic, data)).start()
        else:
            self._deliver(topic, data)

    def _deliver(self, topic: str, data: bytes) -> None:
        if not self._closed:
            self._session.put(topic, data)

    def subscribe(self, pattern: str, handler: Handler) -> None:
        def _on_sample(sample: zenoh.Sample) -> None:
            payload = pickle.loads(bytes(sample.payload.to_bytes()))
            handler(str(sample.key_expr), payload)

        sub = self._session.declare_subscriber(pattern, _on_sample)
        self._subscribers.append(sub)

    def message_count(self) -> int:
        return self._message_count

    def zenoh_id(self) -> str:
        return str(self._session.info.zid())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for sub in self._subscribers:
            sub.undeclare()
        self._session.close()
