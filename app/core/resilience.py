"""Small resilience primitives for outbound calls: a circuit breaker.

Retries + timeouts live next to the HTTP client (app/agent/tools/client.py);
the breaker is shared per-process so a failing dependency is shed quickly.
"""
from __future__ import annotations

import time


class CircuitBreaker:
    """closed -> (N consecutive failures) -> open -> (cool-down) -> half-open -> closed|open."""

    def __init__(self, fail_threshold: int, reset_seconds: float) -> None:
        self.fail_threshold = fail_threshold
        self.reset_seconds = reset_seconds
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if time.monotonic() - self._opened_at >= self.reset_seconds:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        return self.state != "open"

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.fail_threshold or self.state == "half_open":
            self._opened_at = time.monotonic()
