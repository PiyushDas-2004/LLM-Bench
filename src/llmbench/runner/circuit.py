"""Per-provider circuit breaker (closed -> open -> half-open -> closed)."""

from __future__ import annotations

import asyncio
from enum import StrEnum

from llmbench.runner.clock import Clock, SystemClock


class BreakerState(StrEnum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Stops hammering a failing provider.

    Opens after ``failure_threshold`` consecutive failures. After
    ``reset_timeout_s`` one probe call is allowed (half-open); success closes
    the circuit, failure re-opens it. ``trips`` counts how often it opened, so
    the runner can give up on a provider that never recovers.
    """

    def __init__(
        self, failure_threshold: int, reset_timeout_s: float, clock: Clock | None = None
    ) -> None:
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self.clock = clock or SystemClock()
        self.state = BreakerState.CLOSED
        self.trips = 0
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._probe_in_flight = False
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """Wait until a call is permitted; returns seconds waited."""
        waited = 0.0
        while True:
            async with self._lock:
                if self.state is BreakerState.CLOSED:
                    return waited
                if self.state is BreakerState.OPEN:
                    remaining = self._opened_at + self.reset_timeout_s - self.clock.now()
                    if remaining <= 0:
                        self.state = BreakerState.HALF_OPEN
                        self._probe_in_flight = True
                        return waited
                elif not self._probe_in_flight:
                    self._probe_in_flight = True
                    return waited
                else:
                    remaining = min(1.0, self.reset_timeout_s)
            waited += remaining
            await self.clock.sleep(remaining)

    def record_success(self) -> None:
        """A call succeeded: close the circuit."""
        self._consecutive_failures = 0
        self._probe_in_flight = False
        self.state = BreakerState.CLOSED

    def record_failure(self) -> None:
        """A call failed: maybe open the circuit."""
        self._consecutive_failures += 1
        self._probe_in_flight = False
        if self.state is BreakerState.HALF_OPEN or (
            self._consecutive_failures >= self.failure_threshold
            and self.state is BreakerState.CLOSED
        ):
            self.state = BreakerState.OPEN
            self._opened_at = self.clock.now()
            self.trips += 1
