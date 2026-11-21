"""Async token-bucket rate limiting for requests/minute and tokens/minute."""

from __future__ import annotations

import asyncio

from llmbench.config.models import RateLimitConfig
from llmbench.runner.clock import Clock, SystemClock


class TokenBucket:
    """Classic token bucket.

    Args:
        rate_per_s: Refill rate in tokens per second.
        capacity: Maximum burst size.
        clock: Time source.
    """

    def __init__(self, rate_per_s: float, capacity: float, clock: Clock | None = None) -> None:
        if rate_per_s <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self.rate = rate_per_s
        self.capacity = capacity
        self.clock = clock or SystemClock()
        self._tokens = capacity
        self._updated = self.clock.now()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self.clock.now()
        self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
        self._updated = now

    @property
    def available(self) -> float:
        """Tokens available right now."""
        self._refill()
        return self._tokens

    async def acquire(self, amount: float = 1.0) -> float:
        """Take ``amount`` tokens, waiting as needed; returns seconds waited.

        Requests larger than the capacity are clamped to the capacity so a
        single oversized request can still proceed.
        """
        amount = min(amount, self.capacity)
        waited = 0.0
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= amount:
                    self._tokens -= amount
                    return waited
                delay = (amount - self._tokens) / self.rate
                waited += delay
                await self.clock.sleep(delay)


class RateLimiter:
    """Combined requests-per-minute and tokens-per-minute limiter for one provider."""

    def __init__(self, config: RateLimitConfig, clock: Clock | None = None) -> None:
        self.config = config
        rpm, tpm = config.requests_per_minute, config.tokens_per_minute
        # Burst capacity of one second's worth (at least one request) keeps traffic smooth.
        self._requests = TokenBucket(rpm / 60, max(1.0, rpm / 60), clock) if rpm else None
        self._tokens = TokenBucket(tpm / 60, max(1.0, tpm / 60), clock) if tpm else None

    async def acquire(self, tokens: int) -> float:
        """Wait for one request slot and ``tokens`` token budget; returns seconds waited."""
        waited = 0.0
        if self._requests is not None:
            waited += await self._requests.acquire(1)
        if self._tokens is not None:
            waited += await self._tokens.acquire(max(1, tokens))
        return waited
