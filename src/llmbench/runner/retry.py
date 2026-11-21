"""Retry policy: exponential backoff with full jitter that honours ``Retry-After``."""

from __future__ import annotations

import random
from dataclasses import dataclass

from llmbench.config.models import RetryConfig
from llmbench.providers.base import ProviderError

#: Upper bound on how long a server-provided Retry-After is honoured.
MAX_RETRY_AFTER_S = 300.0


@dataclass(frozen=True)
class RetryPolicy:
    """Decides whether and how long to wait before the next attempt."""

    max_attempts: int = 4
    base_delay_s: float = 0.5
    max_delay_s: float = 30.0

    @classmethod
    def from_config(cls, config: RetryConfig) -> RetryPolicy:
        """Build from the experiment's retry section."""
        return cls(config.max_attempts, config.base_delay_s, config.max_delay_s)

    def should_retry(self, error: ProviderError, attempt: int) -> bool:
        """``attempt`` is 1-based; AUTH/INVALID_REQUEST/content errors are never retried."""
        return error.retryable and attempt < self.max_attempts

    def delay(self, attempt: int, error: ProviderError, rng: random.Random) -> float:
        """Full-jitter backoff ``U(0, min(cap, base * 2**(attempt-1)))``, floored by Retry-After."""
        ceiling = min(self.max_delay_s, self.base_delay_s * (2 ** (attempt - 1)))
        backoff = rng.uniform(0, ceiling)
        if error.retry_after_s is not None:
            return max(backoff, min(error.retry_after_s, MAX_RETRY_AFTER_S))
        return backoff
