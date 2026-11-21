"""Budget guard: reserve estimated spend before a call, settle actual spend after."""

from __future__ import annotations

from dataclasses import dataclass, field

from llmbench.config.models import BudgetConfig
from llmbench.errors import BudgetExceededError


@dataclass
class BudgetGuard:
    """Tracks spend and request count against hard limits.

    Reservations make the guard safe under concurrency: in-flight calls count
    against the budget before their actual cost is known.
    """

    config: BudgetConfig
    spent_usd: float = 0.0
    requests: int = 0
    reserved_usd: float = 0.0
    exceeded_reason: str | None = field(default=None)

    def reserve(self, estimated_usd: float) -> None:
        """Reserve budget for one request or raise :class:`BudgetExceededError`."""
        if self.exceeded_reason:
            raise BudgetExceededError(self.exceeded_reason)
        cap_requests = self.config.max_requests
        if cap_requests is not None and self.requests >= cap_requests:
            self.exceeded_reason = f"max_requests={cap_requests} reached"
            raise BudgetExceededError(self.exceeded_reason)
        cap_usd = self.config.max_usd
        if cap_usd is not None and self.spent_usd + self.reserved_usd + estimated_usd > cap_usd:
            self.exceeded_reason = (
                f"max_usd={cap_usd:.4f} would be exceeded (spent {self.spent_usd:.4f}, "
                f"in flight {self.reserved_usd:.4f}, next ~{estimated_usd:.4f})"
            )
            raise BudgetExceededError(self.exceeded_reason)
        self.requests += 1
        self.reserved_usd += estimated_usd

    def settle(self, estimated_usd: float, actual_usd: float | None) -> None:
        """Release a reservation and record the actual cost (estimate if unknown)."""
        self.reserved_usd = max(0.0, self.reserved_usd - estimated_usd)
        self.spent_usd += estimated_usd if actual_usd is None else actual_usd
