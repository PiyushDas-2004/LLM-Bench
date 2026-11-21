"""Injectable time sources so rate limiting, retries and breakers are testable."""

from __future__ import annotations

import asyncio
import time
from typing import Protocol


class Clock(Protocol):
    """Monotonic time plus an async sleep."""

    def now(self) -> float:
        """Monotonic seconds."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Suspend for ``seconds``."""
        ...


class SystemClock:
    """Real time."""

    def now(self) -> float:
        """``time.monotonic()``."""
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        """``asyncio.sleep``."""
        await asyncio.sleep(max(0.0, seconds))


class FakeClock:
    """Virtual time for tests: ``sleep`` advances the clock instantly."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start
        self.sleeps: list[float] = []

    def now(self) -> float:
        """Current virtual time."""
        return self.t

    async def sleep(self, seconds: float) -> None:
        """Advance virtual time and yield to the event loop."""
        self.sleeps.append(seconds)
        self.t += max(0.0, seconds)
        await asyncio.sleep(0)
