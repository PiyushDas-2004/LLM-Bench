"""Helpers shared by HTTP-based provider adapters."""

from __future__ import annotations

import email.utils
import importlib
import time
from collections.abc import Mapping
from types import ModuleType

from llmbench.providers.base import ErrorClass


def parse_retry_after(headers: Mapping[str, str] | None, now: float | None = None) -> float | None:
    """Parse ``retry-after-ms`` / ``retry-after`` (seconds or HTTP date) into seconds."""
    if not headers:
        return None
    lowered = {k.lower(): v for k, v in headers.items()}
    if "retry-after-ms" in lowered:
        try:
            return max(0.0, float(lowered["retry-after-ms"]) / 1000.0)
        except ValueError:
            return None
    value = lowered.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    current = time.time() if now is None else now
    return max(0.0, parsed.timestamp() - current)


def _transport_modules() -> list[ModuleType]:
    modules: list[ModuleType] = []
    for name in ("httpx", "httpx2"):
        try:
            modules.append(importlib.import_module(name))
        except ImportError:  # pragma: no cover - httpx is a hard dependency
            continue
    return modules


def classify_transport_error(exc: BaseException) -> ErrorClass | None:
    """Classify low-level HTTP client errors (``httpx`` or the SDKs' ``httpx2``)."""
    for module in _transport_modules():
        if isinstance(exc, module.TimeoutException):
            return ErrorClass.TIMEOUT
        if isinstance(exc, module.TransportError):
            return ErrorClass.NETWORK
    return None
