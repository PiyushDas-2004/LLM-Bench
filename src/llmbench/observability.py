"""Structured logging (structlog), secret redaction and optional OpenTelemetry spans."""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any

import structlog

_SECRET_KEYS = re.compile(r"(api[_-]?key|secret|token|password|authorization)", re.IGNORECASE)
_SECRET_VALUES = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}|AIza[0-9A-Za-z_\-]{20,}|Bearer\s+[A-Za-z0-9._\-]{8,})"
)
_PROMPT_KEYS = frozenset({"prompt", "rendered_prompt", "output", "output_text", "system_prompt"})
REDACTED = "[REDACTED]"

_state: dict[str, Any] = {"redact_prompts": False, "tracer": None}


def redact_value(value: Any) -> Any:
    """Mask anything that looks like a credential inside a string."""
    if isinstance(value, str):
        return _SECRET_VALUES.sub(REDACTED, value)
    return value


def redact_processor(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """structlog processor: drop secret-named fields, mask key-like strings, optionally prompts."""
    for key in list(event_dict):
        if (_SECRET_KEYS.search(key) and key != "event") or (
            _state["redact_prompts"] and key in _PROMPT_KEYS
        ):
            event_dict[key] = REDACTED
        else:
            event_dict[key] = redact_value(event_dict[key])
    return event_dict


def configure_logging(
    level: str = "INFO",
    fmt: str = "console",
    *,
    redact_prompts: bool = False,
    otel: bool = False,
) -> None:
    """Configure structlog for console or JSON output (logs go to stderr)."""
    _state["redact_prompts"] = redact_prompts
    numeric = logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        numeric = logging.INFO
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            redact_processor,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )
    _state["tracer"] = _make_tracer() if otel else None


def _make_tracer() -> Any:
    try:
        from opentelemetry import trace
    except ImportError:
        structlog.get_logger(__name__).warning(
            "otel_unavailable", hint="pip install 'llmbench[otel]'"
        )
        return None
    return trace.get_tracer("llmbench")


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    """An OpenTelemetry span when tracing is enabled, otherwise a no-op."""
    tracer = _state["tracer"]
    if tracer is None:
        yield
        return
    with tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            current.set_attribute(key, value)
        yield
