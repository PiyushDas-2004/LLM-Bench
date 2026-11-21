from __future__ import annotations

import json

import pytest
import structlog

from llmbench import observability
from llmbench.observability import REDACTED, configure_logging, redact_processor, redact_value, span


def test_redaction_of_secret_fields_and_values() -> None:
    event = {
        "event": "call",
        "api_key": "sk-abcdefghijklmnop",
        "authorization": "Bearer abcdefghijkl",
        "message": "used key sk-abcdefghijklmnop and AIzaSyA1234567890123456789012",
        "prompt": "hello",
    }
    out = redact_processor(None, "info", dict(event))
    assert out["api_key"] == REDACTED
    assert out["authorization"] == REDACTED
    assert "sk-abc" not in out["message"]
    assert "AIza" not in out["message"]
    assert out["prompt"] == "hello"
    assert redact_value(5) == 5


def test_prompt_redaction_and_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "json", redact_prompts=True)
    structlog.get_logger("t").info("trial", prompt="secret question", token="abc123456789")
    line = capsys.readouterr().err.strip().splitlines()[-1]
    record = json.loads(line)
    assert record["prompt"] == REDACTED
    assert record["token"] == REDACTED
    configure_logging("NOT-A-LEVEL", "console")


def test_span_without_and_with_tracer() -> None:
    configure_logging("INFO", "console", otel=False)
    with span("noop", a=1):
        pass
    configure_logging("INFO", "console", otel=True)
    assert observability._state["tracer"] is not None
    with span("traced", run_id="r", n=3):
        pass
    configure_logging("INFO", "console", otel=False)
