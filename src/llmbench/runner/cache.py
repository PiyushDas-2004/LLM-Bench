"""Content-addressed response cache.

The key covers provider, model, generation params, the rendered prompt *and the
trial index*, so repeated trials stay independent samples while re-running an
identical config is free and instant.
"""

from __future__ import annotations

import hashlib
from typing import Any

from llmbench.datasets.loader import canonical_json
from llmbench.providers.base import GenerationRequest, GenerationResult
from llmbench.tracking.store import RunStore


def cache_key(
    provider: str, request: GenerationRequest, model_config: dict[str, Any] | None = None
) -> str:
    """Stable SHA-256 key for a request.

    ``model_config`` carries everything else that can change the output for the
    same prompt: the full model ``params`` (including provider-specific knobs)
    and the ``base_url``.
    """
    payload: dict[str, Any] = {
        "provider": provider,
        "model_config": model_config or {},
        "model": request.model,
        "params": request.params.model_dump(mode="json"),
        "messages": [m.model_dump() for m in request.messages],
        "trial_index": request.trial_index,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class ResponseCache:
    """Read-through cache stored in the run database; only successful results are cached."""

    def __init__(self, store: RunStore, enabled: bool = True) -> None:
        self.store = store
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def get(
        self, provider: str, request: GenerationRequest, model_config: dict[str, Any] | None = None
    ) -> GenerationResult | None:
        """Return a cached result or ``None``."""
        if not self.enabled:
            return None
        payload = self.store.cache_get(cache_key(provider, request, model_config))
        if payload is None:
            self.misses += 1
            return None
        self.hits += 1
        return GenerationResult.model_validate(payload)

    def put(
        self,
        provider: str,
        request: GenerationRequest,
        result: GenerationResult,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        """Store a successful result."""
        if not self.enabled or not result.ok:
            return
        key = cache_key(provider, request, model_config)
        self.store.cache_put(key, provider, request.model, result.model_dump(mode="json"))
