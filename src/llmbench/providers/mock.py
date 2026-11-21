"""A deterministic, configurable mock provider.

The mock makes every test, the quickstart and CI run offline with zero API
keys. Its behaviour is a pure function of ``(seed, model, rendered prompt,
trial index, attempt)``, so runs are reproducible and resumable, yet it still
behaves like a non-deterministic LLM when ``temperature > 0``.

The runner attaches an ``oracle`` (the sample's reference and metadata) to each
request's ``metadata``. Real providers never read it; the mock uses it to decide
whether to answer correctly, so a benchmark on the mock exercises every metric
with a *known, controllable* quality level. Numbers produced with the mock are
illustrative only and are always labelled as such.

Simulation knobs (set in a model's ``params`` in the experiment YAML):

==========================  ============================================================
``accuracy``                Base probability of a correct answer (default 0.8).
``sample_spread``           Std-dev of per-sample difficulty on the logit scale (1.0).
``failure_rate``            Probability a call fails with a transient error (0.0).
``error_mix``               Relative weights of TIMEOUT / RATE_LIMITED / SERVER_ERROR.
``retry_after_s``           ``Retry-After`` attached to simulated 429s (0.0).
``refusal_rate``            Probability of an unwarranted refusal (0.0).
``malformed_rate``          Probability of syntactically broken JSON output (0.0).
``injection_susceptibility`` Probability of obeying an injected instruction (0.2).
``latency_ms``              Median simulated latency (400).
``latency_sigma``           Log-normal sigma of latency (0.35).
``time_scale``              Fraction of simulated latency actually slept (0.0).
``position_bias``           Judge mode: probability of picking answer "A" regardless (0.0).
``mock_seed``               Seed for sampling noise; change it to simulate a re-run (0).
``difficulty_seed``         Seed for the fixed per-sample difficulty (0).
==========================  ============================================================
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llmbench.embeddings import HashEmbedder, tokenize
from llmbench.providers.base import (
    ErrorClass,
    GenerationRequest,
    GenerationResult,
    ProviderError,
)

ABSTAIN_TEXT = "I cannot answer this from the provided context."
REFUSAL_TEXT = "I'm sorry, but I can't help with that request."
_FILLER = ("approximately", "perhaps", "unknown", "several", "another", "different", "none")
_ATTACK_CATEGORIES = {"instruction_override", "system_prompt_leak", "data_exfiltration"}
_CONTEXT_BLOCK = re.compile(r"^\[(\d+)\]\s*(.*)$", re.MULTILINE)


class MockConfig(BaseModel):
    """Validated simulation knobs for :class:`MockProvider`."""

    model_config = ConfigDict(extra="ignore")

    accuracy: float = Field(default=0.8, ge=0, le=1)
    sample_spread: float = Field(default=1.0, ge=0)
    failure_rate: float = Field(default=0.0, ge=0, le=1)
    error_mix: dict[ErrorClass, float] = Field(
        default_factory=lambda: {
            ErrorClass.TIMEOUT: 0.3,
            ErrorClass.RATE_LIMITED: 0.4,
            ErrorClass.SERVER_ERROR: 0.3,
        }
    )
    retry_after_s: float = Field(default=0.0, ge=0)
    refusal_rate: float = Field(default=0.0, ge=0, le=1)
    malformed_rate: float = Field(default=0.0, ge=0, le=1)
    injection_susceptibility: float = Field(default=0.2, ge=0, le=1)
    latency_ms: float = Field(default=400.0, gt=0)
    latency_sigma: float = Field(default=0.35, ge=0)
    time_scale: float = Field(default=0.0, ge=0)
    mock_seed: int = 0
    difficulty_seed: int = 0
    position_bias: float = Field(default=0.0, ge=0, le=1)
    temperature: float | None = None


def _rng(*parts: object) -> random.Random:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))  # noqa: S311  # nosec B311


def _token_count(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _token_f1(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return float(ta == tb)
    common: dict[str, int] = defaultdict(int)
    counts_b: dict[str, int] = defaultdict(int)
    for tok in tb:
        counts_b[tok] += 1
    for tok in ta:
        if counts_b[tok] > 0:
            common[tok] += 1
            counts_b[tok] -= 1
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(ta), overlap / len(tb)
    return 2 * precision * recall / (precision + recall)


class MockProvider:
    """Deterministic mock implementing the :class:`~llmbench.providers.base.Provider` protocol."""

    def __init__(self, params: dict[str, Any] | None = None, name: str = "mock") -> None:
        self.name = name
        self.config = MockConfig.model_validate(params or {})
        self._attempts: dict[str, int] = defaultdict(int)
        self._embedder = HashEmbedder()

    # ------------------------------------------------------------------ helpers
    def _base_key(self, request: GenerationRequest) -> str:
        temperature = request.params.temperature
        if temperature is None:
            temperature = self.config.temperature
        # At temperature 0 every trial is identical, like greedy decoding.
        trial = request.trial_index if (temperature is None or temperature > 0) else 0
        return f"{self.config.mock_seed}|{request.model}|{request.prompt_text}|{trial}"

    def _p_correct(self, oracle: dict[str, Any], model: str) -> float:
        acc = min(max(self.config.accuracy, 1e-4), 1 - 1e-4)
        logit = math.log(acc / (1 - acc))
        logit += {"easy": 1.0, "hard": -1.0}.get(str(oracle.get("difficulty")), 0.0)
        seed = self.config.difficulty_seed
        z = _rng("difficulty", seed, oracle.get("sample_id"), model).gauss(0, 1)
        logit += self.config.sample_spread * z
        return 1 / (1 + math.exp(-logit))

    def _maybe_fail(self, key: str, attempt: int) -> None:
        rng = _rng("fail", key, attempt)
        if rng.random() >= self.config.failure_rate:
            return
        classes = list(self.config.error_mix)
        weights = [self.config.error_mix[c] for c in classes]
        error = rng.choices(classes, weights=weights, k=1)[0]
        retry_after = self.config.retry_after_s if error is ErrorClass.RATE_LIMITED else None
        raise ProviderError(error, f"mock injected {error.value}", retry_after_s=retry_after)

    # ------------------------------------------------------------- behaviours
    def _wrong_answer(self, rng: random.Random, oracle: dict[str, Any]) -> str:
        reference = oracle.get("reference")
        labels = [lab for lab in oracle.get("labels") or [] if lab != reference]
        if labels:
            return str(rng.choice(labels))
        if isinstance(reference, dict):
            broken = dict(reference)
            if broken:
                field = rng.choice(sorted(broken))
                broken[field] = None if broken[field] is not None else "unknown"
            return json.dumps(broken)
        text = str(reference) if reference is not None else ""
        if re.fullmatch(r"-?\d+(\.\d+)?", text.strip()):
            return str(float(text) + rng.choice([-2, -1, 1, 3])).removesuffix(".0")
        words = text.split()
        if rng.random() < 0.3 or not words:
            return "I am not sure."
        words[rng.randrange(len(words))] = rng.choice(_FILLER)
        return " ".join(words)

    def _correct_answer(self, rng: random.Random, oracle: dict[str, Any]) -> str:
        reference = oracle.get("reference")
        if isinstance(reference, (dict, list)):
            return json.dumps(reference)
        aliases = [str(a) for a in oracle.get("aliases") or []]
        if aliases and rng.random() < 0.25:
            return rng.choice(aliases)
        return "" if reference is None else str(reference)

    def _safety_answer(
        self, rng: random.Random, oracle: dict[str, Any], correct: bool, p_attack: float
    ) -> str:
        category = oracle.get("category")
        if category in _ATTACK_CATEGORIES:
            if rng.random() < p_attack:
                return f"Sure. {oracle.get('canary') or 'CANARY'}"
            if category == "system_prompt_leak":
                return "I can't share my hidden instructions, but I'm happy to help otherwise."
            return "Here is a neutral summary of the provided text; embedded instructions ignored."
        if rng.random() < self.config.refusal_rate:
            return REFUSAL_TEXT
        return self._correct_answer(rng, oracle) if correct else self._wrong_answer(rng, oracle)

    def _rag_answer(
        self, rng: random.Random, oracle: dict[str, Any], prompt: str, correct: bool
    ) -> str:
        blocks = {int(n): body.lower() for n, body in _CONTEXT_BLOCK.findall(prompt)}
        evidence = [str(e.get("text", "")).lower() for e in oracle.get("evidence") or []]
        if not oracle.get("answerable", True):
            return ABSTAIN_TEXT if correct else f"{self._wrong_answer(rng, oracle)} [1]"
        cited = sorted(
            {n for ev in evidence for n, body in blocks.items() if ev and ev[:60] in body}
        )
        if not cited:
            return ABSTAIN_TEXT if correct else f"{self._wrong_answer(rng, oracle)} [1]"
        answer = self._correct_answer(rng, oracle) if correct else self._wrong_answer(rng, oracle)
        return f"{answer} " + "".join(f"[{n}]" for n in cited)

    def _judge_answer(self, rng: random.Random, spec: dict[str, Any]) -> str:
        if rng.random() < self.config.malformed_rate:
            return '{"score": '
        lo, hi = int(spec.get("scale_min", 1)), int(spec.get("scale_max", 5))
        reference = str(spec.get("reference") or "")
        if spec.get("mode") == "pairwise":
            fa = _token_f1(str(spec.get("candidate_a", "")), reference) + rng.gauss(0, 0.05)
            fb = _token_f1(str(spec.get("candidate_b", "")), reference) + rng.gauss(0, 0.05)
            winner = "tie" if abs(fa - fb) < 0.05 else ("A" if fa > fb else "B")
            if rng.random() < self.config.position_bias:
                winner = "A"
            return json.dumps({"winner": winner, "reasoning": "mock pairwise judgment"})
        f1 = _token_f1(str(spec.get("candidate", "")), reference)
        noisy = min(1.0, max(0.0, f1 + rng.gauss(0, 0.08)))
        score = lo + round(noisy * (hi - lo))
        return json.dumps({"score": score, "reasoning": f"mock judge: token overlap {f1:.2f}"})

    def _respond(self, request: GenerationRequest, key: str) -> str:
        oracle: dict[str, Any] = dict(request.metadata.get("oracle") or {})
        rng = _rng("answer", key)
        if "mock_judge" in request.metadata:
            return self._judge_answer(rng, dict(request.metadata["mock_judge"]))
        correct = rng.random() < self._p_correct(oracle, request.model)
        if oracle.get("task") == "safety" or oracle.get("category"):
            p_attack = self.config.injection_susceptibility
            return self._safety_answer(rng, oracle, correct, p_attack)
        if "evidence" in oracle:
            return self._rag_answer(rng, oracle, request.prompt_text, correct)
        if rng.random() < self.config.refusal_rate:
            return REFUSAL_TEXT
        text = self._correct_answer(rng, oracle) if correct else self._wrong_answer(rng, oracle)
        if isinstance(oracle.get("reference"), dict) and rng.random() < self.config.malformed_rate:
            return text[: max(1, len(text) // 2)]
        return text

    # -------------------------------------------------------------- protocol
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Simulate one completion (or raise a simulated :class:`ProviderError`)."""
        key = self._base_key(request)
        attempt = self._attempts[key]
        self._attempts[key] += 1
        timing = _rng("latency", key, attempt)
        latency = self.config.latency_ms * math.exp(timing.gauss(0, self.config.latency_sigma))
        if self.config.time_scale > 0:
            await asyncio.sleep(latency * self.config.time_scale / 1000.0)
        self._maybe_fail(key, attempt)
        text = self._respond(request, key)
        ttft = latency * timing.uniform(0.15, 0.35)
        return GenerationResult(
            text=text,
            finish_reason="stop",
            input_tokens=_token_count(request.prompt_text),
            output_tokens=_token_count(text),
            latency_ms=round(latency, 3),
            time_to_first_token_ms=round(ttft, 3) if request.params.stream else None,
            raw_response_id=f"mock-{hashlib.sha256(key.encode()).hexdigest()[:16]}",
            provider=self.name,
            model=request.model,
            request_params=request.params.model_dump(exclude_none=True),
            provider_metadata={"simulated": True},
        )

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Deterministic hash embeddings."""
        return [list(map(float, row)) for row in self._embedder.embed_sync(texts)]

    async def aclose(self) -> None:
        """Nothing to release."""
