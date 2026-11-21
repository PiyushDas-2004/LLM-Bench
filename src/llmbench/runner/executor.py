"""The async benchmark executor.

Design (see ``docs/adr/0002-async-runner.md``):

* A fixed pool of ``concurrency`` workers pulls from a lazily expanded grid,
  so memory stays flat for very large grids.
* Each provider gets its own token-bucket :class:`RateLimiter` and
  :class:`CircuitBreaker`; retries use full-jitter backoff honouring
  ``Retry-After``.
* Every finished trial is written to the store immediately; ``resume`` skips
  trials already present, so an interrupted run continues exactly where it
  stopped and produces the same final results.
* A :class:`BudgetGuard` reserves estimated spend before each call and stops
  the run gracefully (partial results kept) when a limit would be exceeded.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from llmbench.config.loader import parse_experiment
from llmbench.config.models import ExperimentConfig, ModelSpec
from llmbench.config.settings import Settings, get_settings
from llmbench.domain import TrialRecord, utcnow
from llmbench.errors import BudgetExceededError, ConfigError, DatasetDriftError
from llmbench.observability import span
from llmbench.providers.base import (
    ErrorClass,
    GenerationParams,
    GenerationRequest,
    GenerationResult,
    Provider,
    ProviderError,
    looks_like_refusal,
)
from llmbench.providers.registry import create_provider
from llmbench.runner.budget import BudgetGuard
from llmbench.runner.cache import ResponseCache
from llmbench.runner.circuit import CircuitBreaker
from llmbench.runner.clock import Clock, SystemClock
from llmbench.runner.manifest import build_manifest, config_hash, make_run_id
from llmbench.runner.plan import estimate_tokens, max_output_tokens
from llmbench.runner.prepare import PreparedExperiment, WorkItem, prepare
from llmbench.runner.rate_limit import RateLimiter
from llmbench.runner.retry import RetryPolicy
from llmbench.tracking.store import RunStore

log = structlog.get_logger(__name__)

#: A breaker that has opened this many times marks its model as abandoned for the run.
MAX_BREAKER_TRIPS = 3

ProviderFactory = Callable[[ModelSpec, Settings], Provider]
ProgressCallback = Callable[["RunProgress"], None]
Evaluator = Callable[[RunStore, str], Awaitable[None]]


@dataclass
class RunProgress:
    """Live counters reported to progress callbacks."""

    run_id: str
    total: int
    done: int = 0
    skipped_existing: int = 0
    failed: int = 0
    cache_hits: int = 0
    spent_usd: float = 0.0


@dataclass
class RunOutcome:
    """Final state of a run (or of one resume session)."""

    run_id: str
    status: str
    n_expected: int
    n_done: int
    n_failed: int
    cache_hits: int
    spent_usd: float
    stop_reason: str | None = None
    skipped_models: dict[str, str] = field(default_factory=dict)


@dataclass
class _ProviderSlot:
    provider: Provider
    limiter: RateLimiter
    breaker: CircuitBreaker
    abandoned: bool = False


class Runner:
    """Executes experiments against providers and records every trial.

    Args:
        store: Run database.
        settings: Process settings (keys, paths).
        provider_factory: Builds a provider for a model spec (injectable for tests).
        clock: Time source for rate limiting, retries and breakers.
        evaluator: Coroutine computing metrics after execution (``None`` to skip).
        progress: Optional callback invoked after every trial.
    """

    def __init__(
        self,
        store: RunStore,
        settings: Settings | None = None,
        *,
        provider_factory: ProviderFactory | None = None,
        clock: Clock | None = None,
        evaluator: Evaluator | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.store = store
        self.settings = settings or get_settings()
        self.provider_factory = provider_factory or create_provider
        self.clock = clock or SystemClock()
        self.evaluator = evaluator
        self.progress = progress
        self._stop = asyncio.Event()
        self._stop_reason: str | None = None

    def request_stop(self, reason: str = "interrupted") -> None:
        """Ask workers to finish their in-flight trial and stop (used for SIGINT)."""
        if not self._stop.is_set():
            self._stop_reason = reason
            self._stop.set()

    @property
    def stop_reason(self) -> str | None:
        """Why the current run was asked to stop, if it was."""
        return self._stop_reason

    # ----------------------------------------------------------------- public
    async def run(
        self,
        config: ExperimentConfig,
        *,
        use_cache: bool = True,
        tags: dict[str, str] | None = None,
        strict_models: bool = False,
    ) -> RunOutcome:
        """Start a new run of ``config``."""
        prepared = prepare(config, self.settings, strict_models=strict_models)
        created = utcnow()
        dump = config.model_dump(mode="json")
        run_id = make_run_id(config.name, config_hash(dump), created)
        manifest = build_manifest(prepared, run_id, created)
        manifest["cache_enabled"] = use_cache and config.cache
        self.store.create_run(
            run_id=run_id,
            name=config.name,
            task=prepared.task.name,
            dataset_name=prepared.dataset.name,
            dataset_hash=prepared.evaluated_hash,
            config=dump,
            manifest=manifest,
            tags={**config.tags, **(tags or {})},
            n_expected=prepared.grid_size,
        )
        log.info("run_started", run_id=run_id, grid=prepared.grid_size)
        return await self._execute(prepared, run_id, use_cache=use_cache and config.cache)

    async def resume(self, run_id: str, *, use_cache: bool = True) -> RunOutcome:
        """Continue an interrupted run, executing only the missing trials."""
        row = self.store.get_run(run_id)
        manifest = row.manifest
        config = parse_experiment(manifest["config"], Path(manifest["base_dir"]))
        prepared = prepare(config, self.settings)
        prepared.samples = [
            s for s in prepared.dataset.samples if s.id in set(manifest["dataset"]["sample_ids"])
        ]
        if prepared.evaluated_hash != row.dataset_hash:
            raise DatasetDriftError(
                f"dataset changed since run {run_id} started "
                f"({row.dataset_hash[:12]} -> {prepared.evaluated_hash[:12]}); cannot resume"
            )
        expected_models = {m["id"] for m in manifest["models"]}
        missing = expected_models - {m.id for m in prepared.models}
        if missing:
            raise ConfigError(f"cannot resume: models no longer runnable here: {sorted(missing)}")
        prepared.models = [m for m in prepared.models if m.id in expected_models]
        self.store.update_run(run_id, status="running", stop_reason=None, finished_at=None)
        log.info("run_resumed", run_id=run_id)
        return await self._execute(
            prepared, run_id, use_cache=use_cache and bool(manifest.get("cache_enabled", True))
        )

    # --------------------------------------------------------------- internals
    def _build_slots(self, prepared: PreparedExperiment) -> dict[str, _ProviderSlot]:
        config = prepared.config
        slots: dict[str, _ProviderSlot] = {}
        for spec in prepared.models:
            limits = spec.rate_limit or config.rate_limit
            slots[spec.id] = _ProviderSlot(
                provider=self.provider_factory(spec, self.settings),
                limiter=RateLimiter(limits, self.clock),
                breaker=CircuitBreaker(
                    config.circuit_breaker.failure_threshold,
                    config.circuit_breaker.reset_timeout_s,
                    self.clock,
                ),
            )
        return slots

    async def _execute(
        self, prepared: PreparedExperiment, run_id: str, *, use_cache: bool
    ) -> RunOutcome:
        config = prepared.config
        done_keys = self.store.completed_trial_keys(run_id)
        slots = self._build_slots(prepared)
        cache = ResponseCache(self.store, enabled=use_cache)
        budget = BudgetGuard(config.budget)
        policy = RetryPolicy.from_config(config.retry)
        # Seeded, non-cryptographic RNG: only used for retry jitter.
        rng = random.Random(config.seed)  # noqa: S311  # nosec B311
        progress = RunProgress(run_id=run_id, total=prepared.grid_size, done=len(done_keys))
        progress.skipped_existing = len(done_keys)
        items: Iterator[WorkItem] = (i for i in prepared.iter_items() if i.key not in done_keys)
        self._stop.clear()
        self._stop_reason = None

        async def worker() -> None:
            for item in items:
                if self._stop.is_set():
                    return
                slot = slots[item.model.id]
                if slot.abandoned:
                    continue
                try:
                    trial = await self._run_item(
                        prepared, run_id, item, slot, cache, budget, policy, rng
                    )
                except BudgetExceededError as exc:
                    self.request_stop(f"budget_exceeded: {exc}")
                    return
                self.store.add_trial(trial)
                progress.done += 1
                progress.failed += int(not trial.ok)
                progress.cache_hits += int(trial.cached)
                progress.spent_usd = budget.spent_usd
                if self.progress is not None:
                    self.progress(progress)

        with span("llmbench.run", run_id=run_id, grid=prepared.grid_size):
            try:
                async with asyncio.TaskGroup() as group:
                    for _ in range(config.concurrency):
                        group.create_task(worker())
            finally:
                for slot in slots.values():
                    await slot.provider.aclose()

        abandoned = sorted(mid for mid, s in slots.items() if s.abandoned)
        stop_reason = self.stop_reason
        self.store.refresh_run_totals(run_id)
        n_done = len(self.store.completed_trial_keys(run_id))
        if stop_reason and stop_reason.startswith("budget_exceeded"):
            status = "budget_exceeded"
        elif stop_reason:
            status = "interrupted"
        elif n_done < prepared.grid_size:
            status = "partial"
        else:
            status = "completed"
        if abandoned:
            stop_reason = (stop_reason or "") + f" circuit breaker abandoned models: {abandoned}"
        if self.evaluator is not None and n_done:
            await self.evaluator(self.store, run_id)
        self.store.update_run(run_id, status=status, stop_reason=stop_reason)
        log.info("run_finished", run_id=run_id, status=status, done=n_done)
        return RunOutcome(
            run_id=run_id,
            status=status,
            n_expected=prepared.grid_size,
            n_done=n_done,
            n_failed=progress.failed,
            cache_hits=cache.hits,
            spent_usd=budget.spent_usd,
            stop_reason=stop_reason,
            skipped_models=prepared.skipped_models,
        )

    def _request_for(self, prepared: PreparedExperiment, item: WorkItem) -> GenerationRequest:
        rendered = prepared.render(item.prompt_id, item.sample)
        params = GenerationParams.from_mapping(item.model.params)
        if prepared.task.json_output and "json_mode" not in item.model.params:
            params = params.model_copy(update={"json_mode": True})
        return GenerationRequest.from_prompt(
            item.model.model_name,
            rendered.user,
            system=rendered.system,
            params=params,
            trial_index=item.trial_index,
            metadata={"oracle": prepared.task.oracle(item.sample)},
        )

    async def _run_item(
        self,
        prepared: PreparedExperiment,
        run_id: str,
        item: WorkItem,
        slot: _ProviderSlot,
        cache: ResponseCache,
        budget: BudgetGuard,
        policy: RetryPolicy,
        rng: random.Random,
    ) -> TrialRecord:
        request = self._request_for(prepared, item)
        provider_name = item.model.provider
        model_config = {"params": item.model.params, "base_url": item.model.base_url}
        cached = cache.get(provider_name, request, model_config)
        attempts = 0
        queue_wait = 0.0
        if cached is not None:
            result = cached
        else:
            est_in = estimate_tokens(request.prompt_text)
            est_out = max_output_tokens(item.model.params)
            est_cost = prepared.pricing.cost(provider_name, item.model.model_name, est_in, est_out)
            budget.reserve(est_cost or 0.0)
            with span("llmbench.trial", trial_key=item.key):
                result, attempts, queue_wait = await self._call_with_retries(
                    slot, request, est_in + est_out, prepared.config, policy, rng
                )
            actual = prepared.pricing.cost(
                provider_name, item.model.model_name, result.input_tokens, result.output_tokens
            )
            budget.settle(est_cost or 0.0, actual if result.ok else 0.0)
            cache.put(provider_name, request, result, model_config)
        return self._to_trial(
            prepared, run_id, item, request, result, attempts, queue_wait, cached is not None
        )

    async def _call_with_retries(
        self,
        slot: _ProviderSlot,
        request: GenerationRequest,
        token_estimate: int,
        config: ExperimentConfig,
        policy: RetryPolicy,
        rng: random.Random,
    ) -> tuple[GenerationResult, int, float]:
        queue_wait = 0.0
        attempt = 0
        while True:
            attempt += 1
            start = self.clock.now()
            await slot.breaker.acquire()
            await slot.limiter.acquire(token_estimate)
            queue_wait += (self.clock.now() - start) * 1000
            try:
                with span("llmbench.provider_call", provider=slot.provider.name, attempt=attempt):
                    result = await asyncio.wait_for(
                        slot.provider.generate(request), timeout=config.retry.timeout_s
                    )
            except (ProviderError, TimeoutError) as exc:
                error = (
                    exc
                    if isinstance(exc, ProviderError)
                    else ProviderError(
                        ErrorClass.TIMEOUT, f"timed out after {config.retry.timeout_s}s"
                    )
                )
                if error.error_class not in (
                    ErrorClass.INVALID_REQUEST,
                    ErrorClass.CONTENT_FILTERED,
                ):
                    slot.breaker.record_failure()
                    if slot.breaker.trips >= MAX_BREAKER_TRIPS:
                        slot.abandoned = True
                if policy.should_retry(error, attempt) and not slot.abandoned:
                    await self.clock.sleep(policy.delay(attempt, error, rng))
                    continue
                failed = GenerationResult(
                    error=error.error_class,
                    error_message=error.message,
                    provider=slot.provider.name,
                    model=request.model,
                    request_params=request.params.model_dump(exclude_none=True),
                )
                return failed, attempt, queue_wait
            slot.breaker.record_success()
            return result, attempt, queue_wait

    @staticmethod
    def _to_trial(
        prepared: PreparedExperiment,
        run_id: str,
        item: WorkItem,
        request: GenerationRequest,
        result: GenerationResult,
        attempts: int,
        queue_wait_ms: float,
        cached: bool,
    ) -> TrialRecord:
        task = prepared.task
        text = result.text
        parsed: Any = task.parse(text, item.sample) if result.ok else None
        malformed = bool(result.ok and task.json_output and parsed is None)
        refused = bool(
            result.error is ErrorClass.REFUSAL or (result.ok and looks_like_refusal(text))
        )
        passed = bool(result.ok and task.is_pass(text, item.sample))
        cost = prepared.pricing.cost(
            item.model.provider, item.model.model_name, result.input_tokens, result.output_tokens
        )
        system = next((m.content for m in request.messages if m.role == "system"), None)
        return TrialRecord(
            run_id=run_id,
            trial_key=item.key,
            model_id=item.model.id,
            prompt_id=item.prompt_id,
            sample_id=item.sample.id,
            trial_index=item.trial_index,
            system_prompt=system,
            rendered_prompt=request.messages[-1].content,
            output_text=text,
            parsed_output=parsed,
            passed=passed,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms if result.ok else None,
            ttft_ms=result.time_to_first_token_ms,
            queue_wait_ms=round(queue_wait_ms, 3),
            attempts=max(attempts, 1),
            error_class=result.error,
            error_message=result.error_message,
            finish_reason=result.finish_reason,
            refused=refused,
            malformed=malformed,
            cost_usd=cost if result.ok else (0.0 if cost is not None else None),
            cached=cached,
            response_id=result.raw_response_id,
        )
