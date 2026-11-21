from __future__ import annotations

import asyncio
import random

import pytest

from llmbench.config import BudgetConfig, RateLimitConfig
from llmbench.errors import BudgetExceededError
from llmbench.providers import ErrorClass, GenerationParams, GenerationRequest, ProviderError
from llmbench.runner import (
    BreakerState,
    BudgetGuard,
    CircuitBreaker,
    FakeClock,
    RateLimiter,
    RetryPolicy,
    SystemClock,
    TokenBucket,
    cache_key,
)


async def test_token_bucket_waits_for_refill() -> None:
    clock = FakeClock()
    bucket = TokenBucket(rate_per_s=2.0, capacity=2.0, clock=clock)
    assert await bucket.acquire(2) == 0
    waited = await bucket.acquire(1)
    assert waited == pytest.approx(0.5)
    assert clock.t == pytest.approx(0.5)
    assert bucket.available == pytest.approx(0.0)
    # oversized requests are clamped to capacity instead of deadlocking
    clock.t += 10
    assert await bucket.acquire(100) == 0
    with pytest.raises(ValueError, match="positive"):
        TokenBucket(0, 1)


async def test_rate_limiter_enforces_rpm() -> None:
    clock = FakeClock()
    limiter = RateLimiter(RateLimitConfig(requests_per_minute=60, tokens_per_minute=6000), clock)
    for _ in range(5):
        await limiter.acquire(10)
    # 60 rpm = 1/s with burst 1: five requests need ~4 seconds of virtual time
    assert clock.t == pytest.approx(4.0)
    unlimited = RateLimiter(RateLimitConfig(), clock)
    assert await unlimited.acquire(10**9) == 0


async def test_rate_limiter_enforces_tpm() -> None:
    clock = FakeClock()
    limiter = RateLimiter(RateLimitConfig(tokens_per_minute=600), clock)  # 10 tokens/s
    await limiter.acquire(10)
    await limiter.acquire(10)
    assert clock.t == pytest.approx(1.0)


def test_retry_policy_classes_and_backoff() -> None:
    policy = RetryPolicy(max_attempts=4, base_delay_s=1.0, max_delay_s=5.0)
    transient = ProviderError(ErrorClass.SERVER_ERROR, "x")
    assert policy.should_retry(transient, 1)
    assert not policy.should_retry(transient, 4)
    for cls in (ErrorClass.AUTH, ErrorClass.INVALID_REQUEST, ErrorClass.CONTENT_FILTERED):
        assert not policy.should_retry(ProviderError(cls, "x"), 1)
    rng = random.Random(0)
    delays = [policy.delay(a, transient, rng) for a in (1, 2, 3, 10) for _ in range(50)]
    assert all(0 <= d <= 5.0 for d in delays)
    limited = ProviderError(ErrorClass.RATE_LIMITED, "x", retry_after_s=7.0)
    assert policy.delay(1, limited, rng) == 7.0
    huge = ProviderError(ErrorClass.RATE_LIMITED, "x", retry_after_s=10_000)
    assert policy.delay(1, huge, rng) == 300.0


async def test_circuit_breaker_lifecycle() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=2, reset_timeout_s=10, clock=clock)
    assert await breaker.acquire() == 0
    breaker.record_failure()
    assert breaker.state is BreakerState.CLOSED
    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    assert breaker.trips == 1
    waited = await breaker.acquire()  # waits until half-open
    assert waited == pytest.approx(10)
    assert breaker.state is BreakerState.HALF_OPEN
    breaker.record_failure()  # failed probe re-opens
    assert breaker.state is BreakerState.OPEN
    assert breaker.trips == 2
    await breaker.acquire()
    breaker.record_success()
    assert breaker.state is BreakerState.CLOSED


async def test_circuit_breaker_single_probe_in_half_open() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout_s=1, clock=clock)
    breaker.record_failure()
    await breaker.acquire()  # first caller becomes the probe
    second = asyncio.create_task(breaker.acquire())
    await asyncio.sleep(0)
    assert not second.done()
    breaker.record_success()
    await asyncio.wait_for(second, 1)


def test_budget_guard() -> None:
    guard = BudgetGuard(BudgetConfig(max_usd=1.0, max_requests=3))
    guard.reserve(0.4)
    guard.settle(0.4, 0.3)
    guard.reserve(0.4)
    guard.settle(0.4, None)  # unknown actual cost counts the estimate
    assert guard.spent_usd == pytest.approx(0.7)
    with pytest.raises(BudgetExceededError, match="max_usd"):
        guard.reserve(0.5)
    with pytest.raises(BudgetExceededError):
        guard.reserve(0.0)  # stays exceeded
    counter = BudgetGuard(BudgetConfig(max_requests=1))
    counter.reserve(0)
    with pytest.raises(BudgetExceededError, match="max_requests"):
        counter.reserve(0)


def test_cache_key_includes_trial_index_and_params() -> None:
    def req(trial: int, temp: float = 0.5, text: str = "hi") -> GenerationRequest:
        return GenerationRequest.from_prompt(
            "m", text, params=GenerationParams(temperature=temp), trial_index=trial
        )

    assert cache_key("p", req(0)) == cache_key("p", req(0))
    assert cache_key("p", req(0)) != cache_key("p", req(1))
    assert cache_key("p", req(0)) != cache_key("p", req(0, temp=0.6))
    assert cache_key("p", req(0)) != cache_key("p", req(0, text="ho"))
    assert cache_key("p", req(0)) != cache_key("q", req(0))
    assert cache_key("p", req(0), {"params": {"accuracy": 0.9}}) != cache_key(
        "p", req(0), {"params": {"accuracy": 0.3}}
    )


async def test_system_clock() -> None:
    clock = SystemClock()
    t0 = clock.now()
    await clock.sleep(0)
    assert clock.now() >= t0
