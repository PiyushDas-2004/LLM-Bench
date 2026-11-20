"""Exception hierarchy shared across LLMBench."""

from __future__ import annotations


class LLMBenchError(Exception):
    """Base class for all LLMBench errors."""


class ConfigError(LLMBenchError):
    """Invalid configuration, dataset, prompt or pricing file."""


class DatasetError(ConfigError):
    """A dataset failed to load or validate."""


class PromptError(ConfigError):
    """A prompt template failed to load, validate or render."""


class RegistryError(LLMBenchError):
    """A plugin (provider, metric, task) could not be found or registered."""


class BudgetExceededError(LLMBenchError):
    """Raised internally when a run would exceed its configured budget."""


class CircuitOpenError(LLMBenchError):
    """Raised when a provider's circuit breaker is open."""


class DatasetDriftError(LLMBenchError):
    """Two runs were compared on different dataset versions."""


class RunNotFoundError(LLMBenchError):
    """A run id (or baseline name) does not exist in the store."""
