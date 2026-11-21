# ADR 0006: Typed registries with lazy entry-point discovery for plugins

- Status: Accepted
- Date: 2026-11-21

## Context

Users need to add their own providers (an internal gateway, a self-hosted model), metrics
(domain-specific checks) and task types (custom pass criteria) without forking LLMBench.
Built-ins and plugins should be referenced the same way from experiment YAML (`provider:`,
`metrics:`, `task:`), listed by the CLI, and fail with a clear message when a name is unknown.
Importing LLMBench should stay cheap and not import every installed plugin eagerly.

## Decision

- One generic class, `Registry[T]` (`plugins/registry.py`), holds a name -> object mapping for
  one plugin kind and an entry-point group. Built-ins register at import time. On a lookup miss
  (`get`) or when listing (`names`), the registry loads its entry-point group once
  (`importlib.metadata.entry_points`). Built-in names win over entry points with the same name;
  explicit `register` refuses to overwrite an existing different object unless
  `replace=True`; unknown names raise `RegistryError` listing what is registered.
- Three registries and groups, declared (empty) in `pyproject.toml`:
  - `PROVIDERS` / `llmbench.providers` (`providers/registry.py`): a factory
    `(ModelSpec, Settings) -> Provider`. Built-ins `mock`, `openai`, `openai_compat` and
    `gemini` register with `PROVIDERS.decorator(...)`; the SDK-backed factories import their
    adapter module lazily inside the factory. A provider implements the `Provider` protocol
    (`generate`, `embed`, `aclose`, `name`) and raises `ProviderError` with an `ErrorClass` so
    retries and the breaker work.
  - `METRICS` / `llmbench.metrics` (`metrics/base.py`): a callable returning a `Metric`
    (normally the class). `@register_metric` registers a `Metric` subclass;
    `@trial_metric(name, higher_is_better=..., kind=...)` turns `fn(trial, sample)` into a
    registered `TrialMetric`. `GroupMetric` covers per-sample metrics over repeated trials.
    Each metric declares `higher_is_better` and `kind` (`binary` or `continuous`), which the
    comparison and gate use to pick a test and a direction.
  - `TASKS` / `llmbench.tasks` (`tasks/base.py`): a `Task` instance providing `parse`,
    `is_pass`, `oracle`, `default_metrics` and `json_output`.
- `llmbench providers`, `llmbench metrics` and `llmbench tasks` list everything registered,
  including plugins.

A third-party package exposes plugins in its own `pyproject.toml`:

```toml
[project.entry-points."llmbench.providers"]
my_gateway = "my_pkg.llmbench_plugin:gateway_factory"

[project.entry-points."llmbench.metrics"]
mentions_source = "my_pkg.llmbench_plugin:MentionsSource"

[project.entry-points."llmbench.tasks"]
ticket_routing = "my_pkg.llmbench_plugin:TICKET_ROUTING_TASK"
```

## Consequences

Positive:

- One mechanism, three plugin kinds; YAML names resolve identically for built-ins and plugins.
- Entry points load lazily, so a plain `import llmbench` does not import installed plugins.
- Typed protocols and base classes give plugin authors a small, documented surface; metrics
  inherit failure handling (`error_value`) and per-sample result shapes from the base classes.

Negative:

- Entry points are loaded as-is: a provider entry must be a factory function, a metric entry
  a `Metric` class (or factory), and a task entry an instance, not a class. Nothing validates
  the loaded object's type at load time; a mistake surfaces on first use.
- A plugin that fails to import raises on the first registry miss or listing. The registry marks
  its group as loaded before iterating, so entry points after the broken one are not loaded for
  the rest of the process.
- Plugins run in-process with full privileges; installing one is equivalent to running its code.
- `check_availability` only knows the built-in providers' requirements (API keys, `base_url`);
  a plugin provider is treated as available, and missing credentials appear at call time.
- Plugin metrics not registered under the name used in a stored run fall back to
  higher-is-better, continuous defaults in `metric_direction`.

## Alternatives considered

- **Import-path strings in YAML** (`metric: my_pkg.module:Class`). No packaging step, but ties
  configs to module layout and executes arbitrary imports named in data files.
- **pluggy hooks.** Mature, but hook specifications add ceremony for what are plain lookups.
- **A single registry for all kinds.** Fewer objects, but loses typing and allows name clashes
  across kinds.
- **Eager entry-point loading at import.** Simpler control flow, slower startup, and any broken
  plugin would break every command.
