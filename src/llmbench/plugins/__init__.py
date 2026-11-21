"""Plugin discovery for providers, metrics and task types."""

from llmbench.plugins.registry import Registry, load_plugin_modules

__all__ = ["Registry", "load_plugin_modules"]
