"""LLMBench: reproducible evaluation and regression testing for non-deterministic LLM systems."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("llmbench")
except (
    PackageNotFoundError
):  # pragma: no cover - only when running from a source tree without install
    __version__ = "0.0.0"

__all__ = ["__version__"]
