"""A small typed registry with lazy Python entry-point discovery.

Providers, metrics and tasks are each held in a :class:`Registry`. Built-ins
register themselves at import time; third-party packages expose objects through
entry points (``llmbench.providers``, ``llmbench.metrics``, ``llmbench.tasks``)
which are loaded the first time a name is not found.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from importlib.metadata import entry_points
from typing import Generic, TypeVar

from llmbench.errors import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    """Name -> object mapping for one plugin kind."""

    def __init__(self, kind: str, entry_point_group: str) -> None:
        self.kind = kind
        self.entry_point_group = entry_point_group
        self._items: dict[str, T] = {}
        self._entry_points_loaded = False

    def register(self, name: str, item: T, *, replace: bool = False) -> T:
        """Register ``item`` under ``name``; refuses silent overrides unless ``replace``."""
        if not replace and name in self._items and self._items[name] is not item:
            raise RegistryError(f"{self.kind} {name!r} is already registered")
        self._items[name] = item
        return item

    def decorator(self, name: str, *, replace: bool = False) -> Callable[[T], T]:
        """Return a decorator that registers the decorated object under ``name``."""

        def wrap(item: T) -> T:
            return self.register(name, item, replace=replace)

        return wrap

    def unregister(self, name: str) -> None:
        """Remove ``name`` if present (used by tests and plugin reloads)."""
        self._items.pop(name, None)

    def load_entry_points(self) -> None:
        """Import every entry point in this registry's group (idempotent)."""
        if self._entry_points_loaded:
            return
        self._entry_points_loaded = True
        for ep in entry_points(group=self.entry_point_group):
            obj = ep.load()
            if ep.name not in self._items:
                self._items[ep.name] = obj

    def get(self, name: str) -> T:
        """Return the item registered as ``name`` or raise :class:`RegistryError`."""
        if name not in self._items:
            self.load_entry_points()
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "<none>"
            raise RegistryError(f"unknown {self.kind} {name!r}; registered: {known}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._items))

    def names(self) -> list[str]:
        """Sorted names of all registered items, including entry-point plugins."""
        self.load_entry_points()
        return sorted(self._items)
