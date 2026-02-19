"""Registry pattern for scorers, providers, and judges."""

from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Registry:
    """A generic registry for named factories or classes.

    Supports registering callables under string keys and instantiating
    them by name with optional kwargs.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: dict[str, Callable[..., Any]] = {}

    def register(self, key: str, factory: Callable[..., Any]) -> None:
        """Register a factory callable under the given key."""
        if key in self._registry:
            logger.warning("Overwriting existing registry entry '%s' in '%s'", key, self.name)
        self._registry[key] = factory
        logger.debug("Registered '%s' in registry '%s'", key, self.name)

    def register_decorator(self, key: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator factory for registering classes/functions."""

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register(key, fn)
            return fn

        return decorator

    def get(self, key: str) -> Callable[..., Any]:
        """Retrieve a registered factory by key."""
        if key not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"No entry '{key}' in registry '{self.name}'. "
                f"Available: [{available}]"
            )
        return self._registry[key]

    def create(self, key: str, **kwargs: Any) -> Any:
        """Create an instance using the registered factory for the given key."""
        factory = self.get(key)
        return factory(**kwargs)

    def list_keys(self) -> list[str]:
        """Return a sorted list of registered keys."""
        return sorted(self._registry.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def __repr__(self) -> str:
        return f"Registry(name={self.name!r}, keys={self.list_keys()})"


class _GlobalRegistry:
    """Container for the application-wide registries."""

    def __init__(self) -> None:
        self.providers = Registry("providers")
        self.scorers = Registry("scorers")
        self.judges = Registry("judges")
        self.storage_backends = Registry("storage_backends")

    def __repr__(self) -> str:
        return (
            f"GlobalRegistry("
            f"providers={self.providers.list_keys()}, "
            f"scorers={self.scorers.list_keys()}, "
            f"judges={self.judges.list_keys()}, "
            f"storage_backends={self.storage_backends.list_keys()}"
            f")"
        )


# Singleton global registry
global_registry = _GlobalRegistry()
