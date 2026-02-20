"""Providers package — model API adapters."""

from evalkit.providers.base import BaseProvider
from evalkit.providers.mock import MockProvider
from evalkit.providers.openai_provider import OpenAIProvider
from evalkit.providers.anthropic_provider import AnthropicProvider


def create_provider(provider_type: str, **kwargs) -> BaseProvider:
    """Factory function to create a provider by type name.

    Args:
        provider_type: One of "openai", "anthropic", "mock".
        **kwargs: Provider-specific configuration kwargs.

    Returns:
        An instantiated BaseProvider subclass.
    """
    registry: dict[str, type[BaseProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "mock": MockProvider,
    }
    if provider_type not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"Unknown provider type '{provider_type}'. Available: [{available}]"
        )
    return registry[provider_type](**kwargs)


__all__ = [
    "BaseProvider",
    "MockProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "create_provider",
]
