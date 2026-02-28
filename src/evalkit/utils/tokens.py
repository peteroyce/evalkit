"""Token counting utility using tiktoken, with a character-based fallback."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Cost table: model -> (input_cost_per_token, output_cost_per_token) in USD
_COST_TABLE: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50e-6, 10.00e-6),
    "gpt-4o-mini": (0.15e-6, 0.60e-6),
    "gpt-4-turbo": (10.00e-6, 30.00e-6),
    "gpt-4": (30.00e-6, 60.00e-6),
    "gpt-3.5-turbo": (0.50e-6, 1.50e-6),
    "o1": (15.00e-6, 60.00e-6),
    "o1-mini": (3.00e-6, 12.00e-6),
    # Anthropic
    "claude-3-5-sonnet-20241022": (3.00e-6, 15.00e-6),
    "claude-3-5-haiku-20241022": (0.80e-6, 4.00e-6),
    "claude-3-opus-20240229": (15.00e-6, 75.00e-6),
    "claude-3-sonnet-20240229": (3.00e-6, 15.00e-6),
    "claude-3-haiku-20240307": (0.25e-6, 1.25e-6),
    # Fallback approximations for common model families
    "claude-2": (8.00e-6, 24.00e-6),
    "llama": (0.20e-6, 0.20e-6),
    "mistral": (0.20e-6, 0.60e-6),
}

# Approximate characters per token for the fallback estimator
_CHARS_PER_TOKEN = 4.0


class TokenCounter:
    """Counts tokens in text strings using tiktoken when available.

    Falls back to a character-based approximation if tiktoken is not
    installed or if the model's encoding is not recognised.

    Args:
        model: Model name used to select the correct tiktoken encoding.
               Defaults to "gpt-4o" (cl100k_base encoding).
        fallback_chars_per_token: Characters-per-token ratio used when
            tiktoken is unavailable.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        fallback_chars_per_token: float = _CHARS_PER_TOKEN,
    ) -> None:
        self._model = model
        self._fallback_chars_per_token = fallback_chars_per_token
        self._encoder: Any = None
        self._tried_tiktoken = False

    def _get_encoder(self) -> Any:
        if self._tried_tiktoken:
            return self._encoder
        self._tried_tiktoken = True
        try:
            import tiktoken

            try:
                self._encoder = tiktoken.encoding_for_model(self._model)
                logger.debug("TokenCounter: using tiktoken encoding for '%s'", self._model)
            except KeyError:
                # Unknown model — fall back to cl100k_base
                self._encoder = tiktoken.get_encoding("cl100k_base")
                logger.debug(
                    "TokenCounter: unknown model '%s', using cl100k_base encoding",
                    self._model,
                )
        except ImportError:
            logger.info(
                "TokenCounter: tiktoken not installed; using character-based fallback "
                "(%.1f chars/token)",
                self._fallback_chars_per_token,
            )
        return self._encoder

    def count(self, text: str) -> int:
        """Count the number of tokens in a text string.

        Args:
            text: The text to count tokens for.

        Returns:
            Token count (exact if tiktoken is available, estimated otherwise).
        """
        if not text:
            return 0

        encoder = self._get_encoder()
        if encoder is not None:
            return len(encoder.encode(text))

        # Fallback: character-based estimate
        return max(1, round(len(text) / self._fallback_chars_per_token))

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        """Count tokens across a list of chat messages.

        Applies a small per-message overhead (4 tokens) in line with
        the OpenAI token counting guide.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.

        Returns:
            Estimated total token count.
        """
        total = 3  # every reply is primed with <|start|>assistant<|message|>
        for msg in messages:
            total += 4  # every message has: <|start|>{role/name}\n{content}<|end|>\n
            for key, value in msg.items():
                total += self.count(str(value))
                if key == "name":
                    total -= 1  # if name is present, role is omitted
        return total

    def truncate(self, text: str, max_tokens: int, suffix: str = "...") -> str:
        """Truncate text to at most max_tokens tokens.

        Args:
            text: Input text.
            max_tokens: Maximum number of tokens to keep.
            suffix: String appended to truncated text.

        Returns:
            Truncated text (with suffix if truncation occurred).
        """
        encoder = self._get_encoder()
        if encoder is not None:
            tokens = encoder.encode(text)
            if len(tokens) <= max_tokens:
                return text
            truncated = encoder.decode(tokens[:max_tokens])
            return truncated + suffix

        # Fallback: character-based
        max_chars = int(max_tokens * self._fallback_chars_per_token)
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + suffix


def estimate_cost(
    model: str,
    tokens_in: int,
    tokens_out: int,
) -> float | None:
    """Estimate the cost of an API call in USD.

    Args:
        model: Model identifier.
        tokens_in: Number of input/prompt tokens.
        tokens_out: Number of output/completion tokens.

    Returns:
        Estimated cost in USD, or None if the model is not in the cost table.
    """
    # Try exact match first
    costs = _COST_TABLE.get(model)
    if costs is None:
        # Try prefix match (e.g., "claude-3" matches "claude-3-haiku-...")
        for key, val in _COST_TABLE.items():
            if model.startswith(key) or key.startswith(model.split("-")[0]):
                costs = val
                break

    if costs is None:
        logger.debug("estimate_cost: no cost data for model '%s'", model)
        return None

    return tokens_in * costs[0] + tokens_out * costs[1]


def format_cost(cost_usd: float | None) -> str:
    """Format a cost in USD as a human-readable string."""
    if cost_usd is None:
        return "N/A"
    if cost_usd < 0.001:
        return f"${cost_usd * 1000:.4f}m"  # millicents
    return f"${cost_usd:.6f}"
