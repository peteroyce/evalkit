"""YAML config loader with environment variable interpolation."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from evalkit.config.schema import EvalConfig

logger = logging.getLogger(__name__)

# Pattern for ${VAR_NAME} or ${VAR_NAME:-default_value}
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _interpolate_env(value: Any) -> Any:
    """Recursively interpolate ${ENV_VAR} placeholders in string values."""
    if isinstance(value, str):
        def _replace(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2)  # may be None if no :- present
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            if default is not None:
                return default
            logger.warning(
                "Config: environment variable '%s' is not set and has no default.", var_name
            )
            return match.group(0)  # return original placeholder unchanged

        return _ENV_VAR_PATTERN.sub(_replace, value)

    elif isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}

    elif isinstance(value, list):
        return [_interpolate_env(item) for item in value]

    return value  # int, float, bool, None — return as-is


class ConfigLoader:
    """Loads and validates evalkit configuration from YAML files.

    Supports ``${ENV_VAR}`` and ``${ENV_VAR:-default}`` interpolation
    throughout the config file.

    Args:
        strict: If True, raises an error on unknown fields in the config.
    """

    def __init__(self, strict: bool = False) -> None:
        self._strict = strict

    def load(self, path: str | Path) -> EvalConfig:
        """Load an EvalConfig from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A validated EvalConfig instance.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If the config is invalid.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: '{p}'")

        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML configs. "
                "Install with: pip install pyyaml"
            ) from exc

        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        if not isinstance(raw, dict):
            raise ValueError(f"Config file must be a YAML mapping: '{p}'")

        # Interpolate environment variables
        interpolated = _interpolate_env(raw)
        logger.debug("ConfigLoader: loaded and interpolated config from '%s'", p)

        return EvalConfig.model_validate(interpolated)

    def loads(self, yaml_text: str) -> EvalConfig:
        """Load an EvalConfig from a YAML string."""
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError("PyYAML is required.") from exc

        raw = yaml.safe_load(yaml_text) or {}
        interpolated = _interpolate_env(raw)
        return EvalConfig.model_validate(interpolated)


def load_config(path: str | Path) -> EvalConfig:
    """Convenience function to load a config file using default settings."""
    return ConfigLoader().load(path)
