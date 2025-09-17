"""Configuration management for the chat CLI."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

try:  # Python 3.11+
    import tomllib as _tomllib
except ModuleNotFoundError:  # pragma: no cover
    _tomllib = None


DEFAULT_CONFIG_PATH = Path.home() / ".config" / "chat-cli" / "config.toml"
LOCAL_FALLBACK_CONFIG_PATH = Path.cwd() / ".chat-cli" / "config.toml"
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE_PATH = PACKAGE_ROOT / "configs" / "default_config.toml"


class ConfigError(Exception):
    """Raised when the configuration file is missing or malformed."""


@dataclass
class Config:
    """Strongly typed configuration values for the chat client."""

    api_url: str
    api_key: str
    model: str
    temperature: float


def ensure_config_file(path: Path | None = None) -> Path:
    """Ensure the config file exists, writing defaults if necessary."""

    candidates = [path] if path else [DEFAULT_CONFIG_PATH, LOCAL_FALLBACK_CONFIG_PATH]
    last_error: Optional[Exception] = None

    for candidate in filter(None, candidates):
        candidate = Path(candidate)
        if candidate.exists():
            return candidate
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_text(_default_config_text(), encoding="utf-8")
            return candidate
        except PermissionError as exc:
            last_error = exc
            continue
        except OSError as exc:
            last_error = exc
            continue

    target = path or DEFAULT_CONFIG_PATH
    if last_error is not None:
        raise ConfigError(f"failed to create config file at {target}: {last_error}") from last_error
    raise ConfigError("failed to locate or create configuration file")


def load_config(path: Path | None = None) -> Config:
    """Load configuration from TOML, filling default values when needed."""

    config_path = ensure_config_file(path)

    try:
        raw_config = _read_toml(config_path)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc

    defaults = dict(_default_config_values())
    merged: Dict[str, Any] = {**defaults, **raw_config}

    try:
        temperature = float(merged["temperature"])
    except (TypeError, ValueError) as exc:
        raise ConfigError("temperature must be a number") from exc

    return Config(
        api_url=str(merged["api_url"]),
        api_key=str(merged["api_key"]),
        model=str(merged["model"]),
        temperature=temperature,
    )


def _read_toml(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover
        raise ConfigError(f"failed to read config file: {exc}") from exc

    return _parse_toml_text(text)


def _parse_toml_text(text: str) -> Dict[str, Any]:
    if _tomllib is not None:
        try:
            data = _tomllib.loads(text)
        except Exception as exc:  # pragma: no cover - mirrors tomllib errors
            raise ConfigError(f"invalid TOML in config file: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError("config file root must be a table")
        return data

    return _parse_basic_toml(text)


def _parse_basic_toml(text: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigError(f"invalid config line: {raw_line}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ConfigError("config keys cannot be empty")
        data[key] = _parse_scalar(value)
    return data


def _parse_scalar(value: str) -> Any:
    if value.startswith('"') and value.endswith('"'):
        inner = value[1:-1]
        return inner.replace("\\\"", '"').replace("\\n", "\n")

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


@lru_cache(maxsize=1)
def _default_config_text() -> str:
    try:
        return DEFAULT_TEMPLATE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"default configuration template missing at {DEFAULT_TEMPLATE_PATH}: {exc}"
        ) from exc


@lru_cache(maxsize=1)
def _default_config_values() -> Dict[str, Any]:
    return dict(_parse_toml_text(_default_config_text()))
