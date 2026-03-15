"""Configuration management for the chat CLI."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # Python 3.11+
    import tomllib as _tomllib
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as _tomllib
    except ModuleNotFoundError:
        _tomllib = None


DEFAULT_CONFIG_PATH = Path.home() / ".config" / "chat-cli" / "config.toml"
LOCAL_FALLBACK_CONFIG_PATH = Path.cwd() / ".chat-cli" / "config.toml"
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE_PATH = PACKAGE_ROOT / "configs" / "default_config.toml"


class ConfigError(Exception):
    """Raised when the configuration file is missing or malformed."""


@dataclass
class LLMConfig:
    """Configuration for a single LLM endpoint."""

    api_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: Optional[int] = None


@dataclass
class Config:
    """Strongly typed configuration values for the chat client."""

    api_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: Optional[int] = None
    max_retries: int = 0
    fallback_api_url: Optional[str] = None
    fallback_api_key: Optional[str] = None
    fallback_model: Optional[str] = None
    fallback_temperature: Optional[float] = None
    fallback_max_tokens: Optional[int] = None

    @property
    def primary_llm(self) -> LLMConfig:
        return LLMConfig(self.api_url, self.api_key, self.model, self.temperature, self.max_tokens)

    @property
    def fallback_llm(self) -> Optional[LLMConfig]:
        if not self.fallback_api_url or not self.fallback_api_key:
            return None
        return LLMConfig(
            api_url=self.fallback_api_url,
            api_key=self.fallback_api_key,
            model=self.fallback_model or self.model,
            temperature=self.fallback_temperature if self.fallback_temperature is not None else self.temperature,
            max_tokens=self.fallback_max_tokens,
        )

    @property
    def llm_chain(self) -> List[LLMConfig]:
        """Return [primary] or [primary, fallback] depending on configuration."""
        chain = [self.primary_llm]
        if self.fallback_llm:
            chain.append(self.fallback_llm)
        return chain


# Profile directories
PROFILES_DIR = PACKAGE_ROOT / "profiles"


@dataclass
class Profile:
    """Profile configuration containing persona and history."""

    name: str
    path: Path
    system_prompt: str

    @property
    def log_dir(self) -> Path:
        """Return the log directory for this profile."""
        return self.path / "logs"


def find_profile(name: str) -> Path | None:
    """Find a profile directory by name."""
    profile_path = PROFILES_DIR / name
    if profile_path.is_dir():
        return profile_path
    return None


def list_profiles() -> list[str]:
    """List available profile names."""
    if not PROFILES_DIR.exists():
        return []
    return [p.name for p in PROFILES_DIR.iterdir() if p.is_dir()]


def load_profile(name: str) -> Profile:
    """Load a profile by name."""
    profile_path = find_profile(name)
    if not profile_path:
        raise ConfigError(f"profile '{name}' not found in {PROFILES_DIR}")

    config_file = profile_path / "config.toml"
    if not config_file.exists():
        raise ConfigError(f"profile config not found: {config_file}")

    try:
        raw_config = _read_toml(config_file)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc

    system_prompt = str(raw_config.get("system_prompt", ""))

    return Profile(
        name=name,
        path=profile_path,
        system_prompt=system_prompt,
    )


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

    max_tokens: Optional[int] = None
    if "max_tokens" in merged:
        try:
            max_tokens = int(merged["max_tokens"])
        except (TypeError, ValueError) as exc:
            raise ConfigError("max_tokens must be an integer") from exc

    max_retries = int(merged.get("max_retries", 0))

    fallback_api_url = merged.get("fallback_api_url")
    fallback_api_key = merged.get("fallback_api_key")
    fallback_model = merged.get("fallback_model")
    fallback_temperature: Optional[float] = None
    if "fallback_temperature" in merged:
        try:
            fallback_temperature = float(merged["fallback_temperature"])
        except (TypeError, ValueError) as exc:
            raise ConfigError("fallback_temperature must be a number") from exc
    fallback_max_tokens: Optional[int] = None
    if "fallback_max_tokens" in merged:
        try:
            fallback_max_tokens = int(merged["fallback_max_tokens"])
        except (TypeError, ValueError) as exc:
            raise ConfigError("fallback_max_tokens must be an integer") from exc

    return Config(
        api_url=str(merged["api_url"]),
        api_key=str(merged["api_key"]),
        model=str(merged["model"]),
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        fallback_api_url=str(fallback_api_url) if fallback_api_url else None,
        fallback_api_key=str(fallback_api_key) if fallback_api_key else None,
        fallback_model=str(fallback_model) if fallback_model else None,
        fallback_temperature=fallback_temperature,
        fallback_max_tokens=fallback_max_tokens,
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
