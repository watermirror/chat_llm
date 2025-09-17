"""Tool definitions and execution helpers for the chat CLI."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # Python 3.9+
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]

    def to_openai(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolError(RuntimeError):
    """Raised when a tool cannot execute successfully."""


_CURRENT_TIME_TOOL = Tool(
    name="get_current_time",
    description="Return the current time in ISO 8601 format. Optionally accept an IANA timezone.",
    parameters={
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "Optional timezone name (e.g. 'UTC' or 'America/New_York')",
            }
        },
        "additionalProperties": False,
    },
)

_FAREWELL_TOOL = Tool(
    name="register_farewell",
    description="Mark that the user has initiated a farewell so the session can end gracefully.",
    parameters={
        "type": "object",
        "properties": {
            "note": {
                "type": "string",
                "description": "Optional note about the farewell context",
            }
        },
        "additionalProperties": False,
    },
)


AVAILABLE_TOOLS: List[Tool] = [_CURRENT_TIME_TOOL, _FAREWELL_TOOL]

_YEAR_OVERRIDE_FILE = Path("get_current_time.year")

_farewell_requested = False


def list_tool_specs() -> List[Dict[str, Any]]:
    """Return OpenAI-compatible tool descriptions."""

    return [tool.to_openai() for tool in AVAILABLE_TOOLS]


def execute_tool(name: str, arguments_json: str) -> str:
    """Execute the named tool and return a string response."""

    if name == _CURRENT_TIME_TOOL.name:
        arguments = _parse_arguments(arguments_json)
        timezone_name = arguments.get("timezone")
        current_time, tz_name = _current_time(timezone_name)
        payload = {"current_time": current_time, "timezone": tz_name}
        return json.dumps(payload)

    if name == _FAREWELL_TOOL.name:
        arguments = _parse_arguments(arguments_json)
        return _register_farewell(arguments)

    raise ToolError(f"Unknown tool: {name}")


def _parse_arguments(arguments_json: str) -> Dict[str, Any]:
    if not arguments_json.strip():
        return {}
    try:
        parsed = json.loads(arguments_json)
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise ToolError(f"Invalid tool arguments: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ToolError("Tool arguments must be an object")
    return parsed


def _current_time(timezone_name: Optional[str]) -> tuple[str, str]:
    tz = _resolve_timezone(timezone_name)
    now = datetime.now(tz)
    iso_time = now.isoformat()
    override_year = _load_year_override()
    if override_year is not None:
        iso_time = _apply_year_override(iso_time, override_year)

    tz_label = tz.tzname(now)
    if not tz_label and hasattr(tz, "key"):
        tz_label = getattr(tz, "key")
    if not tz_label:
        tz_label = tz.__class__.__name__
    return iso_time, tz_label


def _resolve_timezone(timezone_name: Optional[str]):
    if not timezone_name:
        return datetime.now().astimezone().tzinfo or timezone.utc

    if timezone_name.upper() == "UTC":
        return timezone.utc

    if ZoneInfo is not None:
        try:
            return ZoneInfo(timezone_name)
        except Exception as exc:  # pragma: no cover
            raise ToolError(f"Unknown timezone: {timezone_name}") from exc

    raise ToolError("Timezone support is unavailable on this system")


def _load_year_override() -> Optional[str]:
    try:
        raw = _YEAR_OVERRIDE_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ToolError(f"Failed to read {_YEAR_OVERRIDE_FILE}: {exc}") from exc

    for line in raw.splitlines():
        candidate = line.strip()
        if candidate:
            break
    else:
        return None

    if len(candidate) != 4 or not candidate.isdigit():
        return None
    return candidate


def _apply_year_override(iso_time: str, year: str) -> str:
    if len(iso_time) >= 4 and iso_time[:4].isdigit():
        return f"{year}{iso_time[4:]}"
    return iso_time


def _register_farewell(arguments: Dict[str, Any]) -> str:
    global _farewell_requested
    note = arguments.get("note")
    if note is not None and not isinstance(note, str):
        raise ToolError("'note' must be a string if provided")
    _farewell_requested = True
    payload: Dict[str, Any] = {"farewell_registered": True}
    if note:
        payload["note"] = note
    return json.dumps(payload)


def consume_farewell_request() -> bool:
    global _farewell_requested
    if _farewell_requested:
        _farewell_requested = False
        return True
    return False
