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
    name="check_the_time",
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

_ACT_TOOL = Tool(
    name="act",
    description="描述你在见面时的动作。只能在见面状态下使用。如果同时要说话，请用 speech 参数。",
    parameters={
        "type": "object",
        "properties": {
            "think": {
                "type": "string",
                "description": "可选，你此刻的内心想法",
            },
            "action": {
                "type": "string",
                "description": "你要做的动作，比如 '轻轻握住他的手'",
            },
            "speech": {
                "type": "string",
                "description": "可选，你同时说的话，比如 '想我了吗？'",
            }
        },
        "required": ["action"],
        "additionalProperties": False,
    },
)


_NOOP_TOOL = Tool(
    name="noop",
    description="表示你没有任何反应——不说话、不动作、不回应。只有当你确实不想做任何事情时才调用。",
    parameters={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
)

AVAILABLE_TOOLS: List[Tool] = [_CURRENT_TIME_TOOL, _ACT_TOOL, _NOOP_TOOL]

_YEAR_OVERRIDE_FILE = Path("check_the_time.year")

_is_face_to_face = False
_act_call_count = 0
MAX_ACT_CALLS_PER_TURN = 5

# Face-to-face state file path (set by init_face_to_face_state)
_face_to_face_file: Optional[Path] = None


def _save_face_to_face_state() -> None:
    """Persist face-to-face state to file."""
    if _face_to_face_file is None:
        return
    try:
        _face_to_face_file.parent.mkdir(parents=True, exist_ok=True)
        _face_to_face_file.write_text("1" if _is_face_to_face else "0", encoding="utf-8")
    except Exception:
        pass


def _load_face_to_face_state() -> bool:
    """Load face-to-face state from file."""
    if _face_to_face_file is None:
        return False
    try:
        if _face_to_face_file.exists():
            return _face_to_face_file.read_text(encoding="utf-8").strip() == "1"
    except Exception:
        pass
    return False


def init_face_to_face_state(profile_path: Optional[Path] = None) -> None:
    """Initialize face-to-face state from file on startup.

    Args:
        profile_path: Path to profile directory. If None, uses default location.
    """
    global _is_face_to_face, _face_to_face_file

    if profile_path:
        _face_to_face_file = profile_path / ".face_to_face"
    else:
        _face_to_face_file = Path(__file__).resolve().parent.parent / ".face_to_face"

    _is_face_to_face = _load_face_to_face_state()


def reset_act_call_count() -> None:
    """Reset act call count when user sends a new message."""
    global _act_call_count
    _act_call_count = 0


def set_max_act_calls(n: int) -> None:
    """Set the maximum number of act calls per turn."""
    global MAX_ACT_CALLS_PER_TURN
    MAX_ACT_CALLS_PER_TURN = n


def get_max_act_calls() -> int:
    """Return the current max act calls per turn."""
    return MAX_ACT_CALLS_PER_TURN


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

    if name == _ACT_TOOL.name:
        arguments = _parse_arguments(arguments_json)
        return _execute_act(arguments)

    if name == _NOOP_TOOL.name:
        return json.dumps({"noop": True})

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


def _execute_act(arguments: Dict[str, Any]) -> str:
    """Execute act tool for agent."""
    global _act_call_count

    if not _is_face_to_face:
        raise ToolError("现在不在一起，无法使用 act 工具")

    if _act_call_count >= MAX_ACT_CALLS_PER_TURN:
        return json.dumps({"act_overflow": True})

    think = arguments.get("think")
    action = arguments.get("action")
    if not action or not isinstance(action, str):
        raise ToolError("'action' is required and must be a string")
    action = action.replace("\\n", "\n")

    speech = arguments.get("speech")
    if speech is not None and not isinstance(speech, str):
        raise ToolError("'speech' must be a string if provided")
    if speech:
        speech = speech.replace("\\n", "\n")

    _act_call_count += 1

    if speech:
        result = f"[动作] {action}\n[说话] {speech}"
    else:
        result = f"[动作] {action}"

    ret: Dict[str, Any] = {"act_result": result}
    if think:
        ret["think"] = think
    return json.dumps(ret, ensure_ascii=False)


def handle_face2face(scene: str) -> str:
    """Generate system message for face-to-face meeting."""
    global _is_face_to_face
    _is_face_to_face = True
    _save_face_to_face_state()
    msg = "[系统事件] 你们现在处于见面状态。你可以使用 act 工具来描述动作。"
    if scene:
        msg += f"\n场景：{scene}"
    return msg


def handle_separate(scene: str) -> str:
    """Generate system message for separation."""
    global _is_face_to_face
    _is_face_to_face = False
    _save_face_to_face_state()
    msg = "[系统事件] 你们现在分开了，不再处于见面状态。act 工具已禁用。"
    if scene:
        msg += f"\n场景：{scene}"
    return msg


def is_face_to_face() -> bool:
    """Check if currently in face-to-face mode."""
    return _is_face_to_face


def handle_act(action: str, speech: str = "") -> tuple[str, bool]:
    """Generate message for action. Returns (message, valid)."""
    if not _is_face_to_face:
        return "[系统提示] 现在你们不在一起，无法使用 act 命令。", False

    if speech:
        return f"[动作] {action}\n[说话] {speech}", True
    return f"[动作] {action}", True
