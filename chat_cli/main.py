"""Entry point for the chat CLI application."""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cnlunar

_summary_logger = logging.getLogger("chat_cli.summary")

from .chat import ChatSession
from .client import APIError, ChatClient
from .config import Config, ConfigError, DEFAULT_CONFIG_PATH, Profile, load_config, load_profile, list_profiles
from .tools import ToolError, execute_tool, get_max_act_calls, handle_act, handle_face2face, handle_separate, init_face_to_face_state, is_face_to_face, list_tool_specs, reset_act_call_count, set_max_act_calls

from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.formatted_text import ANSI

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
USER_COLOR = "\033[35m"  # Magenta
AI_COLOR = "\033[36m"  # Cyan
TOOL_COLOR = "\033[33m"  # Yellow
ACTION_COLOR = "\033[32m"  # Green
SYSTEM_COLOR = "\033[34m"  # Blue
GRAY = "\033[90m"  # Gray
TITLE_COLOR = "\033[36;1m"  # Bold Cyan


def style(text: str, color: str, enable: bool) -> str:
    if not enable:
        return text
    return f"{color}{text}{RESET}"


def _prompt_style(text: str, color: str, enable: bool) -> str:
    """Style text for use in input() prompts with readline-safe escape wrapping."""
    if not enable:
        return text
    return f"\001{color}\002{text}\001{RESET}\002"


EXIT_COMMANDS = {"/quit", "/exit", "/q"}


_WEEKDAYS_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def _now_ts() -> str:
    """Return current timestamp with weekday, lunar date, and festivals/solar terms.

    Format: 'YYYY-MM-DD HH:MM:SS 周X 农历X月X日[ 节气/节日]'
    Example: '2026-02-17 09:30:00 周二 农历正月初一 春节'
    """
    now = datetime.now()
    weekday = _WEEKDAYS_CN[now.weekday()]
    base = now.strftime(f"%Y-%m-%d %H:%M:%S {weekday}")

    lunar = cnlunar.Lunar(now)
    # Clean month name: "正月大" -> "正月"
    month = lunar.lunarMonthCn.rstrip("大小")
    lunar_str = f"农历{month}{lunar.lunarDayCn}"

    # Collect festival/solar term tags (deduplicated)
    tags: List[str] = []
    solar_term = lunar.todaySolarTerms
    if solar_term and solar_term != "无":
        tags.append(solar_term)
    legal = lunar.get_legalHolidays()
    if legal and legal not in tags:
        tags.append(legal)
    other = lunar.get_otherHolidays()
    if other and not legal and other not in tags:
        tags.append(other)

    suffix = f" {' '.join(tags)}" if tags else ""
    return f"{base} {lunar_str}{suffix}"


# Match timestamp tags: old format, weekday-only, or full with lunar date/festivals
_TS_PATTERN = re.compile(r"\s*<\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?[^>]*>\s*$")


def _print_ts(ts: str, use_color: bool) -> None:
    """Print a timestamp line in gray."""
    print(style(f"─────────────────── {ts} ───", GRAY, use_color))


_CMD_KEYS = ("face", "sep", "act", "say", "lmt")
_CMD_PATTERN = re.compile(
    r'\b(' + '|'.join(_CMD_KEYS) + r')\s*:\s*',
    re.IGNORECASE,
)


def _parse_commands(text: str) -> Dict[str, str]:
    """Parse command prefixes from input text in any order.

    Returns dict with command names as keys and their values.
    Non-command text is stored under '_text'.
    """
    result: Dict[str, str] = {}
    matches = list(_CMD_PATTERN.finditer(text))
    if not matches:
        result["_text"] = text.strip()
        return result
    # Text before first command
    before = text[:matches[0].start()].strip()
    if before:
        result["_text"] = before
    for i, m in enumerate(matches):
        key = m.group(1).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        val = text[start:end].strip()
        result[key] = val
    return result


def _strip_ai_timestamp(text: str) -> str:
    """Remove trailing timestamp tag from AI reply if present."""
    return _TS_PATTERN.sub("", text).rstrip("\n")
DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent
MAX_LOG_SIZE = 100000  # 100K characters, append to same file if under this limit

# Global profile state (set in main)
_active_profile: Optional[Profile] = None
_log_dir: Path = DEFAULT_LOG_DIR


def _ai_name() -> str:
    if _active_profile:
        return _active_profile.name
    return "AI"


def _ai_label() -> str:
    return f"{_ai_name()}>"


def _setup_summary_logger(log_dir: Path) -> None:
    """Configure summary logger to write to log_dir/summary.log."""
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / "summary.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _summary_logger.addHandler(handler)
    _summary_logger.setLevel(logging.DEBUG)


_LOG_NAME_PATTERN = re.compile(r"^chat_\d{8}_\d{6}\.json$")


def _list_log_files() -> List[Path]:
    """List log files matching the expected naming pattern, sorted newest first."""
    return sorted(
        (f for f in _log_dir.glob("chat_*.json") if _LOG_NAME_PATTERN.match(f.name)),
        reverse=True,
    )


def _generate_log_path() -> Path:
    """Generate a log file path with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _log_dir / f"chat_{timestamp}.json"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with an OpenAI-compatible API")
    parser.add_argument(
        "--config",
        type=Path,
        help=f"Path to configuration file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Profile name to use (loads system prompt and history from profiles/<name>/)",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Do not reuse earlier conversation in subsequent prompts",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors even when stdout is a TTY",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    global _active_profile, _log_dir

    args = parse_args(argv)

    # Load profile if specified
    profile: Optional[Profile] = None
    system_prompt = ""
    profile_path: Optional[Path] = None
    if args.profile:
        try:
            profile = load_profile(args.profile)
            profile_path = profile.path
        except ConfigError as exc:
            print(f"Profile error: {exc}", file=sys.stderr)
            return 1
        _active_profile = profile
        _log_dir = profile.log_dir
        _log_dir.mkdir(parents=True, exist_ok=True)
        system_prompt = profile.system_prompt

    # Setup summary logger to write to log_dir/summary.log
    _setup_summary_logger(_log_dir)

    # Initialize face-to-face state from file (use profile path if available)
    init_face_to_face_state(profile_path)

    try:
        config_path = args.config if args.config else None
        try:
            config = load_config(config_path)
        except ConfigError as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 1

        session = None
        client = ChatClient(config)
        tool_specs = list_tool_specs()
        use_color = sys.stdout.isatty() and not args.no_color

        # Load previous conversation with summaries
        previous_messages, history_summary = _load_history_with_summaries(config)

        # Create session with system prompt, history summary and face-to-face state
        session = ChatSession(
            initial_messages=previous_messages,
            history_summary=history_summary,
            face_to_face=is_face_to_face(),
            system_prompt=system_prompt,
        )

        # Title
        _print_title(use_color)

        # Replay previous conversation history
        _replay_history(session.messages, use_color)

        # Config info after history, rendered in a box
        config_location = _resolve_config_path(config_path)
        profile_name = profile.name if profile else "default"
        info_entries = [
            ("Profile", profile_name),
            ("Model", config.model),
            ("API", config.api_url),
            ("Config", str(config_location)),
        ]
        if history_summary:
            info_entries.append(("History", f"summary loaded ({len(history_summary)} chars)"))
        if is_face_to_face():
            info_entries.append(("Status", "见面中"))
        info_entries.append(("Act limit", f"lmt: N (当前 {get_max_act_calls()})"))
        info_entries.append(("Exit", "/quit /exit /q"))
        _print_info_box(info_entries, use_color)

        while True:
            try:
                prompt_str = style("You>", USER_COLOR, use_color) + " "
                user_input = pt_prompt(ANSI(prompt_str))
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print("\n(Use /quit to exit)")
                continue

            trimmed = user_input.strip()
            if trimmed.lower() in EXIT_COMMANDS:
                break
            if not trimmed:
                continue

            # Show user timestamp
            _print_ts(_now_ts(), use_color)

            if args.no_history:
                session = ChatSession(
                    initial_messages=None,
                    face_to_face=is_face_to_face(),
                    system_prompt=system_prompt,
                )

            history_checkpoint = len(session.messages)

            # Reset act call count for new user turn
            reset_act_call_count()

            # Parse commands (any order): face: sep: act: say: lmt:
            cmds = _parse_commands(trimmed)
            skip = False

            # lmt: (setting, no message)
            if "lmt" in cmds:
                val = cmds["lmt"]
                try:
                    n = int(val)
                    if n < 1:
                        raise ValueError
                    set_max_act_calls(n)
                    print(style(f"[act 次数限制已设为 {n}]", TOOL_COLOR, use_color))
                except ValueError:
                    print(style("[提示] lmt: 须为正整数", TOOL_COLOR, use_color))
                    skip = True

            if not skip and "face" in cmds:
                system_msg = handle_face2face(cmds["face"])
                session.add_message({"role": "system", "content": system_msg, "ts": _now_ts()})
                _print_system_event(system_msg, use_color)
            elif not skip and "sep" in cmds:
                system_msg = handle_separate(cmds["sep"])
                session.add_message({"role": "system", "content": system_msg, "ts": _now_ts()})
                _print_system_event(system_msg, use_color)

            if not skip and "act" in cmds:
                msg, valid = handle_act(cmds.get("act", ""), cmds.get("say", ""))
                if valid:
                    session.add_message({"role": "user", "content": msg, "ts": _now_ts()})
                else:
                    print(style(msg, TOOL_COLOR, use_color))
                    skip = True

            if skip:
                continue

            has_cmd = any(k in cmds for k in ("face", "sep", "act", "lmt"))
            if not has_cmd:
                # Plain text with no commands
                if cmds.get("_text"):
                    session.add_message({"role": "user", "content": user_input, "ts": _now_ts()})
                else:
                    continue
            elif "_text" in cmds and cmds["_text"] and "act" not in cmds:
                # Commands + leftover text: send text as user message too
                session.add_message({"role": "user", "content": cmds["_text"], "ts": _now_ts()})

            # If only lmt: with no other action, skip LLM call
            if has_cmd and not any(k in cmds for k in ("face", "sep", "act", "_text")):
                continue

            try:
                _handle_assistant_interaction(session, client, tool_specs, use_color)
            except KeyboardInterrupt:
                if use_color:
                    print(RESET, end="", flush=True)
                print("[interrupted]")
                _restore_history(session, history_checkpoint)
                continue
            except APIError as exc:
                if use_color:
                    print(RESET, end="", flush=True)
                print(f"[API error] {exc}")
                _restore_history(session, history_checkpoint)
                continue
            except ToolError as exc:
                if use_color:
                    print(RESET, end="", flush=True)
                print(f"[Tool error] {exc}")
                _restore_history(session, history_checkpoint)
                continue
            except Exception as exc:  # pragma: no cover - defensive
                if use_color:
                    print(RESET, end="", flush=True)
                print(f"[error] {exc}")
                _restore_history(session, history_checkpoint)
                continue

        print("Goodbye!")
        return 0
    finally:
        if session is not None:
            final_path = _get_log_path_for_session(session)
            _write_last_log(session, final_path)


def _handle_assistant_interaction(
    session: ChatSession,
    client: ChatClient,
    tool_specs: List[Dict[str, Any]],
    use_color: bool,
) -> None:
    awaiting_final = True

    while awaiting_final:
        ai_label = style(_ai_label(), AI_COLOR, use_color)
        print(f"{ai_label} ", end="", flush=True)
        if use_color:
            print(AI_COLOR, end="", flush=True)

        display_text, tool_calls, ended_nl, raw_content = _stream_response(session, client, tool_specs, use_color)

        if use_color:
            print(RESET, end="", flush=True)
        if not ended_nl:
            print()

        ts = _now_ts()
        # Clean timestamps from content before storing
        store_content = ""
        if raw_content:
            store_content = _strip_ai_timestamp(raw_content)
            # Also strip timestamp inside JSON text field
            if store_content.lstrip().startswith("{"):
                try:
                    parsed = json.loads(store_content)
                except (json.JSONDecodeError, ValueError):
                    # Fix literal newlines inside JSON string values
                    try:
                        parsed = json.loads(store_content.replace("\n", "\\n"))
                        store_content = json.dumps(parsed, ensure_ascii=False)
                    except (json.JSONDecodeError, ValueError):
                        parsed = None
                if parsed and "text" in parsed:
                    cleaned_text = _strip_ai_timestamp(parsed["text"])
                    if cleaned_text != parsed["text"]:
                        parsed["text"] = cleaned_text
                        store_content = json.dumps(parsed, ensure_ascii=False)
        # Detect invalid responses (not valid JSON with "text" field)
        if not tool_calls and _is_invalid_response(store_content):
            print(style("  [无效响应，重试中...]", TOOL_COLOR, use_color))
            continue

        if tool_calls:
            msg: Dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls, "ts": ts}
            if store_content:
                msg["content"] = store_content
                _print_ts(ts, use_color)
            session.add_message(msg)
            awaiting_final = _process_tool_calls(session, tool_calls, use_color)
        else:
            if store_content:
                session.add_message({"role": "assistant", "content": store_content, "ts": ts})
                _print_ts(ts, use_color)
            awaiting_final = False


def _is_invalid_response(raw_content: str) -> bool:
    """Check if response is invalid (not valid JSON with 'text' field)."""
    stripped = raw_content.strip()
    if not stripped:
        return True
    try:
        parsed = json.loads(stripped)
        return "text" not in parsed
    except (json.JSONDecodeError, ValueError):
        # Try fixing literal newlines
        try:
            parsed = json.loads(stripped.replace("\n", "\\n"))
            return "text" not in parsed
        except (json.JSONDecodeError, ValueError):
            return True


_JSON_TEXT_FIELD_RE = re.compile(r'"text"\s*:\s*"')
_JSON_ESCAPE_MAP = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f"}


def _decode_json_string_partial(raw: str) -> Tuple[str, bool]:
    """Decode a partial JSON string value (without surrounding quotes).

    Returns (decoded_text, complete) where complete is True if the closing
    quote was found.
    """
    decoded: List[str] = []
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == '"':
            return "".join(decoded), True
        if ch == '\\':
            if i + 1 >= len(raw):
                break  # incomplete escape, wait for more data
            next_ch = raw[i + 1]
            decoded.append(_JSON_ESCAPE_MAP.get(next_ch, next_ch))
            i += 2
        else:
            decoded.append(ch)
            i += 1
    return "".join(decoded), False


def _stream_response(
    session: ChatSession,
    client: ChatClient,
    tool_specs: List[Dict[str, Any]],
    use_color: bool = True,
) -> Tuple[str, List[Dict[str, Any]], bool, str]:
    """Stream response and return (text, tool_calls, ended_with_newline, think_text)."""
    collected_tool_calls: List[Dict[str, Any]] = []
    full_content = ""
    text_field_start = -1  # position after opening quote of "text" value
    displayed_len = 0  # decoded chars already printed
    display_buffer = ""  # held-back text for timestamp detection
    last_printed = ""

    for event in client.stream_chat(
        session.messages, tools=tool_specs,
        response_format={"type": "json_object"},
    ):
        event_type = event.get("type")
        if event_type == "content":
            full_content += event.get("text", "")
            # Detect text field start
            if text_field_start < 0:
                m = _JSON_TEXT_FIELD_RE.search(full_content)
                if m:
                    text_field_start = m.end()
            # Stream-decode text field incrementally
            if text_field_start >= 0:
                raw = full_content[text_field_start:]
                decoded, _complete = _decode_json_string_partial(raw)
                new_text = decoded[displayed_len:]
                if new_text:
                    display_buffer += new_text
                    displayed_len = len(decoded)
                    safe, display_buffer = _split_safe_output(display_buffer)
                    if safe:
                        print(safe, end="", flush=True)
                        last_printed = safe
        elif event_type == "tool_call":
            collected_tool_calls.append(event["tool_call"])

    # Parse final result — strip timestamp before JSON parsing
    think_text = ""
    final_text = ""
    json_parsed = False
    cleaned_content = _strip_ai_timestamp(full_content)
    try:
        parsed = json.loads(cleaned_content)
        think_text = parsed.get("think", "")
        final_text = parsed.get("text", "")
        json_parsed = True
    except (json.JSONDecodeError, ValueError):
        # Try fixing literal newlines inside JSON string values
        try:
            parsed = json.loads(cleaned_content.replace("\n", "\\n"))
            think_text = parsed.get("think", "")
            final_text = parsed.get("text", "")
            json_parsed = True
        except (json.JSONDecodeError, ValueError):
            # Fallback: treat as plain text
            final_text = cleaned_content

    final_text = _strip_ai_timestamp(final_text)

    # Flush display_buffer and any remaining undisplayed text
    # Only append undisplayed tail when JSON parsed successfully —
    # otherwise final_text is raw content and displayed_len is from
    # the text-field decoder, causing JSON structure to leak.
    remaining = display_buffer
    if json_parsed and final_text and displayed_len < len(final_text):
        remaining += final_text[displayed_len:]
    if remaining:
        clean = _strip_ai_timestamp(remaining).rstrip("\n")
        if clean:
            print(clean, end="", flush=True)
            last_printed = clean

    ended_with_newline = last_printed.endswith("\n")
    return final_text, collected_tool_calls, ended_with_newline, full_content


def _split_safe_output(buffer: str) -> Tuple[str, str]:
    """Split buffer into safe-to-print prefix and held-back suffix.

    Hold back content after the last newline if it could be a timestamp start.
    """
    # If buffer ends mid-potential-timestamp, hold back from last \n or <
    # Timestamp pattern: \n<YYYY-MM-DD HH:MM> or <YYYY-MM-DD HH:MM>
    last_nl = buffer.rfind("\n")
    if last_nl == -1:
        # No newline - check if the whole buffer could be a timestamp start
        if buffer.lstrip().startswith("<"):
            return "", buffer
        return buffer, ""

    after_nl = buffer[last_nl + 1:]
    # If what's after the last newline looks like it could be a timestamp start
    partial = after_nl.lstrip()
    if not partial:
        # Just a trailing newline, safe to print all
        return buffer, ""
    if partial.startswith("<") or (len(partial) <= 20 and partial[0].isdigit()):
        # Could be start of timestamp, hold back
        return buffer[:last_nl + 1], after_nl
    return buffer, ""


def _print_system_event(content: str, use_color: bool) -> None:
    """Print a system event (face/sep) in a readable format."""
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("[系统事件]"):
            text = line[len("[系统事件]"):].strip()
            print(style(f"  ── {text} ──", SYSTEM_COLOR, use_color))
        elif line.startswith("场景："):
            scene = line[len("场景："):]
            print(style(f"  场景：{scene}", SYSTEM_COLOR, use_color))
        elif line.startswith("[当前状态]"):
            text = line[len("[当前状态]"):].strip()
            print(style(f"  {text}", SYSTEM_COLOR, use_color))
        else:
            print(style(f"  {line}", SYSTEM_COLOR, use_color))



def _print_act_result(result_json: str, use_color: bool) -> None:
    """Print act tool result in name do> / name say> format."""
    try:
        data = json.loads(result_json)
        act_result = data.get("act_result", "")
    except (json.JSONDecodeError, AttributeError):
        act_result = result_json

    for line in act_result.split("\n"):
        if line.startswith("[动作] "):
            action = line[5:].replace("\\n", "\n")
            print(f"{style(f'{_ai_name()} do>', ACTION_COLOR, use_color)} {style(action, ACTION_COLOR, use_color)}")
        elif line.startswith("[说话] "):
            speech = line[5:].replace("\\n", "\n")
            print(f"{style(f'{_ai_name()} say>', AI_COLOR, use_color)} {style(speech, AI_COLOR, use_color)}")


def _print_time_result(result_json: str, use_color: bool) -> None:
    """Print check_the_time result in a clean format."""
    try:
        data = json.loads(result_json)
        time_str = data.get("current_time", "")
        tz = data.get("timezone", "")
        # Parse and format nicely
        if "T" in time_str:
            date_part, time_part = time_str.split("T", 1)
            # Trim microseconds and tz offset for display
            time_short = time_part.split(".")[0].split("+")[0].split("-")[0]
            print(style(f"  ⏰ {date_part} {time_short} ({tz})", DIM + TOOL_COLOR, use_color))
        else:
            print(style(f"  ⏰ {time_str} ({tz})", DIM + TOOL_COLOR, use_color))
    except (json.JSONDecodeError, AttributeError):
        print(style(f"  ⏰ {result_json}", DIM + TOOL_COLOR, use_color))


def _get_tool_name_for_result(tool_msg: Dict[str, Any], messages: List[Dict[str, Any]]) -> str:
    """Get the tool name for a tool result message."""
    tool_call_id = tool_msg.get("tool_call_id", "")
    for msg in messages:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                if tc.get("id") == tool_call_id:
                    return tc.get("function", {}).get("name", "")
    return ""


def _print_act_call(arguments_json: str, use_color: bool, show_thinking: bool = False) -> None:
    """Print act tool call in name do> / name say> format from call arguments."""
    try:
        args = json.loads(arguments_json)
    except (json.JSONDecodeError, AttributeError):
        return
    think = args.get("think", "")
    action = args.get("action", "").replace("\\n", "\n")
    speech = args.get("speech", "").replace("\\n", "\n")
    if show_thinking and think:
        print(f"[thinking] {think}")
    if action:
        print(f"{style(f'{_ai_name()} do>', ACTION_COLOR, use_color)} {style(action, ACTION_COLOR, use_color)}")
    if speech:
        print(f"{style(f'{_ai_name()} say>', AI_COLOR, use_color)} {style(speech, AI_COLOR, use_color)}")


def _process_tool_calls(
    session: ChatSession,
    tool_calls: List[Dict[str, Any]],
    use_color: bool,
) -> bool:
    """Process tool calls and return whether to continue awaiting LLM response.

    Returns True if another LLM round-trip is needed, False if the interaction
    should end immediately (e.g. noop called, or act overflow reached).
    """
    should_continue = True
    for tool_call in tool_calls:
        function_data = tool_call.get("function", {})
        name = function_data.get("name", "")
        arguments = function_data.get("arguments", "")
        tool_id = tool_call.get("id", "")

        if name in ("act", "check_the_time", "noop"):
            pass
        else:
            invocation = f"[tool-call] {name}({arguments})"
            print(style(invocation, TOOL_COLOR, use_color))

        try:
            result = execute_tool(name, arguments)
        except ToolError as exc:
            result = json.dumps({"error": str(exc)})
            session.add_tool_message(tool_id, result)
            if name == "act":
                print(style(f"[act-error] {exc}", TOOL_COLOR, use_color))
            else:
                print(style(f"[tool-error] {exc}", TOOL_COLOR, use_color))
                raise
            continue

        if name == "noop":
            print(f"{style(_ai_label(), DIM + AI_COLOR, use_color)} {style('...', DIM + AI_COLOR, use_color)}")
            should_continue = False
            continue
        elif name == "act":
            parsed = json.loads(result)
            if parsed.get("act_overflow"):
                should_continue = False
                continue
            session.add_tool_message(tool_id, result)
            _print_act_result(result, use_color)
        else:
            session.add_tool_message(tool_id, result)
            if name == "check_the_time":
                _print_time_result(result, use_color)
            else:
                output_line = f"[tool-result] {result}"
                print(style(output_line, TOOL_COLOR, use_color))

    return should_continue


def _restore_history(session: ChatSession, checkpoint: int) -> None:
    while len(session.messages) > checkpoint:
        session.remove_last_message()


def _print_info_box(entries: List[Tuple[str, str]], use_color: bool) -> None:
    """Print config info in a styled box."""
    label_width = max(len(k) for k, _ in entries)
    inner_lines = [f" {k:<{label_width}} : {v} " for k, v in entries]
    content_width = max(len(line) for line in inner_lines)
    top = "  ┌" + "─" * content_width + "┐"
    bot = "  └" + "─" * content_width + "┘"
    print(style(top, TOOL_COLOR, use_color))
    for line in inner_lines:
        print(style(f"  │{line:<{content_width}}│", TOOL_COLOR, use_color))
    print(style(bot, TOOL_COLOR, use_color))
    print()


def _print_title(use_color: bool) -> None:
    """Print a styled title banner on startup."""
    title = (
        "\n"
        "  ╔══════════════════════════════════╗\n"
        "  ║" + "Chat CLI".center(34) + "║\n"
        "  ╚══════════════════════════════════╝"
    )
    print(style(title, TITLE_COLOR, use_color))
    print()


def _replay_history(messages: List[Dict[str, Any]], use_color: bool, show_separator: bool = True, show_thinking: bool = False) -> None:
    """Print previous conversation messages so the user can see chat history."""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        ts = msg.get("ts", "")

        if role == "system":
            # Show face/sep events, skip initial system prompt
            if "[系统事件]" in content:
                _print_system_event(content, use_color)
            continue
        elif role == "user":
            print(f"{style('You>', USER_COLOR, use_color)} {content}")
            if ts:
                _print_ts(ts, use_color)
        elif role == "assistant":
            # Parse JSON content: {"think": "...", "text": "..."}
            display_content = content
            think_content = ""
            if content.lstrip().startswith("{"):
                try:
                    parsed = json.loads(content)
                    if "text" in parsed:
                        display_content = parsed["text"]
                        think_content = parsed.get("think", "")
                except (json.JSONDecodeError, ValueError):
                    # Try fixing literal newlines inside JSON string values
                    try:
                        parsed = json.loads(content.replace("\n", "\\n"))
                        if "text" in parsed:
                            display_content = parsed["text"]
                            think_content = parsed.get("think", "")
                    except (json.JSONDecodeError, ValueError):
                        pass
            # Also check legacy reasoning_content field
            if not think_content:
                think_content = msg.get("reasoning_content", "")
            if show_thinking and think_content:
                print(f"[thinking] {think_content}")
            content = display_content
            if content:
                print(f"{style(_ai_label(), AI_COLOR, use_color)} {style(content, AI_COLOR, use_color)}")
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    arguments = func.get("arguments", "")
                    if name == "act":
                        _print_act_call(arguments, use_color, show_thinking=show_thinking)
                    elif name == "noop":
                        print(f"{style(_ai_label(), DIM + AI_COLOR, use_color)} {style('...', DIM + AI_COLOR, use_color)}")
                    elif name == "check_the_time":
                        pass  # result will be shown in tool message
                    else:
                        print(style(f"[tool-call] {name}({arguments})", TOOL_COLOR, use_color))
            if ts:
                _print_ts(ts, use_color)
        elif role == "tool":
            tool_name = _get_tool_name_for_result(msg, messages)
            if tool_name == "act":
                # Show act errors, skip successful results (already shown via _print_act_call)
                try:
                    data = json.loads(content)
                    if "error" in data:
                        print(style(f"[act-error] {data['error']}", TOOL_COLOR, use_color))
                except (json.JSONDecodeError, AttributeError):
                    pass
                continue
            elif tool_name == "check_the_time":
                _print_time_result(content, use_color)
                continue
            elif tool_name == "noop":
                continue  # already shown via tool_calls
            print(style(f"[tool-result] {content}", TOOL_COLOR, use_color))

    if show_separator and len(messages) > 1:
        print(style("--- 以上是之前的对话 ---\n", TOOL_COLOR, use_color))


def _resolve_config_path(provided_path: Optional[Path]) -> Path:
    if provided_path is not None:
        return provided_path

    from .config import ensure_config_file

    return ensure_config_file(None)


def _write_log_to_path(messages: List[Dict[str, Any]], path: Path) -> None:
    """Write messages to a JSON log file and its human-readable companion."""
    payload = {"messages": messages}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - best-effort logging
        print(f"Failed to write chat log to {path}: {exc}", file=sys.stderr)

    # Write human-readable log alongside the JSON
    readable_path = path.with_suffix(".good4read.log")
    try:
        import io
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        _replay_history(messages, use_color=False, show_separator=False, show_thinking=True)
        sys.stdout = old_stdout
        readable_path.write_text(buf.getvalue(), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - best-effort
        sys.stdout = old_stdout  # ensure restore
        print(f"Failed to write readable log to {readable_path}: {exc}", file=sys.stderr)


def _messages_size(messages: List[Dict[str, Any]]) -> int:
    """Estimate serialized size of messages."""
    return len(json.dumps(messages, ensure_ascii=False))


def _write_last_log(session: ChatSession, path: Path) -> None:
    """Save session to log, splitting into a new file if over MAX_LOG_SIZE."""
    messages = session.messages

    if _messages_size(messages) <= MAX_LOG_SIZE:
        _write_log_to_path(messages, path)
        return

    # Over limit: figure out how many messages the previous file already has
    prev_count = 0
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            prev_count = len(data.get("messages", []))
        except Exception:
            pass

    if prev_count > 0 and prev_count < len(messages):
        # Save the old portion back (unchanged) and new portion to a new file
        _write_log_to_path(messages[:prev_count], path)
        new_path = _generate_log_path()
        _write_log_to_path(messages[prev_count:], new_path)
    else:
        # No previous file or nothing to split — write everything
        _write_log_to_path(messages, path)


def _get_log_path_for_session(session: ChatSession) -> Path:
    """Get the latest log file path, or create a new one if none exists."""
    log_files = _list_log_files()
    if log_files:
        return log_files[0]
    return _generate_log_path()


def _load_last_log() -> List[Dict[str, Any]]:
    """Load previous conversation from the most recent log file."""
    try:
        log_files = _list_log_files()
        if not log_files:
            return []
        latest = log_files[0]
        data = json.loads(latest.read_text(encoding="utf-8"))
        messages = data.get("messages", [])
        if messages and isinstance(messages, list):
            return messages
    except Exception:
        pass
    return []


MAX_SUMMARY_SIZE = 10000  # 10K characters per summary
_SUMMARY_MAX_TOKENS = 16384  # default, can be overridden per-LLM in config
_SCENE_SPLIT_THRESHOLD = 30000  # split into scenes when rendered text exceeds this
_BATCH_TEXT_LIMIT = 20000  # max chars of scene text per LLM call
_SCENE_TIME_GAP_MINUTES = 30  # time gap to start a new scene
_SCENE_MIN_MESSAGES = 5  # merge scenes shorter than this

_SUMMARY_SYSTEM_PROMPT = (
    "你是一个互动记录摘要助手。你的任务是为角色扮演聊天生成摘要，"
    "帮助 AI 角色在下次互动时快速回忆之前发生过的事。"
)

_SUMMARY_PERSPECTIVE_RULES = """\
视角规则（非常重要）：
- 你就是这个角色本人，用"我"称呼自己，用"水镜"称呼用户
- 只记录你**清醒时亲身经历或感知到的事件**
- 如果记录中显示你处于睡着、昏迷、失去意识等状态，那段时间内其他人做的事你不应该知道——除非事后有人明确告诉了你
- 你离开现场后发生的事，你同样不应该知道——除非有人转述
- 对于你无法感知的事件，不要写入摘要"""

_SUMMARY_FORMAT_RULES = """\
格式说明：
- "水镜>" 是用户（水镜）的发言
- "我>" 是你自己的发言
- "我 do>" 是你的动作
- "我 say>" 是你做动作时同时说的话
- "─── 时间 ───" 是时间戳，格式包含公历、星期、农历日期，可能附带节气或节日
- "[系统事件]" 是系统事件（见面/分开等）"""

_SUMMARY_CONTENT_RULES = """\
内容要求：
- 按时间顺序组织，用 **时间段标题**（如"上午 10:35-10:41"）分隔
- 每个时间段内列出关键事件：谁说了什么重要的话、做了什么、发生了什么
- 保留重要的对话内容（承诺、约定、告白、争吵、调情、提议的具体内容）
- 保留情感变化和关系发展
- 记录见面/分开事件及场景
- 记录提到的其他人物及相关信息
- 不要省略任何重要事件"""

_SUMMARY_CACHE_FILE = "history_summary.json"


def _render_messages_to_text(messages: List[Dict[str, Any]], for_summary: bool = False) -> str:
    """Render messages to plain text using _replay_history (good4read format).

    If for_summary is True, replace labels to match the AI's first-person perspective:
    - "You>" becomes "水镜>" (the user)
    - "{ai_name}>" / "{ai_name} do>" / "{ai_name} say>" become "我>" / "我 do>" / "我 say>"
    """
    import io
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        _replay_history(messages, use_color=False, show_separator=False, show_thinking=True)
    finally:
        sys.stdout = old_stdout
    text = buf.getvalue()
    if for_summary:
        ai_name = _ai_name()
        text = text.replace(f"{ai_name} do>", "我 do>")
        text = text.replace(f"{ai_name} say>", "我 say>")
        text = text.replace(f"{ai_name}>", "我>")
        text = text.replace("You>", "水镜>")
    return text


def _split_into_scenes(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Split messages into scenes based on system events and time gaps."""
    if not messages:
        return []

    scenes: List[List[Dict[str, Any]]] = [[]]
    prev_ts: Optional[datetime] = None

    for msg in messages:
        content = msg.get("content", "") or ""
        role = msg.get("role", "")

        # Parse timestamp (supports old/new formats with optional weekday/lunar)
        cur_ts = None
        ts_str = msg.get("ts", "")
        if ts_str:
            # Strip everything after seconds or minutes (weekday, lunar, festivals)
            ts_clean = re.sub(r"(\d{2}:\d{2}(?::\d{2})?).*", r"\1", ts_str)
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                try:
                    cur_ts = datetime.strptime(ts_clean, fmt)
                    break
                except ValueError:
                    continue
            if cur_ts is None:
                pass

        # Check if this is a face-to-face START event
        is_f2f_start = (
            role == "system"
            and "[系统事件]" in content
            and "见面状态" in content
        )

        # Check time gap
        has_time_gap = False
        if cur_ts and prev_ts:
            gap_minutes = (cur_ts - prev_ts).total_seconds() / 60
            has_time_gap = gap_minutes > _SCENE_TIME_GAP_MINUTES

        # Start new scene on face-to-face start or time gap
        if (is_f2f_start or has_time_gap) and scenes[-1]:
            scenes.append([])

        scenes[-1].append(msg)
        if cur_ts:
            prev_ts = cur_ts

    # Merge short scenes into previous
    merged: List[List[Dict[str, Any]]] = []
    for scene in scenes:
        if merged and len(scene) < _SCENE_MIN_MESSAGES:
            merged[-1].extend(scene)
        else:
            merged.append(scene)

    return merged


def _batch_scenes(scenes: List[List[Dict[str, Any]]], limit: int = _BATCH_TEXT_LIMIT) -> List[str]:
    """Batch consecutive scenes into text chunks that fit within the limit.

    Returns a list of rendered text strings, each within the limit.
    If a single scene exceeds the limit, it is truncated to fit.
    """
    batches: List[str] = []
    current_text = ""
    for scene in scenes:
        scene_text = _render_messages_to_text(scene, for_summary=True)
        if not scene_text.strip():
            continue
        if not current_text:
            # First scene in batch — always accept, truncate if needed
            if len(scene_text) > limit:
                current_text = scene_text[:limit]
            else:
                current_text = scene_text
        elif len(current_text) + len(scene_text) <= limit:
            # Fits in current batch
            current_text += "\n" + scene_text
        else:
            # Doesn't fit — flush current batch, start new one
            batches.append(current_text)
            if len(scene_text) > limit:
                current_text = scene_text[:limit]
            else:
                current_text = scene_text
    if current_text.strip():
        batches.append(current_text)
    return batches


def _build_initial_summary_prompt(character_background: str, conversation_text: str) -> str:
    """Build prompt for summarizing a scene into the character's memory."""
    bg_section = ""
    if character_background:
        bg_section = f"""你的角色背景（用于理解你是谁，不需要复述）：
{character_background}

"""
    return f"""请为以下互动记录生成详细的时间线摘要。摘要将作为你的"记忆"注入到下次互动中，必须足够详细以便你能准确回忆发生过的事。

{bg_section}{_SUMMARY_PERSPECTIVE_RULES}

{_SUMMARY_CONTENT_RULES}

{_SUMMARY_FORMAT_RULES}

互动记录：
{conversation_text}"""


def _build_merge_prompt(character_background: str, existing_summary: str, new_conversation_text: str) -> str:
    """Build prompt for merging existing summary with new raw conversation."""
    bg_section = ""
    if character_background:
        bg_section = f"""你的角色背景（用于理解你是谁，不需要复述）：
{character_background}

"""
    return f"""以下是你之前的记忆摘要，以及一段新的互动记录原文。请将新的互动内容整合到已有记忆中，生成更新后的完整摘要。

{bg_section}{_SUMMARY_PERSPECTIVE_RULES}

{_SUMMARY_CONTENT_RULES}

整合要求：
- 将新互动内容按时间顺序接续在已有记忆之后
- 如果新内容与已有记忆有重叠，去重保留最完整的版本
- 不要为了精简而丢失已有记忆中的细节
- 更新后的摘要不超过 10KB

{_SUMMARY_FORMAT_RULES}

已有记忆：
{existing_summary}

新的互动记录：
{new_conversation_text}"""


def _call_summary_llm(config: Config, prompt: str, label: str = "") -> str:
    """Call LLM to generate a summary using primary LLM, falling back if configured.

    Tries each LLM in config.llm_chain up to (max_retries + 1) times.
    Raises RuntimeError if all attempts fail.
    """
    tag = f"[{label}] " if label else ""
    prompt_len = len(prompt)
    llm_chain = config.llm_chain
    max_attempts = config.max_retries + 1
    total_llms = len(llm_chain)
    _summary_logger.info("%sprompt length: %d chars, %d LLM(s), max_attempts=%d", tag, prompt_len, total_llms, max_attempts)

    last_error: Optional[Exception] = None
    for llm_idx, llm_config in enumerate(llm_chain):
        llm_label = "primary" if llm_idx == 0 else "fallback"
        is_network_error = False
        attempts = max_attempts
        attempt = 0
        while attempt < attempts:
            attempt += 1
            try:
                client = ChatClient(llm_config)
                result_text = ""
                finish_reason = "unknown"
                for event in client.stream_chat([
                    {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ], max_tokens=llm_config.max_tokens or _SUMMARY_MAX_TOKENS):
                    if event.get("type") == "content":
                        result_text += event.get("text", "")
                    elif event.get("type") == "finish":
                        finish_reason = event.get("finish_reason", "unknown")
                result = result_text.strip()
                _summary_logger.info(
                    "%s%s attempt %d/%d: finish_reason=%s, output=%d chars, model=%s",
                    tag, llm_label, attempt, attempts, finish_reason, len(result), llm_config.model,
                )
                if finish_reason == "stop" and result:
                    if llm_idx > 0:
                        print(f"  [降级] {tag}使用了 {llm_config.model} (原因: {last_error})")
                    return result
                is_network_error = finish_reason == "network_error"
                if is_network_error and attempts == max_attempts:
                    attempts = max(max_attempts, 2)  # network error at least retry once
                _summary_logger.warning(
                    "%s%s attempt %d/%d failed: finish_reason=%s, output=%d chars",
                    tag, llm_label, attempt, attempts, finish_reason, len(result),
                )
                last_error = RuntimeError(
                    f"{llm_label} LLM ({llm_config.model}) finish_reason={finish_reason}"
                )
            except Exception as exc:
                _summary_logger.error(
                    "%s%s attempt %d/%d exception: %s", tag, llm_label, attempt, attempts, exc
                )
                last_error = exc
                # Treat exceptions (connection errors etc.) as network issues
                if attempts == max_attempts:
                    attempts = max(max_attempts, 2)
        _summary_logger.warning("%s%s exhausted %d attempts", tag, llm_label, attempts)

    _summary_logger.error("%sall LLMs exhausted, raising", tag)
    raise RuntimeError(f"Summary generation failed across all LLMs: {last_error}") from last_error


def _summarize_conversation(
    config: Config, messages: List[Dict[str, Any]], character_bg: Optional[str] = None
) -> str:
    """Summarize a conversation using sequential scene-by-scene merging.

    Flow:
    - Split conversation into scenes
    - Scene 1 raw text → LLM → initial summary
    - summary + scene 2 raw text → LLM merge → updated summary
    - summary + scene 3 raw text → LLM merge → updated summary
    - ...and so on

    Args:
        character_bg: Character identity/background text. If None, uses _active_profile.system_prompt.
    """
    if character_bg is None and _active_profile:
        character_bg = _active_profile.system_prompt
    character_bg = character_bg or ""

    # Filter out initial system prompt, keep system events
    dialog_messages = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "system" and i == 0:
            continue
        dialog_messages.append(msg)

    conversation_text = _render_messages_to_text(dialog_messages, for_summary=True)
    if not conversation_text.strip():
        return ""

    _summary_logger.info(
        "conversation rendered: %d chars, %d messages", len(conversation_text), len(dialog_messages)
    )

    # Short conversation: summarize in one shot
    if len(conversation_text) <= _SCENE_SPLIT_THRESHOLD:
        _summary_logger.info("short conversation, summarizing in one shot")
        prompt = _build_initial_summary_prompt(character_bg, conversation_text)
        return _call_summary_llm(config, prompt, label="one-shot")

    # Long conversation: split into scenes, batch, and merge
    scenes = _split_into_scenes(dialog_messages)
    batches = _batch_scenes(scenes)
    _summary_logger.info("split into %d scenes, %d batches", len(scenes), len(batches))

    if len(batches) <= 1:
        _summary_logger.info("single batch, summarizing in one shot")
        prompt = _build_initial_summary_prompt(character_bg, batches[0] if batches else conversation_text)
        return _call_summary_llm(config, prompt, label="one-shot-fallback")

    current_summary = ""
    for idx, batch_text in enumerate(batches, 1):
        _summary_logger.info("batch %d/%d: %d chars", idx, len(batches), len(batch_text))

        if not current_summary:
            prompt = _build_initial_summary_prompt(character_bg, batch_text)
            current_summary = _call_summary_llm(config, prompt, label=f"batch-{idx}/{len(batches)}-init")
        else:
            prompt = _build_merge_prompt(character_bg, current_summary, batch_text)
            current_summary = _call_summary_llm(config, prompt, label=f"batch-{idx}/{len(batches)}-merge")

        _summary_logger.info("summary after batch %d: %d chars", idx, len(current_summary))

    return current_summary


def _merge_summary_with_new(
    config: Config, existing_summary: str, messages: List[Dict[str, Any]],
    character_bg: Optional[str] = None,
) -> str:
    """Merge existing summary with new raw messages via scene-by-scene merging.

    Renders messages to text, splits into scenes, then merges each scene
    into the existing summary sequentially.
    """
    if character_bg is None and _active_profile:
        character_bg = _active_profile.system_prompt
    character_bg = character_bg or ""

    conversation_text = _render_messages_to_text(messages, for_summary=True)
    if not conversation_text.strip():
        return existing_summary

    _summary_logger.info(
        "merge_with_new: existing=%d chars, new messages=%d, new text=%d chars",
        len(existing_summary), len(messages), len(conversation_text),
    )

    # Split into scenes, batch, and merge
    scenes = _split_into_scenes(messages)
    batches = _batch_scenes(scenes)
    _summary_logger.info("merge_with_new: split into %d scenes, %d batches", len(scenes), len(batches))

    current = existing_summary
    for idx, batch_text in enumerate(batches, 1):
        _summary_logger.info("merge batch %d/%d: %d chars", idx, len(batches), len(batch_text))
        prompt = _build_merge_prompt(character_bg, current, batch_text)
        current = _call_summary_llm(config, prompt, label=f"merge-batch-{idx}/{len(batches)}")
        _summary_logger.info("summary after merge batch %d: %d chars", idx, len(current))

    return current


def _load_summary_cache() -> Dict[str, Any]:
    """Load summary cache from history_summary.json."""
    cache_path = _log_dir / _SUMMARY_CACHE_FILE
    try:
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_summary_cache(
    summarized_files: List[str],
    summary: str,
    partial_files: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Save summary cache to history_summary.json."""
    cache_path = _log_dir / _SUMMARY_CACHE_FILE
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "summarized_files": summarized_files,
            "summary": summary,
        }
        if partial_files:
            payload["partial_files"] = partial_files
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _load_file_messages(log_file: Path) -> List[Dict[str, Any]]:
    """Load messages from a log file."""
    try:
        data = json.loads(log_file.read_text(encoding="utf-8"))
        return data.get("messages", [])
    except Exception:
        return []


def _strip_system_head(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove the leading system message (character background) from messages."""
    if messages and messages[0].get("role") == "system":
        return messages[1:]
    return messages


def _load_history_with_summaries(config: Config) -> Tuple[List[Dict[str, Any]], str]:
    """Load latest conversation and summarize older ones with caching.

    Supports partial summarization: the second-newest file may be only partially
    summarized, with its remaining messages prepended to the conversation context.
    """
    log_files = _list_log_files()
    if not log_files:
        return [], ""

    # --- Step 1: File classification ---
    latest_messages = _load_file_messages(log_files[0])

    if _messages_size(latest_messages) > MAX_LOG_SIZE:
        older_files = log_files  # all files become "old", including latest
        latest_messages = []
    else:
        older_files = log_files[1:]
    if not older_files:
        return latest_messages, ""

    # --- Step 2: Read cache ---
    cache = _load_summary_cache()
    cached_summarized = cache.get("summarized_files", [])
    cached_partial: List[Dict[str, Any]] = cache.get("partial_files", [])
    cached_summary = cache.get("summary", "")

    # Build lookup for partial files: filename -> summarized_count
    partial_lookup: Dict[str, int] = {p["file"]: p["summarized_count"] for p in cached_partial}

    # Check if cache is fully up to date (all older files are either summarized or partial)
    cached_all = set(cached_summarized) | set(partial_lookup.keys())
    older_file_names = [f.name for f in older_files]
    if cached_all == set(older_file_names) and cached_summary and not partial_lookup:
        # Fully cached, no partial files to process
        return latest_messages, cached_summary

    # --- Step 3: Collect unsummarized messages ---
    unsummarized_messages: List[Dict[str, Any]] = []
    # Track which messages belong to which file for bookkeeping
    file_message_ranges: List[Tuple[str, int]] = []  # (filename, message_count_in_unsummarized)

    for f in reversed(older_files):  # oldest first for chronological order
        if f.name in cached_summarized:
            continue
        all_msgs = _load_file_messages(f)
        if not all_msgs:
            continue
        stripped = _strip_system_head(all_msgs)
        if f.name in partial_lookup:
            # Skip already-summarized messages (count includes system msg at index 0)
            skip = partial_lookup[f.name] - 1  # -1 because we already stripped system
            if skip > 0:
                stripped = stripped[skip:]
        if stripped:
            unsummarized_messages.extend(stripped)
            file_message_ranges.append((f.name, len(stripped)))

    if not unsummarized_messages and cached_summary:
        # Nothing new to process
        return latest_messages, cached_summary

    # --- Step 4: Trim second-newest file ---
    # second_newest = older_files[0] (newest among older files)
    second_newest_name = older_files[0].name
    need_trim = (
        not latest_messages
        or _messages_size(latest_messages) < MAX_LOG_SIZE // 2
    )

    new_partial_files: List[Dict[str, Any]] = []
    new_summarized_files = list(cached_summarized)

    if need_trim and file_message_ranges:
        # Find the second-newest file's messages in unsummarized list
        last_file, last_count = file_message_ranges[-1]
        if last_file == second_newest_name and last_count > 1:
            half = last_count // 2
            keep_for_summary = last_count - half  # front half stays for summarization
            context_msgs = unsummarized_messages[-half:]  # back half → context
            unsummarized_messages = unsummarized_messages[:-half]
            file_message_ranges[-1] = (last_file, keep_for_summary)

            # Prepend context messages to latest_messages
            latest_messages = context_msgs + latest_messages

            # Calculate total summarized count for this file
            prev_count = partial_lookup.get(second_newest_name, 0)
            if prev_count == 0:
                # First time: system msg (1) + front half
                new_count = 1 + keep_for_summary
            else:
                new_count = prev_count + keep_for_summary
            new_partial_files.append({"file": second_newest_name, "summarized_count": new_count})
            _summary_logger.info(
                "trim %s: keep %d for summary, %d to context, total_summarized=%d",
                second_newest_name, keep_for_summary, half, new_count,
            )
        else:
            # No trimming needed (single message or not the second-newest)
            if second_newest_name not in new_summarized_files:
                new_summarized_files.append(second_newest_name)
    else:
        # No trim: second-newest fully participates
        if second_newest_name not in new_summarized_files and file_message_ranges:
            last_file, _ = file_message_ranges[-1]
            if last_file == second_newest_name:
                new_summarized_files.append(second_newest_name)

    # Mark all other processed files as summarized
    for fname, _ in file_message_ranges:
        if fname != second_newest_name and fname not in new_summarized_files:
            new_summarized_files.append(fname)

    # --- Step 5: Summarize ---
    if not unsummarized_messages:
        combined_summary = cached_summary
    elif not cached_summary:
        _summary_logger.info(
            "initial summarization: %d unsummarized messages", len(unsummarized_messages)
        )
        combined_summary = _summarize_conversation(config, unsummarized_messages)
    else:
        _summary_logger.info(
            "incremental merge: %d unsummarized messages into %d chars summary",
            len(unsummarized_messages), len(cached_summary),
        )
        combined_summary = _merge_summary_with_new(config, cached_summary, unsummarized_messages)

    if len(combined_summary) > MAX_SUMMARY_SIZE:
        _summary_logger.warning(
            "final summary truncated: %d -> %d chars", len(combined_summary), MAX_SUMMARY_SIZE
        )
        combined_summary = combined_summary[:MAX_SUMMARY_SIZE]

    # --- Step 6: Save cache ---
    _summary_logger.info("final summary: %d chars", len(combined_summary))
    if combined_summary:
        _save_summary_cache(new_summarized_files, combined_summary, new_partial_files)

    return latest_messages, combined_summary
