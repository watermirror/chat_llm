"""Entry point for the chat CLI application."""
from __future__ import annotations

import argparse
import json
import re
import sys

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .chat import ChatSession
from .client import APIError, ChatClient
from .config import ConfigError, DEFAULT_CONFIG_PATH, Profile, load_config, load_profile, list_profiles
from .tools import ToolError, consume_farewell_request, execute_tool, handle_act, handle_face2face, handle_separate, init_face_to_face_state, is_face_to_face, list_tool_specs, reset_act_call_count

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


def _now_ts() -> str:
    """Return current timestamp in YYYY-MM-DD HH:MM format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M")


_TS_PATTERN = re.compile(r"\s*<\d{4}-\d{2}-\d{2} \d{2}:\d{2}>\s*$")


def _print_ts(ts: str, use_color: bool) -> None:
    """Print a timestamp line in gray."""
    print(style(f"─────────────────── {ts} ───", GRAY, use_color))


def _strip_ai_timestamp(text: str) -> str:
    """Remove trailing timestamp tag from AI reply if present."""
    return _TS_PATTERN.sub("", text).rstrip("\n")
DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent
MAX_LOG_SIZE = 150000  # 150K characters, append to same file if under this limit

# Global profile state (set in main)
_active_profile: Optional[Profile] = None
_log_dir: Path = DEFAULT_LOG_DIR


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

    # Initialize face-to-face state from file (use profile path if available)
    init_face_to_face_state(profile_path)

    try:
        config_path = args.config if args.config else None
        try:
            config = load_config(config_path)
        except ConfigError as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 1

        client = ChatClient(config)
        tool_specs = list_tool_specs()
        use_color = sys.stdout.isatty() and not args.no_color

        # Load previous conversation with summaries
        previous_messages, history_summary = _load_history_with_summaries(client)

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

            # Check for special commands
            if trimmed.lower().startswith("face:"):
                scene = trimmed[5:].strip()
                system_msg = handle_face2face(scene)
                session.add_message({"role": "system", "content": system_msg, "ts": _now_ts()})
                _print_system_event(system_msg, use_color)
            elif trimmed.lower().startswith("sep:"):
                scene = trimmed[4:].strip()
                system_msg = handle_separate(scene)
                session.add_message({"role": "system", "content": system_msg, "ts": _now_ts()})
                _print_system_event(system_msg, use_color)
            elif trimmed.lower().startswith("act:"):
                # Parse act and optional say
                rest = trimmed[4:].strip()
                speech = ""
                if " say:" in rest.lower():
                    idx = rest.lower().find(" say:")
                    action = rest[:idx].strip()
                    speech = rest[idx + 5:].strip()
                else:
                    action = rest

                msg, valid = handle_act(action, speech)
                if valid:
                    session.add_message({"role": "user", "content": msg, "ts": _now_ts()})
                else:
                    print(style(msg, TOOL_COLOR, use_color))
                    continue
            elif "act:" in trimmed.lower():
                # act: not at the start of message
                print(style("[提示] act: 必须在消息开头使用", TOOL_COLOR, use_color))
                continue
            else:
                session.add_message({"role": "user", "content": user_input, "ts": _now_ts()})

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

            if consume_farewell_request():
                break

        print("Goodbye!")
        return 0
    finally:
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
        ai_label = style("AI>", AI_COLOR, use_color)
        print(f"{ai_label} ", end="", flush=True)
        if use_color:
            print(AI_COLOR, end="", flush=True)

        assistant_reply, tool_calls, ended_nl = _stream_response(session, client, tool_specs, use_color)

        if use_color:
            print(RESET, end="", flush=True)
        if not ended_nl:
            print()

        ts = _now_ts()
        clean_reply = _strip_ai_timestamp(assistant_reply)
        if tool_calls:
            msg: Dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls, "ts": ts}
            if clean_reply:
                msg["content"] = clean_reply
            session.add_message(msg)
            _process_tool_calls(session, tool_calls, use_color)
        else:
            if clean_reply:
                session.add_message({"role": "assistant", "content": clean_reply, "ts": ts})
                _print_ts(ts, use_color)
            awaiting_final = False


def _stream_response(
    session: ChatSession,
    client: ChatClient,
    tool_specs: List[Dict[str, Any]],
    use_color: bool = True,
) -> Tuple[str, List[Dict[str, Any]], bool]:
    """Stream response and return (full_text, tool_calls, ended_with_newline)."""
    assistant_chunks: List[str] = []
    collected_tool_calls: List[Dict[str, Any]] = []
    # Buffer to detect and suppress trailing timestamp
    buffer = ""
    last_printed = ""

    for event in client.stream_chat(session.messages, tools=tool_specs):
        event_type = event.get("type")
        if event_type == "content":
            text = event.get("text", "")
            assistant_chunks.append(text)
            buffer += text
            safe, held = _split_safe_output(buffer)
            if safe:
                print(safe, end="", flush=True)
                last_printed = safe
                buffer = held
        elif event_type == "tool_call":
            collected_tool_calls.append(event["tool_call"])

    # Flush remaining buffer, stripping any timestamp and trailing newlines
    if buffer:
        clean = _strip_ai_timestamp(buffer).rstrip("\n")
        if clean:
            print(clean, end="", flush=True)
            last_printed = clean

    ended_with_newline = last_printed.endswith("\n")
    return "".join(assistant_chunks), collected_tool_calls, ended_with_newline


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
    """Print act tool result in AI Action> / AI Speech> format."""
    try:
        data = json.loads(result_json)
        act_result = data.get("act_result", "")
    except (json.JSONDecodeError, AttributeError):
        act_result = result_json

    for line in act_result.split("\n"):
        if line.startswith("[动作] "):
            action = line[5:]
            print(f"{style('AI Action>', ACTION_COLOR, use_color)} {style(action, ACTION_COLOR, use_color)}")
        elif line.startswith("[说话] "):
            speech = line[5:]
            print(f"{style('AI Speech>', AI_COLOR, use_color)} {style(speech, AI_COLOR, use_color)}")


def _print_time_result(result_json: str, use_color: bool) -> None:
    """Print get_current_time result in a clean format."""
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


def _print_farewell_result(result_json: str, use_color: bool) -> None:
    """Print register_farewell result in a clean format."""
    try:
        data = json.loads(result_json)
        registered = data.get("farewell_registered", False)
        if registered:
            print(style("  👋 告别", DIM + TOOL_COLOR, use_color))
        else:
            reason = data.get("reason", "")
            print(style(f"  👋 告别被拒绝：{reason}", DIM + TOOL_COLOR, use_color))
    except (json.JSONDecodeError, AttributeError):
        print(style(f"  👋 {result_json}", DIM + TOOL_COLOR, use_color))


def _print_act_call(arguments_json: str, use_color: bool) -> None:
    """Print act tool call in AI Action> / AI Speech> format from call arguments."""
    try:
        args = json.loads(arguments_json)
    except (json.JSONDecodeError, AttributeError):
        return
    action = args.get("action", "")
    speech = args.get("speech", "")
    if action:
        print(f"{style('AI Action>', ACTION_COLOR, use_color)} {style(action, ACTION_COLOR, use_color)}")
    if speech:
        print(f"{style('AI Speech>', AI_COLOR, use_color)} {style(speech, AI_COLOR, use_color)}")


def _process_tool_calls(
    session: ChatSession,
    tool_calls: List[Dict[str, Any]],
    use_color: bool,
) -> None:
    for tool_call in tool_calls:
        function_data = tool_call.get("function", {})
        name = function_data.get("name", "")
        arguments = function_data.get("arguments", "")
        tool_id = tool_call.get("id", "")

        if name in ("act", "get_current_time", "register_farewell"):
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

        session.add_tool_message(tool_id, result)

        if name == "act":
            _print_act_result(result, use_color)
        elif name == "get_current_time":
            _print_time_result(result, use_color)
        elif name == "register_farewell":
            _print_farewell_result(result, use_color)
        else:
            output_line = f"[tool-result] {result}"
            print(style(output_line, TOOL_COLOR, use_color))


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


def _replay_history(messages: List[Dict[str, Any]], use_color: bool, show_separator: bool = True) -> None:
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
            if content:
                print(f"{style('AI>', AI_COLOR, use_color)} {style(content, AI_COLOR, use_color)}")
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    arguments = func.get("arguments", "")
                    if name == "act":
                        _print_act_call(arguments, use_color)
                    elif name in ("get_current_time", "register_farewell"):
                        pass  # result will be shown in tool message
                    else:
                        print(style(f"[tool-call] {name}({arguments})", TOOL_COLOR, use_color))
            elif ts:
                # Show timestamp for final assistant message (no tool_calls)
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
            elif tool_name == "get_current_time":
                _print_time_result(content, use_color)
                continue
            elif tool_name == "register_farewell":
                _print_farewell_result(content, use_color)
                continue
            print(style(f"[tool-result] {content}", TOOL_COLOR, use_color))

    if show_separator and len(messages) > 1:
        print(style("--- 以上是之前的对话 ---\n", TOOL_COLOR, use_color))


def _resolve_config_path(provided_path: Optional[Path]) -> Path:
    if provided_path is not None:
        return provided_path

    from .config import ensure_config_file

    return ensure_config_file(None)


def _write_last_log(session: ChatSession, path: Path) -> None:
    payload = {"messages": session.messages}
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
        _replay_history(session.messages, use_color=False, show_separator=False)
        sys.stdout = old_stdout
        readable_path.write_text(buf.getvalue(), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - best-effort
        sys.stdout = old_stdout  # ensure restore
        print(f"Failed to write readable log to {readable_path}: {exc}", file=sys.stderr)


def _get_log_path_for_session(session: ChatSession) -> Path:
    """Get appropriate log path - reuse latest if small, otherwise create new."""
    # Calculate total content size
    total_size = 0
    for msg in session.messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_size += len(content)

    # If small enough, reuse latest log file
    if total_size <= MAX_LOG_SIZE:
        log_files = sorted(_log_dir.glob("chat_*.json"), reverse=True)
        if log_files:
            return log_files[0]

    # Otherwise create new file
    return _generate_log_path()


def _load_last_log() -> List[Dict[str, Any]]:
    """Load previous conversation from the most recent log file."""
    try:
        log_files = sorted(_log_dir.glob("chat_*.json"), reverse=True)
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


def _summarize_conversation(client: ChatClient, messages: List[Dict[str, Any]]) -> str:
    """Use LLM to summarize a conversation."""
    summary_prompt = """请用中文总结以下对话的主要内容，保留重要信息和情感细节。用简洁自然的语言，不要超过500字。

对话内容：
""" + _format_messages_for_summary(messages)

    try:
        result_text = ""
        for event in client.stream_chat([
            {"role": "system", "content": "你是一个对话摘要助手，擅长用简洁的语言总结对话内容。"},
            {"role": "user", "content": summary_prompt}
        ]):
            if event.get("type") == "content":
                result_text += event.get("text", "")
        return result_text.strip()
    except Exception:
        return ""


def _format_messages_for_summary(messages: List[Dict[str, Any]]) -> str:
    """Format messages for summary prompt."""
    lines = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue
        if role == "user":
            lines.append(f"用户: {content}")
        elif role == "assistant":
            lines.append(f"爱丽丝: {content}")
    return "\n".join(lines)


def _summarize_all_summaries(client: ChatClient, summaries: List[str]) -> str:
    """Summarize multiple summaries into one if too long."""
    combined = "\n\n---\n\n".join(summaries)
    if len(combined) <= MAX_SUMMARY_SIZE:
        return combined

    summary_prompt = f"""请将以下多段对话摘要合并成一个整体摘要，保留重要的时间线和情感发展。不超过800字。

摘要内容：
{combined}"""

    try:
        result_text = ""
        for event in client.stream_chat([
            {"role": "system", "content": "你是一个对话摘要助手，擅长合并多个摘要。"},
            {"role": "user", "content": summary_prompt}
        ]):
            if event.get("type") == "content":
                result_text += event.get("text", "")
        return result_text.strip()
    except Exception:
        return combined[:MAX_SUMMARY_SIZE]


def _load_history_with_summaries(client: ChatClient) -> Tuple[List[Dict[str, Any]], str]:
    """Load latest conversation and summarize older ones."""
    log_files = sorted(_log_dir.glob("chat_*.json"), reverse=True)
    if not log_files:
        return [], ""

    # Load latest conversation
    latest_messages = []
    try:
        data = json.loads(log_files[0].read_text(encoding="utf-8"))
        latest_messages = data.get("messages", [])
    except Exception:
        pass

    # Summarize older conversations
    summaries: List[str] = []
    for log_file in log_files[1:]:
        try:
            data = json.loads(log_file.read_text(encoding="utf-8"))
            messages = data.get("messages", [])
            if messages:
                summary = _summarize_conversation(client, messages)
                if summary:
                    # Truncate if too long
                    if len(summary) > MAX_SUMMARY_SIZE:
                        summary = summary[:MAX_SUMMARY_SIZE]
                    summaries.append(f"[{log_file.stem}]\n{summary}")
        except Exception:
            continue

    # Combine summaries
    combined_summary = ""
    if summaries:
        combined_summary = _summarize_all_summaries(client, summaries)
        if len(combined_summary) > MAX_SUMMARY_SIZE:
            combined_summary = combined_summary[:MAX_SUMMARY_SIZE]

    return latest_messages, combined_summary
