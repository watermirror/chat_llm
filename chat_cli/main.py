"""Entry point for the chat CLI application."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .chat import ChatSession
from .client import APIError, ChatClient
from .config import ConfigError, DEFAULT_CONFIG_PATH, Profile, load_config, load_profile, list_profiles
from .tools import ToolError, consume_farewell_request, execute_tool, handle_act, handle_face2face, handle_separate, init_face_to_face_state, is_face_to_face, list_tool_specs, reset_act_call_count

RESET = "\033[0m"
USER_COLOR = "\033[35m"  # Magenta
AI_COLOR = "\033[36m"  # Cyan
TOOL_COLOR = "\033[33m"  # Yellow


def style(text: str, color: str, enable: bool) -> str:
    if not enable:
        return text
    return f"{color}{text}{RESET}"


EXIT_COMMANDS = {"/quit", "/exit", "/q"}
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

        config_location = _resolve_config_path(config_path)

        config_line = f"Using API URL: {config.api_url} | Model: {config.model}"
        print(style(config_line, TOOL_COLOR, use_color))
        print(style(f"Config file: {config_location}", TOOL_COLOR, use_color))
        if profile:
            print(style(f"Profile: {profile.name}", TOOL_COLOR, use_color))
        if history_summary:
            print(style(f"Loaded history summary ({len(history_summary)} chars)", TOOL_COLOR, use_color))
        if is_face_to_face():
            print(style("当前状态: 见面中", TOOL_COLOR, use_color))

        print("Type /quit, /exit, or /q to leave the chat.")

        # Debug: print face-to-face state and system prompt
        print(style(f"\n[Debug] is_face_to_face: {is_face_to_face()}", TOOL_COLOR, use_color))
        if session.messages and session.messages[0].get("role") == "system":
            print(style("=== System Prompt ===", TOOL_COLOR, use_color))
            print(session.messages[0]["content"])
            print(style("=== End System Prompt ===\n", TOOL_COLOR, use_color))

        while True:
            try:
                prompt = style("You>", USER_COLOR, use_color) + " "
                user_input = input(prompt)
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
                session.add_message({"role": "system", "content": system_msg})
            elif trimmed.lower().startswith("sep:"):
                scene = trimmed[4:].strip()
                system_msg = handle_separate(scene)
                session.add_message({"role": "system", "content": system_msg})
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
                    session.add_user_message(msg)
                else:
                    session.add_message({"role": "system", "content": msg})
            else:
                session.add_user_message(user_input)

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

        assistant_reply, tool_calls = _stream_response(session, client, tool_specs)

        if use_color:
            print(RESET, end="", flush=True)
        print()

        if tool_calls:
            session.add_assistant_tool_call(tool_calls, assistant_reply)
            _process_tool_calls(session, tool_calls, use_color)
        else:
            if assistant_reply:
                session.add_assistant_message(assistant_reply)
            awaiting_final = False


def _stream_response(
    session: ChatSession,
    client: ChatClient,
    tool_specs: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    assistant_chunks: List[str] = []
    collected_tool_calls: List[Dict[str, Any]] = []

    for event in client.stream_chat(session.messages, tools=tool_specs):
        event_type = event.get("type")
        if event_type == "content":
            text = event.get("text", "")
            print(text, end="", flush=True)
            assistant_chunks.append(text)
        elif event_type == "tool_call":
            collected_tool_calls.append(event["tool_call"])

    return "".join(assistant_chunks), collected_tool_calls


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

        invocation = f"[tool-call] {name}({arguments})"
        print(style(invocation, TOOL_COLOR, use_color))

        try:
            result = execute_tool(name, arguments)
        except ToolError as exc:
            result = json.dumps({"error": str(exc)})
            session.add_tool_message(tool_id, result)
            error_line = f"[tool-error] {exc}"
            print(style(error_line, TOOL_COLOR, use_color))
            raise

        session.add_tool_message(tool_id, result)
        output_line = f"[tool-result] {result}"
        print(style(output_line, TOOL_COLOR, use_color))


def _restore_history(session: ChatSession, checkpoint: int) -> None:
    while len(session.messages) > checkpoint:
        session.remove_last_message()


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
