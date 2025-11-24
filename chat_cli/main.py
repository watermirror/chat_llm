"""Entry point for the chat CLI application."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .chat import ChatSession
from .client import APIError, ChatClient
from .config import ConfigError, DEFAULT_CONFIG_PATH, load_config
from .tools import ToolError, consume_farewell_request, execute_tool, list_tool_specs

RESET = "\033[0m"
USER_COLOR = "\033[35m"  # Magenta
AI_COLOR = "\033[36m"  # Cyan
TOOL_COLOR = "\033[33m"  # Yellow


def style(text: str, color: str, enable: bool) -> str:
    if not enable:
        return text
    return f"{color}{text}{RESET}"


EXIT_COMMANDS = {"/quit", "/exit", "/q"}
LAST_LOG_PATH = Path(__file__).resolve().parent.parent / "last_log.json"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with an OpenAI-compatible API")
    parser.add_argument(
        "--config",
        type=Path,
        help=f"Path to configuration file (default: {DEFAULT_CONFIG_PATH})",
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
    session = ChatSession()
    args = parse_args(argv)

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

        config_location = _resolve_config_path(config_path)

        config_line = f"Using API URL: {config.api_url} | Model: {config.model}"
        print(style(config_line, TOOL_COLOR, use_color))
        print(style(f"Config file: {config_location}", TOOL_COLOR, use_color))

        print("Type /quit, /exit, or /q to leave the chat.")

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
                session = ChatSession()

            history_checkpoint = len(session.messages)
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
        _write_last_log(session)


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


def _write_last_log(session: ChatSession, path: Path = LAST_LOG_PATH) -> None:
    payload = {"messages": session.messages}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - best-effort logging
        print(f"Failed to write chat log to {path}: {exc}", file=sys.stderr)
