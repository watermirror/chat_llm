# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python CLI chat application that interfaces with OpenAI-compatible APIs (like DeepSeek). The application provides an interactive chat interface with streaming responses and tool calling capabilities.

## Architecture

The codebase follows a modular design with clear separation of concerns:

- **`chat_cli/main.py`**: Entry point and CLI interface management, handles user input/output, colors, and command parsing
- **`chat_cli/client.py`**: OpenAI API client wrapper with streaming support and tool integration via Pydantic models
- **`chat_cli/chat.py`**: Conversation state management and message history tracking
- **`chat_cli/config.py`**: Configuration loading from TOML files with fallback mechanisms
- **`chat_cli/tools.py`**: Tool definitions and execution (currently: time tools and farewell handling)
- **`configs/default_config.toml`**: Default configuration template

## Development Commands

Run the application:
```bash
python3 -m chat_cli
```

Run with custom config:
```bash
python3 -m chat_cli --config path/to/config.toml
```

Run without conversation history:
```bash
python3 -m chat_cli --no-history
```

Run without colors:
```bash
python3 -m chat_cli --no-color
```

## Configuration

Configuration is managed via TOML files. The application looks for config in this order:
1. Provided `--config` path
2. `~/.config/chat-cli/config.toml`
3. `./.chat-cli/config.toml` (local fallback)

Required configuration fields:
- `api_url`: The API endpoint URL
- `api_key`: Authentication key
- `model`: Model name to use
- `temperature`: Sampling temperature (float)

## Key Components

### ChatSession (chat.py)
Manages conversation history with methods to add user/assistant/tool messages and maintain state.

### ChatClient (client.py)
Handles API communication with streaming support. Uses OpenAI SDK with custom base URL configuration. Implements tool calling via Pydantic function tools.

### Tool System (tools.py)
Extensible tool framework. Current tools:
- `get_current_time`: Returns current time with optional timezone
- `register_farewell`: Handles graceful session termination

### Configuration Management (config.py)
Robust config loading with:
- TOML parsing (uses built-in `tomllib` on Python 3.11+, falls back to basic parser)
- Default value merging
- Config file creation from templates
- Multiple search paths

## Dependencies

The application requires:
- `openai`: For API client functionality
- `pydantic`: For tool schema validation and function calling

## Code Patterns

- Uses `from __future__ import annotations` for forward references
- Type hints throughout with `typing` module
- Dataclasses for structured data (`@dataclass`)
- Error handling with custom exception classes
- Streaming response processing with generators
- Tool calling integration via OpenAI function calling protocol