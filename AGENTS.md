# Repository Guidelines

## Project Structure & Module Organization
- `chat_cli/` holds the CLI: `main.py` (entrypoint/IO loop), `chat.py` (session state), `client.py` (OpenAI-compatible streaming client), `config.py` (config loading/defaults), and `tools.py` (built-in function tools).
- `chat_cli_2/` is the Kotlin + Spring AI CLI: `ChatCliApplication.kt` (entrypoint), `cli/` (interactive loop, logging), `config/` (config loading/defaults), `tool/` (tools + transcript recorder), and `run2` (wrapper script).
- `configs/` contains provider-specific TOML templates; the CLI will fall back to `.chat-cli/config.toml` in the repo root or `~/.config/chat-cli/config.toml` if present.
- `config` is a helper script that copies `configs/config.toml.<name>` into `~/.config/chat-cli/config.toml`.
- `run` is a convenience wrapper for `python -m chat_cli`.
- `last_log.json` captures the most recent conversation for debugging; keep it out of commits unless it is intentionally needed.

## Build, Test, and Development Commands
- Install runtime deps: `python -m pip install openai pydantic`.
- Run locally with a specific config: `./run --config path/to/config.toml` (or omit `--config` to use the first available default).
- Copy a provider config: `./config deepseek` (replace `deepseek` with another `<name>` matching `configs/config.toml.<name>`).
- No automated tests exist yet; once added, run them with `pytest`.
- Kotlin/Spring AI CLI: build `mvn -f chat_cli_2/pom.xml -DskipTests package`; run `./run2 --config path/to/config.toml` (honors `--no-history`, `--no-color`).

## Coding Style & Naming Conventions
- Python 3.11+ codebase; use 4-space indentation, type hints, and dataclasses where they clarify structure.
- Keep modules and variables snake_case; prefer small, single-purpose functions that raise custom errors (`ConfigError`, `APIError`, `ToolError`) on failure.
- Preserve streaming-friendly patterns in `client.py` and avoid blocking calls in the chat loop.
- Document non-obvious behavior with short docstrings rather than inline comments.

## Testing Guidelines
- Add tests under `tests/` using `pytest`; name files `test_*.py` and mirror module paths (e.g., `tests/test_client.py`).
- Mock network calls to the OpenAI SDK when asserting request shapes or retry behavior; avoid real API calls in CI.
- When adding tools, cover argument parsing failures and happy-path payload shapes.

## Commit & Pull Request Guidelines
- Follow the existing Conventional Commit style (`feat:`, `chore:`, etc.) seen in `git log`.
- PRs should describe behavior changes, config expectations (API URL/model), and any new CLI flags; include repro steps or screenshots when CLI output changes.
- Run the relevant commands before opening a PR (`./run ...`, `pytest` when present) and note any limitations or required env vars (`OPENAI_API_KEY`/TOML config).
