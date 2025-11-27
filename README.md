# Chat LLM CLI Toolkit

This repo provides two CLIs for chatting with OpenAI-compatible APIs:
- `chat_cli/` (Python)
- `chat_cli_2/` (Kotlin + Spring AI)

## Quick Start
1) Prepare a config file (TOML) with `api_url`, `api_key`, `model`, `temperature`. You can copy a template via the helper script:
```
./config deepseek   # copies configs/config.toml.deepseek -> ~/.config/chat-cli/config.toml
```
2) Run the Python CLI:
```
./run --config path/to/config.toml   # or omit --config to use ~/.config/chat-cli/config.toml
```
3) Run the Kotlin CLI:
```
./run2 --config path/to/config.toml  --no-history --no-color
```

## Notes
- Both CLIs write the most recent conversation to `last_log.json` (ignored by git).
- `get_current_time.year` (optional) in the repo root can override the year used by the time tool for testing.
- Additional provider templates live under `configs/` (e.g., `config.toml.deepseek`, `config.toml.gpt-4o`).

## Common Flags
- `--config <path>`: explicit config file.
- `--no-history`: start each turn without prior messages (Python: resets per turn; Kotlin: clears session).
- `--no-color`: disable ANSI colors.
