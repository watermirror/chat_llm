# Chat LLM CLI Toolkit

Two CLIs for OpenAI-compatible chat:
- `chat_cli/` (Python)
- `chat_cli_2/` (Kotlin + Spring AI)

## Quick Start
1) Create a config (TOML), default path: `~/.config/chat-cli/config.toml`. Copy a template with:
```
./config deepseek   # copies configs/config.toml.deepseek -> ~/.config/chat-cli/config.toml
```
2) Run the Python CLI (uses the default config above):
```
./run
```
3) Run the Kotlin CLI (same default config):
```
./run2
```

## Notes
- Config file fields example:
  ```
  api_url = "https://api.openai.com/v1/chat/completions"
  api_key = "sk-xxxx"
  model = "gpt-4o"
  temperature = 1.0
  ```
  You can also place it at repo root `.chat-cli/config.toml`; both CLIs look there if present.
- `last_log.json` stores the most recent conversation (ignored by git).
- `get_current_time.year` (optional) in repo root or `chat_cli_2/` overrides the year for the time tool (testing only).
- More templates in `configs/` (e.g., `config.toml.deepseek`, `config.toml.gpt-4o`).

## Common Flags
- `--config <path>`: explicit config file.
- `--no-history`: start each turn without prior messages.
- `--no-color`: disable ANSI colors.
