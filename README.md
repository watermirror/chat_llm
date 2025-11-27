# Chat LLM CLI Toolkit

Two CLIs for OpenAI-compatible chat:
- `chat_cli/` (Python)
- `chat_cli_2/` (Kotlin + Spring AI)

## Quick Start
1) 生成配置（TOML），默认位置：`~/.config/chat-cli/config.toml`。可用脚本快速拷贝模板：
```
./config deepseek   # 拷贝 configs/config.toml.deepseek -> ~/.config/chat-cli/config.toml
```
2) 运行 Python CLI（默认读取上面的配置）：
```
./run
```
3) 运行 Kotlin CLI（默认读取同一配置）：
```
./run2
```

## Notes
- 配置文件字段示例：
  ```
  api_url = "https://api.openai.com/v1/chat/completions"
  api_key = "sk-xxxx"
  model = "gpt-4o"
  temperature = 1.0
  ```
  也可放在仓库根下 `.chat-cli/config.toml`，两端 CLI 都会按顺序查找。
- `last_log.json` 记录最近一次对话，已在 .gitignore。
- `get_current_time.year`（可选）放在仓库根或 `chat_cli_2/`，可覆盖时间工具返回的年份，用于测试。
- 更多配置模板在 `configs/`（如 `config.toml.deepseek`, `config.toml.gpt-4o`）。

## Common Flags
- `--config <path>`: explicit config file.
- `--no-history`: start each turn without prior messages.
- `--no-color`: disable ANSI colors.
