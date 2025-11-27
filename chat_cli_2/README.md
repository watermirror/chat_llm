# chat_cli_2 (Kotlin + Spring AI)

Kotlin/Spring Boot CLI mirroring `chat_cli` features: streaming chat, tool calling (`get_current_time`, `register_farewell`), config discovery, and `last_log.json` capture.

## Quick start
- Install JDK 17+ and Maven.
- Configure your API endpoint/key in `~/.config/chat-cli/config.toml` (auto-created from `src/main/resources/configs/default_config.toml` if missing) or pass `--config /path/to/config.toml`.
- Run: `mvn -f chat_cli_2/pom.xml spring-boot:run` (add `-Dspring-boot.run.arguments="--no-history --no-color"` as needed).

## Flags
- `--config /path/to/config.toml` – load an explicit config.
- `--no-history` – clear session state between turns.
- `--no-color` – disable ANSI colors.

## Notes
- Tools: `get_current_time` (optional `timezone`; respects `get_current_time.year` override file) and `register_farewell` (sets a goodbye flag to exit gracefully).
- Conversation log: `chat_cli_2/last_log.json` is written on exit.
