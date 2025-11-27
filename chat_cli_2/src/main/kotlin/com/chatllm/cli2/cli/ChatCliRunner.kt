package com.chatllm.cli2.cli

import com.chatllm.cli2.config.AppSettings
import com.chatllm.cli2.core.ChatSession
import com.chatllm.cli2.tool.FarewellRegistry
import com.chatllm.cli2.tool.ToolTranscriptRecorder
import com.chatllm.cli2.tool.toAssistantMessage
import com.chatllm.cli2.tool.toToolResponseMessage
import org.springframework.ai.chat.messages.AssistantMessage
import org.springframework.ai.chat.messages.Message
import org.springframework.ai.chat.prompt.Prompt
import org.springframework.ai.chat.model.ChatModel
import org.springframework.boot.ApplicationArguments
import org.springframework.boot.CommandLineRunner
import org.springframework.stereotype.Component
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import org.springframework.ai.chat.messages.ToolResponseMessage
import org.springframework.ai.chat.messages.UserMessage
import java.nio.file.Files
import java.nio.file.Path
import java.time.format.DateTimeFormatter

@Component
class ChatCliRunner(
    private val chatModel: ChatModel,
    private val appSettings: AppSettings,
    private val farewellRegistry: FarewellRegistry,
    private val toolRecorder: ToolTranscriptRecorder,
    private val arguments: ApplicationArguments,
) : CommandLineRunner {

    private val exitCommands = setOf("/quit", "/exit", "/q")
    private val lastLogPath: Path = Path.of("last_log.json")
    private val mapper = jacksonObjectMapper().registerKotlinModule()

    override fun run(vararg args: String?) {
        val noHistory = arguments.containsOption("no-history")
        val useColor = !arguments.containsOption("no-color")

        println(style("Using API URL: ${appSettings.apiUrl} | Model: ${appSettings.model}", TOOL_COLOR, useColor))
        println(style("Config file: ${appSettings.configPath}", TOOL_COLOR, useColor))
        println("Type /quit, /exit, or /q to leave the chat.")

        val session = ChatSession()

        while (true) {
            try {
                if (noHistory) {
                    session.clear()
                }

                val userPrompt = promptUser(useColor) ?: break
                val trimmed = userPrompt.trim()
                if (trimmed.lowercase() in exitCommands) {
                    break
                }
                if (trimmed.isEmpty()) {
                    continue
                }

                session.addUserMessage(userPrompt)

                try {
                    val assistantMessage = streamAssistant(session, useColor)
                    appendToolEventsToSession(session)
                    if (assistantMessage != null) {
                        val hasTools = assistantMessage.toolCalls.isNotEmpty()
                        val hasContent = !assistantMessage.text.isNullOrEmpty()
                        if (hasTools && !hasContent) {
                            // Tool calls already appended; skip empty assistant content.
                        } else {
                            session.addAssistantMessage(assistantMessage)
                        }
                    }
                } catch (ex: Exception) {
                    session.removeLastMessage()
                    System.err.println("[error] ${ex.message}")
                    continue
                }

                if (farewellRegistry.consume() != null) {
                    break
                }
            } catch (ex: InterruptedException) {
                println("\n(Use /quit to exit)")
            } catch (ex: Exception) {
                System.err.println("[error] ${ex.message}")
            }
        }

        println("Goodbye!")
        writeLastLog(session)
    }

    private fun promptUser(useColor: Boolean): String? {
        val prefix = style("You>", USER_COLOR, useColor)
        print("$prefix ")
        return readlnOrNull()
    }

    private fun streamAssistant(session: ChatSession, useColor: Boolean): AssistantMessage? {
        val prompt = Prompt(session.messages)
        val label = style("AI>", AI_COLOR, useColor)
        print("$label ")
        var lastAssistant: AssistantMessage? = null
        var printed = ""

        chatModel.stream(prompt).doOnNext { response ->
            val assistant = response.result.output as? AssistantMessage ?: return@doOnNext
            val text = assistant.text ?: ""
            when {
                text.startsWith(printed) -> {
                    val chunk = text.substring(printed.length)
                    print(style(chunk, AI_COLOR, useColor))
                    printed = text
                }
                printed.startsWith(text) -> {
                    // duplicate or shorter prefix, ignore
                }
                else -> {
                    // treat as delta
                    print(style(text, AI_COLOR, useColor))
                    printed += text
                }
            }
            lastAssistant = assistant
        }.blockLast()

        println()
        val toolCalls = lastAssistant?.toolCalls ?: emptyList()
        val content = if (printed.isNotEmpty()) printed else lastAssistant?.text ?: ""
        if (content.isEmpty() && toolCalls.isEmpty()) return null

        return AssistantMessage.builder()
            .content(content)
            .toolCalls(toolCalls)
            .build()
    }

    private fun appendToolEventsToSession(session: ChatSession) {
        val events = toolRecorder.drain()
        events.forEach { ev ->
            when (ev.kind) {
                com.chatllm.cli2.tool.ToolEvent.Kind.CALL -> session.addMessage(ev.toAssistantMessage())
                com.chatllm.cli2.tool.ToolEvent.Kind.RESULT -> session.addMessage(ev.toToolResponseMessage())
            }
        }
    }

    private fun writeLastLog(session: ChatSession) {
        try {
            val payload = mapOf(
                "saved_at" to DateTimeFormatter.ISO_INSTANT.format(java.time.Instant.now()),
                "messages" to session.messages.map { toWireMessage(it) },
            )
            lastLogPath.parent?.let { Files.createDirectories(it) }
            mapper.writerWithDefaultPrettyPrinter().writeValue(lastLogPath.toFile(), payload)
        } catch (_: Exception) {
            // Best-effort log; ignore failures.
        }
    }

    private fun toWireMessage(message: Message): Map<String, Any?> {
        val role = message.messageType.name.lowercase()
        val base = mutableMapOf<String, Any?>("role" to role)
        when (message) {
            is UserMessage -> base["content"] = message.text ?: ""
            is AssistantMessage -> {
                base["content"] = message.text ?: ""
                if (message.toolCalls.isNotEmpty()) {
                    base["tool_calls"] = message.toolCalls.map {
                        mapOf(
                            "id" to it.id(),
                            "type" to it.type(),
                            "name" to it.name(),
                            "arguments" to it.arguments(),
                        )
                    }
                }
            }
            is ToolResponseMessage -> {
                val firstResponse = message.responses.firstOrNull()
                if (firstResponse != null) {
                    base["tool_call_id"] = firstResponse.id()
                    base["name"] = firstResponse.name()
                    base["content"] = firstResponse.responseData()
                } else {
                    base["content"] = message.text ?: ""
                }
            }
            else -> base["content"] = message.text ?: ""
        }
        return base
    }

    private fun style(text: String, color: String, enabled: Boolean): String {
        return if (enabled) color + text + RESET else text
    }

    companion object {
        private const val RESET = "\u001B[0m"
        private const val USER_COLOR = "\u001B[35m" // Magenta
        private const val AI_COLOR = "\u001B[36m"   // Cyan
        private const val TOOL_COLOR = "\u001B[33m" // Yellow
    }
}
