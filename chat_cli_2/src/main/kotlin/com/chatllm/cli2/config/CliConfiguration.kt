package com.chatllm.cli2.config

import org.springframework.ai.chat.model.ChatModel
import org.springframework.ai.openai.OpenAiChatModel
import org.springframework.ai.openai.OpenAiChatOptions
import org.springframework.ai.openai.api.OpenAiApi
import org.springframework.ai.model.tool.ToolCallingManager
import org.springframework.ai.tool.ToolCallback
import org.springframework.boot.ApplicationArguments
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import java.net.URI

@Configuration
open class CliConfiguration(
    private val configLoader: ConfigLoader,
    private val toolCallingManager: ToolCallingManager,
    private val toolCallbacks: List<ToolCallback>,
) {
    @Bean
    open fun appSettings(args: ApplicationArguments): AppSettings {
        val customConfig = args.getOptionValues("config")?.firstOrNull()
        return configLoader.load(customConfig)
    }

    @Bean
    open fun chatModel(appSettings: AppSettings): ChatModel {
        val (baseUrl, completionsPath) = normalizeBaseAndPath(appSettings.apiUrl)

        val api = OpenAiApi.builder()
            .baseUrl(baseUrl)
            .completionsPath(completionsPath)
            .apiKey(appSettings.apiKey)
            .build()

        val defaultOptions = OpenAiChatOptions.builder()
            .model(appSettings.model)
            .temperature(appSettings.temperature)
            .toolCallbacks(toolCallbacks)
            .parallelToolCalls(true)
            .toolChoice("auto")
            .internalToolExecutionEnabled(true)
            .build()

        return OpenAiChatModel.builder()
            .openAiApi(api)
            .defaultOptions(defaultOptions)
            .toolCallingManager(toolCallingManager)
            .build()
    }

    private fun normalizeBaseAndPath(rawUrl: String): Pair<String, String> {
        val uri = URI(rawUrl)
        val hostPort = buildString {
            append(uri.scheme).append("://").append(uri.host)
            if (uri.port != -1) append(":").append(uri.port)
        }
        val path = uri.rawPath ?: ""

        // Case 1: full path already includes chat/completions
        val completionIndex = path.indexOf("/chat/completions")
        if (completionIndex >= 0) {
            val basePath = path.substring(0, completionIndex)
            val completionsPath = path.substring(completionIndex)
            return Pair(hostPort + basePath, completionsPath)
        }

        // Case 2: path ends with /v1 to avoid double v1
        if (path.endsWith("/v1") || path.endsWith("/v1/")) {
            val trimmed = path.removeSuffix("/").removeSuffix("/v1")
            val base = hostPort + trimmed
            return Pair(base.ifEmpty { hostPort }, "/v1/chat/completions")
        }

        // Default: keep path as base prefix; use standard completions path
        val base = hostPort + path
        return Pair(base.ifEmpty { hostPort }, "/v1/chat/completions")
    }
}
