package com.chatllm.cli2

import org.springframework.ai.model.openai.autoconfigure.OpenAiAudioSpeechAutoConfiguration
import org.springframework.ai.model.openai.autoconfigure.OpenAiAudioTranscriptionAutoConfiguration
import org.springframework.ai.model.openai.autoconfigure.OpenAiChatAutoConfiguration
import org.springframework.ai.model.openai.autoconfigure.OpenAiEmbeddingAutoConfiguration
import org.springframework.ai.model.openai.autoconfigure.OpenAiImageAutoConfiguration
import org.springframework.ai.model.openai.autoconfigure.OpenAiModerationAutoConfiguration
import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication

@SpringBootApplication(
    exclude = [
        OpenAiAudioSpeechAutoConfiguration::class,
        OpenAiAudioTranscriptionAutoConfiguration::class,
        OpenAiChatAutoConfiguration::class,
        OpenAiEmbeddingAutoConfiguration::class,
        OpenAiImageAutoConfiguration::class,
        OpenAiModerationAutoConfiguration::class,
    ],
)
open class ChatCliApplication

fun main(args: Array<String>) {
    runApplication<ChatCliApplication>(*args)
}
