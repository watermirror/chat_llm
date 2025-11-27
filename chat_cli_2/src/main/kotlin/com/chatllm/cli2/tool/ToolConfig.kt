package com.chatllm.cli2.tool

import com.fasterxml.jackson.annotation.JsonProperty
import org.springframework.ai.model.tool.ToolCallingManager
import org.springframework.ai.tool.ToolCallback
import org.springframework.ai.tool.function.FunctionToolCallback
import org.springframework.ai.tool.resolution.StaticToolCallbackResolver
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.core.ParameterizedTypeReference
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import java.nio.file.Files
import java.nio.file.Path
import java.time.ZoneId
import java.time.ZonedDateTime
import java.util.concurrent.atomic.AtomicBoolean

@Configuration
open class ToolConfig {

    private val mapper = jacksonObjectMapper().registerKotlinModule()

    @Bean
    open fun farewellRegistry(): FarewellRegistry = FarewellRegistry()

    @Bean
    open fun toolCallbacks(
        farewellRegistry: FarewellRegistry,
        recorder: ToolTranscriptRecorder,
    ): List<ToolCallback> = listOf(
        currentTimeTool(recorder),
        farewellTool(farewellRegistry, recorder)
    )

    @Bean
    open fun toolCallingManager(toolCallbacks: List<ToolCallback>): ToolCallingManager {
        val resolver = StaticToolCallbackResolver(toolCallbacks)
        return ToolCallingManager.builder().toolCallbackResolver(resolver).build()
    }

    private fun currentTimeTool(recorder: ToolTranscriptRecorder): ToolCallback {
        return FunctionToolCallback.builder<CurrentTimeRequest, CurrentTimeResponse>(
            "get_current_time"
        ) { request, _ ->
            printToolCall("get_current_time", request.toString())
            recorder.recordCall("get_current_time", request.toString())
            val zone = resolveZone(request.timezone)
            val now = ZonedDateTime.now(zone)
            val overrideYear = loadYearOverride()
            val iso = now.toOffsetDateTime().toString()
            val finalIso = overrideYear?.let { applyYearOverride(iso, it) } ?: iso
            val payload = CurrentTimeResponse(currentTime = finalIso, timezone = zone.id)
            printToolResult(payload.toString())
            recorder.recordResult("get_current_time", mapper.writeValueAsString(payload))
            payload
        }
            .description("Return the current time in ISO 8601 format. Optionally accept an IANA timezone.")
            .inputType(object : ParameterizedTypeReference<CurrentTimeRequest>() {})
            .build()
    }

    private fun farewellTool(registry: FarewellRegistry, recorder: ToolTranscriptRecorder): ToolCallback {
        return FunctionToolCallback.builder<FarewellRequest, FarewellResponse>(
            "register_farewell"
        ) { request, _ ->
            printToolCall("register_farewell", request.toString())
            recorder.recordCall("register_farewell", mapper.writeValueAsString(request))
            registry.register(request.note)
            val payload = FarewellResponse(farewellRegistered = true, note = request.note)
            printToolResult(payload.toString())
            recorder.recordResult("register_farewell", mapper.writeValueAsString(payload))
            payload
        }
            .description("Mark that the user has initiated a farewell so the session can end gracefully.")
            .inputType(object : ParameterizedTypeReference<FarewellRequest>() {})
            .build()
    }

    private fun resolveZone(timezone: String?): ZoneId {
        if (timezone.isNullOrBlank()) {
            return ZoneId.systemDefault()
        }
        return try {
            ZoneId.of(timezone)
        } catch (ex: Exception) {
            throw IllegalArgumentException("Unknown timezone: $timezone", ex)
        }
    }

    private fun loadYearOverride(): String? {
        val cwd = Path.of("").toAbsolutePath()
        val candidates = listOf(
            cwd.resolve("get_current_time.year"),
            cwd.parent?.resolve("get_current_time.year")
        ).filterNotNull()

        val overrideFile = candidates.firstOrNull { Files.exists(it) } ?: return null

        val firstLine = Files.readAllLines(overrideFile).firstOrNull { it.isNotBlank() }?.trim()
        return if (firstLine != null && firstLine.length == 4 && firstLine.all { it.isDigit() }) firstLine else null
    }

    private fun applyYearOverride(isoTime: String, year: String): String {
        return if (isoTime.length >= 4 && isoTime.take(4).all { it.isDigit() }) {
            "$year${isoTime.drop(4)}"
        } else isoTime
    }

    private fun printToolCall(name: String, args: String) {
        val yellow = "\u001B[33m"
        val reset = "\u001B[0m"
        println("\n${yellow}[tool-call] $name($args)$reset")
    }

    private fun printToolResult(result: String) {
        val yellow = "\u001B[33m"
        val reset = "\u001B[0m"
        println("${yellow}[tool-result] $result$reset")
    }

}

data class CurrentTimeRequest(val timezone: String? = null)

data class CurrentTimeResponse(
    @JsonProperty("current_time") val currentTime: String,
    val timezone: String,
)

data class FarewellRequest(val note: String? = null)

data class FarewellResponse(
    @JsonProperty("farewell_registered") val farewellRegistered: Boolean,
    val note: String? = null,
)

class FarewellRegistry {
    private val requested = AtomicBoolean(false)
    @Volatile
    private var note: String? = null

    fun register(note: String?) {
        this.note = note
        requested.set(true)
    }

    fun consume(): FarewellState? {
        return if (requested.getAndSet(false)) FarewellState(note) else null
    }
}

data class FarewellState(val note: String?)
