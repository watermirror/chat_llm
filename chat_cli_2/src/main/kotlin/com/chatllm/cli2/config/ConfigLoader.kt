package com.chatllm.cli2.config

import org.springframework.stereotype.Component
import org.tomlj.Toml
import java.io.IOException
import java.io.InputStream
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths

private const val DEFAULT_TEMPLATE_PATH = "/configs/default_config.toml"

@Component
class ConfigLoader {
    private val defaultConfigPath: Path = Paths.get(System.getProperty("user.home"), ".config", "chat-cli", "config.toml")
    private val localFallbackPath: Path = Paths.get(".chat-cli", "config.toml")

    fun load(customPath: String?): AppSettings {
        val candidates = if (customPath != null) {
            listOf(Paths.get(customPath))
        } else {
            listOf(defaultConfigPath, localFallbackPath)
        }

        var lastError: Exception? = null
        for (candidate in candidates) {
            try {
                val resolved = ensureConfigFile(candidate)
                return parseConfig(resolved)
            } catch (ex: Exception) {
                lastError = ex as? Exception ?: RuntimeException(ex)
            }
        }
        throw IllegalStateException("Failed to load configuration", lastError)
    }

    private fun ensureConfigFile(path: Path): Path {
        if (Files.exists(path)) {
            return path
        }

        try {
            Files.createDirectories(path.parent)
            writeTemplate(path)
            return path
        } catch (ex: IOException) {
            throw IllegalStateException("Could not create config at $path", ex)
        }
    }

    private fun writeTemplate(target: Path) {
        val template: InputStream = javaClass.getResourceAsStream(DEFAULT_TEMPLATE_PATH)
            ?: throw IllegalStateException("Missing default template resource at $DEFAULT_TEMPLATE_PATH")
        template.use { input -> Files.copy(input, target) }
    }

    private fun parseConfig(path: Path): AppSettings {
        val result = Toml.parse(path)
        if (result.hasErrors()) {
            val message = result.errors().joinToString("; ") { it.message ?: it.toString() }
            throw IllegalArgumentException("Invalid TOML in $path: $message")
        }
        val apiUrl = result.getString("api_url") ?: throw IllegalArgumentException("api_url is required")
        val apiKey = result.getString("api_key") ?: throw IllegalArgumentException("api_key is required")
        val model = result.getString("model") ?: throw IllegalArgumentException("model is required")
        val temperature = when {
            result.isDouble("temperature") -> result.getDouble("temperature") ?: 1.0
            result.isLong("temperature") -> result.getLong("temperature")?.toDouble() ?: 1.0
            else -> 1.0
        }
        return AppSettings(apiUrl = apiUrl, apiKey = apiKey, model = model, temperature = temperature, configPath = path)
    }
}

data class AppSettings(
    val apiUrl: String,
    val apiKey: String,
    val model: String,
    val temperature: Double,
    val configPath: Path,
)
