package com.chatllm.cli2.tool

import org.springframework.ai.chat.messages.AssistantMessage
import org.springframework.ai.chat.messages.ToolResponseMessage
import org.springframework.stereotype.Component
import java.util.UUID
import java.util.concurrent.ConcurrentLinkedQueue

data class ToolEvent(
    val id: String,
    val name: String,
    val kind: Kind,
    val payload: String,
) {
    enum class Kind { CALL, RESULT }
}

@Component
class ToolTranscriptRecorder {
    private val events = ConcurrentLinkedQueue<ToolEvent>()
    private val currentId = ThreadLocal<String?>()

    fun recordCall(name: String, args: String) {
        val id = UUID.randomUUID().toString()
        currentId.set(id)
        events.add(ToolEvent(id = id, name = name, kind = ToolEvent.Kind.CALL, payload = args))
    }

    fun recordResult(name: String, result: String) {
        val id = currentId.get() ?: UUID.randomUUID().toString()
        events.add(ToolEvent(id = id, name = name, kind = ToolEvent.Kind.RESULT, payload = result))
        currentId.remove()
    }

    fun drain(): List<ToolEvent> {
        val drained = mutableListOf<ToolEvent>()
        while (true) {
            val ev = events.poll() ?: break
            drained.add(ev)
        }
        return drained
    }
}

fun ToolEvent.toAssistantMessage(): AssistantMessage =
    AssistantMessage.builder()
        .content("")
        .toolCalls(listOf(AssistantMessage.ToolCall(id, "function", name, payload)))
        .build()

fun ToolEvent.toToolResponseMessage(): ToolResponseMessage =
    ToolResponseMessage.builder()
        .responses(listOf(ToolResponseMessage.ToolResponse(id, name, payload)))
        .build()
