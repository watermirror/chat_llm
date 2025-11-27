package com.chatllm.cli2.core

import org.springframework.ai.chat.messages.AssistantMessage
import org.springframework.ai.chat.messages.Message
import org.springframework.ai.chat.messages.SystemMessage
import org.springframework.ai.chat.messages.UserMessage

class ChatSession {
    private val _messages: MutableList<Message> = mutableListOf()

    val messages: List<Message>
        get() = _messages

    fun addMessage(message: Message) = _messages.add(message)

    fun addUserMessage(content: String) = addMessage(UserMessage(content))

    fun addAssistantMessage(message: AssistantMessage) = addMessage(message)

    fun addSystemMessage(content: String) = addMessage(SystemMessage(content))

    fun clear() {
        _messages.clear()
    }

    fun removeLastMessage() {
        if (_messages.isNotEmpty()) {
            _messages.removeLast()
        }
    }
}
