"""Conversation management utilities for the chat CLI."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


Message = Dict[str, Any]


@dataclass
class ChatSession:
    """Track chat history and provide helpers for message management."""

    _messages: List[Message] = field(default_factory=list)

    @property
    def messages(self) -> List[Message]:
        """Return the current chat history."""

        return self._messages

    def add_message(self, message: Message) -> None:
        self._messages.append(message)

    def add_user_message(self, content: str) -> None:
        self.add_message({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.add_message({"role": "assistant", "content": content})

    def add_assistant_tool_call(self, tool_calls: List[Dict[str, Any]], content: str = "") -> None:
        message: Message = {"role": "assistant", "tool_calls": tool_calls}
        if content:
            message["content"] = content
        self.add_message(message)

    def add_tool_message(self, tool_call_id: str, content: str) -> None:
        self.add_message({"role": "tool", "tool_call_id": tool_call_id, "content": content})

    def remove_last_message(self) -> None:
        if self._messages:
            self._messages.pop()

    def is_empty(self) -> bool:
        return not self._messages
