"""Conversation management utilities for the chat CLI."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from chat_cli.base_prompt import BASE_SYSTEM_PROMPT


Message = Dict[str, Any]


@dataclass
class ChatSession:
    """Track chat history and provide helpers for message management."""

    _messages: List[Message] = field(default_factory=list)

    def __init__(
        self,
        initial_messages: Optional[List[Message]] = None,
        history_summary: str = "",
        face_to_face: bool = False,
        system_prompt: str = "",
    ) -> None:
        # Build system prompt: profile identity → universal rules → history → state
        system_content = system_prompt + BASE_SYSTEM_PROMPT
        if history_summary:
            system_content += f"\n\n以下是你们之前的一些对话记录摘要：\n{history_summary}"
        if face_to_face:
            system_content += "\n\n[当前状态] 你们现在处于见面状态。你可以使用 act 工具来描述动作。"
        else:
            system_content += "\n\n[当前状态] 你们现在不在一起，是线上聊天状态。act 工具已禁用。"

        if initial_messages:
            self._messages = list(initial_messages)
            # Replace old system prompt with new one
            if self._messages and self._messages[0].get("role") == "system":
                self._messages[0] = {"role": "system", "content": system_content}
            else:
                # If no system message, prepend one
                self._messages.insert(0, {"role": "system", "content": system_content})
        else:
            self._messages = [{"role": "system", "content": system_content}]

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
