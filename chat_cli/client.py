"""OpenAI-powered chat client using the Chat Completions API."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional
from urllib.parse import urlparse

try:
    from openai import OpenAI, OpenAIError, pydantic_function_tool
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "The openai package is required. Install it with `pip install openai`."
    ) from exc

try:
    from pydantic import BaseModel, Field, create_model
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "The pydantic package is required. Install it with `pip install pydantic`."
    ) from exc

from .config import Config


class APIError(RuntimeError):
    """Raised when the OpenAI SDK reports an error."""


@dataclass
class _ToolBuffer:
    """Accumulate streaming tool-call deltas."""

    id: str = ""
    type: str = "function"
    name: str = ""
    arguments: str = ""

    def merge_delta(self, delta: Any) -> None:
        if not delta:
            return

        if isinstance(delta, dict):
            if delta.get("id"):
                self.id = str(delta["id"])
            if delta.get("type"):
                self.type = str(delta["type"])
            function = delta.get("function") or {}
            if isinstance(function, dict):
                name = function.get("name")
                if name:
                    self.name = str(name)
                arguments = function.get("arguments")
                if arguments:
                    self.arguments += str(arguments)
            return

        if getattr(delta, "id", None):
            self.id = delta.id  # type: ignore[assignment]
        if getattr(delta, "type", None):
            self.type = delta.type  # type: ignore[assignment]
        function = getattr(delta, "function", None)
        if function is None:
            return
        if getattr(function, "name", None):
            self.name = function.name  # type: ignore[assignment]
        if getattr(function, "arguments", None):
            self.arguments += function.arguments  # type: ignore[assignment]

    def to_event(self) -> Dict[str, Any]:
        return {
            "type": "tool_call",
            "tool_call": {
                "id": self.id,
                "type": self.type,
                "function": {
                    "name": self.name,
                    "arguments": self.arguments,
                },
            },
        }


class ChatClient:
    """Chat completion client backed by the OpenAI SDK."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=self._compute_base_url(config.api_url),
        )

    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Iterator[Dict[str, Any]]:
        request: Dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": self._config.temperature,
        }

        prepared_tools = self._prepare_tools(tools)
        if prepared_tools:
            request["tools"] = prepared_tools
            request["tool_choice"] = "auto"

        try:
            stream = self._client.chat.completions.create(stream=True, **request)
            yield from self._consume_stream(stream)
        except OpenAIError as exc:
            if self._should_retry_without_stream(str(exc)):
                yield from self._non_stream_completion(request)
            else:
                raise APIError(str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise APIError(str(exc)) from exc

    def _consume_stream(self, stream: Iterable[Any]) -> Iterator[Dict[str, Any]]:
        tool_buffers: Dict[int, _ToolBuffer] = {}
        try:
            for chunk in stream:
                choices = getattr(chunk, "choices", [])
                finish_reasons: List[str] = []

                for choice in choices:
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue

                    content = getattr(delta, "content", None)
                    if content:
                        yield {"type": "content", "text": content}

                    tool_deltas = getattr(delta, "tool_calls", None) or []
                    for tool_delta in tool_deltas:
                        index = getattr(tool_delta, "index", 0)
                        buffer = tool_buffers.setdefault(index, _ToolBuffer())
                        buffer.merge_delta(tool_delta)

                    finish_reason = getattr(choice, "finish_reason", None)
                    if finish_reason:
                        finish_reasons.append(finish_reason)

                if any(reason == "tool_calls" for reason in finish_reasons):
                    for buffer in tool_buffers.values():
                        yield buffer.to_event()
                    tool_buffers.clear()

                if any(reason for reason in finish_reasons if reason and reason != "tool_calls"):
                    break
        finally:
            close = getattr(stream, "close", None)
            if callable(close):  # pragma: no branch
                close()

    def _non_stream_completion(self, request: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        completion = self._client.chat.completions.create(stream=False, **request)
        yield from self._yield_from_completion(completion)

    def _yield_from_completion(self, completion: Any) -> Iterator[Dict[str, Any]]:
        for choice in getattr(completion, "choices", []) or []:
            message = getattr(choice, "message", None)
            if message is None:
                continue

            content = getattr(message, "content", None)
            if content:
                yield {"type": "content", "text": content}

            for tool_call in getattr(message, "tool_calls", []) or []:
                function = getattr(tool_call, "function", None)
                yield {
                    "type": "tool_call",
                    "tool_call": {
                        "id": getattr(tool_call, "id", ""),
                        "type": getattr(tool_call, "type", "function"),
                        "function": {
                            "name": getattr(function, "name", "") if function else "",
                            "arguments": getattr(function, "arguments", "") if function else "",
                        },
                    },
                }

    @staticmethod
    def _should_retry_without_stream(message: str) -> bool:
        lowered = message.lower()
        if "stream" not in lowered:
            return False
        keywords = (
            "not supported",
            "unsupported",
            "does not support",
            "unrecognized request argument",
        )
        return any(keyword in lowered for keyword in keywords)

    def _prepare_tools(self, tools: Optional[List[Dict[str, Any]]]) -> List[Any]:
        if not tools:
            return []

        prepared: List[Any] = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            function = tool.get("function", {})
            name = function.get("name")
            if not name:
                continue
            description = function.get("description", "")
            schema = function.get("parameters") or {}
            model = self._schema_to_model(name, schema)
            prepared.append(pydantic_function_tool(model, name=name, description=description))
        return prepared

    def _schema_to_model(self, name: str, schema: Dict[str, Any]) -> type[BaseModel]:
        properties: Dict[str, Any] = schema.get("properties", {}) if isinstance(schema, dict) else {}
        required = set(schema.get("required", [])) if isinstance(schema, dict) else set()

        fields: Dict[str, tuple] = {}
        for field_name, definition in properties.items():
            field_type = self._json_type_to_python(definition.get("type"))
            field_description = definition.get("description")
            default = ... if field_name in required else None
            fields[field_name] = (
                field_type,
                Field(default=default, description=field_description),
            )

        if not fields:
            fields["_unused"] = (Optional[str], Field(default=None))

        model_name = f"ToolArgs_{name}"
        return create_model(model_name, **fields)  # type: ignore[return-value]

    @staticmethod
    def _json_type_to_python(type_name: Optional[str]):
        mapping = {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        return mapping.get(type_name or "string", str)

    @staticmethod
    def _compute_base_url(api_url: str) -> str:
        parsed = urlparse(api_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("api_url must include scheme and host")

        base = f"{parsed.scheme}://{parsed.netloc}"
        path = (parsed.path or "").rstrip("/")
        if path.endswith("/chat/completions"):
            path = path[: -len("/chat/completions")]
        if path:
            base = base + path
        return base or api_url
