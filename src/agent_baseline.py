from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


SYSTEM_PROMPT = (
    "Bạn là trợ lý hữu ích. Hãy trả lời ngắn gọn bằng tiếng Việt và chỉ dùng "
    "thông tin trong hội thoại hiện tại."
)


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent with conversation history limited to each individual thread."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Reply using the configured live model, or the deterministic offline path."""
        if not thread_id or not thread_id.strip():
            raise ValueError("thread_id must not be empty")
        if not isinstance(message, str):
            raise TypeError("message must be a string")

        if self.langchain_agent is None:
            return self._reply_offline(thread_id, message)

        return self._reply_live(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Generate a deterministic response using only this thread's prior turns."""
        session = self.sessions.setdefault(thread_id, SessionState())
        prompt_messages = [*session.messages, {"role": "user", "content": message}]
        prompt_text = "\n".join(
            f"{item['role']}: {item['content']}" for item in prompt_messages
        )
        session.prompt_tokens_processed += estimate_tokens(
            f"system: {SYSTEM_PROMPT}\n{prompt_text}"
        )

        response = self._offline_response(session.messages, message)
        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": response})
        session.token_usage += estimate_tokens(response)
        return self._result(response, session)

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        prompt_messages = [*session.messages, {"role": "user", "content": message}]
        prompt_text = "\n".join(
            f"{item['role']}: {item['content']}" for item in prompt_messages
        )
        estimated_prompt_tokens = estimate_tokens(
            f"system: {SYSTEM_PROMPT}\n{prompt_text}"
        )

        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={
                "configurable": {
                    "thread_id": self._checkpoint_thread_id(user_id, thread_id)
                }
            },
        )
        messages = result.get("messages")
        if not messages:
            raise RuntimeError("LangChain agent returned no messages")
        response = self._message_content(messages[-1])
        if not response:
            raise RuntimeError("LangChain agent returned an empty response")

        usage = getattr(messages[-1], "usage_metadata", None) or {}
        output_tokens = usage.get("output_tokens")
        input_tokens = usage.get("input_tokens")
        session.token_usage += (
            output_tokens if isinstance(output_tokens, int) and output_tokens >= 0
            else estimate_tokens(response)
        )
        session.prompt_tokens_processed += (
            input_tokens if isinstance(input_tokens, int) and input_tokens >= 0
            else estimated_prompt_tokens
        )

        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": response})
        return self._result(response, session)

    @staticmethod
    def _offline_response(messages: list[dict[str, str]], message: str) -> str:
        facts: dict[str, str] = {}
        for item in messages:
            if item.get("role") == "user":
                facts.update(extract_profile_updates(item.get("content", "")))

        question = message.casefold()
        fact_key: str | None = None
        if any(term in question for term in ("tên gì", "tên mình", "tên tôi")):
            fact_key = "name"
        elif any(term in question for term in ("nghề", "công việc", "làm gì")):
            fact_key = "profession"
        elif any(term in question for term in ("ở đâu", "nơi ở", "đang ở")):
            fact_key = "location"
        elif any(term in question for term in ("style", "phong cách", "trả lời")):
            fact_key = "response_style"
        elif any(term in question for term in ("đồ uống", "uống gì", "thức uống")):
            fact_key = "favorite_drink"
        elif any(term in question for term in ("sở thích", "quan tâm", "yêu thích")):
            fact_key = "interests"

        if fact_key and fact_key in facts:
            return facts[fact_key]
        if fact_key:
            return "Mình chưa có thông tin đó trong hội thoại này."
        return "Mình đã ghi nhận. Bạn muốn mình hỗ trợ thêm điều gì?"

    def _maybe_build_langchain_agent(self):
        """Create an optional checkpointed agent only when live use is configured."""
        model_config = self.config.model
        has_live_endpoint = model_config.provider == "ollama" or (
            model_config.provider == "custom" and model_config.base_url
        )
        if not model_config.api_key and not has_live_endpoint:
            return None

        from langchain.agents import create_agent
        from langgraph.checkpoint.memory import InMemorySaver

        model = build_chat_model(model_config)
        return create_agent(
            model,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
        )

    @staticmethod
    def _message_content(message: Any) -> str:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts = [
                block["text"]
                for block in content
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            ]
            return "\n".join(text_parts).strip()
        raise TypeError("LangChain agent returned a message with unsupported content")

    @staticmethod
    def _checkpoint_thread_id(user_id: str, thread_id: str) -> str:
        return f"{len(user_id)}:{user_id}{thread_id}"

    def _result(self, response: str, session: SessionState) -> dict[str, Any]:
        return {
            "response": response,
            "agent_tokens_only": session.token_usage,
            "token_usage": session.token_usage,
            "prompt_tokens_processed": session.prompt_tokens_processed,
        }
