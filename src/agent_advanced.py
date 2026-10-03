from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    PROFILE_FIELDS,
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import build_chat_model


SYSTEM_PROMPT = (
    "Bạn là trợ lý hữu ích, trả lời bằng tiếng Việt. Dùng hồ sơ người dùng và "
    "tóm tắt hội thoại bên dưới làm ngữ cảnh; không suy đoán những thông tin "
    "không có trong đó."
)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent combining thread memory, a persistent user profile, and compaction."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route a turn through the live model when configured, otherwise offline."""
        if not user_id or not user_id.strip():
            raise ValueError("user_id must not be empty")
        if not thread_id or not thread_id.strip():
            raise ValueError("thread_id must not be empty")
        if not isinstance(message, str):
            raise TypeError("message must be a string")

        if self.langchain_agent is None:
            return self._reply_offline(user_id, thread_id, message)
        return self._reply_live(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        self._persist_profile_updates(user_id, message)

        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        response = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", response)

        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + estimate_tokens(
            response
        )
        return self._result(user_id, thread_id, response)

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        self._persist_profile_updates(user_id, message)

        self.compact_memory.append(thread_id, "user", message)
        context = self.compact_memory.context(thread_id)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        system_message = self._system_context(user_id, context)
        input_messages = [
            {"role": "system", "content": system_message},
            *context["messages"],
        ]
        result = self.langchain_agent.invoke({"messages": input_messages})
        messages = result.get("messages")
        if not messages:
            raise RuntimeError("LangChain agent returned no messages")
        response = self._message_content(messages[-1])
        if not response:
            raise RuntimeError("LangChain agent returned an empty response")

        usage = getattr(messages[-1], "usage_metadata", None) or {}
        output_tokens = usage.get("output_tokens")
        input_tokens = usage.get("input_tokens")
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + (
            output_tokens
            if isinstance(output_tokens, int) and output_tokens >= 0
            else estimate_tokens(response)
        )
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0)
            + (
                input_tokens
                if isinstance(input_tokens, int) and input_tokens >= 0
                else prompt_tokens
            )
        )
        self.compact_memory.append(thread_id, "assistant", response)
        return self._result(user_id, thread_id, response)

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        context = self.compact_memory.context(thread_id)
        return estimate_tokens(self._system_context(user_id, context)) + sum(
            estimate_tokens(f"{message['role']}: {message['content']}")
            for message in context["messages"]
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        facts = self.profile_store.facts(user_id)
        context = self.compact_memory.context(thread_id)
        if not facts:
            recent_user_messages = [
                item["content"]
                for item in context["messages"]
                if item.get("role") == "user"
            ]
            if recent_user_messages and any(
                term in message.casefold()
                for term in ("vừa nói", "vừa rồi", "trong thread này", "cuộc hội thoại này")
            ):
                return f"Trong thread này bạn vừa nói: {recent_user_messages[-1]}"
            return "Mình chưa có thông tin hồ sơ để nhắc lại."

        question = message.casefold()
        memory_question_terms = (
            "nhớ",
            "nhắc lại",
            "tên",
            "nghề",
            "công việc",
            "ở đâu",
            "nơi ở",
            "sở thích",
            "quan tâm",
            "đồ uống",
            "món ăn",
            "món ruột",
            "thú cưng",
            "corgi",
            "style",
            "phong cách",
            "3 bullet",
        )
        if not any(term in question for term in memory_question_terms):
            recent_user_messages = [
                item["content"]
                for item in context["messages"]
                if item.get("role") == "user"
            ]
            if recent_user_messages and any(
                term in question
                for term in ("vừa nói", "vừa rồi", "trong thread này", "cuộc hội thoại này")
            ):
                return f"Trong thread này bạn vừa nói: {recent_user_messages[-1]}"
            return "Mình đã ghi nhận. Bạn muốn mình hỗ trợ thêm điều gì?"

        # Return the profile together so compound recall questions are answered.
        details = [
            f"{PROFILE_FIELDS[key]}: {facts[key]}"
            for key in facts
        ]
        if "response_style" in facts and "ngắn gọn" not in facts["response_style"].casefold():
            details.append(f"Phong cách trả lời ngắn gọn: {facts['response_style']}")
        return "Mình nhớ: " + "; ".join(details) + "."

    def _persist_profile_updates(self, user_id: str, message: str) -> None:
        updates = extract_profile_updates(message)
        existing = self.profile_store.facts(user_id)
        for key, value in updates.items():
            if key == "interests" and key in existing:
                prior = existing[key]
                if value.casefold() in prior.casefold():
                    continue
                value = f"{prior}, {value}"
            self.profile_store.upsert_fact(user_id, key, value)

    def _maybe_build_langchain_agent(self):
        """Build a live agent only when credentials or a local endpoint are set."""
        model_config = self.config.model
        has_live_endpoint = model_config.provider == "ollama" or (
            model_config.provider == "custom" and model_config.base_url
        )
        if not model_config.api_key and not has_live_endpoint:
            return None

        from langchain.agents import create_agent

        return create_agent(build_chat_model(model_config), system_prompt=SYSTEM_PROMPT)

    def _system_context(self, user_id: str, context: dict[str, object]) -> str:
        profile = self.profile_store.read_text(user_id).strip()
        summary = str(context["summary"]).strip()
        sections = [SYSTEM_PROMPT]
        if profile and profile != "# User Profile":
            sections.append(f"Hồ sơ người dùng:\n{profile}")
        if summary:
            sections.append(f"Tóm tắt hội thoại trước:\n{summary}")
        return "\n\n".join(sections)

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

    def _result(self, user_id: str, thread_id: str, response: str) -> dict[str, Any]:
        return {
            "response": response,
            "agent_tokens_only": self.token_usage(thread_id),
            "token_usage": self.token_usage(thread_id),
            "prompt_tokens_processed": self.prompt_token_usage(thread_id),
            "memory_file_size": self.memory_file_size(user_id),
            "compactions": self.compaction_count(thread_id),
            "memory_path": str(self.profile_store.path_for(user_id)),
        }
