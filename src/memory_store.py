from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict


PROFILE_FIELDS = {
    "name": "Tên",
    "location": "Nơi ở hiện tại",
    "profession": "Nghề nghiệp",
    "response_style": "Phong cách trả lời",
    "interests": "Sở thích và mối quan tâm",
    "favorite_drink": "Đồ uống yêu thích",
    "favorite_food": "Món ăn yêu thích",
    "pet": "Thú cưng",
}


class ThreadMemory(TypedDict):
    messages: list[dict[str, str]]
    summary: str
    compactions: int


def estimate_tokens(text: str) -> int:
    """Estimate tokens consistently using roughly one token per four characters."""
    stripped = text.strip()
    if not stripped:
        return 0
    return (len(stripped) + 3) // 4


@dataclass
class UserProfileStore:
    """Persistent markdown storage for each user's `User.md` profile."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        if not user_id or not user_id.strip():
            raise ValueError("user_id must not be empty")

        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", user_id).strip("-_")
        if not slug:
            slug = "user"
        if slug != user_id:
            digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:10]
            slug = f"{slug}-{digest}"
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return "# User Profile\n"
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        if not search_text:
            raise ValueError("search_text must not be empty")
        content = self.read_text(user_id)
        updated = content.replace(search_text, replacement, 1)
        if updated == content:
            return False
        self.write_text(user_id, updated)
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Read the known structured facts from a profile markdown file."""
        text = self.read_text(user_id)
        facts: dict[str, str] = {}
        for key, label in PROFILE_FIELDS.items():
            pattern = re.compile(
                rf"(?m)^-\s+\*\*{re.escape(label)}:\*\*\s*(.*?)\s*$"
            )
            match = pattern.search(text)
            if match:
                facts[key] = match.group(1)
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        """Insert or replace one known fact without duplicating its markdown row."""
        if key not in PROFILE_FIELDS:
            raise ValueError(f"unsupported profile fact: {key!r}")
        value = value.strip()
        if not value:
            raise ValueError("profile fact value must not be empty")

        label = PROFILE_FIELDS[key]
        line = f"- **{label}:** {value}"
        content = self.read_text(user_id)
        pattern = re.compile(
            rf"(?m)^-\s+\*\*{re.escape(label)}:\*\*.*$"
        )
        if pattern.search(content):
            content = pattern.sub(lambda _: line, content, count=1)
        else:
            content = content.rstrip() + f"\n{line}\n"
        return self.write_text(user_id, content)


_VALUE_STOP = re.compile(
    r"\b(?:chứ|nhưng|mà|để|vì|do|nên|còn|tuy nhiên|"
    r"mỗi ngày|vài tháng|trong vài tháng)\b",
    re.IGNORECASE,
)
_LOCATION_STOP = re.compile(
    r"\b(?:chứ|nhưng|mà|và|để|vì|do|nên|còn|tuy nhiên|"
    r"mỗi ngày|vài tháng|trong vài tháng|trong giai đoạn(?: này)?|dù)\b",
    re.IGNORECASE,
)
_PROFESSION_STOP = re.compile(
    r"\b(?:chứ|nhưng|mà|và|để|vì|do|nên|còn|tuy nhiên|cho)\b",
    re.IGNORECASE,
)
_QUESTION_VALUES = {
    "ai",
    "đâu",
    "gì",
    "nào",
    "bao nhiêu",
    "như thế nào",
    "không",
    "mình",
    "tôi",
    "và",
    "hoặc",
    "style",
    "trả lời",
}
_HYPOTHETICAL_OR_JOKE = re.compile(
    r"\b(?:đùa|nói đùa|giả sử|ví dụ|nếu|hay là|chuyển sang)\b",
    re.IGNORECASE,
)


def _clean_fact_value(value: str, stop_words: re.Pattern[str] = _VALUE_STOP) -> str:
    value = re.split(r"[.;!?\n]", value, maxsplit=1)[0].strip()
    value = stop_words.split(value, maxsplit=1)[0].strip(" \t:-")
    return re.sub(r"\s+", " ", value)


def _valid_fact_value(value: str) -> bool:
    normalized = value.casefold().strip()
    return bool(normalized) and not any(
        normalized == question
        or normalized.startswith(f"{question} ")
        for question in _QUESTION_VALUES
    )


def _first_fact(
    pattern: str,
    message: str,
    stop_words: re.Pattern[str] = _VALUE_STOP,
) -> tuple[re.Match[str], str] | None:
    match = re.search(pattern, message, flags=re.IGNORECASE)
    if not match:
        return None
    value = _clean_fact_value(match.group("value"), stop_words)
    if not _valid_fact_value(value):
        return None
    return match, value


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract explicitly stated profile facts from a single user message.

    This deliberately uses conservative patterns: questions, travel mentions,
    hypothetical roles, and joke statements are not treated as durable facts.
    """
    if not message or not message.strip():
        return {}
    normalized_message = message.casefold()
    if (
        "?" in message
        or "？" in message
        or re.search(r"\b(?:gì|đâu|nào|như thế nào)\b", normalized_message)
    ):
        return {}

    updates: dict[str, str] = {}

    name_match = _first_fact(
        r"\b(?:mình|tôi)\s+tên\s+(?:là\s+)?(?P<value>[^,.;!?\n]+)",
        message,
    ) or _first_fact(
        r"\btên\s+(?:mình|tôi)\s+là\s+(?P<value>[^,.;!?\n]+)",
        message,
    )
    if not name_match:
        candidate_name = _first_fact(
            r"\btên\s+(?P<value>[A-ZÀ-Ỹ][^,.;!?\n]+)",
            message,
        )
        if candidate_name:
            match, candidate = candidate_name
            preceding_context = message[max(0, match.start() - 50) : match.start()]
            if (
                candidate[0].isupper()
                and not re.search(r"\b(?:corgi|chó|mèo|thú\s+cưng)\b", preceding_context, re.I)
            ):
                name_match = candidate_name
    if name_match:
        updates["name"] = name_match[1]

    location_patterns = (
        r"\b(?:nơi\s+ở|địa\s+điểm)\s+"
        r"(?:hiện\s+tại\s+)?(?:đã\s+)?(?:được\s+)?"
        r"(?:cập\s+nhật|chuyển|đổi)\s+"
        r"(?:từ\s+[^,.;!?\n]+\s+)?sang\s+(?P<value>[^,.;!?\n]+)",
        r"\b(?:hiện(?:\s+tại)?|từ\s+tuần\s+này|giờ)\s+"
        r"(?:mình|tôi)\s+(?:đang\s+)?(?:sống|ở|sinh\s+sống)\s+"
        r"(?:tại\s+)?(?P<value>[^,.;!?\n]+)",
        r"\b(?:từ\s+tuần\s+này\s+)?(?:mình|tôi)\s+"
        r"(?:hiện\s+)?đang\s+làm\s+việc\s+ở\s+"
        r"(?P<value>[^,.;!?\n]+)",
        r"\b(?:mình|tôi)\s+(?:(?:hiện(?:\s+tại)?|đang|vẫn)\s+)*"
        r"(?:sống|ở|sinh\s+sống)\s+(?:tại\s+)?(?P<value>[^,.;!?\n]+)",
        r"\b(?:hiện(?:\s+tại)?|từ\s+tuần\s+này)\s+ở\s+"
        r"(?P<value>[^,.;!?\n]+)",
    )
    for pattern in location_patterns:
        location_match = _first_fact(pattern, message, _LOCATION_STOP)
        if not location_match:
            continue
        match, location = location_match
        following_context = message[match.end() : match.end() + 80]
        if re.search(
            r"\b(?:đi\s+họp|họp|bay|du\s+lịch|công\s+tác|chỉ\s+là\s+nơi)\b",
            following_context,
            flags=re.IGNORECASE,
        ):
            continue
        updates["location"] = location
        break

    profession_patterns = (
        r"\b(?:giờ|nay|hiện(?:\s+tại)?)\s+chuyển\s+sang\s+"
        r"(?P<value>[^,.;!?\n]+)",
        r"\b(?:nghề\s+nghiệp|công\s+việc)\s+"
        r"(?:(?:hiện\s+tại|hiện\s+giờ)\s+)?"
        r"(?:(?:của\s+(?:mình|tôi)|thì)\s+)?"
        r"(?:vẫn\s+)?(?:là|:)\s*(?P<value>[^,.;!?\n]+)",
        r"\b(?:mình|tôi)\s+(?:(?:hiện(?:\s+tại)?|vẫn|đang)\s+)*"
        r"làm\s+(?:nghề\s+)?(?:là\s+)?(?P<value>[^,.;!?\n]+)",
    )
    for pattern in profession_patterns:
        profession_match = _first_fact(pattern, message, _PROFESSION_STOP)
        if not profession_match:
            continue
        match, profession = profession_match
        preceding_context = message[max(0, match.start() - 60) : match.start()]
        preceding_context = re.split(r"[.!?\n]", preceding_context)[-1]
        if (
            profession.casefold().startswith(("việc ", "việc từ ", "việc ở "))
            or _HYPOTHETICAL_OR_JOKE.search(preceding_context)
            or re.match(r"^(?:từ\s+xa|ở\s+)", profession, flags=re.IGNORECASE)
            or "?" in message[match.start() :].split(".", maxsplit=1)[0]
        ):
            continue
        updates["profession"] = profession
        break

    style_patterns = (
        r"\b(?:style|phong\s+cách)\s+(?:trả\s+lời|phản\s+hồi)\s*"
        r"(?:(?:của\s+(?:mình|tôi))\s*)?(?:là|:)?\s*(?P<value>[^.!?\n]+)",
        r"\b(?:muốn|thích|mong)\s+(?:bạn\s+)?(?:hãy\s+)?"
        r"(?:cách\s+)?(?:trả\s+lời|phản\s+hồi|giải\s+thích)\s+"
        r"(?P<value>[^.!?\n]+)",
        r"\b(?:muốn|thích|mong)\s+(?:câu\s+)?(?:trả\s+lời|phản\s+hồi)"
        r"\s+(?P<value>[^.!?\n]+)",
        r"\b(?:hãy|vui\s+lòng)\s+(?:trả\s+lời|phản\s+hồi|giải\s+thích)\s+"
        r"(?P<value>[^.!?\n]+)",
    )
    for pattern in style_patterns:
        style_match = _first_fact(pattern, message)
        if style_match:
            style = style_match[1]
            if not re.match(r"(?:mình|tôi)\s+(?:thích|muốn|mong)\b", style, re.I):
                updates["response_style"] = style
            break

    drink_patterns = (
        r"\bđồ\s+uống\s+yêu\s+thích"
        r"(?:\s+của\s+(?:mình|tôi))?\s*(?:là|:)\s*"
        r"(?P<value>[^,.;!?\n]+)",
        r"\b(?:mình|tôi)\s+(?:vẫn\s+)?uống\s+"
        r"(?P<value>[^,.;!?\n]+?)(?:\s+như\s+cũ)?$",
    )
    for pattern in drink_patterns:
        drink_match = _first_fact(pattern, message)
        if drink_match:
            updates["favorite_drink"] = drink_match[1]
            break

    food_match = _first_fact(
        r"\b(?:món\s+ăn\s+yêu\s+thích|món\s+ruột)"
        r"(?:\s+của\s+(?:mình|tôi))?\s*(?:là|:)?\s+"
        r"(?P<value>[^,.;!?\n]+)",
        message,
    )
    if food_match:
        updates["favorite_food"] = food_match[1]

    pet_match = _first_fact(
        r"\b(?:mình|tôi)\s+(?:đang\s+)?nuôi\s+"
        r"(?P<value>[^,.;!?\n]+)",
        message,
    )
    if pet_match:
        pet = pet_match[1]
        updates["pet"] = re.sub(
            r"^(?:một\s+)?(?:con|bé|chú)\s+",
            "",
            pet,
            flags=re.IGNORECASE,
        )

    interest_match = _first_fact(
        r"\b(?:mình|tôi)\s+(?:rất\s+)?"
        r"(?:(?:đang|vẫn)\s+)*(?:thích|quan\s+tâm\s+(?:nhiều\s+)?(?:đến|tới)|"
        r"yêu\s+thích)\s+"
        r"(?P<value>[^.!?\n]+)",
        message,
    )
    if interest_match:
        interests = interest_match[1]
        if not re.match(
            r"(?:cách\s+giải\s+thích|câu\s+trả\s+lời|việc\s+trả\s+lời|"
            r"trả\s+lời|phản\s+hồi|kiểu\s+trả\s+lời)\b",
            interests,
            flags=re.IGNORECASE,
        ):
            updates["interests"] = interests

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Build a deterministic, bounded summary from the most recent old messages."""
    if max_items < 1:
        raise ValueError("max_items must be positive")

    lines: list[str] = []
    for message in messages[-max_items:]:
        role = message.get("role", "unknown").strip() or "unknown"
        content = re.sub(r"\s+", " ", message.get("content", "")).strip()
        if not content:
            continue
        if len(content) > 360:
            content = f"{content[:220].rstrip()} … {content[-120:].lstrip()}"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _bound_summary(summary: str, max_characters: int) -> str:
    if len(summary) <= max_characters:
        return summary
    if max_characters < 32:
        return summary[-max_characters:]
    start_length = max_characters // 3
    end_length = max_characters - start_length - 5
    return f"{summary[:start_length].rstrip()} ... {summary[-end_length:].lstrip()}"


@dataclass
class CompactMemoryManager:
    """Keep recent messages in full and summarize older thread history."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, ThreadMemory] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threshold_tokens < 1:
            raise ValueError("threshold_tokens must be positive")
        if self.keep_messages < 0:
            raise ValueError("keep_messages must be non-negative")

    def append(self, thread_id: str, role: str, content: str) -> None:
        if not thread_id or not thread_id.strip():
            raise ValueError("thread_id must not be empty")
        normalized_role = role.strip().lower()
        if normalized_role not in {"user", "assistant", "system"}:
            raise ValueError(f"unsupported message role: {role!r}")

        thread = self.state.setdefault(
            thread_id,
            {"messages": [], "summary": "", "compactions": 0},
        )
        thread["messages"].append({"role": normalized_role, "content": content})

        if self._context_tokens(thread) <= self.threshold_tokens:
            return

        messages = thread["messages"]
        if len(messages) <= self.keep_messages:
            return

        compact_count = len(messages) - self.keep_messages
        old_messages = messages[:compact_count]
        thread["messages"] = messages[compact_count:]

        new_summary = summarize_messages(old_messages)
        if thread["summary"]:
            new_summary = f"{thread['summary']}\n{new_summary}"
        thread["summary"] = _bound_summary(
            new_summary,
            max_characters=max(160, self.threshold_tokens * 2),
        )
        thread["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self.state.get(
            thread_id,
            {"messages": [], "summary": "", "compactions": 0},
        )
        return {
            "messages": [message.copy() for message in thread["messages"]],
            "summary": thread["summary"],
            "compactions": thread["compactions"],
        }

    def compaction_count(self, thread_id: str) -> int:
        thread = self.state.get(thread_id)
        return thread["compactions"] if thread else 0

    @staticmethod
    def _context_tokens(thread: ThreadMemory) -> int:
        message_text = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in thread["messages"]
        )
        return estimate_tokens(f"{thread['summary']}\n{message_text}")
