from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Load and validate the conversation benchmark JSON."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise OSError(f"Unable to read benchmark dataset {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in benchmark dataset {path}: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError(f"Benchmark dataset {path} must contain a JSON list")

    for index, conversation in enumerate(payload):
        if not isinstance(conversation, dict):
            raise ValueError(f"Conversation {index} in {path} must be a JSON object")
        conversation_id = conversation.get("id")
        user_id = conversation.get("user_id")
        turns = conversation.get("turns")
        recall_questions = conversation.get("recall_questions", [])
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise ValueError(f"Conversation {index} in {path} has no valid id")
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError(f"Conversation {conversation_id!r} in {path} has no valid user_id")
        if not isinstance(turns, list) or any(not isinstance(turn, str) for turn in turns):
            raise ValueError(f"Conversation {conversation_id!r} must have a list of string turns")
        if not isinstance(recall_questions, list):
            raise ValueError(
                f"Conversation {conversation_id!r} recall_questions must be a list"
            )
        for question_index, item in enumerate(recall_questions):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Recall question {question_index} in {conversation_id!r} must be an object"
                )
            if not isinstance(item.get("question"), str):
                raise ValueError(
                    f"Recall question {question_index} in {conversation_id!r} needs a string question"
                )
            expected = item.get("expected_contains")
            if not isinstance(expected, list) or any(
                not isinstance(fact, str) for fact in expected
            ):
                raise ValueError(
                    f"Recall question {question_index} in {conversation_id!r} needs "
                    "a list of expected_contains strings"
                )
    return payload


def recall_points(answer: str, expected: list[str]) -> float:
    """Score a recall answer as 0, 0.5, or 1 for none, partial, or full recall."""
    if not expected:
        return 1.0 if answer.strip() else 0.0
    normalized_answer = answer.casefold()
    matched = sum(1 for fact in expected if fact.casefold() in normalized_answer)
    if matched == len(expected):
        return 1.0
    return 0.5 if matched else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Estimate offline response quality from relevance and a non-empty response."""
    if not answer.strip():
        return 0.0
    recall = (
        sum(1 for fact in expected if fact.casefold() in answer.casefold()) / len(expected)
        if expected
        else 1.0
    )
    return round(0.8 * recall + 0.2, 4)


def run_agent_benchmark(
    agent_name: str,
    agent: Any,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    """Run training turns and cross-thread recall questions for one agent."""
    del config  # Agent instances already carry their configured state/data paths.
    user_ids = {conversation["user_id"] for conversation in conversations}
    initial_memory_bytes = _profile_bytes(agent, user_ids)
    thread_ids: set[str] = set()
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conversation in conversations:
        user_id = conversation["user_id"]
        training_thread = f"benchmark:{conversation['id']}"
        thread_ids.add(training_thread)
        for turn in conversation["turns"]:
            agent.reply(user_id, training_thread, turn)

        for index, recall_question in enumerate(conversation.get("recall_questions", [])):
            recall_thread = f"benchmark:{conversation['id']}:recall:{index}"
            thread_ids.add(recall_thread)
            result = agent.reply(user_id, recall_thread, recall_question["question"])
            answer = result.get("response")
            if not isinstance(answer, str):
                raise TypeError(
                    f"{agent_name} returned a non-string response for "
                    f"{conversation['id']} recall question {index}"
                )
            expected = recall_question["expected_contains"]
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(thread_id) for thread_id in thread_ids),
        prompt_tokens_processed=sum(
            agent.prompt_token_usage(thread_id) for thread_id in thread_ids
        ),
        recall_score=_average(recall_scores),
        response_quality=_average(quality_scores),
        memory_growth_bytes=max(0, _profile_bytes(agent, user_ids) - initial_memory_bytes),
        compactions=sum(agent.compaction_count(thread_id) for thread_id in thread_ids),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a readable Markdown table."""
    headers = (
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    )
    table_rows = [
        (
            row.agent_name,
            str(row.agent_tokens_only),
            str(row.prompt_tokens_processed),
            f"{row.recall_score:.2f}",
            f"{row.response_quality:.2f}",
            str(row.memory_growth_bytes),
            str(row.compactions),
        )
        for row in rows
    ]
    widths = [
        max(len(header), *(len(row[column]) for row in table_rows))
        for column, header in enumerate(headers)
    ]

    def format_row(values: tuple[str, ...]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        ) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join(
        [format_row(headers), separator, *(format_row(row) for row in table_rows)]
    )


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _profile_bytes(agent: Any, user_ids: set[str]) -> int:
    profile_store = getattr(agent, "profile_store", None)
    if profile_store is None:
        return 0
    return sum(profile_store.file_size(user_id) for user_id in user_ids)


def _run_suite(
    conversations: list[dict[str, Any]],
    config: LabConfig,
    suite_name: str,
    state_root: Path,
) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for agent_name, agent_type in (
        ("Baseline", BaselineAgent),
        ("Advanced", AdvancedAgent),
    ):
        agent_config = replace(
            config,
            state_dir=state_root / suite_name / agent_name.lower(),
        )
        agent_config.state_dir.mkdir(parents=True, exist_ok=True)
        agent = agent_type(config=agent_config, force_offline=True)
        rows.append(run_agent_benchmark(agent_name, agent, conversations, agent_config))
    return rows


def main() -> None:
    """Run the standard and long-context benchmark deterministically offline."""
    config = load_config(Path(__file__).resolve().parent.parent)
    standard = load_conversations(config.data_dir / "conversations.json")
    stress = load_conversations(config.data_dir / "advanced_long_context.json")

    with tempfile.TemporaryDirectory(prefix="memory-agent-benchmark-") as temp_dir:
        state_root = Path(temp_dir)
        print("Standard Benchmark")
        print(format_rows(_run_suite(standard, config, "standard", state_root)))
        print()
        print("Long-Context Stress Benchmark")
        print(format_rows(_run_suite(stress, config, "stress", state_root)))


if __name__ == "__main__":
    main()
