from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


def make_config(tmp_path: Path):
    """Create an isolated offline config with a low compaction threshold."""
    config = load_config(tmp_path)
    config.state_dir = tmp_path / "test-state"
    config.state_dir.mkdir(parents=True, exist_ok=True)
    config.compact_threshold_tokens = 120
    config.compact_keep_messages = 4
    return config


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)

    result = agent.reply("user-1", "profile-thread", "Mình tên là Linh.")
    profile_path = agent.profile_store.path_for("user-1")

    assert profile_path.is_file()
    assert "Linh" in agent.profile_store.read_text("user-1")
    assert result["memory_file_size"] == profile_path.stat().st_size
    assert agent.profile_store.edit_text("user-1", "Linh", "Linh Nguyễn")
    assert "Linh Nguyễn" in agent.profile_store.read_text("user-1")
    assert not agent.profile_store.edit_text("user-1", "missing", "replacement")


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)
    long_message = "Mình đang đọc về memory architecture. " * 12

    for _ in range(12):
        agent.reply("user-1", "long-thread", long_message)

    context = agent.compact_memory.context("long-thread")
    assert agent.compaction_count("long-thread") > 0
    assert len(context["messages"]) <= config.compact_keep_messages
    assert context["summary"]


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config=config, force_offline=True)
    baseline = BaselineAgent(config=replace(config), force_offline=True)

    advanced.reply("user-1", "advanced-training", "Mình tên là Linh.")
    baseline.reply("user-1", "baseline-training", "Mình tên là Linh.")

    advanced_answer = advanced.reply(
        "user-1",
        "advanced-recall",
        "Mình tên gì?",
    )["response"]
    baseline_answer = baseline.reply(
        "user-1",
        "baseline-recall",
        "Mình tên gì?",
    )["response"]

    assert "Linh" in advanced_answer
    assert "Linh" not in baseline_answer
    assert "chưa có thông tin" in baseline_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config=config, force_offline=True)
    baseline = BaselineAgent(config=replace(config), force_offline=True)
    long_messages = [
        (
            f"Đoạn hội thoại {index}: "
            "Mình đang so sánh chi phí lưu toàn bộ lịch sử với memory compact. "
            "Hãy ghi nhận các chi tiết về thiết kế, benchmark, token và trade-off. "
        )
        * 5
        for index in range(18)
    ]

    for message in long_messages:
        advanced.reply("user-1", "advanced-long", message)
        baseline.reply("user-1", "baseline-long", message)

    assert advanced.compaction_count("advanced-long") > 0
    assert (
        advanced.prompt_token_usage("advanced-long")
        < baseline.prompt_token_usage("baseline-long")
    )
