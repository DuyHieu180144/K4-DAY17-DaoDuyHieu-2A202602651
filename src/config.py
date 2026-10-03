from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig


SUPPORTED_PROVIDERS = ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter")
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-3-5-haiku-latest",
    "ollama": "llama3.2",
    "openrouter": "openai/gpt-4o-mini",
}


@dataclass
class LabConfig:
    """Shared paths, compact-memory settings, and model configurations."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load lab settings from the environment and an optional root-level .env."""
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError as exc:
        if exc.name != "dotenv":
            raise
    else:
        load_dotenv(dotenv_path=root / ".env", override=False)

    def env_value(name: str, default: str | None = None) -> str | None:
        value = os.getenv(name)
        return value.strip() if value and value.strip() else default

    def config_path(name: str, default: Path) -> Path:
        value = env_value(name)
        path = Path(value).expanduser() if value else default
        if not path.is_absolute():
            path = root / path
        return path.resolve()

    def positive_int(name: str, default: int, *, allow_zero: bool = False) -> int:
        value = env_value(name)
        try:
            parsed = int(value) if value is not None else default
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {value!r}") from exc
        minimum = 0 if allow_zero else 1
        if parsed < minimum:
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be {qualifier}, got {parsed}")
        return parsed

    def provider_config(prefix: str, provider: str, model_name: str) -> ProviderConfig:
        if provider not in SUPPORTED_PROVIDERS:
            choices = ", ".join(SUPPORTED_PROVIDERS)
            raise ValueError(f"{prefix} provider must be one of: {choices}; got {provider!r}")

        api_keys = {
            "openai": env_value("OPENAI_API_KEY"),
            "custom": env_value("CUSTOM_API_KEY"),
            "gemini": env_value("GEMINI_API_KEY", env_value("GOOGLE_API_KEY")),
            "anthropic": env_value("ANTHROPIC_API_KEY"),
            "ollama": env_value("OLLAMA_API_KEY"),
            "openrouter": env_value("OPENROUTER_API_KEY"),
        }
        base_urls = {
            "openai": env_value("OPENAI_BASE_URL"),
            "custom": env_value("CUSTOM_BASE_URL"),
            "gemini": env_value("GEMINI_BASE_URL"),
            "anthropic": env_value("ANTHROPIC_BASE_URL"),
            "ollama": env_value("OLLAMA_BASE_URL", "http://localhost:11434"),
            "openrouter": env_value(
                "OPENROUTER_BASE_URL",
                "https://openrouter.ai/api/v1",
            ),
        }
        api_key = env_value(f"{prefix}_API_KEY", api_keys[provider])
        base_url = env_value(f"{prefix}_BASE_URL", base_urls[provider])
        temperature_value = env_value(f"{prefix}_TEMPERATURE", "0")
        try:
            temperature = float(temperature_value)
        except ValueError as exc:
            raise ValueError(
                f"{prefix}_TEMPERATURE must be a number, got {temperature_value!r}"
            ) from exc
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError(
                f"{prefix}_TEMPERATURE must be a finite, non-negative number"
            )

        return ProviderConfig(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
        )

    data_dir = config_path("DATA_DIR", root / "data")
    state_dir = config_path("STATE_DIR", root / "state")
    state_dir.mkdir(parents=True, exist_ok=True)

    model_provider = (env_value("LLM_PROVIDER", "openai") or "openai").lower()
    model_name = env_value("LLM_MODEL", DEFAULT_MODELS.get(model_provider, ""))
    if not model_name:
        raise ValueError(f"LLM_PROVIDER must be one of: {', '.join(SUPPORTED_PROVIDERS)}")

    judge_provider = (env_value("JUDGE_PROVIDER", model_provider) or model_provider).lower()
    judge_model_name = env_value("JUDGE_MODEL", model_name) or model_name

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=positive_int("COMPACT_THRESHOLD_TOKENS", 1200),
        compact_keep_messages=positive_int(
            "COMPACT_KEEP_MESSAGES",
            6,
            allow_zero=True,
        ),
        model=provider_config("LLM", model_provider, model_name),
        judge_model=provider_config("JUDGE", judge_provider, judge_model_name),
    )
