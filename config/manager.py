"""
Manages ~/.astra/config.json — persists provider selection, API keys, and model choices.
"""
import json
import os
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".astra"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "active_provider": "ollama",
    "providers": {
        "ollama": {
            "base_url": "http://localhost:11434",
            "model": "qwen2.5-coder:7b",
        },
        "anthropic": {
            "api_key": "",
            "model": "claude-sonnet-4-6",
        },
        "openai": {
            "api_key": "",
            "model": "gpt-4o",
        },
        "groq": {
            "api_key": "",
            "model": "llama-3.3-70b-versatile",
        },
        "minmax": {
            "api_key": "",
            "model": "minimax/minimax-m2.5"
        },
    },
}


def load_config() -> dict:
    """Load config from disk, filling in any missing keys with defaults."""
    if not CONFIG_FILE.exists():
        return _deep_copy(DEFAULT_CONFIG)

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        # Merge: fill missing keys from defaults without clobbering saved values
        return _deep_merge(DEFAULT_CONFIG, saved)
    except Exception:
        return _deep_copy(DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    """Persist config to ~/.astra/config.json."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def get_active_provider(cfg: dict) -> str:
    return cfg.get("active_provider", "ollama")


def get_provider_cfg(cfg: dict, provider: str | None = None) -> dict:
    provider = provider or get_active_provider(cfg)
    return cfg.get("providers", {}).get(provider, {})


def set_active_provider(cfg: dict, provider: str) -> dict:
    cfg["active_provider"] = provider
    return cfg


def set_provider_field(cfg: dict, provider: str, key: str, value: str) -> dict:
    cfg.setdefault("providers", {}).setdefault(provider, {})[key] = value
    return cfg


def mask_key(key: str) -> str:
    if not key:
        return "[dim](not set)[/dim]"
    if len(key) <= 8:
        return "****"
    return key[:4] + "****" + key[-4:]


def _deep_copy(d: dict) -> dict:
    return json.loads(json.dumps(d))


def _deep_merge(base: dict, override: dict) -> dict:
    result = _deep_copy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
