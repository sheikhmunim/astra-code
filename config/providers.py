"""
LLM factory — builds the right LangChain chat model based on active provider config.
Returns (llm, tool_mode) where tool_mode is "react" (Ollama) or "native" (cloud APIs).
"""
from __future__ import annotations
from typing import Tuple, Any

# Models known to support native function calling via Groq
GROQ_TOOL_MODELS = {
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-groq-70b-8192-tool-use-preview",
    "llama3-groq-8b-8192-tool-use-preview",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
}

PROVIDER_DISPLAY = {
    "ollama": "Ollama (local)",
    "anthropic": "Anthropic (Claude)",
    "openai": "OpenAI",
    "groq": "Groq",
}

MODEL_SUGGESTIONS = {
    "ollama": ["qwen2.5-coder:7b", "qwen2.5-coder:14b", "llama3.1:8b", "deepseek-coder-v2:16b"],
    "anthropic": ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5-20251001"],
    "openai": ["gpt-4o", "gpt-4o-mini", "o3-mini"],
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
}


def build_llm(cfg: dict) -> Tuple[Any, str]:
    """
    Build LangChain chat model from config.
    Returns (llm, tool_mode) — tool_mode is "react" or "native".
    """
    from config.manager import get_active_provider, get_provider_cfg

    provider = get_active_provider(cfg)
    pcfg = get_provider_cfg(cfg, provider)

    if provider == "ollama":
        return _build_ollama(pcfg)
    elif provider == "anthropic":
        return _build_anthropic(pcfg)
    elif provider == "openai":
        return _build_openai(pcfg)
    elif provider == "groq":
        return _build_groq(pcfg)
    else:
        raise ValueError(f"Unknown provider: {provider!r}")


def _build_ollama(pcfg: dict):
    from langchain_ollama import ChatOllama
    llm = ChatOllama(
        model=pcfg.get("model", "qwen2.5-coder:7b"),
        base_url=pcfg.get("base_url", "http://localhost:11434"),
        temperature=0,
    )
    return llm, "react"


def _build_anthropic(pcfg: dict):
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError("Run: pip install langchain-anthropic")

    api_key = pcfg.get("api_key") or ""
    if not api_key:
        raise ValueError("Anthropic API key not set. Run: astra config")

    llm = ChatAnthropic(
        model=pcfg.get("model", "claude-sonnet-4-6"),
        api_key=api_key,
        temperature=0,
    )
    return llm, "native"


def _build_openai(pcfg: dict):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError("Run: pip install langchain-openai")

    api_key = pcfg.get("api_key") or ""
    if not api_key:
        raise ValueError("OpenAI API key not set. Run: astra config")

    llm = ChatOpenAI(
        model=pcfg.get("model", "gpt-4o"),
        api_key=api_key,
        temperature=0,
    )
    return llm, "native"


def _build_groq(pcfg: dict):
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        raise ImportError("Run: pip install langchain-groq")

    api_key = pcfg.get("api_key") or ""
    if not api_key:
        raise ValueError("Groq API key not set. Run: astra config")

    model = pcfg.get("model", "llama-3.3-70b-versatile")
    tool_mode = "native" if model in GROQ_TOOL_MODELS else "react"

    llm = ChatGroq(
        model=model,
        api_key=api_key,
        temperature=0,
    )
    return llm, tool_mode
