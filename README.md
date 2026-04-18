# Astra Code

A Claude Code-style coding agent for your terminal — runs on local LLMs via Ollama or cloud APIs (Anthropic, OpenAI, Groq).

## Install

```bash
pip install astra-code
```

Astra will automatically detect if Ollama is installed and guide you through setup on first run.

## Usage

```bash
astra
```

That's it. Astra picks up your current working directory and is ready to help with coding tasks.

## What it can do

- Read, write, and edit files
- Run shell commands
- Search codebases with glob and grep
- Work with any local model via Ollama
- Switch to cloud APIs (Anthropic, OpenAI, Groq) with `/configure`

## Configure

Type `/configure` inside the app to switch providers, set API keys, or change models.

```
┌──────────────────────────────────────────┐
│  1  Ollama (local)    qwen2.5-coder:7b  ✓│
│  2  Anthropic         claude-sonnet-4-6  │
│  3  OpenAI            gpt-4o             │
│  4  Groq              llama-3.3-70b      │
└──────────────────────────────────────────┘
```

## Cloud providers

Install the extras you need:

```bash
pip install "astra-code[anthropic]"
pip install "astra-code[openai]"
pip install "astra-code[groq]"
pip install "astra-code[all]"   # everything
```

## Local models (Ollama)

Install [Ollama](https://ollama.com) then pull a model:

```bash
ollama pull qwen2.5-coder:7b    # recommended — fast, good at code
ollama pull qwen2.5-coder:14b   # slower, more capable
ollama pull deepseek-coder-v2:16b
```

## Requirements

- Python 3.10+
- Ollama (auto-detected on first run) or a cloud API key
