import os

# Ollama connection
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Default model — qwen2.5-coder:7b is ideal for coding tasks
DEFAULT_MODEL: str = os.environ.get("ASTRA_MODEL", "qwen2.5-coder:7b")

# Agent loop safety limit
MAX_ITERATIONS: int = int(os.environ.get("ASTRA_MAX_ITER", "50"))

# Bash tool timeout (seconds)
BASH_TIMEOUT: int = int(os.environ.get("ASTRA_BASH_TIMEOUT", "30"))
