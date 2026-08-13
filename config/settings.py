import os

# Ollama connection
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Default model — phi4-mini is small, fast, and low on RAM requirements
DEFAULT_MODEL: str = os.environ.get("ASTRA_MODEL", "phi4-mini")

# Agent loop safety limit
MAX_ITERATIONS: int = int(os.environ.get("ASTRA_MAX_ITER", "20"))

# Bash tool timeout (seconds)
BASH_TIMEOUT: int = int(os.environ.get("ASTRA_BASH_TIMEOUT", "120"))
