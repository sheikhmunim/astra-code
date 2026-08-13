import os

# Ollama connection
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Default model — phi4-mini is small, fast, and low on RAM requirements
DEFAULT_MODEL: str = os.environ.get("ASTRA_MODEL", "phi4-mini")

# Agent loop safety limit
MAX_ITERATIONS: int = int(os.environ.get("ASTRA_MAX_ITER", "20"))

# Bash tool timeout (seconds)
BASH_TIMEOUT: int = int(os.environ.get("ASTRA_BASH_TIMEOUT", "120"))

# Ollama context window (tokens). Ollama's own default is only 2048, which is
# too small for a system prompt + tool descriptions + history — causes silent
# truncation, which leads to bad/incomplete responses and extra retry loops.
OLLAMA_NUM_CTX: int = int(os.environ.get("ASTRA_OLLAMA_NUM_CTX", "8192"))
