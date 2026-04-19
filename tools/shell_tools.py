import os
import subprocess
from langchain_core.tools import tool
from config.settings import BASH_TIMEOUT


@tool
def bash(command: str) -> str:
    """Execute a shell command and return stdout, stderr, and exit code.
    Timeout: 30 seconds. For interactive or long-running commands, append & to run in background.
    Returns a formatted string with all output.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=BASH_TIMEOUT,
            cwd=os.getcwd(),
        )
        parts = []
        if result.stdout.strip():
            parts.append(f"STDOUT:\n{result.stdout.rstrip()}")
        if result.stderr.strip():
            parts.append(f"STDERR:\n{result.stderr.rstrip()}")
        parts.append(f"EXIT CODE: {result.returncode}")
        return "\n\n".join(parts) if parts else f"(no output)\nEXIT CODE: {result.returncode}"
    except subprocess.TimeoutExpired:
        return f"ERROR: Command timed out after {BASH_TIMEOUT} seconds.\nConsider running long commands in the background with & or breaking them into smaller steps."
    except Exception as e:
        return f"ERROR executing command: {e}"
