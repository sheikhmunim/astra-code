"""
Startup checks — runs once when astra launches.
Detects Ollama, offers to install it, and shows model pull commands.
"""
import platform
import subprocess
import sys
from rich.console import Console
from rich.panel import Panel

console = Console(highlight=False)

OLLAMA_DOWNLOAD = {
    "Windows": "https://ollama.com/download/OllamaSetup.exe",
    "Darwin":  "https://ollama.com/download/Ollama-darwin.zip",
    "Linux":   "curl -fsSL https://ollama.com/install.sh | sh",
}

RECOMMENDED_MODELS = [
    ("qwen2.5-coder:7b",      "recommended — fast, great at code"),
    ("qwen2.5-coder:14b",     "more capable, needs ~10GB RAM"),
    ("deepseek-coder-v2:16b", "strongest local coder, needs ~12GB RAM"),
    ("llama3.1:8b",           "general purpose"),
]


def check_ollama() -> bool:
    """
    Check if Ollama is installed and running.
    If not installed: show download instructions and exit.
    If installed but no models pulled: show pull commands.
    Returns True if Ollama is ready to use.
    """
    if not _ollama_installed():
        _show_install_instructions()
        return False

    if not _ollama_running():
        _show_start_instructions()
        return False

    pulled = _list_models()
    if not pulled:
        _show_pull_instructions()
        return False

    return True


# ── Internal checks ───────────────────────────────────────────────────────────

def _ollama_installed() -> bool:
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _ollama_running() -> bool:
    """Check if the Ollama server is reachable on localhost:11434."""
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434", timeout=3)
        return True
    except Exception:
        return False


def _list_models() -> list[str]:
    """Return list of pulled model names."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().splitlines()
        # Skip header line, extract model names
        return [l.split()[0] for l in lines[1:] if l.strip()]
    except Exception:
        return []


# ── Instructions ──────────────────────────────────────────────────────────────

def _show_install_instructions():
    os_name = platform.system()
    link = OLLAMA_DOWNLOAD.get(os_name, "https://ollama.com/download")

    if os_name == "Linux":
        install_line = f"[cyan]{link}[/cyan]"
    else:
        install_line = f"Download from: [cyan]{link}[/cyan]"

    console.print(Panel(
        f"[yellow]Ollama is not installed.[/yellow]\n\n"
        f"{install_line}\n\n"
        f"After installing, run [bold]astra[/bold] again.",
        title="[yellow]Ollama Required[/yellow]",
        border_style="yellow",
        expand=False,
    ))


def _show_start_instructions():
    console.print(Panel(
        "[yellow]Ollama is installed but not running.[/yellow]\n\n"
        "Start it with:\n"
        "  [cyan]ollama serve[/cyan]\n\n"
        "Or on Windows/Mac, open the Ollama app from your system tray.",
        title="[yellow]Start Ollama[/yellow]",
        border_style="yellow",
        expand=False,
    ))


def _show_pull_instructions():
    model_lines = "\n".join(
        f"  [cyan]ollama pull {name}[/cyan]   [dim]{desc}[/dim]"
        for name, desc in RECOMMENDED_MODELS
    )
    console.print(Panel(
        "[yellow]No models found. Pull one to get started:[/yellow]\n\n"
        + model_lines + "\n\n"
        "Then run [bold]astra[/bold] again.",
        title="[yellow]Pull a Model[/yellow]",
        border_style="yellow",
        expand=False,
    ))
