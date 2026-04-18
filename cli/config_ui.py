"""
Numbered menu config — no free-text guessing, no LLM involvement.
"""
import click
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

from config.manager import (
    load_config, save_config,
    get_active_provider, get_provider_cfg,
    set_active_provider, set_provider_field,
    mask_key,
)
from config.providers import PROVIDER_DISPLAY, MODEL_SUGGESTIONS

console = Console()

PROVIDERS = ["ollama", "anthropic", "openai", "groq", "minmax"]
CANCEL_WORDS = {"exit", "quit", "cancel", "q", "back", ""}


def show_table(cfg: dict):
    active = get_active_provider(cfg)
    table = Table(border_style="cyan", show_lines=True, expand=False)
    table.add_column("#",        style="bold cyan", width=3)
    table.add_column("Provider", style="bold",      min_width=12)
    table.add_column("Model",                        min_width=24)
    table.add_column("API Key / URL",                min_width=20)
    table.add_column("",         justify="center",  width=3)

    for i, provider in enumerate(PROVIDERS, 1):
        pcfg   = get_provider_cfg(cfg, provider)
        model  = pcfg.get("model", "—")
        key    = pcfg.get("base_url", "localhost:11434") if provider == "ollama" else mask_key(pcfg.get("api_key", ""))
        active_mark = "[bold green]✓[/bold green]" if provider == active else ""
        table.add_row(str(i), PROVIDER_DISPLAY.get(provider, provider), model, key, active_mark)

    console.print(table)


def run_configure() -> dict | None:
    """
    Numbered menu config wizard.
    Returns updated cfg if something changed, else None.
    """
    cfg = load_config()
    console.print()
    show_table(cfg)
    console.print(
        "\n[dim]Enter a number to switch provider, "
        "[bold]delete <number>[/bold] to remove an API key, "
        "or leave blank to cancel.[/dim]"
    )

    raw = _ask("\n> ").strip()
    if not raw or raw.lower() in CANCEL_WORDS:
        return None

    # ── delete by number or name ──────────────────────────────────────────────
    low = raw.lower()
    if low.startswith("delete "):
        arg = low.split(None, 1)[1].strip()
        target = _resolve_provider(arg)
        if target is None:
            return None
        return _handle_delete(cfg, target)

    # ── select provider by number ─────────────────────────────────────────────
    provider = _resolve_provider(raw)
    if provider is None:
        return None

    # ── pick a model ──────────────────────────────────────────────────────────
    model = _pick_model(cfg, provider)
    if model is None:
        return None

    # ── Auto-install provider package if missing ──────────────────────────────
    if provider != "ollama":
        _ensure_provider_package(provider)

    # ── API key (always shown, update optional) ───────────────────────────────
    if provider != "ollama":
        pcfg = get_provider_cfg(cfg, provider)
        existing = pcfg.get("api_key", "")
        if existing:
            console.print(f"\n[dim]Current API key: {mask_key(existing)}[/dim]")
            console.print("[dim]Press Enter to keep it, or paste a new key to replace:[/dim]")
        else:
            console.print(f"\n[yellow]API key required for {PROVIDER_DISPLAY.get(provider, provider)}.[/yellow]")
        key = _prompt_api_key()
        if key:
            set_provider_field(cfg, provider, "api_key", key)
        elif not existing:
            console.print("[dim]Cancelled — no API key set.[/dim]")
            return None

    set_provider_field(cfg, provider, "model", model)
    set_active_provider(cfg, provider)
    save_config(cfg)

    console.print(
        f"\n[green]✓ Switched to[/green] "
        f"[cyan]{PROVIDER_DISPLAY.get(provider, provider)}[/cyan] / "
        f"[bold]{model}[/bold]\n"
    )
    show_table(load_config())
    return load_config()


# ── helpers ───────────────────────────────────────────────────────────────────

def _resolve_provider(text: str) -> str | None:
    """Accept a number (1-4) or provider name. Returns provider key or None."""
    text = text.strip()
    if text.isdigit():
        idx = int(text) - 1
        if 0 <= idx < len(PROVIDERS):
            return PROVIDERS[idx]
        console.print(f"[red]Invalid number. Pick 1–{len(PROVIDERS)}.[/red]")
        return None
    if text.lower() in PROVIDERS:
        return text.lower()
    console.print(f"[red]Unknown provider: {text!r}[/red]")
    return None


def _pick_model(cfg: dict, provider: str) -> str | None:
    """Show numbered model menu and return chosen model, or None to cancel."""
    suggestions = MODEL_SUGGESTIONS.get(provider, [])
    current     = get_provider_cfg(cfg, provider).get("model", "")

    console.print(f"\n[bold]Model[/bold] [dim](current: {current})[/dim]")
    for i, m in enumerate(suggestions, 1):
        marker = " [green]← current[/green]" if m == current else ""
        console.print(f"  [cyan]{i}[/cyan]. {m}{marker}")
    console.print(f"  [cyan]{len(suggestions) + 1}[/cyan]. Enter custom model name")
    console.print("  [dim]Leave blank or type 'cancel' to go back.[/dim]")

    raw = _ask("> ").strip()
    if not raw or raw.lower() in CANCEL_WORDS:
        console.print("[dim]Cancelled.[/dim]")
        return None

    # Number selection
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(suggestions):
            return suggestions[idx]
        if idx == len(suggestions):          # "custom" option
            return _ask_custom_model(current)
        console.print(f"[red]Invalid number.[/red]")
        return None

    # Typed a model name directly
    return raw


def _ask_custom_model(current: str) -> str | None:
    console.print("[dim]  Type the exact model name (e.g. claude-opus-4-6). Blank to cancel.[/dim]")
    val = _ask("  Model name: ").strip()
    if not val or val.lower() in CANCEL_WORDS:
        console.print("[dim]Cancelled.[/dim]")
        return None
    return val


def _handle_delete(cfg: dict, provider: str) -> dict | None:
    if provider == "ollama":
        console.print("[yellow]Ollama has no API key to delete.[/yellow]")
        return None
    set_provider_field(cfg, provider, "api_key", "")
    if get_active_provider(cfg) == provider:
        set_active_provider(cfg, "ollama")
        console.print("[dim]Fell back to Ollama.[/dim]")
    save_config(cfg)
    console.print(f"[green]✓ Deleted API key for {PROVIDER_DISPLAY.get(provider, provider)}.[/green]")
    return load_config()


def _prompt_api_key() -> str:
    """Paste-friendly API key prompt using click (supports Ctrl+V on Windows)."""
    console.print("  [dim]Paste or type your API key, then press Enter:[/dim]")
    try:
        return click.prompt("  API key", hide_input=True, default="", show_default=False).strip()
    except (click.Abort, KeyboardInterrupt):
        return ""


_PROVIDER_PACKAGES = {
    "anthropic": "langchain-anthropic",
    "openai":    "langchain-openai",
    "groq":      "langchain-groq",
    "minmax":    "langchain-openai",
}

_PROVIDER_IMPORTS = {
    "anthropic": "langchain_anthropic",
    "openai":    "langchain_openai",
    "groq":      "langchain_groq",
    "minmax":    "langchain-openai",
}


def _ensure_provider_package(provider: str) -> None:
    """Auto-install the provider's package if it isn't already installed."""
    import importlib, subprocess, sys

    module = _PROVIDER_IMPORTS.get(provider)
    package = _PROVIDER_PACKAGES.get(provider)
    if not module or not package:
        return

    try:
        importlib.import_module(module)
        return  # already installed
    except ImportError:
        pass

    console.print(f"\n[dim]Installing [bold]{package}[/bold]...[/dim]")
    with console.status(f"[cyan]pip install {package}[/cyan]", spinner="dots"):
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True, text=True
        )

    if result.returncode == 0:
        console.print(f"[green]✓ Installed {package}[/green]")
    else:
        console.print(f"[red]Failed to install {package}:[/red]\n{result.stderr.strip()}")


def _ask(prompt: str) -> str:
    """Plain input — never reaches the LLM."""
    try:
        return console.input(prompt)
    except (KeyboardInterrupt, EOFError):
        return ""
