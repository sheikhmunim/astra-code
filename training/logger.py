"""
Logs (user prompt, agent response, rating) to ~/.astra/training_log.jsonl
for future fine-tuning.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

LOG_FILE = Path.home() / ".astra" / "training_log.jsonl"


def log_example(
    user: str,
    response: str,
    rating: int,
    model: str = "",
    provider: str = "",
) -> None:
    """Append one rated example to the training log."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "provider":  provider,
        "model":     model,
        "user":      user,
        "response":  response,
        "rating":    rating,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def ask_rating() -> int | None:
    """
    Show a compact rating bar and wait for a single keypress.
    Returns 1-5, or None if the user pressed Enter to skip.
    Uses msvcrt on Windows, tty/termios on Unix.
    """
    from rich.console import Console
    _c = Console(highlight=False)
    _c.print("\n  [dim]Rate this response:[/dim] "
             "[cyan]1[/cyan] · [cyan]2[/cyan] · [cyan]3[/cyan] · "
             "[cyan]4[/cyan] · [cyan]5[/cyan]   "
             "[dim](enter to skip)[/dim]",
             end="")

    ch = _getch()
    if ch in ("1", "2", "3", "4", "5"):
        _c.print(f"  [dim]→ {ch} saved[/dim]")
        return int(ch)

    _c.print()   # blank line on skip / any other key
    return None


# ── Single-keypress helpers ───────────────────────────────────────────────────

def _getch() -> str:
    """Read one character without requiring Enter."""
    if sys.platform == "win32":
        import msvcrt
        ch = msvcrt.getwch()
        return ch
    else:
        import tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch
