import threading
import random
from rich.console import Console
from rich.syntax import Syntax
from rich.rule import Rule
from rich.markup import escape
from rich.text import Text
from rich.panel import Panel
from rich.align import Align
from rich.columns import Columns

from agent.parser import parse_react_action, parse_final_answer, parse_thought

console = Console(highlight=False, emoji=True)

# ── Cycling words (shown while model works, like Claude Code) ─────────────────

_THINKING_WORDS = [
    "Thinking", "Analyzing", "Planning", "Considering",
    "Reasoning", "Processing", "Evaluating", "Reflecting",
    "Exploring", "Reviewing", "Examining", "Determining",
    "Understanding", "Calculating", "Investigating",
]

_TOOL_WORDS = {
    "bash":        ["Running",   "Executing",  "Processing"],
    "read_file":   ["Reading",   "Loading",    "Scanning"],
    "write_file":  ["Writing",   "Saving",     "Creating"],
    "edit_file":   ["Editing",   "Updating",   "Modifying"],
    "glob_search": ["Searching", "Scanning",   "Globbing"],
    "grep_search": ["Searching", "Matching",   "Grepping"],
}

# Tool icons (Claude Code uses ⎿)
_TOOL_ICONS = {
    "bash":        "❯",
    "read_file":   "⎿",
    "write_file":  "⎿",
    "edit_file":   "⎿",
    "glob_search": "⎿",
    "grep_search": "⎿",
}

SPINNER = "dots"


# ── Banner ────────────────────────────────────────────────────────────────────

LOGO = """\
 ▄▄▄· .▄▄ · ▄▄▄▄▄▄▄▄   ▄▄▄·
▐█ ▀█ ▐█ ▀. •██  ▀▄ █·▐█ ▀█
▄█▀▀█ ▄▀▀▀█▄ ▐█.▪▐▀▀▄ ▄█▀▀█
▐█ ▪▐▌▐█▄▪▐█ ▐█▌·▐█•█▌▐█ ▪▐▌
 ▀  ▀  ▀▀▀▀  ▀▀▀ .▀  ▀ ▀  ▀"""


def print_banner(model: str, cwd: str):
    console.print()
    console.print(Align.center(Text(LOGO, style="bold cyan")))
    console.print()
    console.print(Panel(
        f"  [dim]model[/dim]   [green]{model}[/green]\n"
        f"  [dim]cwd  [/dim]   [yellow]{cwd}[/yellow]\n\n"
        f"  [dim]/help · /configure · /status · exit[/dim]",
        border_style="cyan",
        expand=False,
        padding=(0, 2),
    ))
    console.print()


# ── Tool display (compact, Claude Code style) ─────────────────────────────────

def print_tool_call(tool_name: str, tool_args: dict):
    icon = _TOOL_ICONS.get(tool_name, "⎿")
    # Show the most relevant arg inline
    key_arg = _key_arg(tool_name, tool_args)
    console.print(
        f"  [dim cyan]{icon}[/dim cyan] [bold]{tool_name}[/bold]"
        + (f"[dim]({escape(key_arg)})[/dim]" if key_arg else ""),
        highlight=False,
    )


def _key_arg(tool_name: str, args: dict) -> str:
    """Pick the single most informative arg to show inline."""
    priority = {
        "bash":        ["command"],
        "read_file":   ["file_path"],
        "write_file":  ["file_path"],
        "edit_file":   ["file_path"],
        "glob_search": ["pattern"],
        "grep_search": ["pattern"],
    }
    for key in priority.get(tool_name, []):
        if key in args:
            val = str(args[key])
            return val if len(val) <= 60 else val[:57] + "..."
    return ""


def print_tool_result(result: str):
    lines = result.strip().splitlines()
    # Show up to 8 lines, then truncate
    shown = lines[:8]
    rest  = len(lines) - 8
    for ln in shown:
        console.print(f"  [dim]{escape(ln)}[/dim]", highlight=False)
    if rest > 0:
        console.print(f"  [dim]… {rest} more lines[/dim]")
    console.print()


# ── Cycling spinner ───────────────────────────────────────────────────────────

class _CyclingSpinner:
    """
    Spinner that rotates through random words every ~2 s,
    exactly like Claude Code does while the model is working.
    """
    def __init__(self, words: list[str], prefix: str = ""):
        self._words  = words
        self._prefix = prefix
        self._idx    = random.randrange(len(words))
        self._status = None
        self._stop   = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._status = console.status(self._text(), spinner=SPINNER)
        self._status.start()
        self._thread = threading.Thread(target=self._cycle, daemon=True)
        self._thread.start()

    def update_words(self, words: list[str], prefix: str = ""):
        self._words  = words
        self._prefix = prefix
        self._idx    = random.randrange(len(words))
        if self._status:
            self._status.update(self._text())

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        if self._status:
            self._status.stop()
            self._status = None

    def _text(self) -> str:
        word = self._words[self._idx % len(self._words)]
        return f"[dim]{self._prefix}{word}…[/dim]"

    def _cycle(self):
        while not self._stop.wait(2.0):
            self._idx = (self._idx + 1) % len(self._words)
            if self._status:
                self._status.update(self._text())


# ── Streaming renderer ────────────────────────────────────────────────────────

class StreamingRenderer:
    def __init__(self, show_tools: bool = True):
        self.show_tools     = show_tools
        self._spinner       = None
        self._line_buf      = ""
        self._full_buf      = ""
        self._in_final      = False
        self._final_started = False
        self._final_printed = 0
        self._action_det    = False

    # ── Spinner ───────────────────────────────────────────────────────────────

    def thinking(self):
        self._reset()
        self._spinner = _CyclingSpinner(_THINKING_WORDS)
        self._spinner.start()

    def running_tool(self, name: str):
        words = _TOOL_WORDS.get(name, ["Running"])
        if self._spinner:
            self._spinner.update_words(words)
        else:
            self._spinner = _CyclingSpinner(words)
            self._spinner.start()

    def _stop_spinner(self):
        if self._spinner:
            self._spinner.stop()
            self._spinner = None

    def _reset(self):
        self._line_buf      = ""
        self._full_buf      = ""
        self._in_final      = False
        self._final_started = False
        self._final_printed = 0
        self._action_det    = False

    # ── Token streaming ───────────────────────────────────────────────────────

    def on_agent_token(self, token):
        if isinstance(token, list):
            token = "".join(b.get("text","") if isinstance(b, dict) else str(b) for b in token)
        if not isinstance(token, str):
            token = str(token)

        self._stop_spinner()
        self._full_buf += token
        self._line_buf += token

        while "\n" in self._line_buf:
            line, self._line_buf = self._line_buf.split("\n", 1)
            self._render_line(line)
            self._final_printed = 0

        if self._in_final and self._line_buf:
            new = self._line_buf[self._final_printed:]
            if new:
                console.print(new, end="", highlight=False)
                self._final_printed = len(self._line_buf)

    def _render_line(self, line: str):
        stripped = line.strip()
        low      = stripped.lower()

        if low.startswith("thought:"):
            if self.show_tools:
                console.print(f"[dim italic]  {escape(stripped[8:].strip())}[/dim italic]")

        elif low.startswith("action:"):
            self._action_det = True  # actual tool display happens in on_tool_run

        elif low.startswith("action input:"):
            pass

        elif low.startswith("final answer:"):
            self._in_final = True
            if not self._final_started:
                console.print()
                self._final_started = True
            start = stripped[13:].strip()
            if start:
                console.print(start, end="", highlight=False)

        elif self._in_final:
            _render_response_line(line)

        elif stripped and not low.startswith(("observation:", "action input:")):
            console.print(line, highlight=False)

    # ── Tool display ──────────────────────────────────────────────────────────

    def on_tool_run(self, tool_name: str, tool_args: dict):
        if self.show_tools:
            print_tool_call(tool_name, tool_args)
        self.running_tool(tool_name)

    def on_tool_result(self, result: str):
        self._stop_spinner()
        if self.show_tools:
            print_tool_result(result)

    # ── Flush ─────────────────────────────────────────────────────────────────

    def flush(self):
        if self._line_buf.strip():
            self._render_line(self._line_buf)
        if self._in_final:
            console.print()
        self._stop_spinner()


# ── Response line renderer (handles code blocks inline) ───────────────────────

_in_code_block   = False
_code_block_lang = "text"
_code_lines      = []


def _render_response_line(line: str):
    global _in_code_block, _code_block_lang, _code_lines

    if line.strip().startswith("```"):
        if not _in_code_block:
            _in_code_block   = True
            _code_block_lang = line.strip()[3:].strip() or "text"
            _code_lines      = []
        else:
            # End of block — render it
            console.print(Syntax(
                "\n".join(_code_lines),
                _code_block_lang,
                theme="monokai",
                line_numbers=False,
                background_color="default",
            ))
            _in_code_block = False
            _code_lines    = []
    elif _in_code_block:
        _code_lines.append(line)
    else:
        console.print(line, highlight=False)
