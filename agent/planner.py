"""
Planner — decides if a task needs a plan and generates one.
Runs once per user turn before the ReAct loop starts.
"""
import re

# ── Complexity heuristics ─────────────────────────────────────────────────────

_COMPLEX_KEYWORDS = {
    "refactor", "implement", "build", "rewrite",
    "migrate", "convert", "optimize", "redesign", "restructure",
    "integrate", "setup", "configure", "deploy",
    "split", "merge", "extract", "upgrade", "scaffold",
    "system", "pipeline", "architecture", "framework",
}

_SIMPLE_PREFIXES = (
    "what is", "what are", "what does", "what's",
    "why is", "why does", "why are",
    "how does", "how is", "how do",
    "show me", "show the",
    "read ", "list ", "tell me", "explain",
    "describe", "where is", "who is",
    "create a function", "write a function", "add a function",
    "create a class", "write a class",
    "fix ", "debug ",
)

# Single-file or single-function tasks — never plan these
_TRIVIAL_PATTERNS = (
    r"create\s+a\s+(function|method|class|script|file)",
    r"write\s+a\s+(function|method|class|script|file)",
    r"add\s+a\s+(function|method|class)",
    r"fix\s+(a\s+)?(bug|error|issue|typo)",
    r"rename\s+\w+",
    r"print\s+",
)


def should_plan(query: str, force: bool = False) -> bool:
    """Return True if this query warrants a planning step."""
    if force:
        return True

    q = query.lower().strip()

    # Always skip for short queries
    if len(q.split()) <= 6:
        return False

    # Skip for known simple prefixes
    if any(q.startswith(p) for p in _SIMPLE_PREFIXES):
        return False

    # Skip for trivial single-unit tasks
    if any(re.search(p, q) for p in _TRIVIAL_PATTERNS):
        return False

    # Complex keyword found
    words = set(re.findall(r'\w+', q))
    if words & _COMPLEX_KEYWORDS:
        return True

    # Multi-part task across multiple files/components
    if " and " in q and len(q.split()) > 12:
        return True

    # Long query is likely complex
    if len(q.split()) > 20:
        return True

    return False


# ── Plan generation ───────────────────────────────────────────────────────────

_PLAN_PROMPT = """\
You are a planning assistant for a coding agent.
Given a task, output ONLY a numbered list of concrete steps to complete it.
Rules:
- 3 to 7 steps maximum
- Each step is one short sentence
- Start each line with a number and period: "1. Do X"
- No preamble, no explanation, no markdown — just the numbered list

Task: {query}
"""


def generate_plan(llm, query: str, cwd: str) -> list[str]:
    """
    Call the LLM to generate a step-by-step plan.
    Returns a list of step strings (without numbering).
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = _PLAN_PROMPT.format(query=query)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in content
            )
        return _parse_steps(content or "")
    except Exception:
        return []


def _parse_steps(text: str) -> list[str]:
    """Extract numbered steps from model output."""
    steps = []
    for line in text.strip().splitlines():
        line = line.strip()
        # Match "1. Step text" or "1) Step text"
        match = re.match(r'^\d+[.)]\s+(.+)', line)
        if match:
            steps.append(match.group(1).strip())
    return steps[:7]  # cap at 7
