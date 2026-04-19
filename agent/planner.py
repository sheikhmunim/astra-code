"""
Planner — decides if a task needs a plan and generates one.
Runs once per user turn before the ReAct loop starts.
"""
import re

# ── Complexity heuristics ─────────────────────────────────────────────────────

_COMPLEX_KEYWORDS = {
    "refactor", "implement", "build", "create", "add", "fix", "rewrite",
    "migrate", "convert", "optimize", "redesign", "restructure", "update",
    "integrate", "setup", "configure", "deploy", "remove", "delete",
    "change", "modify", "move", "rename", "split", "merge", "extract",
    "replace", "upgrade", "debug", "test", "generate", "scaffold",
}

_SIMPLE_PREFIXES = (
    "what is", "what are", "what does", "what's",
    "why is", "why does", "why are",
    "how does", "how is", "how do",
    "show me", "show the",
    "read ", "list ", "tell me", "explain",
    "describe", "where is", "who is",
)


def should_plan(query: str, force: bool = False) -> bool:
    """Return True if this query warrants a planning step."""
    if force:
        return True

    q = query.lower().strip()

    # Short simple questions — skip planning
    if len(q.split()) <= 5:
        return False
    if any(q.startswith(p) for p in _SIMPLE_PREFIXES) and len(q.split()) < 10:
        return False

    # Complex keyword found
    words = set(re.findall(r'\w+', q))
    if words & _COMPLEX_KEYWORDS:
        return True

    # Multi-part task ("X and Y")
    if " and " in q and len(q.split()) > 8:
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
