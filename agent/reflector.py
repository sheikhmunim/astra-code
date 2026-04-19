"""
Self-reflection — reviews the agent's final answer against the original request.
Returns (is_good, feedback). If not good, feedback is injected back into the
agent so it can fix the issue.
"""

MAX_REFLECTIONS = 2

_REFLECT_PROMPT = """\
You are reviewing a coding agent's response to a user request.

User request: {query}

Agent response: {response}

Check:
1. Did the agent fully complete what was asked?
2. Are there any errors, missing steps, or incorrect logic?
3. Is anything left unfinished?

Reply with ONLY one of these two formats:
- GOOD
- NEEDS_WORK: <one specific issue to fix, one sentence>

No explanation, no preamble — just GOOD or NEEDS_WORK: ...
"""


def reflect(llm, query: str, response: str) -> tuple[bool, str]:
    """
    Ask the LLM to review the response.
    Returns (is_good, feedback).
    is_good=True  → response is acceptable, proceed to END
    is_good=False → feedback describes what needs fixing
    """
    from langchain_core.messages import HumanMessage

    # Skip reflection for very short responses — not worth the extra call
    if len(response.strip()) < 80:
        return True, ""

    prompt = _REFLECT_PROMPT.format(query=query, response=response)
    try:
        result = llm.invoke([HumanMessage(content=prompt)])
        content = result.content
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in content
            )
        content = (content or "").strip()

        if content.upper().startswith("GOOD"):
            return True, ""

        if content.upper().startswith("NEEDS_WORK"):
            feedback = content.split(":", 1)[-1].strip()
            return False, feedback or "The response is incomplete."

        # Ambiguous reply — treat as good to avoid spurious loops
        return True, ""

    except Exception:
        return True, ""
