"""
Self-reflection — reviews the agent's final answer against the original request.
Returns (is_good, feedback). If not good, feedback is injected back into the
agent so it can fix the issue.
"""

MAX_REFLECTIONS = 2

_REFLECT_PROMPT = """\
You are reviewing a coding agent's response to a user request.

User request: {query}

Tool results (what actually ran):
{tool_context}

Agent final response: {response}

Check:
1. Did the agent fully complete what was asked?
2. Are there any errors, missing steps, or incorrect logic?
3. Is anything left unfinished?
4. Did the agent hallucinate success for a command that clearly failed (e.g. "not recognized", "command not found", "not installed")?
5. Did the agent try to run Linux/macOS-only tools on Windows, or vice versa, without telling the user?

IMPORTANT: If the tool results show the task was completed successfully (e.g. "Wrote N lines to file.py", "EXIT CODE: 0"), treat it as GOOD even if the final response text is brief.

Reply with ONLY one of these two formats:
- GOOD
- NEEDS_WORK: <one specific issue to fix, one sentence>

Special cases — always NEEDS_WORK:
- If the response claims success but tool output shows an error or "not recognized".
- If the agent attempted OS-incompatible commands without explaining.

No explanation, no preamble — just GOOD or NEEDS_WORK: ...
"""


def reflect(llm, query: str, response: str, tool_context: str = "") -> tuple[bool, str]:
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

    tool_context_str = tool_context if tool_context else "(no tools were called)"
    prompt = _REFLECT_PROMPT.format(query=query, response=response, tool_context=tool_context_str)
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
