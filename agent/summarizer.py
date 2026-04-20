"""
Context summarizer — fires when message history grows too long.
Compresses old messages into a compact summary, removes them from state,
and injects the summary so the agent always works with a short context window.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

SUMMARIZE_THRESHOLD = 20   # summarize when message count exceeds this
KEEP_RECENT        = 6     # always keep this many recent messages untouched

_SUMMARIZE_PROMPT = """\
Summarize the following conversation history between a coding agent and a user.
Be extremely concise — under 120 words.
Focus only on facts: what was asked, what files were created/edited, what commands ran, what succeeded or failed, current state of the work.
No narration, no filler. Output plain text.

Conversation:
{history}
"""


def should_summarize(messages: list) -> bool:
    return len(messages) > SUMMARIZE_THRESHOLD


def summarize_history(llm, messages: list) -> str:
    """Compress a list of messages into a short summary string."""
    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "User" if not (msg.content or "").startswith("Observation:") else "Tool"
        elif isinstance(msg, AIMessage):
            role = "Agent"
        elif isinstance(msg, SystemMessage):
            continue   # skip system messages — they're rebuilt fresh each turn
        else:
            role = "Tool"

        content = msg.content or ""
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in content
            )
        # Truncate very long tool outputs to avoid bloating the summarizer prompt
        if len(content) > 400:
            content = content[:400] + "…"
        lines.append(f"{role}: {content.strip()}")

    history_text = "\n".join(lines)
    prompt = _SUMMARIZE_PROMPT.format(history=history_text)

    try:
        result = llm.invoke([HumanMessage(content=prompt)])
        content = result.content or ""
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in content
            )
        return content.strip()
    except Exception:
        return "(summary unavailable)"
