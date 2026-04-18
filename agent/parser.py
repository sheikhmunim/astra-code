import re
import json


def _fix_json_newlines(raw: str) -> str:
    """Escape literal newlines/tabs inside JSON string values.
    Local models often output actual newlines in JSON strings instead of \\n.
    """
    result = []
    in_string = False
    escape_next = False

    for ch in raw:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            result.append(ch)
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string:
            if ch == '\n':
                result.append('\\n')
                continue
            if ch == '\r':
                result.append('\\r')
                continue
            if ch == '\t':
                result.append('\\t')
                continue
        result.append(ch)

    return ''.join(result)


def _try_parse_json(raw: str):
    """Try to parse JSON, with a fallback that fixes literal newlines in strings."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_fix_json_newlines(raw))
    except json.JSONDecodeError:
        pass
    return None


# Words that look like tool names but are actually the model signalling it's done
_PSEUDO_TOOLS = {"final", "answer", "done", "none", "n/a"}


def parse_react_action(text: str):
    """Extract (tool_name, tool_args) from a ReAct-format model response.
    Returns (None, None) if no Action is found OR if the model wrote
    'Action: Final Answer' (a common local-model mistake meaning it is done).
    """
    action_match = re.search(r'\bAction\s*:\s*(\w+)', text, re.IGNORECASE)
    input_match = re.search(r'\bAction\s+Input\s*:\s*(\{.*)', text, re.IGNORECASE | re.DOTALL)

    if not action_match:
        return None, None

    tool_name = action_match.group(1).strip()

    # Treat "Action: Final Answer" / "Action: Final" / "Action: Done" as no-op
    if tool_name.lower() in _PSEUDO_TOOLS:
        return None, None

    if input_match:
        raw = input_match.group(1).strip()
        # Extract the balanced JSON object
        depth = 0
        end = 0
        for i, ch in enumerate(raw):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        json_str = raw[:end] if end > 0 else raw
        args = _try_parse_json(json_str)
        return tool_name, args if args is not None else {}

    return tool_name, {}


def parse_final_answer(text: str):
    """Extract the Final Answer from model response, or None.
    Handles both correct format ('Final Answer: ...') and the common local-model
    mistake of 'Action: Final Answer' followed by content on the next line.
    """
    # Standard format
    match = re.search(r'Final Answer\s*:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    # Local model variant: "Action: Final Answer\nAction Input: {...}"
    # Extract the content from Action Input if it looks like a plain response
    action_final = re.search(
        r'Action\s*:\s*Final(?:\s+Answer)?\s*\nAction\s+Input\s*:\s*(\{.*?\})',
        text, re.IGNORECASE | re.DOTALL
    )
    if action_final:
        args = _try_parse_json(action_final.group(1))
        if isinstance(args, dict):
            return args.get("response") or args.get("answer") or args.get("text") or str(args)

    return None


def parse_thought(text: str):
    """Extract the Thought line(s) from model response."""
    match = re.search(r'Thought\s*:\s*(.*?)(?=\nAction|\nFinal Answer|$)', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
