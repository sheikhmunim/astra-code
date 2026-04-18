import os
from langchain_core.tools import tool


@tool
def read_file(file_path: str, start_line: int = None, end_line: int = None) -> str:
    """Read a file and return its contents with line numbers.
    Optionally provide start_line and end_line (1-indexed) to read a specific range.
    """
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        return f"ERROR: File not found: {abs_path}"
    if not os.path.isfile(abs_path):
        return f"ERROR: Path is not a file: {abs_path}"

    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        # Apply line range if requested (1-indexed, inclusive)
        if start_line is not None or end_line is not None:
            sl = (start_line - 1) if start_line else 0
            el = end_line if end_line else len(lines)
            lines = lines[sl:el]
            offset = sl
        else:
            offset = 0

        numbered = "".join(
            f"{offset + i + 1}\t{line}" for i, line in enumerate(lines)
        )
        return numbered if numbered else "(empty file)"
    except Exception as e:
        return f"ERROR reading file: {e}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Create or fully overwrite a file with the given content.
    Use for new files or intentional full rewrites. Prefer edit_file for partial changes.
    """
    abs_path = os.path.abspath(file_path)
    try:
        os.makedirs(os.path.dirname(abs_path), exist_ok=True) if os.path.dirname(abs_path) else None
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        lines = content.count("\n") + 1
        return f"OK: Wrote {lines} lines to {abs_path}"
    except Exception as e:
        return f"ERROR writing file: {e}"


@tool
def edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """Surgically replace an exact string in a file.
    old_string must match the file content exactly (including indentation and whitespace).
    If it doesn't match, re-read the file and try again with the correct text.
    """
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        return f"ERROR: File not found: {abs_path}"

    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        count = content.count(old_string)
        if count == 0:
            # Give a useful hint showing nearby content
            return (
                f"ERROR: old_string not found in {abs_path}.\n"
                f"The string you provided does not exactly match any text in the file.\n"
                f"Re-read the file with read_file and copy the exact text you want to replace."
            )
        if count > 1:
            return (
                f"ERROR: old_string appears {count} times in {abs_path}. "
                f"Provide a larger, unique context to make the match unambiguous."
            )

        new_content = content.replace(old_string, new_string, 1)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"OK: Edit applied to {abs_path}"
    except Exception as e:
        return f"ERROR editing file: {e}"
