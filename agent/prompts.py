import platform
from datetime import date


TOOL_DESCRIPTIONS = """
## Available Tools

### write_file
Creates or fully overwrites a file with the given content.
Parameters: file_path (str), content (str)

### read_file
Reads a file and returns its contents with line numbers.
Parameters: file_path (str), start_line (int, optional), end_line (int, optional)

### edit_file
Surgically replaces an exact string in a file. old_string must match exactly (including whitespace).
If match fails, re-read the file first.
Parameters: file_path (str), old_string (str), new_string (str)

### bash
Executes a shell command. Returns stdout, stderr, and exit code.
Parameters: command (str)

### glob_search
Finds files matching a glob pattern (e.g. **/*.py, src/**/*.ts).
Parameters: pattern (str), base_dir (str, optional)

### grep_search
Searches file contents for a regex pattern. Returns file:line matches.
Parameters: pattern (str), path (str, optional), file_glob (str, optional)
"""

REACT_FORMAT = """
## Response Format

Always respond in this exact ReAct format:

Thought: <your reasoning>
Action: <tool_name>
Action Input: {"param": "value"}

After receiving an Observation, continue or finish:

Thought: <reasoning>
Final Answer: <response to user>
Memory: <one specific fact worth remembering about this project or user preference — or omit if nothing new>

## Rules
- NEVER skip the Thought step.
- Action Input must be valid JSON (escape newlines as \\n, use double quotes).
- Always READ a file before editing it.
- When done, use Final Answer.
- Memory is optional — only add it if you learned something genuinely useful to remember.

## Example

User: create hello.py that prints hello world

Thought: I need to create a new Python file.
Action: write_file
Action Input: {"file_path": "hello.py", "content": "print('Hello, World!')\\n"}

Observation: OK: Wrote 1 lines to hello.py

Thought: File created successfully.
Final Answer: Created `hello.py` with `print('Hello, World!')`.
Memory: Project uses plain Python scripts with no framework.
"""

NATIVE_INSTRUCTIONS = """
## Instructions
- Think step by step before using tools.
- Always READ a file before editing it — never guess at its contents.
- Prefer edit_file over write_file for existing files.
- After running bash commands, check the output before proceeding.
- When a task is complete, give a concise summary of what you did.
- Never run destructive commands (rm -rf, drop table, etc.) without explicit confirmation.
- After your response, optionally add: Memory: <one fact worth remembering about this project>
"""


def build_system_prompt(
    cwd: str,
    model: str,
    native_tools: bool = False,
    memories: list[str] | None = None,
    plan: str = "",
) -> str:
    os_info = f"{platform.system()} {platform.release()}"
    today = date.today().isoformat()

    base = f"""You are Astra, a coding agent running locally on the user's machine.
You help with software engineering tasks: reading, writing, editing code, running commands, and searching codebases.

## Environment
- Working directory: {cwd}
- OS: {os_info}
- Date: {today}
- Model: {model}
"""

    memory_section = ""
    if memories:
        memory_section = "\n## What I remember about this project\n"
        for m in memories:
            memory_section += f"- {m}\n"

    plan_section = ""
    if plan:
        plan_section = f"\n## Plan for this task\n{plan}\n\nFollow this plan step by step.\n"

    if native_tools:
        return base + memory_section + plan_section + NATIVE_INSTRUCTIONS
    else:
        return base + memory_section + plan_section + TOOL_DESCRIPTIONS + REACT_FORMAT
