import os
import re
import glob as _glob
import fnmatch
from langchain_core.tools import tool


@tool
def glob_search(pattern: str, base_dir: str = ".") -> str:
    """Find files matching a glob pattern (e.g. **/*.py, src/**/*.ts).
    Searches recursively from base_dir (defaults to current directory).
    Returns a list of matching file paths, sorted by modification time (newest first).
    """
    base = os.path.abspath(base_dir)
    if not os.path.isdir(base):
        return f"ERROR: Directory not found: {base}"

    # Use Python's glob with recursive=True — correctly handles ** patterns
    search_pattern = os.path.join(base, pattern)
    raw_matches = _glob.glob(search_pattern, recursive=True)

    # Filter to files only, skip hidden paths
    matches = [
        p for p in raw_matches
        if os.path.isfile(p) and not any(
            part.startswith(".") for part in p.replace("\\", "/").split("/")
        )
    ]

    if not matches:
        return f"No files found matching '{pattern}' in {base}"

    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    rel_matches = [os.path.relpath(p, base).replace("\\", "/") for p in matches]
    return "\n".join(rel_matches)


@tool
def grep_search(pattern: str, path: str = ".", file_glob: str = "*") -> str:
    """Search file contents for a regex pattern.
    Args:
        pattern: Regular expression to search for
        path: File or directory to search in (default: current directory)
        file_glob: Only search files matching this glob (e.g. *.py, *.ts)
    Returns matching lines as: filepath:line_number: line_content
    """
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return f"ERROR: Path not found: {abs_path}"

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"ERROR: Invalid regex pattern: {e}"

    results = []
    max_results = 100

    def search_file(filepath: str):
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f, 1):
                    if regex.search(line):
                        rel = os.path.relpath(filepath, abs_path if os.path.isdir(abs_path) else os.path.dirname(abs_path))
                        results.append(f"{rel.replace(chr(92), '/')}:{i}: {line.rstrip()}")
                        if len(results) >= max_results:
                            return
        except Exception:
            pass

    if os.path.isfile(abs_path):
        search_file(abs_path)
    else:
        for root, dirs, files in os.walk(abs_path):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fname in files:
                if fnmatch.fnmatch(fname, file_glob):
                    search_file(os.path.join(root, fname))
                if len(results) >= max_results:
                    break

    if not results:
        return f"No matches found for '{pattern}' in {abs_path}"

    output = "\n".join(results)
    if len(results) >= max_results:
        output += f"\n... (results truncated at {max_results} matches)"
    return output
