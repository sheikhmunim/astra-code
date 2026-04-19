import re

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.prompts import build_system_prompt
from agent.parser import parse_react_action, parse_final_answer
from agent.memory import MemoryStore
from agent.planner import should_plan, generate_plan
from agent.reflector import reflect, MAX_REFLECTIONS
from tools.file_tools import read_file, write_file, edit_file
from tools.shell_tools import bash
from tools.search_tools import glob_search, grep_search
from config.settings import MAX_ITERATIONS

ALL_TOOLS = [read_file, write_file, edit_file, bash, glob_search, grep_search]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}

_STUCK_THRESHOLD = 3  # same tool called this many times in a row → break out


def _detect_stuck_loop_react(messages: list, tool_name: str, threshold: int = _STUCK_THRESHOLD) -> bool:
    """Return True if tool_name appears in ≥ threshold consecutive recent AIMessages."""
    if not tool_name:
        return False
    count = 1  # current call counts as 1
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue  # skip HumanMessage observations
        t, _ = parse_react_action(msg.content or "")
        if t == tool_name:
            count += 1
            if count >= threshold:
                return True
        else:
            break  # streak broken
    return False


def _detect_stuck_loop_native(messages: list, tool_calls: list, threshold: int = _STUCK_THRESHOLD) -> bool:
    """Return True if the same set of tools appears in ≥ threshold consecutive AIMessages."""
    if not tool_calls:
        return False
    current_names = frozenset(tc["name"] for tc in tool_calls)
    count = 1
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        if not (hasattr(msg, "tool_calls") and msg.tool_calls):
            break
        names = frozenset(tc["name"] for tc in msg.tool_calls)
        if names == current_names:
            count += 1
            if count >= threshold:
                return True
        else:
            break
    return False


def _parse_memory_line(text: str) -> str | None:
    match = re.search(r'\bMemory\s*:\s*(.+)', text, re.IGNORECASE)
    if match:
        fact = match.group(1).strip()
        if fact.lower() not in ("none", "n/a", ""):
            return fact
    return None


def _get_last_user_message(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        role = getattr(msg, "type", None) or getattr(msg, "role", "")
        if role in ("human", "user"):
            return msg.content if hasattr(msg, "content") else str(msg)
    return ""


def build_graph(llm, tool_mode: str = "react", model_label: str = "", cwd: str = ""):
    memory = MemoryStore(cwd) if cwd else None

    if tool_mode == "native":
        return _build_native_graph(llm, model_label, memory), memory
    else:
        return _build_react_graph(llm, model_label, memory), memory


# ── Shared reflector node ────────────────────────────────────────────────────

def _make_reflector_node(llm):
    from cli.interface import console
    from rich.markup import escape

    def reflector_node(state: AgentState) -> dict:
        # Get original user query
        query = _get_last_user_message(state)

        # Get agent's final answer (last AI message content)
        response = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                content = msg.content or ""
                if isinstance(content, list):
                    content = " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in content
                    )
                response = content
                break

        is_good, feedback = reflect(llm, query, response)

        if not is_good and feedback:
            console.print(
                f"  [dim yellow]↻ Reflecting:[/dim yellow] [dim]{escape(feedback)}[/dim]"
            )
            # Frame as Observation so ReAct models stay in tool-use mode
            return {
                "messages": [HumanMessage(
                    content=f"Observation: Issue found — {feedback}\n"
                            f"Use the appropriate tools to fix this, then give a Final Answer."
                )],
                "reflection_count": state.get("reflection_count", 0) + 1,
            }

        return {"reflection_count": state.get("reflection_count", 0)}

    return reflector_node


# ── Shared planner node ───────────────────────────────────────────────────────

def _make_planner_node(llm):
    from cli.interface import print_plan

    def planner_node(state: AgentState) -> dict:
        query      = _get_last_user_message(state)
        force      = state.get("force_plan", False)

        if not should_plan(query, force=force):
            return {"plan": "", "force_plan": False}

        steps = generate_plan(llm, query, state.get("cwd", ""))
        if steps:
            print_plan(steps)
            plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
        else:
            plan_text = ""

        return {"plan": plan_text, "force_plan": False}

    return planner_node


# ── ReAct graph ───────────────────────────────────────────────────────────────

def _build_react_graph(llm, model_label: str, memory: MemoryStore | None):
    planner_node   = _make_planner_node(llm)
    reflector_node = _make_reflector_node(llm)

    def agent_node(state: AgentState) -> dict:
        import os
        os.chdir(state["cwd"])   # keep all relative paths correct
        iterations = state.get("iterations", 0)
        if iterations >= MAX_ITERATIONS:
            return {
                "messages":  [AIMessage(content=f"Final Answer: Reached maximum iteration limit ({MAX_ITERATIONS}).")],
                "iterations": iterations,
            }

        memories = []
        if memory and memory.ready:
            query = _get_last_user_message(state)
            if query:
                memories = memory.retrieve(query)

        system = SystemMessage(content=build_system_prompt(
            state["cwd"], model_label,
            memories=memories,
            plan=state.get("plan", ""),
        ))
        messages = [system] + list(state["messages"])
        response = llm.invoke(messages)

        if memory and memory.ready and hasattr(response, "content"):
            fact = _parse_memory_line(response.content or "")
            if fact:
                memory.save(fact, category="auto")

        return {"messages": [response], "iterations": iterations + 1}

    def tool_node(state: AgentState) -> dict:
        last_msg = state["messages"][-1]
        text = last_msg.content if hasattr(last_msg, "content") else ""
        tool_name, tool_args = parse_react_action(text)

        # Break out of stuck loops before executing the tool again
        if _detect_stuck_loop_react(state["messages"], tool_name):
            return {"messages": [HumanMessage(
                content=(
                    f"Observation: You have called '{tool_name}' {_STUCK_THRESHOLD} times in a row "
                    f"with the same approach and it is not working. Stop repeating this. "
                    f"Give a Final Answer now that explains what you tried, what failed, "
                    f"and what the user should do next."
                )
            )]}

        if tool_name in TOOL_MAP:
            try:
                result = TOOL_MAP[tool_name].invoke(tool_args or {})
            except Exception as e:
                result = f"ERROR: {e}"
        else:
            result = f"ERROR: Unknown tool '{tool_name}'. Available: {list(TOOL_MAP.keys())}"

        return {"messages": [HumanMessage(content=f"Observation: {result}")]}

    def should_continue(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        if not isinstance(last_msg, AIMessage):
            return END
        tool_name, _ = parse_react_action(last_msg.content or "")
        if tool_name:
            return "tools"
        # Route to reflector if under the reflection limit
        if state.get("reflection_count", 0) < MAX_REFLECTIONS:
            return "reflector"
        return END

    def after_reflect(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        if isinstance(last_msg, HumanMessage) and "observation:" in (last_msg.content or "").lower():
            return "agent"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("planner",   planner_node)
    graph.add_node("agent",     agent_node)
    graph.add_node("tools",     tool_node)
    graph.add_node("reflector", reflector_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools":     "tools",
        "reflector": "reflector",
        END:         END,
    })
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("reflector", after_reflect, {
        "agent": "agent",
        END:     END,
    })
    return graph.compile(checkpointer=MemorySaver())


# ── Native graph ──────────────────────────────────────────────────────────────

def _build_native_graph(llm, model_label: str, memory: MemoryStore | None):
    planner_node   = _make_planner_node(llm)
    reflector_node = _make_reflector_node(llm)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    tool_node      = ToolNode(ALL_TOOLS)

    def agent_node(state: AgentState) -> dict:
        import os
        os.chdir(state["cwd"])   # keep all relative paths correct
        iterations = state.get("iterations", 0)
        if iterations >= MAX_ITERATIONS:
            return {
                "messages":  [AIMessage(content=f"Reached maximum iteration limit ({MAX_ITERATIONS}).")],
                "iterations": iterations,
            }

        memories = []
        if memory and memory.ready:
            query = _get_last_user_message(state)
            if query:
                memories = memory.retrieve(query)

        system = SystemMessage(content=build_system_prompt(
            state["cwd"], model_label, native_tools=True,
            memories=memories,
            plan=state.get("plan", ""),
        ))
        messages = [system] + list(state["messages"])
        response = llm_with_tools.invoke(messages)

        if memory and memory.ready and hasattr(response, "content"):
            content = response.content
            if isinstance(content, list):
                content = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
            fact = _parse_memory_line(content or "")
            if fact:
                memory.save(fact, category="auto")

        return {"messages": [response], "iterations": iterations + 1}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            if _detect_stuck_loop_native(state["messages"], last.tool_calls):
                return END
            return "tools"
        if state.get("reflection_count", 0) < MAX_REFLECTIONS:
            return "reflector"
        return END

    def after_reflect(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        if isinstance(last_msg, HumanMessage) and "observation:" in (last_msg.content or "").lower():
            return "agent"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("planner",   planner_node)
    graph.add_node("agent",     agent_node)
    graph.add_node("tools",     tool_node)
    graph.add_node("reflector", reflector_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools":     "tools",
        "reflector": "reflector",
        END:         END,
    })
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("reflector", after_reflect, {
        "agent": "agent",
        END:     END,
    })
    return graph.compile(checkpointer=MemorySaver())
