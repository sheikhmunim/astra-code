from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.prompts import build_system_prompt
from agent.parser import parse_react_action, parse_final_answer
from tools.file_tools import read_file, write_file, edit_file
from tools.shell_tools import bash
from tools.search_tools import glob_search, grep_search
from config.settings import MAX_ITERATIONS

ALL_TOOLS = [read_file, write_file, edit_file, bash, glob_search, grep_search]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}


def build_graph(llm, tool_mode: str = "react", model_label: str = ""):
    """
    Build the LangGraph agent.

    Args:
        llm: LangChain chat model instance (ChatOllama, ChatAnthropic, etc.)
        tool_mode: "react"   — text-based ReAct parsing (for local models)
                   "native"  — LangChain bind_tools + ToolNode (for cloud APIs)
        model_label: display name injected into the system prompt
    """
    if tool_mode == "native":
        return _build_native_graph(llm, model_label)
    else:
        return _build_react_graph(llm, model_label)


# ── ReAct graph (Ollama / models without native function calling) ─────────────

def _build_react_graph(llm, model_label: str):
    def agent_node(state: AgentState) -> dict:
        iterations = state.get("iterations", 0)
        if iterations >= MAX_ITERATIONS:
            return {
                "messages": [AIMessage(content=f"Final Answer: Reached maximum iteration limit ({MAX_ITERATIONS}).")],
                "iterations": iterations,
            }
        system = SystemMessage(content=build_system_prompt(state["cwd"], model_label))
        messages = [system] + list(state["messages"])
        response = llm.invoke(messages)
        return {"messages": [response], "iterations": iterations + 1}

    def tool_node(state: AgentState) -> dict:
        last_msg = state["messages"][-1]
        text = last_msg.content if hasattr(last_msg, "content") else ""
        tool_name, tool_args = parse_react_action(text)

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
        return "tools" if tool_name else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=MemorySaver())


# ── Native graph (Anthropic, OpenAI, Groq — real function calling) ────────────

def _build_native_graph(llm, model_label: str):
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    tool_node = ToolNode(ALL_TOOLS)

    def agent_node(state: AgentState) -> dict:
        iterations = state.get("iterations", 0)
        if iterations >= MAX_ITERATIONS:
            return {
                "messages": [AIMessage(content=f"Reached maximum iteration limit ({MAX_ITERATIONS}).")],
                "iterations": iterations,
            }
        system = SystemMessage(content=build_system_prompt(state["cwd"], model_label, native_tools=True))
        messages = [system] + list(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response], "iterations": iterations + 1}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=MemorySaver())
