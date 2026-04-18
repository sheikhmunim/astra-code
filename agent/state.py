from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    cwd: str          # project working directory (injected at session start)
    iterations: int   # tracks loop count for safety limit
