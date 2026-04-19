from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages:         Annotated[list, add_messages]
    cwd:              str   # project working directory
    iterations:       int   # loop count for safety limit
    plan:             str   # generated plan (empty if not planned)
    force_plan:       bool  # /plan command sets this for one turn
    reflection_count: int   # how many reflections have run this turn
