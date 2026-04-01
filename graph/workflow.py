"""LangGraph workflow — orchestrates multi-agent collaboration.

Optimized flow with parallel execution:
  - recommend: Router → [Scout ∥ Tactics] → Analyst → Reporter
  - scout:     Router → Scout → Analyst → Reporter
  - compare:   Router → Analyst → Reporter
  - tactics:   Router → Tactics → Reporter

Supports multi-turn conversation via MemorySaver checkpointing.
"""

from typing import TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agents.router import route_query
from agents.scout import run_scout
from agents.analyst import run_analyst
from agents.tactics import run_tactics
from agents.reporter import run_reporter
from knowledge.memory import VectorMemory
from utils import clean_surrogates

# Global vector memory instance — shared across turns within a session
_vector_memory = VectorMemory()


class FootballState(TypedDict):
    query: str
    intent: str
    parameters: dict
    scout_data: str
    analysis: str
    tactical_context: str
    report: str
    chat_history: Annotated[list[dict], add]  # accumulates across turns


# ---- Node functions ----

def router_node(state: FootballState) -> dict:
    """Classify intent and extract parameters, with vector memory for context."""
    # Retrieve semantically relevant past turns instead of fixed sliding window
    relevant_history = _vector_memory.retrieve(state["query"], k=3)
    result = route_query(state["query"], relevant_history)
    return {
        "intent": result["intent"],
        "parameters": result.get("parameters", {}),
    }


def scout_node(state: FootballState) -> dict:
    """Find players matching criteria."""
    data = clean_surrogates(run_scout(state["parameters"], state["query"]))
    return {"scout_data": data}


def analyst_node(state: FootballState) -> dict:
    """Analyze player data."""
    data = clean_surrogates(run_analyst(state["scout_data"], state["parameters"], state["query"]))
    return {"analysis": data}


def tactics_node(state: FootballState) -> dict:
    """Evaluate tactical fit."""
    player_data = state.get("scout_data", "") or state.get("analysis", "")
    data = clean_surrogates(run_tactics(state["query"], state["parameters"], player_data))
    return {"tactical_context": data}


def reporter_node(state: FootballState) -> dict:
    """Generate final report and store turn in vector memory."""
    report = clean_surrogates(run_reporter(
        state["query"],
        state.get("scout_data", ""),
        state.get("analysis", ""),
        state.get("tactical_context", ""),
    ))
    # Store this turn in vector memory for semantic retrieval in future turns
    _vector_memory.add_turn(state["query"], state["intent"], report)

    turn_record = {
        "query": state["query"],
        "intent": state["intent"],
        "report_summary": report[:500],
    }
    return {"report": report, "chat_history": [turn_record]}


# ---- Conditional routing ----

def route_by_intent(state: FootballState) -> list[str]:
    """Route to the correct agent(s) based on intent. Returns a list for parallel fan-out."""
    intent = state["intent"]
    if intent == "recommend":
        return ["scout", "tactics"]  # parallel fan-out
    elif intent == "scout":
        return ["scout"]
    elif intent == "compare":
        return ["analyst"]
    elif intent == "tactics":
        return ["tactics"]
    return ["scout"]


def after_scout(state: FootballState) -> str:
    if state["intent"] == "recommend":
        return END  # scout done, will join at analyst
    return "analyst"


def after_tactics(state: FootballState) -> str:
    if state["intent"] == "recommend":
        return END  # tactics done, will join at analyst
    return "reporter"


# ---- Build the graph ----

def build_workflow():
    """Build and compile the multi-agent workflow graph with memory."""
    memory = MemorySaver()
    workflow = StateGraph(FootballState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("scout", scout_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("tactics", tactics_node)
    workflow.add_node("reporter", reporter_node)
    workflow.add_node("join_recommend", lambda state: {})  # no-op join node

    # Entry point
    workflow.set_entry_point("router")

    # Router → fan-out based on intent
    workflow.add_conditional_edges(
        "router",
        route_by_intent,
        ["scout", "analyst", "tactics"],
    )

    # Scout → join or analyst
    workflow.add_conditional_edges("scout", after_scout, {
        END: "join_recommend",
        "analyst": "analyst",
    })

    # Tactics → join or reporter
    workflow.add_conditional_edges("tactics", after_tactics, {
        END: "join_recommend",
        "reporter": "reporter",
    })

    # Join node (both scout & tactics done) → analyst
    workflow.add_edge("join_recommend", "analyst")

    # Analyst always goes to reporter
    workflow.add_edge("analyst", "reporter")

    # Reporter is the end
    workflow.add_edge("reporter", END)

    return workflow.compile(checkpointer=memory)
