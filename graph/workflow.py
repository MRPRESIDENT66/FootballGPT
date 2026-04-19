"""LangGraph workflow — orchestrates multi-agent collaboration.

Optimized flow with parallel execution:
  - recommend: Router → [Scout ∥ Tactics] → Analyst → Reporter
  - scout:     Router → Scout → Analyst → Reporter
  - compare:   Router → Analyst → Reporter
  - tactics:   Router → Tactics → Reporter

Supports multi-turn conversation via MemorySaver checkpointing.
LangSmith tracing enabled when LANGCHAIN_TRACING_V2=true (zero-code integration).
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
from knowledge.retrieval import build_shared_knowledge
from utils import clean_surrogates
from config.settings import settings

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
    shared_knowledge: str
    analyst_knowledge: str
    tactics_knowledge: str
    reporter_knowledge: str
    chat_history: Annotated[list[dict], add]  # accumulates across turns


# ---- Node functions ----

def _validate_routing(intent: str, params: dict) -> bool:
    """Validate that Router output is consistent — intent matches parameters."""
    if intent == "compare":
        names = params.get("player_names", [])
        if not names or len(names) < 2:
            return False
    elif intent == "recommend":
        if not params.get("team"):
            return False
    elif intent == "scout":
        # Scout should have at least one filter criterion
        filters = ["position", "max_age", "min_age", "league", "club",
                   "nationality", "min_pace", "max_market_value", "min_goals"]
        if not any(params.get(f) for f in filters):
            return False
    return True


def router_node(state: FootballState) -> dict:
    """Classify intent and extract parameters, with validation and retry."""
    relevant_history = _vector_memory.retrieve(state["query"], k=3)

    result = route_query(state["query"], relevant_history)
    intent = result["intent"]
    params = result.get("parameters", {})

    # Validate: if intent and parameters don't match, retry once
    if not _validate_routing(intent, params):
        result = route_query(state["query"], relevant_history)
        intent = result["intent"]
        params = result.get("parameters", {})

        # Still invalid → fallback to scout (safest default)
        if not _validate_routing(intent, params):
            intent = "scout"

    return {
        "intent": intent,
        "parameters": params,
    }


def scout_node(state: FootballState) -> dict:
    """Find players matching criteria."""
    data = clean_surrogates(run_scout(
        state["parameters"],
        state["query"],
        state.get("shared_knowledge", ""),
    ))
    return {"scout_data": data}


def analyst_node(state: FootballState) -> dict:
    """Analyze player data."""
    data, knowledge = run_analyst(
        state["scout_data"],
        state["parameters"],
        state["query"],
        state.get("shared_knowledge", ""),
    )
    return {
        "analysis": clean_surrogates(data),
        "analyst_knowledge": clean_surrogates(knowledge),
    }


def tactics_node(state: FootballState) -> dict:
    """Evaluate tactical fit."""
    player_data = state.get("scout_data", "") or state.get("analysis", "")
    data, knowledge = run_tactics(
        state["query"],
        state["parameters"],
        player_data,
        state.get("shared_knowledge", ""),
    )
    return {
        "tactical_context": clean_surrogates(data),
        "tactics_knowledge": clean_surrogates(knowledge),
    }


def reporter_node(state: FootballState) -> dict:
    """Generate final report and store turn in vector memory."""
    report, knowledge = run_reporter(
        state["query"],
        state.get("scout_data", ""),
        state.get("analysis", ""),
        state.get("tactical_context", ""),
        state.get("parameters", {}),
        state.get("shared_knowledge", ""),
    )
    report = clean_surrogates(report)
    # Store this turn in vector memory for semantic retrieval in future turns
    _vector_memory.add_turn(state["query"], state["intent"], report)

    turn_record = {
        "query": state["query"],
        "intent": state["intent"],
        "report_summary": report[:500],
    }
    return {
        "report": report,
        "reporter_knowledge": clean_surrogates(knowledge),
        "chat_history": [turn_record],
    }


def shared_retrieval_node(state: FootballState) -> dict:
    """Retrieve shared baseline knowledge once per turn."""
    knowledge = clean_surrogates(build_shared_knowledge(
        state["query"],
        state["intent"],
        state.get("parameters", {}),
    ))
    return {"shared_knowledge": knowledge}


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


def fan_out_after_shared(state: FootballState) -> list[str]:
    """Route after shared retrieval has populated baseline context."""
    return route_by_intent(state)


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
    """Build and compile the multi-agent workflow graph with memory.

    LangSmith tracing is automatic when LANGCHAIN_TRACING_V2=true.
    All LLM calls, tool invocations, and state transitions are captured.
    """
    if settings.LANGSMITH_TRACING:
        print(f"🔍 LangSmith tracing enabled → project: {settings.LANGSMITH_PROJECT}")

    memory = MemorySaver()
    workflow = StateGraph(FootballState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("shared_retrieval", shared_retrieval_node)
    workflow.add_node("scout", scout_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("tactics", tactics_node)
    workflow.add_node("reporter", reporter_node)
    workflow.add_node("join_recommend", lambda state: {})  # no-op join node

    # Entry point
    workflow.set_entry_point("router")

    # Router → shared retrieval
    workflow.add_edge("router", "shared_retrieval")

    # Shared retrieval → fan-out based on intent
    workflow.add_conditional_edges(
        "shared_retrieval",
        fan_out_after_shared,
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
