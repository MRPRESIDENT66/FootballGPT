"""Tactics Agent — evaluates tactical fit using RAG knowledge base and team data."""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from config.settings import settings
from knowledge.rag import search_football_knowledge
from knowledge.retrieval import run_agent_retrieval
from tools.player_db import get_team_stats, get_team_roster

TACTICS_PROMPT = """You are a football tactics analyst. Evaluate tactical fit between players and teams.

You have these tools — use up to 2 to gather context, then give your assessment:
- search_football_knowledge: search Wikipedia knowledge base for team history, tactics, playing style
- get_team_stats: get aggregated squad data (average age, position breakdown, top scorer, etc.)
- get_team_roster: get full list of current squad players

Be concise (200 words max). Focus on:
1. Team's formation and style
2. Squad composition and gaps
3. How candidates fit the system
4. Key tactical considerations"""


def run_tactics(
    query: str,
    criteria: dict,
    player_data: str = "",
    shared_knowledge: str = "",
) -> tuple[str, str]:
    """Run tactics agent. Max 2 tool call rounds."""
    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.DASHSCOPE_BASE_URL,
        temperature=settings.TEMPERATURE,
        **settings.NO_THINKING,
    ).bind_tools([search_football_knowledge, get_team_stats, get_team_roster])

    task_knowledge = run_agent_retrieval("tactics", query, criteria, shared_knowledge)

    context = f"User query: {query}\nCriteria: {criteria}"
    if player_data:
        context += f"\n\nPlayer data:\n{player_data}"
    if shared_knowledge:
        context += f"\n\nShared knowledge:\n{shared_knowledge}"
    if task_knowledge:
        context += f"\n\nTactics-specific knowledge:\n{task_knowledge}"

    messages = [
        SystemMessage(content=TACTICS_PROMPT),
        HumanMessage(content=context),
    ]

    tool_map = {
        "search_football_knowledge": search_football_knowledge,
        "get_team_stats": get_team_stats,
        "get_team_roster": get_team_roster,
    }

    # Max 2 tool call rounds
    for _ in range(2):
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content, task_knowledge

        for tc in response.tool_calls:
            result = tool_map[tc["name"]].invoke(tc["args"])
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    messages.append(HumanMessage(content="Now give your tactical assessment. Be concise."))
    return llm.invoke(messages).content, task_knowledge
