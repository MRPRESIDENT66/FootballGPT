"""Analyst Agent — analyzes player data with Wikipedia context. Limited to 2 tool calls max."""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from config.settings import settings
from tools.player_db import compare_players, get_player_details, find_similar_players, get_top_scorers
from knowledge.rag import retrieve_knowledge

ANALYST_PROMPT = """You are a professional football data analyst. Analyze the player data provided.

You have these tools available — use them when they add value:
- compare_players: side-by-side stat comparison of two players
- get_player_details: get full stats for a specific player
- find_similar_players: find players with similar statistical profiles (uses per-90 stats)
- get_top_scorers: get top scorers by league or position

When analyzing:
1. Compare key stats relevant to the position
2. Evaluate attributes (pace, shooting, passing, dribbling, defending, physical)
3. Consider age, development potential, and market value
4. Highlight standout stats and weaknesses
5. Use find_similar_players when the user asks for alternatives or "players like X"
6. Reference the Wikipedia background (career history, honours, playing style) when available

Be concise and data-driven. Limit to 300 words."""


def run_analyst(player_data: str, criteria: dict, query: str) -> str:
    """Run the analyst agent. Max 2 tool call rounds."""
    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.DASHSCOPE_BASE_URL,
        temperature=settings.TEMPERATURE,
        **settings.NO_THINKING,
    ).bind_tools([compare_players, get_player_details, find_similar_players, get_top_scorers])

    # RAG: retrieve relevant player backgrounds for richer analysis
    knowledge = retrieve_knowledge(query, entity_type="player", limit=5)

    content = f"User query: {query}\nCriteria: {criteria}\nPlayer data:\n{player_data}\n\n"
    if knowledge:
        content += f"Wikipedia background:\n{knowledge}\n\n"
    content += "Analyze briefly. Use tools only if needed for direct comparison."

    messages = [
        SystemMessage(content=ANALYST_PROMPT),
        HumanMessage(content=content),
    ]

    tool_map = {
        "compare_players": compare_players,
        "get_player_details": get_player_details,
        "find_similar_players": find_similar_players,
        "get_top_scorers": get_top_scorers,
    }

    for _ in range(2):  # max 2 rounds
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content

        for tc in response.tool_calls:
            result = tool_map[tc["name"]].invoke(tc["args"])
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    messages.append(HumanMessage(content="Summarize your analysis now. No more tool calls."))
    return llm.invoke(messages).content
