"""Router Agent — classifies user intent and extracts search criteria."""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import settings

ROUTER_PROMPT = """You are a football analysis router. Your job is to classify the user's intent and extract structured parameters.

Classify into exactly ONE of these intents:
- "scout": User wants to find/discover players matching certain criteria (position, age, pace, etc.)
- "compare": User wants to compare two or more specific players
- "tactics": User wants tactical analysis of a team, formation, or playing style
- "recommend": User wants a player recommendation for a specific team (combines scouting + tactical fit)

Output ONLY valid JSON with this structure:
{
  "intent": "scout|compare|tactics|recommend",
  "parameters": {
    "position": "position if mentioned, e.g. RW, ST, CDM",
    "max_age": null or integer,
    "min_age": null or integer,
    "league": "league if mentioned",
    "club": "club if mentioned",
    "nationality": "nationality if mentioned",
    "min_pace": null or integer,
    "max_market_value": null or integer (in million EUR),
    "min_goals": null or integer,
    "player_names": ["list of player names if comparing"],
    "team": "team name if asking about a specific team",
    "query_summary": "brief summary of what the user wants"
  }
}

Important rules:
- Extract as many parameters as possible from the user's query
- For "compare" intent, player_names must have exactly 2 names
- For "recommend" intent, team must be specified
- For "scout" intent, at least one filter criterion should be extracted
- If the user mentions speed/fast/quick, map to min_pace (e.g., "fast" = min_pace 85)
- If the user's query contains pronouns like "he", "him", "them", "that player", "compare him with", etc., resolve them using conversation history
- Output ONLY the JSON, no other text"""


def route_query(query: str, chat_history: list[dict] | None = None) -> dict:
    """Classify user intent and extract parameters, using chat history for context."""
    import json

    llm = ChatOpenAI(
        model=settings.ROUTER_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.DASHSCOPE_BASE_URL,
        temperature=0.1,
    )

    # Build context from semantically relevant past turns (vector memory)
    context = ""
    if chat_history:
        history_lines = []
        for turn in chat_history:
            history_lines.append(f"[Turn {turn.get('turn', '?')}] {turn['content']}")
        context = "Relevant past conversation:\n" + "\n\n".join(history_lines) + "\n\n"

    messages = [
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=f"{context}Current query: {query}" if context else query),
    ]

    response = llm.invoke(messages)
    text = response.content.strip()

    # Clean up markdown code blocks if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    return json.loads(text)
