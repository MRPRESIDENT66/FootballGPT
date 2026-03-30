"""Tactics Agent — evaluates tactical fit using RAG knowledge base. Max 1 tool call."""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from config.settings import settings
from knowledge.rag import search_football_knowledge

TACTICS_PROMPT = """You are a football tactics analyst. Evaluate tactical fit between players and teams.

You have one tool: search_football_knowledge — use it ONCE to get team info, then give your assessment.

Be concise (200 words max). Focus on:
1. Team's formation and style
2. How candidates fit the system
3. Key tactical considerations"""


def run_tactics(query: str, criteria: dict, player_data: str = "") -> str:
    """Run tactics agent. Max 1 tool call."""
    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.DASHSCOPE_BASE_URL,
        temperature=settings.TEMPERATURE,
        **settings.NO_THINKING,
    ).bind_tools([search_football_knowledge])

    context = f"User query: {query}\nCriteria: {criteria}"
    if player_data:
        context += f"\n\nPlayer data:\n{player_data}"

    messages = [
        SystemMessage(content=TACTICS_PROMPT),
        HumanMessage(content=context),
    ]

    # Max 1 tool call round
    response = llm.invoke(messages)
    messages.append(response)

    if response.tool_calls:
        for tc in response.tool_calls:
            result = search_football_knowledge.invoke(tc["args"])
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        messages.append(HumanMessage(content="Now give your tactical assessment. Be concise."))
        response = llm.invoke(messages)

    return response.content
