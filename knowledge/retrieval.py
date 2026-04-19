"""Shared retrieval and task-specific query rewriting helpers."""

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config.settings import settings
from knowledge.rag import retrieve_knowledge

_REWRITE_PROMPT = """You rewrite football retrieval queries for a vector search system.

Given:
- the user's original query
- the current agent role
- structured criteria
- already retrieved shared knowledge

Return ONLY valid JSON:
{
  "needs_additional_retrieval": true,
  "queries": ["up to 3 short retrieval queries"]
}

Rules:
- Queries must be concise, retrieval-oriented, and entity-specific.
- Do not answer the user's question.
- Prefer football entities, tactical roles, playing style, squad gaps, career background, or honours.
- If shared knowledge already looks sufficient, set needs_additional_retrieval to false and queries to [].
- For analyst, focus on player/candidate background.
- For tactics, focus on team system, role requirements, squad gaps.
- For reporter, focus on missing summary background that would improve the final writeup."""


def build_shared_knowledge(query: str, intent: str, criteria: dict) -> str:
    """Retrieve a shared baseline context once per turn."""
    sections: list[str] = []
    team = criteria.get("team")
    league = criteria.get("league")
    position = criteria.get("position")

    if team:
        team_query = f"{team} tactical system squad strengths weaknesses"
        team_knowledge = retrieve_knowledge(
            team_query,
            entity_type="team",
            league=league,
            limit=3,
        )
        if team_knowledge:
            sections.append(f"## Team Knowledge\n{team_knowledge}")

    if intent in {"compare", "recommend", "scout"}:
        player_query = query
        player_knowledge = retrieve_knowledge(
            player_query,
            entity_type="player",
            league=league,
            position=position,
            limit=4,
        )
        if player_knowledge:
            sections.append(f"## Player Knowledge\n{player_knowledge}")

    if intent == "tactics" and not team:
        general_team_knowledge = retrieve_knowledge(query, entity_type="team", limit=3)
        if general_team_knowledge:
            sections.append(f"## Team Knowledge\n{general_team_knowledge}")

    return "\n\n".join(sections)


def _contains_term(text: str, term: str) -> bool:
    """Case-insensitive exact-ish term containment check."""
    if not text or not term:
        return False
    pattern = r"\b" + re.escape(term.lower()) + r"\b"
    return re.search(pattern, text.lower()) is not None


def _extract_candidate_names(text: str, max_names: int = 3) -> list[str]:
    """Pull candidate names from scout/analysis text lines like '- Name ('."""
    if not text:
        return []

    names: list[str] = []
    patterns = [
        r"(?:^|\n)[-*]?\s*([A-Z][A-Za-zÀ-ÿ'`.-]+(?:\s+[A-Z][A-Za-zÀ-ÿ'`.-]+){0,2})\s*\(",
        r'"name"\s*:\s*"([^"]+)"',
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text):
            candidate = match.strip()
            if candidate and candidate not in names:
                names.append(candidate)
            if len(names) >= max_names:
                return names
    return names


def should_retrieve_for_analyst(
    query: str,
    criteria: dict,
    shared_knowledge: str,
    player_data: str = "",
) -> bool:
    """Rule-based gate for analyst-specific retrieval."""
    intent_terms = ["compare", "like", "similar", "alternative", "replace", "recommend"]
    if any(term in query.lower() for term in intent_terms):
        player_names = criteria.get("player_names", []) or []
        if player_names and any(not _contains_term(shared_knowledge, name) for name in player_names):
            return True

        candidates = _extract_candidate_names(player_data)
        if candidates and any(not _contains_term(shared_knowledge, name) for name in candidates):
            return True

    return False


def should_retrieve_for_tactics(query: str, criteria: dict, shared_knowledge: str) -> bool:
    """Rule-based gate for tactics-specific retrieval."""
    team = criteria.get("team")
    position = criteria.get("position")
    tactical_terms = ["system", "fit", "weakness", "need", "replace", "role", "tactic"]

    if team and not _contains_term(shared_knowledge, team):
        return True

    if team and position:
        role_phrase = f"{position.lower()} role"
        if role_phrase not in shared_knowledge.lower():
            return True

    return any(term in query.lower() for term in tactical_terms) and bool(team)


def should_retrieve_for_reporter(
    query: str,
    criteria: dict,
    shared_knowledge: str,
    scout_data: str = "",
    analysis: str = "",
) -> bool:
    """Rule-based gate for reporter-specific retrieval."""
    if criteria.get("player_names"):
        names = criteria["player_names"]
    else:
        names = _extract_candidate_names(analysis) or _extract_candidate_names(scout_data)

    if not names:
        return False

    # Reporter only supplements when named entities are still missing from prior context.
    return any(not _contains_term(shared_knowledge, name) for name in names[:2])


def should_retrieve_for_agent(
    agent_role: str,
    query: str,
    criteria: dict,
    shared_knowledge: str,
    player_data: str = "",
    analysis: str = "",
) -> bool:
    """Dispatch rule-based retrieval gating by agent role."""
    if agent_role == "analyst":
        return should_retrieve_for_analyst(query, criteria, shared_knowledge, player_data)
    if agent_role == "tactics":
        return should_retrieve_for_tactics(query, criteria, shared_knowledge)
    if agent_role == "reporter":
        return should_retrieve_for_reporter(query, criteria, shared_knowledge, player_data, analysis)
    return False


def rewrite_retrieval_queries(
    agent_role: str,
    query: str,
    criteria: dict,
    shared_knowledge: str,
) -> list[str]:
    """Use a lightweight model to generate task-specific retrieval queries."""
    llm = ChatOpenAI(
        model=settings.RETRIEVAL_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.DASHSCOPE_BASE_URL,
        temperature=0.1,
        **settings.NO_THINKING,
    )

    shared_preview = shared_knowledge[:1500] if shared_knowledge else ""
    payload = {
        "agent_role": agent_role,
        "query": query,
        "criteria": criteria,
        "shared_knowledge_preview": shared_preview,
    }
    response = llm.invoke(
        [
            SystemMessage(content=_REWRITE_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]
    )

    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not parsed.get("needs_additional_retrieval"):
        return []

    queries = parsed.get("queries", [])
    return [q.strip() for q in queries if isinstance(q, str) and q.strip()][:3]


def run_agent_retrieval(
    agent_role: str,
    query: str,
    criteria: dict,
    shared_knowledge: str,
    player_data: str = "",
    analysis: str = "",
) -> str:
    """Retrieve incremental, task-specific context for one agent role."""
    if not should_retrieve_for_agent(
        agent_role,
        query,
        criteria,
        shared_knowledge,
        player_data=player_data,
        analysis=analysis,
    ):
        return ""

    queries = rewrite_retrieval_queries(agent_role, query, criteria, shared_knowledge)
    if not queries:
        return ""

    league = criteria.get("league")
    position = criteria.get("position")
    team = criteria.get("team")
    sections: list[str] = []

    for q in queries:
        entity_type = None
        if agent_role == "tactics":
            entity_type = "team"
        elif agent_role == "analyst":
            entity_type = "player"

        result = retrieve_knowledge(
            q,
            entity_type=entity_type,
            league=league,
            position=position if agent_role == "analyst" else None,
            limit=3,
        )
        if result:
            sections.append(f"### Query: {q}\n{result}")

        if team and agent_role == "tactics" and not result:
            fallback = retrieve_knowledge(f"{team} {q}", entity_type="team", league=league, limit=2)
            if fallback:
                sections.append(f"### Query: {team} {q}\n{fallback}")

    return "\n\n".join(sections)
