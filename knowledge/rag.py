"""RAG knowledge base — retrieves football knowledge from Qdrant.

Replaces the old FAISS + team_profiles.json approach with Qdrant-backed
Wikipedia knowledge (players, teams, seasons) with payload filtering.

Two interfaces:
  - search_football_knowledge: LangChain @tool for Tactics agent (tool call)
  - retrieve_knowledge: plain function for Scout/Analyst/Reporter (direct injection)
"""

from typing import Optional, List

from langchain_core.tools import tool

from knowledge.qdrant_store import FootballKnowledgeStore

_store = None


def _get_store() -> FootballKnowledgeStore:
    global _store
    if _store is None:
        _store = FootballKnowledgeStore()
    return _store


def _format_results(results: List[dict]) -> str:
    """Format Qdrant search results into readable text."""
    if not results:
        return ""

    pieces = []
    for r in results:
        source = r.get("source_url", "")
        header = f"[{r.get('entity_type', '')}] {r.get('name', '')} — {r.get('section', '')}"
        pieces.append(f"{header}\n{r['text']}\n({source})")

    return "\n\n---\n\n".join(pieces)


def retrieve_knowledge(
    query: str,
    entity_type: Optional[str] = None,
    league: Optional[str] = None,
    position: Optional[str] = None,
    section: Optional[str] = None,
    limit: int = 5,
) -> str:
    """Retrieve knowledge from Qdrant with optional filtering.

    Used by Scout/Analyst/Reporter to directly inject context into prompts.
    """
    store = _get_store()
    results = store.search(
        query,
        limit=limit,
        entity_type=entity_type,
        league=league,
        position=position,
        section=section,
    )
    return _format_results(results)


@tool
def search_football_knowledge(query: str) -> str:
    """Search the football knowledge base for player backgrounds, team history,
    tactical styles, youth academy info, transfer history, and season data.
    Use this to get real-world context beyond raw statistics."""
    store = _get_store()
    results = store.search(query, limit=5)
    return _format_results(results) or "No relevant knowledge found."
