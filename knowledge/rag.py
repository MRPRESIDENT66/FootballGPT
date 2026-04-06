"""RAG knowledge base — retrieves football knowledge from Qdrant.

Replaces the old FAISS + team_profiles.json approach with Qdrant-backed
Wikipedia knowledge (players, teams, seasons) with payload filtering.
"""

from langchain_core.tools import tool

from knowledge.qdrant_store import FootballKnowledgeStore

_store = None


def _get_store() -> FootballKnowledgeStore:
    global _store
    if _store is None:
        _store = FootballKnowledgeStore()
    return _store


@tool
def search_football_knowledge(query: str) -> str:
    """Search the football knowledge base for player backgrounds, team history,
    tactical styles, youth academy info, transfer history, and season data.
    Use this to get real-world context beyond raw statistics."""
    store = _get_store()
    results = store.search(query, limit=5)

    if not results:
        return "No relevant knowledge found."

    pieces = []
    for r in results:
        source = r.get("source_url", "")
        header = f"[{r.get('entity_type', '')}] {r.get('name', '')} — {r.get('section', '')}"
        pieces.append(f"{header}\n{r['text']}\n({source})")

    return "\n\n---\n\n".join(pieces)
