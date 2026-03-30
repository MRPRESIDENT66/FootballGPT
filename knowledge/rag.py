"""RAG knowledge base — loads team profiles into FAISS for retrieval."""

import json
import os

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool

from config.settings import settings

_vectorstore = None


def _build_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    path = os.path.join(settings.KNOWLEDGE_DIR, "team_profiles.json")
    with open(path, "r", encoding="utf-8") as f:
        teams = json.load(f)

    documents = []
    for team in teams:
        content = (
            f"Team: {team['team']}\n"
            f"League: {team['league']}\n"
            f"Manager: {team['manager']}\n"
            f"Formation: {team['formation']}\n"
            f"Style: {team['style']}\n"
            f"Strengths: {', '.join(team['strengths'])}\n"
            f"Weaknesses: {', '.join(team['weaknesses'])}\n"
            f"Key Players: {', '.join(team['key_players'])}\n"
            f"Transfer Needs: {', '.join(team['transfer_needs'])}\n"
            f"Budget: {team['budget_estimate_million_eur']}M EUR\n"
            f"Tactical Notes: {team['tactical_notes']}"
        )
        documents.append(Document(page_content=content, metadata={"team": team["team"]}))

    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.DASHSCOPE_BASE_URL,
        check_embedding_ctx_length=False,
    )
    _vectorstore = FAISS.from_documents(documents, embeddings)
    return _vectorstore


@tool
def search_football_knowledge(query: str) -> str:
    """Search the football knowledge base for tactical info, team profiles, and transfer needs.
    Use this to understand team styles, formations, strengths, and weaknesses."""
    vs = _build_vectorstore()
    docs = vs.similarity_search(query, k=3)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
