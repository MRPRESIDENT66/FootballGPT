"""Vector-based conversation memory — stores and retrieves relevant chat history via FAISS.

Instead of a fixed sliding window (last N turns), this module embeds each conversation
turn and retrieves the most semantically relevant ones for the current query.
"""

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from config.settings import settings


class VectorMemory:
    """Semantic memory that stores conversation turns as vectors for retrieval."""

    def __init__(self):
        self._embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.DASHSCOPE_BASE_URL,
            check_embedding_ctx_length=False,
        )
        self._store = None
        self._turn_count = 0

    def add_turn(self, query: str, intent: str, report_summary: str):
        """Add a conversation turn to vector memory."""
        self._turn_count += 1
        content = f"Q: {query}\nIntent: {intent}\nA: {report_summary[:500]}"
        doc = Document(
            page_content=content,
            metadata={"turn": self._turn_count, "query": query, "intent": intent},
        )

        if self._store is None:
            self._store = FAISS.from_documents([doc], self._embeddings)
        else:
            self._store.add_documents([doc])

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """Retrieve the most relevant past turns for the current query."""
        if self._store is None or not query or not isinstance(query, str):
            return []

        docs = self._store.similarity_search(str(query), k=k)
        return [
            {
                "turn": doc.metadata["turn"],
                "query": doc.metadata["query"],
                "intent": doc.metadata["intent"],
                "content": doc.page_content,
            }
            for doc in docs
        ]

    @property
    def is_empty(self) -> bool:
        return self._store is None
