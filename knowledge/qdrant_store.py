"""Qdrant vector store — stores and retrieves football knowledge chunks.

Uses local file-based Qdrant (no Docker needed). Can be switched to
remote Qdrant server/cloud by changing the client initialization.
"""

import uuid
import logging
from typing import Optional, List, Dict
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)
from langchain_openai import OpenAIEmbeddings

from config.settings import settings
from knowledge.wiki_scraper import WikiChunk

logger = logging.getLogger(__name__)

_DEFAULT_QDRANT_PATH = str(
    Path(__file__).resolve().parent.parent / "data" / "qdrant_store"
)
_COLLECTION_NAME = "football_knowledge"
_VECTOR_DIM = 1024  # text-embedding-v4 outputs 1024-dim vectors


class FootballKnowledgeStore:
    """Qdrant-backed vector store for football Wikipedia knowledge.

    Usage:
        store = FootballKnowledgeStore()
        store.upsert_chunks(chunks)          # write
        results = store.search("fast winger", entity_type="player", league="Premier League")
    """

    def __init__(self, qdrant_path: Optional[str] = None):
        self._path = qdrant_path or _DEFAULT_QDRANT_PATH
        self._client = QdrantClient(path=self._path)
        self._embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.DASHSCOPE_BASE_URL,
            check_embedding_ctx_length=False,
        )
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = [c.name for c in self._client.get_collections().collections]
        if _COLLECTION_NAME not in collections:
            self._client.create_collection(
                collection_name=_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=_VECTOR_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {_COLLECTION_NAME}")

    # ---- Write ----

    def upsert_chunks(self, chunks: List[WikiChunk], batch_size: int = 10):
        """Embed and upsert chunks into Qdrant in batches."""
        if not chunks:
            return

        total = len(chunks)
        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]

            # Batch embed
            vectors = self._embeddings.embed_documents(texts)

            points = []
            for chunk, vector in zip(batch, vectors):
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=chunk.to_payload(),
                ))

            self._client.upsert(
                collection_name=_COLLECTION_NAME,
                points=points,
            )
            logger.info(f"Upserted batch {i // batch_size + 1} ({len(batch)} chunks)")

        logger.info(f"Total upserted: {total} chunks")

    # ---- Read ----

    def search(
        self,
        query: str,
        limit: int = 5,
        entity_type: Optional[str] = None,
        league: Optional[str] = None,
        position: Optional[str] = None,
        section: Optional[str] = None,
        nationality: Optional[str] = None,
    ) -> List[dict]:
        """Semantic search with optional payload filtering.

        Args:
            query: Natural language search query.
            limit: Max results to return.
            entity_type: Filter by "player", "team", or "season".
            league: Filter by league name.
            position: Filter by position code (FW, MF, DF, GK).
            section: Filter by section type (career, style_of_play, etc.).
            nationality: Filter by nationality.

        Returns:
            List of dicts with 'text', 'score', and payload fields.
        """
        query_vector = self._embeddings.embed_query(query)

        # Build filter conditions
        must_conditions = []
        if entity_type:
            must_conditions.append(
                FieldCondition(key="entity_type", match=MatchValue(value=entity_type))
            )
        if league:
            must_conditions.append(
                FieldCondition(key="league", match=MatchValue(value=league))
            )
        if position:
            must_conditions.append(
                FieldCondition(key="position", match=MatchValue(value=position))
            )
        if section:
            must_conditions.append(
                FieldCondition(key="section", match=MatchValue(value=section))
            )
        if nationality:
            must_conditions.append(
                FieldCondition(key="nationality", match=MatchValue(value=nationality))
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        response = self._client.query_points(
            collection_name=_COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        )

        return [
            {
                "score": hit.score,
                **hit.payload,
            }
            for hit in response.points
        ]

    # ---- Stats ----

    def count(self) -> int:
        """Return total number of vectors in the collection."""
        info = self._client.get_collection(_COLLECTION_NAME)
        return info.points_count

    def count_by_type(self) -> Dict[str, int]:
        """Return vector counts grouped by entity_type."""
        counts = {}
        for entity_type in ["player", "team", "season"]:
            result = self._client.count(
                collection_name=_COLLECTION_NAME,
                count_filter=Filter(
                    must=[FieldCondition(key="entity_type", match=MatchValue(value=entity_type))]
                ),
            )
            counts[entity_type] = result.count
        return counts
