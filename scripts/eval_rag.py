"""Evaluate RAG retrieval quality for tactical knowledge base.

Tests whether the FAISS vector store retrieves the correct team profiles
for given queries. Measures Recall@K and MRR (Mean Reciprocal Rank).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge.rag import _build_vectorstore

TEST_CASES = [
    # Direct team queries
    {"query": "Liverpool high pressing style", "relevant": ["Liverpool"]},
    {"query": "Barcelona tiki-taka possession", "relevant": ["Barcelona"]},
    {"query": "Guardiola's positional play at Man City", "relevant": ["Manchester City"]},
    {"query": "Atletico Madrid defensive block counter-attack", "relevant": ["Atletico Madrid"]},
    {"query": "Bayern Munich formation under Kompany", "relevant": ["Bayern Munich"]},

    # Style-based queries (should find matching teams)
    {"query": "Which team plays with inverted fullbacks?", "relevant": ["Manchester City"]},
    {"query": "Teams that use a back three formation", "relevant": ["Manchester United", "Bayer Leverkusen", "Atalanta"]},
    {"query": "Who plays gegenpressing?", "relevant": ["Liverpool"]},
    {"query": "Best counter-attacking teams in Europe", "relevant": ["Atletico Madrid", "Real Madrid"]},
    {"query": "Youth academy development focus", "relevant": ["Barcelona", "Ajax", "Benfica"]},

    # Transfer/need queries
    {"query": "Teams that need a striker", "relevant": ["Arsenal", "Paris Saint Germain", "Chelsea"]},
    {"query": "Who needs a defensive midfielder?", "relevant": ["Manchester City", "Real Madrid"]},

    # Tactical concept queries
    {"query": "Wing-back system in Serie A", "relevant": ["Inter", "Napoli", "Atalanta"]},
    {"query": "High defensive line risk", "relevant": ["Barcelona", "Tottenham", "Benfica"]},
    {"query": "Set piece specialists Premier League", "relevant": ["Arsenal", "Brentford"]},
]


def evaluate_retrieval(k: int = 3):
    print(f"⚽ RAG Retrieval Quality Evaluation (k={k})")
    print(f"📊 Test cases: {len(TEST_CASES)}")
    print("=" * 60)

    vs = _build_vectorstore()

    total_recall = 0
    total_mrr = 0
    results_detail = []

    for i, case in enumerate(TEST_CASES, 1):
        query = case["query"]
        relevant = set(case["relevant"])

        docs = vs.similarity_search(query, k=k)
        retrieved = [doc.metadata.get("team", "") for doc in docs]

        # Recall@K: how many relevant docs were found
        hits = set(retrieved) & relevant
        recall = len(hits) / len(relevant) if relevant else 0

        # MRR: position of first relevant result
        mrr = 0
        for rank, team in enumerate(retrieved, 1):
            if team in relevant:
                mrr = 1.0 / rank
                break

        total_recall += recall
        total_mrr += mrr

        icon = "✅" if recall > 0 else "❌"
        print(f"  [{i:2d}] {icon} \"{query[:55]}\"")
        print(f"       Expected: {relevant}")
        print(f"       Got:      {retrieved}")
        print(f"       Recall: {recall:.2f} | MRR: {mrr:.2f}")

        results_detail.append({
            "query": query,
            "recall": recall,
            "mrr": mrr,
        })

    # Summary
    n = len(TEST_CASES)
    avg_recall = total_recall / n
    avg_mrr = total_mrr / n

    print("\n" + "=" * 60)
    print(f"📊 Average Recall@{k}: {avg_recall:.3f}")
    print(f"📊 Average MRR:       {avg_mrr:.3f}")
    print()

    if avg_recall >= 0.7:
        print("✅ RAG retrieval quality is strong.")
    elif avg_recall >= 0.5:
        print("⚠️  RAG retrieval quality is acceptable but could improve.")
    else:
        print("❌ RAG retrieval quality needs improvement. Consider adding more data or tuning embeddings.")


if __name__ == "__main__":
    evaluate_retrieval(k=3)
