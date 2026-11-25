"""Market Pattern Retrieval-Augmented Generation (RAG) & Vector Intelligence Engine.

Performs high-dimensional cosine similarity search across historical microstructural vertex embeddings:
- Finds identical past price action, liquidity sweeps, and volatility bursts.
- Analyzes empirical forward return distributions (win probability, expectancy).
- Feeds retrieved vector context directly into the AI Copilot for grounded quant intelligence.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

from core.vertex_store import VERTEX_STORE, NoTickDataError


class MarketRAGEngine:
    """Vector database and semantic search for market microstructures."""

    def __init__(self):
        self.store = VERTEX_STORE

    def _vectors_and_meta(self) -> tuple[np.ndarray, List[Dict[str, Any]]]:
        """Return embeddings, or empty results when no real ticks exist."""
        try:
            return self.store.get_vectors_and_meta()
        except NoTickDataError:
            return np.empty((0, 32), dtype=np.float32), []

    def search_similar_patterns(
        self,
        query_vector: Optional[np.ndarray] = None,
        top_k: int = 5,
        min_similarity: float = 0.70,
    ) -> List[Dict[str, Any]]:
        """Searches historical embeddings for the closest microstructural patterns."""
        vectors, metadata = self._vectors_and_meta()
        if len(vectors) == 0:
            return []

        # If no query vector provided, use the most recent historical embedding as query
        if query_vector is None:
            query_vector = vectors[-1]

        # Use the first 25 dimensions (normalized features) for cosine similarity
        q_feat = query_vector[:25]
        q_norm = np.linalg.norm(q_feat)
        if q_norm > 0:
            q_feat = q_feat / q_norm

        db_feats = vectors[:, :25]

        # Dot product of unit vectors = Cosine Similarity
        similarities = np.dot(db_feats, q_feat)

        # Exclude the exact same query window if querying recent
        if query_vector is vectors[-1]:
            similarities[-1] = -1.0

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            sim = float(similarities[idx])
            if sim < min_similarity:
                continue

            meta = metadata[idx]
            t_dt = datetime.fromtimestamp(meta["time_ms"] / 1000.0, timezone.utc)

            # Determine forward outcome
            fwd_bps = meta["fwd_return_bps"]
            outcome = "BULLISH_EXPANSION" if fwd_bps > 2.0 else "BEARISH_EXPANSION" if fwd_bps < -2.0 else "CHOP_RANGE"

            results.append({
                "rank": rank + 1,
                "similarity_pct": round(sim * 100.0, 1),
                "timestamp": t_dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "price": round(meta["price"], 2),
                "regime": meta["regime"],
                "volatility": round(meta["volatility"], 2),
                "fwd_return_bps": round(fwd_bps, 1),
                "fwd_return_usd": round(meta["price"] * (fwd_bps / 10000.0), 2),
                "outcome": outcome,
                "vector_id": int(idx),
            })

        return results

    def get_market_pattern_rag_context(self, current_price: float = 3945.50) -> Dict[str, Any]:
        """Generates statistical forward expectancy from top-10 historical matches."""
        matches = self.search_similar_patterns(top_k=10, min_similarity=0.75)
        if not matches:
            return {
                "rag_available": False,
                "summary": (
                    "No real tick history is ingested; pattern retrieval is "
                    "unavailable until a real tick dataset is loaded."
                ),
                "matches": [],
            }

        fwd_returns = [m["fwd_return_bps"] for m in matches]
        bullish_count = sum(1 for r in fwd_returns if r > 0)
        win_rate = (bullish_count / len(fwd_returns)) * 100.0 if fwd_returns else 50.0
        avg_fwd_bps = float(np.mean(fwd_returns)) if fwd_returns else 0.0

        summary = (
            f"Retrieved {len(matches)} historical microstructural patterns with average similarity of "
            f"{np.mean([m['similarity_pct'] for m in matches]):.1f}%. "
            f"Historical forward outcome: {win_rate:.0f}% Bullish with average drift of {avg_fwd_bps:+.1f} bps."
        )

        return {
            "rag_available": True,
            "summary": summary,
            "win_rate_pct": round(win_rate, 1),
            "avg_forward_return_bps": round(avg_fwd_bps, 1),
            "sample_size": len(matches),
            "matches": matches,
        }

    def augment_copilot_prompt(self, user_prompt: str) -> str:
        """Enriches user query with historical RAG pattern intelligence."""
        rag_data = self.get_market_pattern_rag_context()
        if not rag_data.get("rag_available"):
            return user_prompt

        context = (
            f"\n[HISTORICAL VERTEX RAG CONTEXT]\n"
            f"- Pattern Similarity Matches: {rag_data['sample_size']} episodes found in MT5 tick database\n"
            f"- Historical Win Rate: {rag_data['win_rate_pct']}% Bullish\n"
            f"- Average Forward Move: {rag_data['avg_forward_return_bps']} basis points\n"
            f"- Top Match Episode: {rag_data['matches'][0]['timestamp']} (Similarity: {rag_data['matches'][0]['similarity_pct']}%)\n"
            f"[/HISTORICAL VERTEX RAG CONTEXT]\n\n"
            f"User Question: {user_prompt}"
        )
        return context


# Global Market RAG Engine Singleton
MARKET_RAG_ENGINE = MarketRAGEngine()
