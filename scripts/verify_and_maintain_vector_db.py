#!/usr/bin/env python3
"""Persistent Vector DB & Real Tick Dataset Health & Maintenance CLI.

Verifies and manages the 2.88-year real XAUUSD tick dataset (205M+ ticks)
and the 32-dimensional Market RAG unit-sphere vector embedding index.

Usage:
    python scripts/verify_and_maintain_vector_db.py --check
    python scripts/verify_and_maintain_vector_db.py --rebuild
    python scripts/verify_and_maintain_vector_db.py --test-rag
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Prototype root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from core.marketdata import store
from core.vertex_store import VERTEX_STORE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vector_db_maintenance")


def verify_partitioned_ticks() -> dict:
    """Verify partitioned Parquet tick store integrity and coverage."""
    cov = store.coverage("XAUUSD")
    total_ticks = cov.get("ticks", 0)
    span_years = cov.get("span_years", 0.0)
    months = cov.get("months", [])
    meets_min = cov.get("meets_two_year_minimum", False)

    first_dt = (
        datetime.fromtimestamp(cov["first_ms"] / 1000, timezone.utc).isoformat()
        if cov.get("first_ms")
        else "N/A"
    )
    last_dt = (
        datetime.fromtimestamp(cov["last_ms"] / 1000, timezone.utc).isoformat()
        if cov.get("last_ms")
        else "N/A"
    )

    logger.info("── Partitioned Parquet Tick Store ──")
    logger.info("  Total Ticks:       {:,}".format(total_ticks))
    logger.info("  Partitions:        {} monthly files".format(len(months)))
    logger.info("  Span:              {:.2f} years (Meets 2-Year Min: {})".format(span_years, meets_min))
    logger.info("  First Tick:        {}".format(first_dt))
    logger.info("  Last Tick:         {}".format(last_dt))

    return {
        "total_ticks": total_ticks,
        "partitions_count": len(months),
        "span_years": span_years,
        "first_tick": first_dt,
        "last_tick": last_dt,
        "meets_two_year_minimum": meets_min,
    }


def verify_vector_store() -> dict:
    """Verify 32-D Vector DB embeddings and manifest."""
    st = VERTEX_STORE.get_status()
    vecs, meta = VERTEX_STORE.get_vectors_and_meta()

    logger.info("── Vertex Vector DB (Market RAG) ──")
    logger.info("  Status:            {}".format(st.get("status")))
    logger.info("  Sampled Ticks:     {:,}".format(st.get("tick_count", 0)))
    logger.info("  Vector Count:      {:,} embeddings".format(len(vecs)))
    logger.info("  Vector Dimensions: {}".format(vecs.shape[1] if len(vecs) else 0))
    logger.info("  Parquet Disk Size: {} KB".format(st.get("parquet_size_kb", 0)))
    logger.info("  Vector Disk Size:  {} KB".format(st.get("vector_size_kb", 0)))
    logger.info("  Coverage Range:    {}".format(st.get("time_range", "N/A")))

    # Verify unit-sphere normalization on sample
    norms = np.linalg.norm(vecs[:, :25], axis=1)
    norm_valid = bool(np.allclose(norms, 1.0, atol=1e-3))
    logger.info("  Unit-Sphere Norm:  {} (mean: {:.4f})".format("PASSED" if norm_valid else "FAILED", float(np.mean(norms))))

    return {
        "status": st.get("status"),
        "sampled_ticks": st.get("tick_count", 0),
        "vector_count": len(vecs),
        "dimensions": int(vecs.shape[1]) if len(vecs) else 0,
        "parquet_size_kb": st.get("parquet_size_kb", 0),
        "vector_size_kb": st.get("vector_size_kb", 0),
        "time_range": st.get("time_range", "N/A"),
        "unit_sphere_normalized": norm_valid,
    }


def test_rag_query() -> list:
    """Run a test cosine-similarity pattern retrieval across the Vector DB."""
    vecs, meta = VERTEX_STORE.get_vectors_and_meta()
    if len(vecs) == 0:
        logger.error("No vector embeddings available for RAG test.")
        return []

    # Query vector: use the latest market trajectory vector
    query_vec = vecs[-1, :25]
    all_traj = vecs[:, :25]

    # Cosine similarity (both are already unit-norm)
    sims = np.dot(all_traj, query_vec)

    # Exclude trivial identity match of the exact same index
    top_indices = np.argsort(sims)[::-1][1:6]

    logger.info("── Test Market RAG Pattern Retrieval (Top-5 Matches) ──")
    results = []
    for rank, idx in enumerate(top_indices, 1):
        m = meta[idx]
        score = float(sims[idx])
        dt = datetime.fromtimestamp(m["time_ms"] / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        logger.info("  #{} Similarity: {:.1f}% | Date: {} | Price: ${:.2f} | Regime: {} | Fwd: {:+.1f} bps".format(
            rank, score * 100.0, dt, m["price"], m["regime"], m["fwd_return_bps"]
        ))
        results.append({
            "rank": rank,
            "similarity_pct": round(score * 100.0, 1),
            "date": dt,
            "price": m["price"],
            "regime": m["regime"],
            "fwd_return_bps": m["fwd_return_bps"],
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Vector DB & Real Ticks Maintenance Tool")
    parser.add_argument("--check", action="store_true", help="Perform integrity check on ticks and vectors")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild Vertex Store vector embeddings from raw ticks")
    parser.add_argument("--test-rag", action="store_true", help="Run test Market RAG pattern retrieval")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON status")

    args = parser.parse_args()

    # Default to check if no arguments given
    if not args.check and not args.rebuild and not args.test_rag:
        args.check = True
        args.test_rag = True

    if args.rebuild:
        logger.info("Rebuilding Vertex Store vector index with force_refresh=True...")
        t0 = time.time()
        res = VERTEX_STORE.build_and_compress_vertex_store(force_refresh=True)
        logger.info("Rebuild complete in {:.1f}s: {}".format(time.time() - t0, res))

    report = {}
    if args.check:
        t_res = verify_partitioned_ticks()
        v_res = verify_vector_store()
        report["ticks"] = t_res
        report["vertex_store"] = v_res

    if args.test_rag:
        rag_res = test_rag_query()
        report["rag_sample"] = rag_res

    if args.json:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
