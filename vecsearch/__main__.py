"""Benchmark HNSW against exact search: python -m vecsearch [--n 20000] [--dim 64] [--queries 200]"""
from __future__ import annotations

import argparse
import time

import numpy as np

from . import ExactIndex, HNSWIndex


def clustered(n: int, dim: int, clusters: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic data shaped like embeddings: points gathered around cluster centres."""
    centres = rng.normal(size=(clusters, dim))
    return (centres[rng.integers(0, clusters, size=n)] + 0.35 * rng.normal(size=(n, dim))).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    # Queries are held-out points from the same distribution as the data.
    everything = clustered(args.n + args.queries, args.dim, clusters=50, rng=rng)
    data, queries = everything[: args.n], everything[args.n :]

    exact = ExactIndex(args.dim)
    exact.add(data)
    start = time.perf_counter()
    truth = [{i for i, _ in exact.search(q, args.k)} for q in queries]
    exact_qps = args.queries / (time.perf_counter() - start)

    index = HNSWIndex(args.dim, M=16, ef_construction=200)
    start = time.perf_counter()
    index.add_batch(data)
    build = time.perf_counter() - start

    print(f"{args.n:,} vectors, {args.dim} dimensions, cosine distance, recall@{args.k} over {args.queries} queries")
    print(f"HNSW build: {build:.1f} s ({args.n / build:,.0f} inserts/s). Graph: {index.stats()}\n")
    print("| Method | Recall@10 | Vectors compared per query | Share of dataset | Queries / second |")
    print("|---|---:|---:|---:|---:|")
    print(f"| Exact (NumPy brute force) | 1.000 | {args.n:,} | 100% | {exact_qps:,.0f} |")
    for ef in (10, 20, 50, 100, 200):
        index.distance_computations = 0
        start = time.perf_counter()
        results = [{i for i, _ in index.search(q, args.k, ef=ef)} for q in queries]
        qps = args.queries / (time.perf_counter() - start)
        compared = index.distance_computations / args.queries
        recall = sum(len(r & t) for r, t in zip(results, truth)) / (args.k * args.queries)
        print(f"| HNSW, ef={ef} | {recall:.3f} | {compared:,.0f} | {compared / args.n:.1%} | {qps:,.0f} |")

    # Harder case: queries from clusters the index has never seen.
    strange = clustered(args.queries, args.dim, clusters=50, rng=np.random.default_rng(99))
    strange_truth = [{i for i, _ in exact.search(q, args.k)} for q in strange]
    print("\nOut-of-distribution queries (drawn from different clusters than the data):\n")
    print("| Method | Recall@10 | Vectors compared per query |")
    print("|---|---:|---:|")
    for ef in (50, 200, 800):
        index.distance_computations = 0
        results = [{i for i, _ in index.search(q, args.k, ef=ef)} for q in strange]
        recall = sum(len(r & t) for r, t in zip(results, strange_truth)) / (args.k * args.queries)
        print(f"| HNSW, ef={ef} | {recall:.3f} | {index.distance_computations / args.queries:,.0f} |")


if __name__ == "__main__":
    main()
