"""Hierarchical Navigable Small World graphs (Malkov & Yashunin, 2016).

The index is a stack of proximity graphs. The top layers are sparse and act as
express lanes; layer 0 holds every point. A search starts at the top, greedily
walks towards the query, drops a layer, and repeats, then does a wider beam
search on layer 0. Cost grows roughly with log(n) instead of n.
"""
from __future__ import annotations

import heapq
import json
import math
import random

import numpy as np

from .exact import normalize


class HNSWIndex:
    def __init__(self, dim: int, metric: str = "cosine", M: int = 16, ef_construction: int = 200, seed: int = 0) -> None:
        if metric not in ("cosine", "l2"):
            raise ValueError("metric must be 'cosine' or 'l2'")
        if M < 2:
            raise ValueError("M must be at least 2")
        self.dim, self.metric = dim, metric
        self.M = M                    # max neighbours per node on layers >= 1
        self.M0 = 2 * M               # layer 0 is denser, as the paper recommends
        self.ef_construction = ef_construction
        self._level_mult = 1.0 / math.log(M)
        self._rng = random.Random(seed)
        self._vectors = np.empty((0, dim), dtype=np.float32)
        self._count = 0
        self._graphs: list[dict[int, list[int]]] = []  # one adjacency map per layer
        self._entry: int | None = None
        self._deleted: set[int] = set()
        self.distance_computations = 0  # running total, for measuring work per query

    def __len__(self) -> int:
        return self._count - len(self._deleted)

    # ---- distances -------------------------------------------------------
    def _dist(self, query: np.ndarray, ids: list[int]) -> np.ndarray:
        self.distance_computations += len(ids)
        vectors = self._vectors[ids]
        if self.metric == "cosine":
            return 1.0 - vectors @ query
        return np.linalg.norm(vectors - query, axis=1)

    def _prepare(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32).reshape(self.dim)
        return normalize(vector) if self.metric == "cosine" else vector

    # ---- core routines ---------------------------------------------------
    def _search_layer(self, query: np.ndarray, entry_points: list[tuple[float, int]], ef: int, layer: int) -> list[tuple[float, int]]:
        """Beam search on one layer. Returns up to ef (distance, id) pairs, closest first."""
        graph = self._graphs[layer]
        visited = {i for _, i in entry_points}
        candidates = list(entry_points)           # min-heap: closest unexplored first
        heapq.heapify(candidates)
        best = [(-d, i) for d, i in entry_points]  # max-heap: worst kept result on top
        heapq.heapify(best)
        while candidates:
            dist, node = heapq.heappop(candidates)
            if dist > -best[0][0] and len(best) >= ef:
                break  # everything left is farther than our worst result
            fresh = [n for n in graph[node] if n not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            for d, n in zip(self._dist(query, fresh).tolist(), fresh):
                if len(best) < ef or d < -best[0][0]:
                    heapq.heappush(candidates, (d, n))
                    heapq.heappush(best, (-d, n))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted((-d, i) for d, i in best)

    def _select_neighbours(self, candidates: list[tuple[float, int]], m: int) -> list[int]:
        """The paper's heuristic (Algorithm 4).

        Take candidates closest-first, but skip one if it is closer to an already
        chosen neighbour than to the new point. This keeps edges pointing in
        different directions, which is what lets the graph bridge clusters.
        """
        chosen: list[int] = []
        skipped: list[int] = []
        for dist, cand in candidates:
            if len(chosen) >= m:
                break
            if chosen and float(self._dist(self._vectors[cand], chosen).min()) < dist:
                skipped.append(cand)
            else:
                chosen.append(cand)
        chosen.extend(skipped[: m - len(chosen)])  # top up so nodes are not left under-connected
        return chosen

    # ---- public API ------------------------------------------------------
    def add(self, vector: np.ndarray) -> int:
        """Insert one vector and return its id."""
        vector = self._prepare(vector)
        if self._count == len(self._vectors):  # grow geometrically
            grown = np.empty((max(1024, 2 * len(self._vectors)), self.dim), dtype=np.float32)
            grown[: self._count] = self._vectors[: self._count]
            self._vectors = grown
        node = self._count
        self._vectors[node] = vector
        self._count += 1

        level = int(-math.log(1.0 - self._rng.random()) * self._level_mult)
        while len(self._graphs) <= level:
            self._graphs.append({})
        for layer in range(level + 1):
            self._graphs[layer][node] = []

        if self._entry is None:
            self._entry = node
            return node

        top = self._top_level(self._entry)
        entry = [(float(self._dist(vector, [self._entry])[0]), self._entry)]
        for layer in range(top, level, -1):  # greedy descent through layers above the node's level
            entry = self._search_layer(vector, entry, 1, layer)
        for layer in range(min(level, top), -1, -1):
            found = self._search_layer(vector, entry, self.ef_construction, layer)
            limit = self.M0 if layer == 0 else self.M
            neighbours = self._select_neighbours(found, self.M)
            graph = self._graphs[layer]
            graph[node] = neighbours
            for other in neighbours:  # edges are bidirectional; prune anyone now over the limit
                links = graph[other]
                links.append(node)
                if len(links) > limit:
                    dists = self._dist(self._vectors[other], links).tolist()
                    graph[other] = self._select_neighbours(sorted(zip(dists, links)), limit)
            entry = found
        if level > top:
            self._entry = node
        return node

    def add_batch(self, vectors: np.ndarray) -> list[int]:
        return [self.add(v) for v in np.asarray(vectors, dtype=np.float32).reshape(-1, self.dim)]

    def _top_level(self, node: int) -> int:
        for layer in range(len(self._graphs) - 1, -1, -1):
            if node in self._graphs[layer]:
                return layer
        return 0

    def search(self, query: np.ndarray, k: int = 10, ef: int | None = None) -> list[tuple[int, float]]:
        """Approximate k nearest neighbours as (id, distance), closest first.

        ``ef`` is the beam width on layer 0: higher is more accurate and slower.
        """
        if self._entry is None:
            return []
        query = self._prepare(query)
        ef = max(ef or 2 * k, k)
        if self._deleted:
            ef += min(len(self._deleted), ef)  # leave room for tombstones we will filter out
        entry = [(float(self._dist(query, [self._entry])[0]), self._entry)]
        for layer in range(len(self._graphs) - 1, 0, -1):
            entry = self._search_layer(query, entry, 1, layer)
        found = self._search_layer(query, entry, ef, 0)
        return [(i, d) for d, i in found if i not in self._deleted][:k]

    def delete(self, node: int) -> None:
        """Soft delete: the node stays in the graph as a waypoint but is never returned."""
        if not 0 <= node < self._count:
            raise KeyError(node)
        self._deleted.add(node)

    # ---- persistence -----------------------------------------------------
    def save(self, path: str) -> None:
        meta = {"dim": self.dim, "metric": self.metric, "M": self.M, "ef_construction": self.ef_construction,
                "entry": self._entry, "deleted": sorted(self._deleted),
                "graphs": [{str(k): v for k, v in g.items()} for g in self._graphs]}
        np.savez_compressed(path, vectors=self._vectors[: self._count], meta=np.array(json.dumps(meta)))

    @classmethod
    def load(cls, path: str) -> "HNSWIndex":
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        index = cls(meta["dim"], meta["metric"], meta["M"], meta["ef_construction"])
        index._vectors = data["vectors"].astype(np.float32)
        index._count = len(index._vectors)
        index._entry = meta["entry"]
        index._deleted = set(meta["deleted"])
        index._graphs = [{int(k): v for k, v in g.items()} for g in meta["graphs"]]
        return index

    def stats(self) -> dict:
        degrees = [len(v) for v in self._graphs[0].values()] if self._graphs else [0]
        return {"vectors": len(self), "layers": len(self._graphs),
                "nodes_per_layer": [len(g) for g in self._graphs],
                "mean_degree_layer0": round(sum(degrees) / max(len(degrees), 1), 2)}
