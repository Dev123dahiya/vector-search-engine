import numpy as np
import pytest

from vecsearch import ExactIndex, HNSWIndex


def make_data(n=2000, dim=24, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(20, dim))
    return (centres[rng.integers(0, 20, size=n)] + 0.3 * rng.normal(size=(n, dim))).astype(np.float32)


@pytest.fixture(scope="module")
def built():
    data = make_data()
    exact = ExactIndex(24)
    exact.add(data)
    index = HNSWIndex(24, M=12, ef_construction=100, seed=1)
    index.add_batch(data)
    return data, exact, index


def recall(index, exact, queries, k=10, ef=None):
    hits = 0
    for q in queries:
        truth = {i for i, _ in exact.search(q, k)}
        hits += len(truth & {i for i, _ in index.search(q, k, ef=ef)})
    return hits / (k * len(queries))


def test_exact_search_is_sorted_and_correct():
    index = ExactIndex(2, metric="l2")
    index.add(np.array([[0, 0], [3, 4], [1, 0], [10, 10]], dtype=np.float32))
    assert index.search(np.array([0, 0]), k=3) == [(0, 0.0), (2, 1.0), (1, 5.0)]
    assert len(index.search(np.array([0, 0]), k=99)) == 4


def test_a_stored_vector_finds_itself_first(built):
    data, _, index = built
    for i in (0, 17, 999, 1999):
        top_id, top_dist = index.search(data[i], k=1, ef=50)[0]
        assert top_dist < 1e-5  # an exact duplicate may tie, so compare distance not id


def test_recall_is_high_and_grows_with_ef(built):
    data, exact, index = built
    queries = data[:100] + 0.05 * np.random.default_rng(7).normal(size=(100, 24)).astype(np.float32)
    low, high = recall(index, exact, queries, ef=10), recall(index, exact, queries, ef=100)
    assert high >= 0.97
    assert high >= low


def test_results_are_sorted_and_distances_match_exact(built):
    data, exact, index = built
    query = make_data(1, seed=3)[0]
    results = index.search(query, k=10, ef=200)
    dists = [d for _, d in results]
    assert dists == sorted(dists) and len({i for i, _ in results}) == 10
    truth = dict(exact.search(query, k=2000))
    for i, d in results:
        assert d == pytest.approx(truth[i], abs=1e-5)


def test_search_touches_a_small_fraction_of_the_data(built):
    """The point of the index: far fewer distance computations than a full scan."""
    data, _, index = built
    index.distance_computations = 0
    index.search(data[123] + 0.01, k=10, ef=50)
    assert 0 < index.distance_computations < 0.4 * len(data)


def test_graph_respects_degree_limits(built):
    _, _, index = built
    assert max(len(v) for v in index._graphs[0].values()) <= index.M0
    for layer in index._graphs[1:]:
        assert all(len(v) <= index.M for v in layer.values())
    sizes = index.stats()["nodes_per_layer"]
    assert sizes == sorted(sizes, reverse=True) and sizes[0] == 2000


def test_soft_delete_hides_results(built):
    data, _, _ = built
    index = HNSWIndex(24, M=8, ef_construction=60)
    index.add_batch(data[:300])
    target = index.search(data[5], k=1)[0][0]
    index.delete(target)
    assert target not in {i for i, _ in index.search(data[5], k=10)}
    assert len(index.search(data[5], k=10)) == 10 and len(index) == 299
    with pytest.raises(KeyError):
        index.delete(10_000)


def test_save_and_load_round_trip(tmp_path, built):
    data, _, index = built
    path = str(tmp_path / "index.npz")
    index.save(path)
    loaded = HNSWIndex.load(path)
    query = make_data(1, seed=9)[0]
    assert loaded.search(query, k=10, ef=80) == index.search(query, k=10, ef=80)
    assert loaded.stats() == index.stats()


def test_l2_metric_and_edge_cases():
    index = HNSWIndex(3, metric="l2", M=4, ef_construction=20)
    assert index.search(np.zeros(3), k=5) == []
    index.add(np.array([1.0, 0, 0]))
    assert index.search(np.array([1.0, 0, 0]), k=5) == [(0, 0.0)]
    index.add_batch(np.eye(3, dtype=np.float32) * 5)
    assert index.search(np.array([0, 5.0, 0]), k=1)[0][0] == 2
    with pytest.raises(ValueError):
        HNSWIndex(3, metric="manhattan")
