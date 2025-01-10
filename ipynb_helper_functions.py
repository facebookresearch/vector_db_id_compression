import numpy as np
from numpy.typing import NDArray

import faiss
from faiss.contrib.datasets import Dataset as FaissDataset
from faiss.contrib.inspect_tools import get_invlist
from faiss.contrib.inspect_tools import get_NSG_neighbors

from typing import Optional

def reconstruct_ivfs_from_start_ids(start_ids: NDArray[np.integer]) -> list[NDArray[np.integer]]:
    return [
        np.arange(start_ids[j], start_ids[j+1])
        for j in range(len(start_ids) - 1)
    ]

def make_start_ids_array(index: faiss.IndexIVF) -> NDArray[np.uint32]:
    num_clusters = index.nlist
    ivfs = get_ivfs(index)
    start_ids = np.zeros(num_clusters + 1, dtype=np.uint32)
    for c, ivf in enumerate(ivfs, 1):
        cluster_size = len(ivf)
        start_ids[c] = start_ids[c-1] + cluster_size 
    return start_ids

def compute_empirical_entropy_from_freqs(arr: NDArray[np.integer]) -> np.floating:
    _, freqs = np.unique(arr, return_counts=True)
    total = len(arr)
    p = freqs/total
    return -(p*np.log2(p)).sum()

def get_ivfs(index: faiss.IndexIVF) -> list[NDArray[np.integer]]:
    num_clusters = index.invlists.nlist
    return [get_invlist(index.invlists, c)[0] for c in range(num_clusters)]


def prepare_index(ds: FaissDataset, index_string: str, relabel_ids_sequentially: bool = False) -> faiss.Index:
    index = faiss.index_factory(ds.d, index_string)
    index.train(ds.get_train())
    database = ds.get_database()
    index.add(database)

    if relabel_ids_sequentially:
        permutation = np.concatenate(get_ivfs(index))
        index.reset()
        index.add(database[permutation])
        assert np.all(np.concatenate(get_ivfs(index)) == np.arange(index.ntotal))
        print('Successfully relabelled index sequentially.')

    return index


def get_cluster_ids_as_list_of_arrays(index: faiss.Index) -> list[np.floating]:
    return [
        get_invlist(index.invlists, cluster_id)[0] for cluster_id in range(index.nlist)
    ]


def compute_gaps(arr: NDArray[np.integer]) -> NDArray[np.integer]:
    assert arr.ndim == 1
    diffs: NDArray[np.integer] = np.diff(np.sort(arr)).astype(np.integer)
    assert np.all(diffs > 0)
    return diffs


def compute_avg_gap_value(arr: NDArray[np.integer]) -> np.floating:
    assert arr.ndim == 1
    diffs: NDArray[np.floating] = np.diff(np.sort(arr)).astype(np.float32)
    assert np.all(diffs > 0)
    return diffs.mean()  # type: ignore


def compute_gap_stats(
    arr: NDArray[np.integer], bound: float | np.floating
) -> dict[str, np.floating | float]:
    num_gaps = arr.shape[0] - 1
    uniform_gap_value_from_bound = bound / num_gaps
    uniform_gap_value_from_max = arr.max() / num_gaps
    avg_gap_value = compute_avg_gap_value(arr)
    ratio_max = avg_gap_value / uniform_gap_value_from_max
    ratio_bound = avg_gap_value / uniform_gap_value_from_bound

    return {
        "uniform_gap_value_from_bound": uniform_gap_value_from_bound,
        "uniform_gap_value_from_max": uniform_gap_value_from_max,
        "num_gaps": num_gaps,
        "avg_gap_value": avg_gap_value,
        "ratio_max": ratio_max,
        "ratio_bound": ratio_bound,
    }


def _test_compute_avg_gap_value() -> None:
    arr = np.arange(1000)
    assert compute_avg_gap_value(arr).mean() == 1.0
    assert compute_avg_gap_value(23 * arr).mean() == 23


def _test_compute_avg_gap_value_uniform() -> None:
    for size in range(100, 1000):
        np.random.seed(size)
        max_value = (1 << 32) - 1
        arr = np.random.randint(0, max_value + 1, size=size)
        num_gaps = arr.shape[0] - 1

        assert np.diff(arr).shape[0] == num_gaps

        uniform_gap_value_from_bound = max_value / num_gaps
        uniform_gap_value_from_max = arr.max() / num_gaps
        avg_gap_value = compute_avg_gap_value(arr)

        assert 0.95 < uniform_gap_value_from_bound / uniform_gap_value_from_max < 1.05
        assert 0.95 < avg_gap_value / uniform_gap_value_from_max < 1.05
        assert 0.94 < avg_gap_value / uniform_gap_value_from_bound < 1.0


def _get_HNSW_neighbor(
    hnsw: faiss.IndexHNSWFlat, i: int, level: int
) -> NDArray[np.int32]:
    "list the neighbors for node i at level"
    assert i < hnsw.levels.size()
    assert level < hnsw.levels.at(i)
    be = np.empty(2, "uint64")
    hnsw.neighbor_range(i, level, faiss.swig_ptr(be), faiss.swig_ptr(be[1:]))
    return np.array([hnsw.neighbors.at(j) for j in range(be[0], be[1])], dtype=np.int32)


def get_HNSW_neighbors(hnsw: faiss.IndexHNSWFlat) -> NDArray[np.int32]:
    levels_per_node = faiss.vector_to_array(hnsw.levels)
    num_nodes = len(levels_per_node)
    neighbors = list()
    for node in range(num_nodes):
        max_level = levels_per_node[node]
        neighbors.append(
            np.concatenate(
                [
                    (c := _get_HNSW_neighbor(hnsw, node, level))[c >= 0]
                    for level in range(max_level)
                ]
            )
        )

    largest_edge = max(map(len, neighbors))
    neighbors = [
        np.pad(
            edges,
            pad_width=(0, largest_edge - len(edges)),
            constant_values=-1,
        )
        for edges in neighbors
    ]
    neighbors = np.sort(np.vstack(neighbors), axis=1)  # type: ignore
    return neighbors  # type: ignore


if __name__ == "__main__":
    _test_compute_avg_gap_value()
    _test_compute_avg_gap_value_uniform()
