import datetime
from pathlib import Path

import faiss
from faiss.contrib.datasets import DatasetDeep1B, DatasetSIFT1M, SyntheticDataset
from faiss.contrib.inspect_tools import get_NSG_neighbors
from qinco_datasets import DatasetFB_ssnpp


def friend_to_edgelist_repr(graph_friends):
    return list(
        sorted(
            [v, w]
            for v, friends in enumerate(graph_friends)
            for w in friends
            if w != -1
        )
    )


def vector_to_array(v):
    """make a vector visible as a numpy array (without copying data)"""
    return faiss.rev_swig_ptr(v.data(), v.size())


def get_hnsw_links(hnsw, vno):
    """get link strcutre for vertex vno"""

    # make arrays visible from Python
    levels = vector_to_array(hnsw.levels)
    cum_nneighbor_per_level = vector_to_array(hnsw.cum_nneighbor_per_level)
    offsets = vector_to_array(hnsw.offsets)
    neighbors = vector_to_array(hnsw.neighbors)

    # all neighbors of vno
    neigh_vno = neighbors[offsets[vno] : offsets[vno + 1]]

    # break down per level
    nlevel = levels[vno]
    return [
        neigh_vno[cum_nneighbor_per_level[l] : cum_nneighbor_per_level[l + 1]]
        for l in range(nlevel)
    ]


if __name__ == "__main__":
    import sys

    _dataset = int(sys.argv[1])
    _max_degree = int(sys.argv[2])

    indexing_params = dict(
        index_str=[
            f"{graph_type}{max_degree},{code}"
            # for code in ["Flat", "PQ4", "PQ8", "PQ16"]
            for code in ["Flat"]
            for max_degree in [_max_degree]
            for graph_type in ["NSG", "HNSW"]
        ],
        dataset=[
            [
                SyntheticDataset(d=32, nt=10_000, nq=1, nb=1_000),
                DatasetSIFT1M(),
                DatasetDeep1B(nb=10**6),
                DatasetFB_ssnpp(basedir="/checkpoint/dsevero/data/fb_ssnpp/"),
            ][_dataset]
        ],
    )

    for dataset in indexing_params["dataset"]:
        for index_str in indexing_params["index_str"]:
            print(f"Indexing {dataset} / {index_str}", flush=True)
            index = faiss.index_factory(dataset.d, index_str)
            index.verbose = True
            database = dataset.get_database()
            index.add(database)

            # first run will be with no compression
            if "NSG" in index_str:
                graph_friends = get_NSG_neighbors(index.nsg)
            elif "HNSW" in index_str:
                graph_friends = [
                    get_hnsw_links(index.hnsw, v)[0] for v in range(dataset.nb)
                ]

            graph_edgelist = friend_to_edgelist_repr(graph_friends)
            graphel = "\n".join(map(lambda e: f"{e[0]} {e[1]}", graph_edgelist))
            dataset_name = type(dataset).__name__
            el_path = Path(f"graphs/{dataset_name}-{index_str}.el")
            el_path.parent.mkdir(parents=True, exist_ok=True)
            with open(el_path, "w") as f:
                f.write(graphel)
