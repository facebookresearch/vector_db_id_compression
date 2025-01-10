import datetime

import faiss
import numpy as np
import pandas as pd
from faiss.contrib.datasets import DatasetDeep1B, DatasetSIFT1M, SyntheticDataset
from faiss.contrib.inspect_tools import get_NSG_neighbors
from qinco_datasets import DatasetFB_ssnpp
from rec.definitions import Graph
from rec.models import PolyasUrnModel, UniformNodesModel


def friend_to_edgelist_repr(graph_friends):
    return np.array(
        [[v, w] for v, friends in enumerate(graph_friends) for w in friends if w != -1]
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

    now = datetime.datetime.now()
    _dataset = int(sys.argv[1])
    _max_degree = int(sys.argv[2])
    csv_path = (
        f"benchmark-results-graphs/graph-results-{now}-{_dataset}-{_max_degree}.csv"
    )

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
        compression_method=[("uniform", {})]
        + [
            ("rec-pu", dict(bias=bias)) for bias in [1, 10, 100, 1_000, 10_000, 100_000]
        ],
    )

    results = []
    i = 0
    for dataset in indexing_params["dataset"]:
        for index_str in indexing_params["index_str"]:
            print(f"Indexing {dataset} / {index_str}", flush=True)
            index = faiss.index_factory(dataset.d, index_str)
            index.verbose = True
            database = dataset.get_database()
            index.add(database)

            # first run will be with no compression
            for comp_method, comp_params in [
                (None, None),
                *indexing_params["compression_method"],
            ]:
                if comp_method is None:
                    continue

                if "NSG" in index_str:
                    graph_friends = get_NSG_neighbors(index.nsg)
                elif "HNSW" in index_str:
                    graph_friends = [
                        get_hnsw_links(index.hnsw, v)[0] for v in range(dataset.nb)
                    ]

                graph_edgelist = friend_to_edgelist_repr(graph_friends)
                graph = Graph(
                    edge_array=graph_edgelist,
                    num_nodes=len(graph_friends),
                    num_edges=len(graph_edgelist),
                )

                if comp_method.startswith("rec-"):
                    model_pu = PolyasUrnModel(
                        graph.num_nodes,
                        graph.num_edges,
                        **comp_params,
                        undirected=False,
                    )
                    _, graph_bpe = model_pu.compute_bpe(graph)

                elif comp_method == "uniform":
                    model_unif = UniformNodesModel(
                        graph.num_nodes,
                        graph.num_edges,
                        **comp_params,
                        undirected=False,
                    )
                    _, graph_bpe = model_unif.compute_bpe(graph)

                results.append(
                    {
                        "index_str": index_str,
                        "comp_method": comp_method or "ref",
                        "dataset": type(dataset).__name__,
                        "nb": dataset.nb,
                        "nt": dataset.nt,
                        "bpe": graph_bpe,
                        "num_edges": graph.num_edges,
                        **comp_params,
                    }
                )
                print(results[-1])
                i += 1

                df = pd.DataFrame(results)
                df.to_csv(csv_path, index=False)
                print(f"Saved to {csv_path} with {i} entries")
                print(df)
