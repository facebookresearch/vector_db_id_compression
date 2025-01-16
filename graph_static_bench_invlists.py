# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from faiss.contrib.datasets import DatasetDeep1B, DatasetSIFT1M, SyntheticDataset
from faiss.contrib.inspect_tools import get_NSG_neighbors
from rec.definitions import Graph
from rec.models import PolyasUrnModel

from qinco_datasets import DatasetFB_ssnpp


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
    dataset_idx = int(sys.argv[1])
    max_degree = int(sys.argv[2])
    fb_ssnpp_dir = None

    if dataset_idx == 3:
        assert (
            len(sys.argv) == 4
        ), "Path to fb_ssnpp/ directory is needed for DatasetFB_ssnpp (index 3)"
        fb_ssnpp_dir = sys.argv[3]

    AVAILABLE_DATASETS = [
        (SyntheticDataset, dict(d=32, nt=10_000, nq=1, nb=1_000)),
        (DatasetSIFT1M, {}),
        (DatasetDeep1B, dict(nb=10**6)),
        (DatasetFB_ssnpp, dict(basedir=fb_ssnpp_dir)),
    ]

    dataset_cls, dataset_kwargs = AVAILABLE_DATASETS[dataset_idx]
    dataset = dataset_cls(**dataset_kwargs)

    index_strs = [f"NSG{max_degree},Flat", f"HNSW{max_degree},Flat"]
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    csv_path = Path(f"results-offline-graphs-rec/{now}-{dataset_cls.__name__}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for index_str in index_strs:
        print(f"Indexing {dataset_cls.__name__} / {index_str}", flush=True)
        index = faiss.index_factory(dataset.d, index_str)
        index.verbose = True
        database = dataset.get_database()
        index.add(database)

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

        model_pu = PolyasUrnModel(
            graph.num_nodes,
            graph.num_edges,
            undirected=False,
        )

        # Compute results directly as BPE. See REC paper for details.
        _, graph_bpe = model_pu.compute_bpe(graph)

        results.append(
            {
                "index_str": index_str,
                "comp_method": "rec",
                "dataset": type(dataset).__name__,
                "nb": dataset.nb,
                "nt": dataset.nt,
                "bpe": graph_bpe,
                "num_edges": graph.num_edges,
            }
        )
        print(results[-1])

        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        print(df)
