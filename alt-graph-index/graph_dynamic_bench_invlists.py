# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import sys
import time
from pathlib import Path

import altid
import faiss
import numpy as np
import pandas as pd
from faiss.contrib.datasets import DatasetDeep1B, DatasetSIFT1M, SyntheticDataset
from faiss.contrib.inspect_tools import get_NSG_neighbors

from ..qinco_datasets import DatasetFB_ssnpp

AVAILABLE_COMPRESSED_IVFS = {
    "elias-fano": altid.EliasFanoNSGGraph,
    "roc": altid.ROCNSGGraph,
    "compact": altid.CompactBitNSGGraph,
    "ref": None,
}


def get_ids_size(dataset, graph_comp, comp_method, num_edges):
    if comp_method is None:
        return 8 * num_edges
    elif comp_method == "compact":
        return np.log2(dataset.nb) / 8 * num_edges
    else:
        return graph_comp.compressed_ids_size_in_bytes


def get_overhead_size(dataset, graph_comp, comp_method, num_edges):
    if comp_method in ["roc", "elias-fano"]:
        return graph_comp.overhead_in_bytes
    elif comp_method == "ref":
        return 0
    elif comp_method == "compact":
        return 0
    else:
        return None


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
    index_str = f"NSG{max_degree},Flat"

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    csv_path = Path(
        f"results-online-graphs/graph-dynamic-results-{now}-{index_str.replace(",", "_")}-{dataset_cls.__name__}.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    compression_methods = ["elias-fano", "roc", "compact"]

    search_time_params = dict(k=[20], nq=[None], nprobe=[16])
    num_runs = 100

    results = []
    i = 0

    print(f"Indexing {type(dataset).__name__} / {index_str}", flush=True)
    index = faiss.index_factory(dataset.d, index_str)
    index.verbose = True
    database = dataset.get_database()
    index.add(database)
    num_edges = (get_NSG_neighbors(index.nsg) != -1).sum()

    # precompute graphs for compression methods
    print("Compressing database ...")
    graph = index.nsg.get_final_graph()
    invlists_comp = {
        comp_method: AVAILABLE_COMPRESSED_IVFS[comp_method](graph)
        for comp_method in compression_methods
    }

    # first run will be with no compression
    print("Running search ...")
    for comp_method in [None, *compression_methods]:
        if comp_method is not None:
            graph_comp = invlists_comp[comp_method]
            index.nsg.replace_final_graph(graph_comp)
        else:
            graph_comp = graph

        for k in search_time_params["k"]:
            for nq in search_time_params["nq"]:
                for nprobe in search_time_params["nprobe"]:
                    index.nprobe = nprobe
                    queries = dataset.get_queries()[:nq]

                    for run_id in range(num_runs):
                        t0 = time.time()
                        _, Iref = index.search(queries, k)
                        t1 = time.time()
                        dt_search = t1 - t0

                        results.append(
                            {
                                "dt_search": dt_search,
                                "nprobe": nprobe,
                                "run_id": run_id,
                                "index_str": index_str,
                                "k": k,
                                "nq": queries.shape[0],
                                "comp_method": comp_method or "ref",
                                "dataset": type(dataset).__name__,
                                "ids_size": get_ids_size(
                                    dataset, graph_comp, comp_method, num_edges
                                ),
                                "overhead_size": get_overhead_size(
                                    dataset, graph_comp, comp_method, num_edges
                                ),
                                "nb": dataset.nb,
                                "nt": dataset.nt,
                                "num_edges": num_edges,
                            }
                        )
                        print(results[-1], flush=True)
                        i += 1

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path} with {i} entries")
    print(df)
