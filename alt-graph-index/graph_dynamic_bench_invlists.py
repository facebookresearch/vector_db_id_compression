import datetime
import time

import altid
import faiss
import numpy as np
import pandas as pd
from faiss.contrib.datasets import DatasetDeep1B, DatasetSIFT1M, SyntheticDataset
from faiss.contrib.inspect_tools import get_NSG_neighbors
from qinco_datasets import DatasetFB_ssnpp

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
    import sys

    _max_degree = int(sys.argv[1])
    _dataset = int(sys.argv[2])

    now = datetime.datetime.now()
    csv_path = f"benchmark-results-graphs-dynamic/graph-dynamic-results-{now}-{_max_degree}-{_dataset}.csv"

    indexing_params = dict(
        index_str=[
            f"NSG{max_degree},{code}"
            for code in ["Flat"]
            for max_degree in [_max_degree]
        ],
        dataset=[
            [
                DatasetSIFT1M(),
                DatasetDeep1B(nb=10**6),
                DatasetFB_ssnpp(basedir="/checkpoint/dsevero/data/fb_ssnpp/"),
            ][_dataset],
        ],
        compression_method=["elias-fano", "roc", "compact"],
    )

    search_time_params = dict(k=[20], nq=[None], nprobe=[16])
    num_runs = 100

    results = []
    i = 0
    for dataset in indexing_params["dataset"]:
        for index_str in indexing_params["index_str"]:
            print(f"Indexing {dataset} / {index_str}", flush=True)
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
                for comp_method in indexing_params["compression_method"]
            }

            # first run will be with no compression
            print("Running search ...")
            for comp_method in [None, *indexing_params["compression_method"]]:
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
