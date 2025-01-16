# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import sys
import time
from pathlib import Path

import custom_invlists
import faiss
import pandas as pd
from faiss.contrib.datasets import DatasetDeep1B, DatasetSIFT1M, SyntheticDataset

from ..qinco_datasets import DatasetFB_ssnpp

AVAILABLE_COMPRESSED_IVFS = {
    "elias-fano": custom_invlists.CompressedIDInvertedListsEliasFano,
    "roc": custom_invlists.CompressedIDInvertedListsFenwickTree,
    "packed-bits": custom_invlists.CompressedIDInvertedListsPackedBits,
    "wavelet-tree": custom_invlists.CompressedIDInvertedListsWaveletTree,
    "ref": None,  # must be last
}


def get_ids_size(dataset, invlist, comp_method):
    if comp_method is None:
        return 8 * dataset.nb
    else:
        return invlist.compressed_ids_size_in_bytes


def get_overhead_size(dataset, invlist, comp_method):
    if comp_method is None:
        return 0
    elif comp_method in ["roc", "elias-fano"]:
        return invlist.overhead_in_bytes


if __name__ == "__main__":

    dataset_idx = int(sys.argv[1])
    index_str = str(sys.argv[2])
    fb_ssnpp_dir = None

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

    AVAILABLE_DATASETS = [
        (SyntheticDataset, dict(d=32, nt=10_000, nq=1, nb=1_000)),
        (DatasetSIFT1M, {}),
        (DatasetDeep1B, dict(nb=10**6)),
        (DatasetFB_ssnpp, dict(basedir=fb_ssnpp_dir)),
    ]

    compression_methods = list(AVAILABLE_COMPRESSED_IVFS.keys())[:-1]

    search_time_params = dict(
        k=[20],
        nq=[None],
        nprobe=[1, 4, 16],
    )
    num_runs = 100

    results = []
    i = 0

    dataset_cls, dataset_kwargs = AVAILABLE_DATASETS[dataset_idx]
    dataset = dataset_cls(**dataset_kwargs)

    csv_path = Path(
        f"results-online-ivf/ivf-results-{now}-{index_str}-{dataset_cls.__name__}.csv".replace(
            ",", "_"
        ).replace(
            " ", "_"
        )
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    index = faiss.index_factory(dataset.d, index_str)
    index.train(dataset.get_train())
    database = dataset.get_database()
    index.add(database)

    # for deferred decoding
    index.parallel_mode = 3

    # precompute invlists for compression methods
    invlists_comp = {
        comp_method: AVAILABLE_COMPRESSED_IVFS[comp_method](index.invlists)
        for comp_method in compression_methods
    }

    # first run will be with no compression
    for comp_method in [None, *compression_methods]:
        if comp_method is None:
            invlist = index.invlists
        else:
            invlist = invlists_comp[comp_method]
            index.replace_invlists(invlist, False)

        decode_1by1 = comp_method in ("wavelet-tree", "packed-bits", None)

        for k in search_time_params["k"]:
            for nq in search_time_params["nq"]:
                for nprobe in search_time_params["nprobe"]:
                    index.nprobe = nprobe
                    queries = dataset.get_queries()[:nq]
                    print(queries.shape, flush=True)

                    for run_id in range(num_runs):
                        t0 = time.time()
                        index.search_defer_id_decoding(
                            queries, k=k, decode_1by1=decode_1by1
                        )
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
                                "ids_size": get_ids_size(dataset, invlist, comp_method),
                                "overhead_size": get_overhead_size(
                                    dataset, invlist, comp_method
                                ),
                                "nb": dataset.nb,
                                "nt": dataset.nt,
                            }
                        )
                        i += 1
                        print(results[-1], flush=True)

                    df = pd.DataFrame(results)
                    df.to_csv(csv_path, index=False)
                    print(
                        f"Saved to {csv_path} with {i} entries",
                        flush=True,
                    )
                    print(df, flush=True)
