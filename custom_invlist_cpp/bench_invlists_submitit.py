import datetime
import time

import custom_invlists
import faiss
import pandas as pd
import submitit
from faiss.contrib.datasets import DatasetDeep1B, DatasetSIFT1M, SyntheticDataset
from qinco_datasets import DatasetFB_ssnpp

AVAILABLE_COMPRESSED_IVFS = {
    "elias-fano": custom_invlists.CompressedIDInvertedListsEliasFano,
    "roc": custom_invlists.CompressedIDInvertedListsFenwickTree,
    "packed-bits": custom_invlists.CompressedIDInvertedListsPackedBits,
    "wavelet-tree": custom_invlists.CompressedIDInvertedListsWaveletTree,
    "ref": None,
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

    indexing_params = dict(
        index_str=[
            f"IVF{num_clusters},{code}"
            # for num_clusters in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
            # for num_clusters in [64, 128, 256, 512, 1024]
            for num_clusters in [256, 512, 1024, 2048]
            for code in ["Flat"]
            # for code in [
            #     # "Flat",
            #     "PQ2",
            #     "PQ4",
            #     "PQ8",
            #     "PQ16",
            #     "PQ32",
            #     # "PQ4np",
            #     # "PQ8x10np",
            # ]
            # for code in ["PQ8x10np", "PQ8x12np", "PQ8", "PQ16"]
        ],
        dataset=[
            SyntheticDataset(32, 10_000, 1000_000, 10_000),
            DatasetSIFT1M(),
            DatasetFB_ssnpp(basedir="/checkpoint/dsevero/data/fb_ssnpp/"),
            DatasetDeep1B(nb=10**6),
        ],
        compression_method=["elias-fano", "roc", "packed-bits", "wavelet-tree"],
    )

    search_time_params = dict(
        # k=[1, 2, 4, 8, 16, 32], nq=[1, 2, 4, 8, 16], nprobe=[1, 2, 4, 8, 16]
        k=[20],
        nq=[None],
        nprobe=[1, 4, 16],
    )
    num_runs = 100

    executor = submitit.AutoExecutor(folder="slurm_jobs")
    executor.update_parameters(
        slurm_job_name="faiss_ivf_benchmark",
        slurm_constraint="bldg1",  # Schedule on E5-2698 2.20GHz nodes
        tasks_per_node=1,  # Number of tasks (MPI processes)
        cpus_per_task=10,  # Number of CPUs per task
        slurm_time="48:00:00",  # Time limit (HH:MM:SS)
        slurm_additional_parameters={
            "output": "%x-%j.out",
            "error": "%x-%j.err",
        },
    )
    jobs = []
    for index_str in indexing_params["index_str"]:
        for dataset in indexing_params["dataset"]:
            for compression_method in [None, *indexing_params["compression_method"]]:
                job = executor.submit(
                    index_str,
                    dataset,
                    compression_method,
                    num_runs,
                    search_time_params,
                )
                time.sleep(1)
                print(job)
                jobs.append(job)


def run_iteration(index_str, dataset, num_runs, search_time_params):
    now = datetime.datetime.now()
    csv_path = f"benchmark-results/ivf-results-{now}.csv"
    results = []
    i = 0
    index = faiss.index_factory(dataset.d, index_str)
    index.train(dataset.get_train())
    database = dataset.get_database()
    index.add(database)

    # for deferred decoding
    index.parallel_mode = 3

    # precompute invlists for compression methods
    invlists_comp = {
        comp_method: AVAILABLE_COMPRESSED_IVFS[comp_method](index.invlists)
        for comp_method in indexing_params["compression_method"]
    }

    # first run will be with no compression
    for comp_method in [None, *indexing_params["compression_method"]]:
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
