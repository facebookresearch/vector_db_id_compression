import pickle
from datetime import datetime

import faiss
import numpy as np
from faiss.contrib.datasets import DatasetDeep1B, DatasetSIFT1M
from faiss.contrib.inspect_tools import get_invlist
from qinco_datasets import DatasetFB_ssnpp


def batched_log_ascending_factorial(a, k):
    cummulative_log_factorial = np.cumsum(np.log2(a + np.arange(np.max(k))))
    return cummulative_log_factorial[k - 1]


def compute_bits(seq, alphabet_size=256):
    _, counts = np.unique(seq, return_counts=True)
    seq_info_content = (
        batched_log_ascending_factorial(alphabet_size, len(seq))
        - batched_log_ascending_factorial(1, counts).sum()
    )
    return seq_info_content


def compute_entropy(X):
    num_examples, num_vars = X.shape
    entropies = np.zeros(num_vars)
    for i in range(num_vars):
        x = X[:, i]

        _, x_counts = np.unique(x, return_counts=True)

        x_prob = x_counts / num_examples

        entropies[i] = -np.sum(x_prob * np.log2(x_prob))
    return entropies


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
datasets = [
    DatasetDeep1B(nb=10**6),
    DatasetFB_ssnpp(basedir="/checkpoint/dsevero/data/fb_ssnpp/"),
    DatasetSIFT1M(),
]

results = []
for ds in datasets:
    for index_str in [f"IVF1024,PQ{2**c}" for c in [1, 2, 3, 4, 5]]:
        try:
            index = faiss.index_factory(ds.d, index_str)
            index.train(ds.get_train())
            database = ds.get_database()
            index.add(database)

            num_clusters = index.invlists.nlist
            codes_per_cluster = [
                get_invlist(index.invlists, c)[1] for c in range(num_clusters)
            ]
            num_samples, num_vars = np.concatenate(codes_per_cluster).shape

            bpe_per_component = np.zeros(num_vars)
            for c, codes in enumerate(codes_per_cluster):
                for i in range(num_vars):
                    bpe_per_component[i] += compute_bits(codes[:, i])
            bpe_per_component /= num_samples
            results.append(
                {
                    "bpes": bpe_per_component,
                    "index_str": index_str,
                    "dataset": type(ds).__name__,
                }
            )
            print(results, flush=True)
            with open(f"results-code-compression_{timestamp}.pickle", "wb") as f:
                pickle.dump(results, f)
        except Exception as e:
            print(index_str, e, flush=True)
