from ml_collections import config_flags
from ml_collections import config_dict
from rec.datasets import load_dataset
from absl import app
from absl import flags

import craystack as cs
import json
import os

assert (
    "DATA_DIR" in os.environ
), "Please set the DATA_DIR environment variable to the path of the data directory."
print(f'Using data directory: {os.environ["DATA_DIR"]}')

_CONFIG = config_flags.DEFINE_config_file("config")
flags.DEFINE_boolean("decode", False, "Decompress the graph.")
flags.DEFINE_boolean("eval_likelihood_only", False, "Evaluate likelihood only.")
_FLAGS = flags.FLAGS


def main(_):
    print(f"Running experiment with config: \n{_CONFIG.value}")
    print(f'Data directory: {os.environ["DATA_DIR"]}')
    
    results = run_experiment(_CONFIG.value)
    print("\n------------------RESULTS------------------")
    print(f"Dataset:", results["dataset_name"])
    print(f"Number of nodes:", results["num_nodes"])
    print(f"Number of edges:", results["num_edges"])
    print(f"BPE of the vertex-sequence under Pólya's Urn (theoretical):", results["seq_bpe"])
    print(f"BPE of the graph under Pólya's Urn (theoretical):", results["graph_bpe"])

    if results["ans_bpe"] is not None:
        print(f"Random Edge Coding message length in BPE:", results["ans_bpe"])
    
    if results["decoding_correct"] is not None:
        print(f"Encoded graph == decoded graph?:", results["decoding_correct"])

    print("\n-------------------------------------------")

def run_experiment(config: config_dict.ConfigDict):
    graph = load_dataset(
        dataset_name=config.dataset_name, max_num_edges=config.max_num_edges
    )

    model = config.model(
        num_nodes=graph.num_nodes,
        num_edges=graph.num_edges,
        seed=config.seed,
        bias=config.bias,
    )
    seq_bpe, graph_bpe = model.compute_bpe(graph)

    ans_bpe = None
    if not _FLAGS.eval_likelihood_only:
        ans_state = cs.rans.base_message(shape=(1,))
        ans_state = model.push(ans_state, graph)
        ans_bpe = (32 + 32 * len(cs.flatten(ans_state))) / graph.num_edges
    bpe_gap = ans_bpe / graph_bpe - 1 if ans_bpe is not None else None

    decoding_correct = None
    if (not _FLAGS.eval_likelihood_only) and _FLAGS.decode:
        _, graph_decoded = model.pop(ans_state)
        decoding_correct = graph_decoded == graph

    return {
        "dataset_name": config.dataset_name,
        "model": str(config.model),
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "seq_bpe": seq_bpe,
        "graph_bpe": graph_bpe,
        "ans_bpe": ans_bpe,
        "bpe_gap": bpe_gap,
        "decoding_correct": decoding_correct,
        "bias": config.bias,
        "degree_entropy": graph.degree_entropy,
        "degree_entropy_per_edge": graph.degree_entropy / graph.num_edges,
    }


if __name__ == "__main__":
    app.run(main)
