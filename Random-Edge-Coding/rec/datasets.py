from tqdm import tqdm
from .definitions import Graph
from joblib import Memory

import fastremap
import numpy as np
import random
import os


DATA_DIR = os.environ["DATA_DIR"]
memory = Memory(f"{DATA_DIR}/rec/")

AVAILABLE_DATASETS = [
    # large social networks
    "youtube",
    "foursquare",
    "digg",
    "gowalla",

    # large non-social networks
    "skitter",
    "dblp",
]


@memory.cache
def load_dataset(dataset_name: str, max_num_edges, seed=0) -> Graph:
    assert (
        dataset_name in AVAILABLE_DATASETS
    ), f"dataset_name must be one of {AVAILABLE_DATASETS}"

    rec_data_dir = f"{DATA_DIR}/rec"

    if dataset_name == "youtube":
        edge_list = load_youtube(
            f"{rec_data_dir}/youtube-u-growth/out.youtube-u-growth"
        )
    elif dataset_name == "foursquare":
        edge_list = load_network_repo_dataset(f"{rec_data_dir}/soc-FourSquare.mtx")
    elif dataset_name == "digg":
        edge_list = load_network_repo_dataset(f"{rec_data_dir}/soc-digg.mtx")
    elif dataset_name == "gowalla":
        edge_list = load_SNAP_dataset(f"{rec_data_dir}/loc-gowalla_edges.txt")
        # Has each undirected edge twice (fwd/bwd), deduplicate here:
        num_edges = len(edge_list)
        edge_list = list(set(map(tuple, map(sorted, edge_list))))
        assert 2 * len(edge_list) == num_edges
    elif dataset_name == "skitter":
        edge_list = load_SNAP_dataset(f"{rec_data_dir}/as-skitter.txt")
    elif dataset_name == "dblp":
        edge_list = load_SNAP_dataset(f"{rec_data_dir}/com-dblp.ungraph.txt")
    else:
        raise ValueError(f"Dataset {dataset_name} is not available.")

    if max_num_edges > -1:
        print(f"Sampling (at most) {max_num_edges} edges ...", flush=True)
        random.seed(seed)
        random.shuffle(edge_list)
        edge_list = edge_list[:max_num_edges]

    print("Sorting edge list")
    edge_list = sorted(map(tuple, map(sorted, edge_list)))
    assert len(set(edge_list)) == len(edge_list), "Duplicate edges found."

    print(f"Relabeling vertices...", flush=True)
    edge_array = relabel_vertices(edge_list)
    num_nodes = int(np.max(edge_array) + 1)

    return Graph(edge_array=edge_array, num_nodes=num_nodes, num_edges=len(edge_array))


def relabel_vertices(edge_list) -> np.ndarray:
    vertex_array, _ = fastremap.renumber(np.array(edge_list, dtype=np.uint32))
    vertex_array = np.array(vertex_array)

    # Guarantees the smallest vertex is 0
    vertex_array = vertex_array - np.min(vertex_array)
    return vertex_array


def load_network_repo_dataset(path: str) -> list:
    with open(path, "r") as f:
        txt = f.readlines()[2:]

    edge_list = list()
    for _, line in tqdm(enumerate(txt, start=1)):
        v, w = line.replace("\n", "").split(" ")[:2]
        edge_list.append((int(v), int(w)))

    return edge_list


def load_youtube(path: str) -> list:
    with open(path, "r") as f:
        txt = f.readlines()

    edge_list = list()
    i = 0
    for line in tqdm(txt):
        if line.startswith("%"):
            continue

        i += 1
        v, w = line.replace("\n", "").split(" ")[:2]
        edge_list.append((int(v), int(w)))

    return edge_list


def load_SNAP_dataset(path: str) -> list:
    with open(path, "r") as f:
        txt = f.readlines()

    # parse file to get vertex pairs
    edge_list = list()
    i = 0
    for line in tqdm(txt):
        if line.startswith("#"):
            continue

        i += 1
        v, w = line.replace("\n", "").split("\t")
        edge_list.append((int(v), int(w)))

    print(f"Finished parsing {i} lines.", flush=True)
    return edge_list


def load_small_szip_graphs(path: str) -> list:
    with open(path, "r") as f:
        txt = f.readlines()

    # parse file to get vertex pairs
    edge_list = list()
    i = 0
    if "\t" in txt[0]:
        sep = "\t"
    elif " " in txt[0]:
        sep = " "
    else:
        raise ValueError("Unknown separator.")

    for line in tqdm(txt):
        i += 1
        v, w = line.strip().split(sep)
        edge_list.append((int(v), int(w)))

    print(f"Finished parsing {i} lines.", flush=True)
    return edge_list
