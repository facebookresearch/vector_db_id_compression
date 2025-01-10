import random
from typing import Tuple

import craystack as cs
import numpy as np
from craystack.rans import pop_with_finer_prec as rans_pop
from craystack.rans import push_with_finer_prec as rans_push
from tqdm import tqdm

from .definitions import Graph, Model
from .utils import (
    batched_log_ascending_factorial,
    batched_log_factorial,
    bulk_insert_into_multiset,
    edge_and_vertex_multisets_from_graph,
    graph_from_multiset,
)


def UniformModel(prec: int) -> Model:
    def push(ans_state, symbol):
        ans_state = cs.rans.push_with_finer_prec(ans_state, symbol, 1, prec)
        return ans_state

    def pop(ans_state, *context):
        symbol, pop_ = cs.rans.pop_with_finer_prec(ans_state, prec)
        ans_state = pop_(symbol, 1)
        return ans_state, int(symbol[0])

    return Model(push, pop)


def PolyasUrnModel(
    num_nodes: int,
    num_edges: int,
    bias: int = 1,
    seed: int = 0,
    undirected: bool = True,
) -> Model:
    assert bias == int(bias), "Bias must be an integer"

    # Define codecs
    swor_codec = cs.multiset.SamplingWithoutReplacement()
    frequency_codec = FrequencyCountCodec()
    binary_codec = UniformModel(2)

    def push(ans_state, graph: Graph):
        # Build multisets
        edge_multiset, vertex_multiset = edge_and_vertex_multisets_from_graph(
            graph=graph, seed=seed
        )
        vertex_multiset = bulk_insert_into_multiset(range(num_nodes), vertex_multiset)

        step = 2 * num_edges - 1
        with tqdm(total=num_edges, desc=f"Encoding") as pbar:
            for _ in range(graph.num_edges):
                # 1) Sample an edge without replacement
                ans_state, edge, edge_multiset = swor_codec.pop(
                    ans_state, edge_multiset
                )

                if undirected:
                    # 2) Pick an order for the vertices
                    ans_state, b = binary_codec.pop(ans_state)
                else:
                    b = 0

                # 3) Update vertex multiset
                vertex_multiset = cs.multiset.remove(vertex_multiset, edge[b])

                # 4) Encode vertex
                ans_state = frequency_codec.push(ans_state, edge[b], vertex_multiset)
                step -= 1

                # 5) Update vertex multiset
                vertex_multiset = cs.multiset.remove(vertex_multiset, edge[1 - b])

                # 6) Encode vertex
                ans_state = frequency_codec.push(
                    ans_state, edge[1 - b], vertex_multiset
                )
                step -= 1

                if step // 2 % (num_edges // 20) == 0:
                    pbar.update((num_edges // 20))

        return ans_state

    def pop(ans_state):
        alphabet = list(range(num_nodes))
        random.seed(seed)
        random.shuffle(alphabet)

        empty_multiset = ()
        vertex_multiset = bulk_insert_into_multiset(alphabet, empty_multiset)
        edge_multiset = ()

        step = 0
        with tqdm(total=num_edges, desc=f"Decoding") as pbar:
            for _ in range(num_edges):
                # 6) Decode vertex
                step += 1
                ans_state, w = frequency_codec.pop(ans_state, vertex_multiset)

                # 5) Update vertex multiset
                vertex_multiset = cs.multiset.insert(vertex_multiset, w)

                # 4) Decode vertex
                step += 1
                ans_state, v = frequency_codec.pop(ans_state, vertex_multiset)

                # 3) Update vertex multiset
                vertex_multiset = cs.multiset.insert(vertex_multiset, v)

                if undirected:
                    # 2) Bits-back: infer `b`, from step 2 of push, defining the order of the vertices
                    if v < w:
                        b = 0
                        edge = [v, w]
                    else:
                        b = 1
                        edge = [w, v]
                    ans_state = binary_codec.push(ans_state, b)
                else:
                    b = 0

                # 1) Bits-back: input an edge without replacement
                ans_state, edge_multiset = swor_codec.push(
                    ans_state, edge, edge_multiset
                )

                if step // 2 % (num_edges // 20) == 0:
                    pbar.update((num_edges // 20))

        graph = graph_from_multiset(edge_multiset, num_nodes, num_edges)
        return ans_state, graph

    def compute_bpe(graph: Graph) -> Tuple[float, float]:
        vertex_array = graph.edge_array.flatten()
        _, counts = np.unique(vertex_array, return_counts=True)
        seq_info_content = (
            batched_log_ascending_factorial(graph.num_nodes * bias, len(vertex_array))
            - batched_log_ascending_factorial(bias, counts).sum()
        )
        num_bits_back = batched_log_factorial(graph.num_edges)
        if undirected:
            num_bits_back += graph.num_edges
        graph_info_content = seq_info_content - num_bits_back
        return (
            float(seq_info_content / graph.num_edges),
            float(graph_info_content / graph.num_edges),
        )

    return Model(push, pop, compute_bpe)


def FrequencyCountCodec():
    def push(ans_state, symbol, multiset):
        start, freq = cs.multiset.forward_lookup(multiset, symbol)
        multiset_size, *_ = multiset
        ans_state = rans_push(ans_state, start, freq, multiset_size)
        return ans_state

    def pop(ans_state, multiset):
        multiset_size, *_ = multiset
        cdf_value, pop_ = rans_pop(ans_state, multiset_size)
        (start, freq), symbol = cs.multiset.reverse_lookup(multiset, cdf_value[0])
        ans_state = pop_(start, freq)
        return ans_state, symbol

    return cs.Codec(push, pop)


def UniformNodesModel(
    num_nodes: int,
    num_edges: int,
    seed: int = 0,
    undirected: bool = True,
) -> Model:

    def push(ans_state, graph: Graph):
        raise NotImplementedError

    def pop(ans_state):
        raise NotImplementedError

    def compute_bpe(graph: Graph) -> Tuple[float, float]:
        nodes = np.unique(graph.edge_array.flatten())
        num_nodes = len(nodes)
        seq_info_content = 2 * graph.num_edges * np.log2(num_nodes)
        num_bits_back = batched_log_factorial(graph.num_edges)
        if undirected:
            num_bits_back += graph.num_edges
        graph_info_content = seq_info_content - num_bits_back
        return (
            float(seq_info_content / graph.num_edges),
            float(graph_info_content / graph.num_edges),
        )

    return Model(push, pop, compute_bpe)
