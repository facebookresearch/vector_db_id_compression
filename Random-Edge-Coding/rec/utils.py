from .definitions import Graph
from functools import reduce

import numpy as np
import craystack as cs
import random


def flatten(l):
    return [item for sublist in l for item in sublist]


def edge_and_vertex_multisets_from_graph(graph, seed=0):
    edge_list = graph.edge_array.tolist()

    # Shuffle the edge list to create a reasonably balanced multiset BST
    random.seed(seed)
    random.shuffle(edge_list)

    empty_multiset = ()
    vertex_multiset = bulk_insert_into_multiset(flatten(edge_list), empty_multiset)
    edge_multiset = cs.multiset.build_multiset(map(sorted, edge_list))
    return edge_multiset, vertex_multiset


def graph_from_multiset(multiset, num_nodes, num_edges):
    vertex_array = np.array(cs.multiset.to_sequence(multiset))
    edge_array = vertex_array.reshape((num_edges, 2))
    return Graph(
        edge_array=edge_array,
        num_nodes=num_nodes,
        num_edges=num_edges,
    )


def batched_log_ascending_factorial(a, k):
    cummulative_log_factorial = np.cumsum(np.log2(a + np.arange(np.max(k))))
    return cummulative_log_factorial[k - 1]


def batched_log_factorial(k):
    return batched_log_ascending_factorial(1, k)


def bulk_insert_into_multiset(sequence, multiset):
    return tuple(reduce(cs.multiset.insert, sequence, multiset))
