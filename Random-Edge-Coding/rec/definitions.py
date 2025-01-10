from dataclasses import dataclass
from collections import namedtuple

import numpy as np


@dataclass
class Graph:
    edge_array: np.ndarray
    num_nodes: int
    num_edges: int

    def __eq__(self, other):
        self_sorted_edge_array = np.sort(np.sort(self.edge_array, axis=1), axis=0)
        other_sorted_edge_array = np.sort(np.sort(other.edge_array, axis=1), axis=0)
        return bool((self_sorted_edge_array == other_sorted_edge_array).all())

    @property
    def sorted_edge_list(self):
        return sorted(map(sorted, self.edge_array))

    @property
    def degree_entropy(self):
        vertex_array = self.edge_array.flatten()
        degrees = np.bincount(vertex_array)
        total_degree = len(vertex_array)
        probs = degrees / total_degree
        entropy = -np.sum(probs * np.log2(probs))
        return entropy


Model = namedtuple("Model", ["push", "pop", "compute_bpe"])
Model.__new__.__defaults__ = (None, None, None)
