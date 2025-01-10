import unittest

# importing this module adds various methods to the IndexIVF and IndexNSG objects...
import altid
import faiss
import numpy as np
from faiss.contrib.datasets import SyntheticDataset
from faiss.contrib.inspect_tools import get_NSG_neighbors




class TestCompressedNSG(unittest.TestCase):

    def test_compact_bit(self):
        self.do_test(altid.CompactBitNSGGraph)

    def test_elias_fano(self):
        self.do_test(altid.EliasFanoNSGGraph)

    def test_roc_graph(self):
        self.do_test(altid.ROCNSGGraph)

    def do_test(self, graph_class):

        ds = SyntheticDataset(32, 0, 1000, 10)

        index = faiss.index_factory(ds.d, "NSG32,Flat")
        index.add(ds.get_database())
        Dref, Iref = index.search(ds.get_queries(), 10)

        gr = index.nsg.get_final_graph()

        compactbit_graph = graph_class(gr)
        index.nsg.replace_final_graph(compactbit_graph)

        D, I = index.search(ds.get_queries(), 10)

        np.testing.assert_array_equal(I, Iref)
        np.testing.assert_array_equal(D, Dref)


class TestSearchTraced(unittest.TestCase):

    def test_traced(self):
        ds = SyntheticDataset(32, 0, 10000, 10)

        index_nsg = faiss.index_factory(ds.d, "NSG32,Flat")
        index_nsg.add(ds.get_database())
        q = ds.get_queries()[:1]  # just one query
        Dref, Iref = index_nsg.search(q, 10)

        D, I, trace = index_nsg.search_and_trace(q, 10)
        np.testing.assert_array_equal(I, Iref)
        np.testing.assert_array_equal(D, Dref)

        # at least, all result vectors should be in the trace
        assert set(I.ravel().tolist()) <= set(trace)
