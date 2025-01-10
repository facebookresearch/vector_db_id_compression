
import unittest
import numpy as np
import faiss
import custom_invlists 

from faiss.contrib.datasets import SyntheticDataset


class TestEquiv(unittest.TestCase): 

    def test_equiv(self): 
        ds = SyntheticDataset(32, 1000, 2000, 50)
        index = faiss.index_factory(ds.d, "IVF32,PQ4np")
        index.train(ds.get_train())
        index.add(ds.get_database())
        Dref, Iref = index.search(ds.get_queries(), 10)

        invlists2 = custom_invlists.CompressedIDInvertedLists(index.invlists)
        print("invlists2 compression:", invlists2.precision)
        index.replace_invlists(invlists2, False)

        Dnew, Inew = index.search(ds.get_queries(), 10)

        np.testing.assert_array_equal(Iref, Inew)
