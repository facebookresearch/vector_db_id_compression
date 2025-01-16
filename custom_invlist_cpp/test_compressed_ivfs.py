# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest
from ctypes import c_uint64
from pathlib import Path

import custom_invlists  # type: ignore
import faiss  # type: ignore
import numpy as np
from faiss.contrib.datasets import SyntheticDataset  # type: ignore
from faiss.contrib.inspect_tools import get_invlist  # type: ignore

LOGGER = logging.getLogger(Path(__file__).name)
logging.basicConfig(level=logging.INFO)


def make_compressed_wt(il):
    return custom_invlists.CompressedIDInvertedListsWaveletTree(il, 1)


class TestCompressedIDInvertedLists(unittest.TestCase):

    def test_packed_bits(self):
        self.do_test(custom_invlists.CompressedIDInvertedListsPackedBits)

    def test_fenwick_tree(self):
        self.do_test(custom_invlists.CompressedIDInvertedListsFenwickTree)

    def test_elias_fano(self):
        self.do_test(custom_invlists.CompressedIDInvertedListsEliasFano)

    def test_wavelet_tree(self):
        self.do_test(custom_invlists.CompressedIDInvertedListsWaveletTree)

    def test_wavelet_tree_compressed(self):
        self.do_test(make_compressed_wt)

    def do_test(self, CompressedIVF):
        ds = SyntheticDataset(d=4, nt=1_000, nb=100, nq=1)
        database = ds.get_database()
        queries = ds.get_queries()
        k = 5
        index_string = "IVF8,Flat"

        index = faiss.index_factory(ds.d, index_string)
        index.train(ds.get_train())
        index.add(database)

        LOGGER.info(f"TESTING {CompressedIVF}")

        index_comp = faiss.index_factory(ds.d, index_string)
        index_comp.train(ds.get_train())
        index_comp.add(database)

        print(get_invlist(index.invlists, 0)[0])

        for c in range(index.nlist):
            ids_comp = get_invlist(index_comp.invlists, c)[0]
            ids_ref = get_invlist(index.invlists, c)[0]
            # print(c, ids_comp, ids_ref)
            assert np.all(np.sort(ids_comp) == ids_ref)
        LOGGER.info(
            "Clusters in index and index_comp contain the same elements, before compression"
        )

        invlists_comp = CompressedIVF(index_comp.invlists)
        index_comp.replace_invlists(invlists_comp, False)

        for c in range(index.nlist):
            n = invlists_comp.list_size(c)
            p = int(invlists_comp.get_ids(c))
            ids_comp = np.ctypeslib.as_array((c_uint64 * n).from_address(p))
            ids_ref = get_invlist(index.invlists, c)[0]
            assert np.all(np.sort(ids_comp) == ids_ref)
        LOGGER.info(
            "Clusters in index and index_comp contain the same elements, after compression"
        )

        _, Iref = index.search(queries, k)
        _, Icomp = index_comp.search(queries, k)
        np.testing.assert_array_equal(Iref, Icomp)
        LOGGER.info(
            f"Search results are the same for compressed and uncompressed IVFs."
        )
        LOGGER.info(f"All tests passed for {CompressedIVF}!")


class TestDeferredIVFDecoding(unittest.TestCase):

    def test_ivf(self):

        ds = SyntheticDataset(32, 10000, 10000, 10)

        index = faiss.index_factory(ds.d, "IVF32,PQ4np")
        index.train(ds.get_train())
        index.add(ds.get_database())
        index.nprobe = 4
        Dref, Iref = index.search(ds.get_queries(), 10)

        index.parallel_mode = 3

        D, I = index.search_defer_id_decoding(ds.get_queries(), 10)

        np.testing.assert_array_equal(I, Iref)
        np.testing.assert_array_equal(D, Dref)

        # test return codes
        D, I, codes = index.search_defer_id_decoding(
            ds.get_queries(), 10, return_codes=2
        )

        assert codes.shape == (ds.nq, 10, 5)
        for q in range(ds.nq):
            for ki in range(10):
                if I[q, ki] < 0:
                    continue
                list_no = int(codes[q, ki, 0])
                code = codes[q, ki, 1:]
                il_ids, il_codes = get_invlist(index.invlists, list_no)
                offset = np.where(il_ids == I[q, ki])[0][0]
                assert np.all(code == il_codes[offset])

    def test_1by1_wavelet_tree(self):
        self.do_1by1_test(custom_invlists.CompressedIDInvertedListsWaveletTree)

    def test_1by1_wavelet_tree_compressed(self):
        self.do_1by1_test(make_compressed_wt)

    def test_1by1_packed_bits(self):
        self.do_1by1_test(custom_invlists.CompressedIDInvertedListsPackedBits)

    def test_1by1_elias_fano(self):
        self.do_1by1_test(custom_invlists.CompressedIDInvertedListsEliasFano)

    def do_1by1_test(self, CompressedIDInvertedLists):
        ds = SyntheticDataset(32, 10000, 10000, 10)

        index = faiss.index_factory(ds.d, "IVF32,PQ4np")
        index.train(ds.get_train())
        index.add(ds.get_database())
        index.nprobe = 4
        Dref, Iref = index.search(ds.get_queries(), 10)

        invlists2 = CompressedIDInvertedLists(index.invlists)
        index.replace_invlists(invlists2, False)
        index.parallel_mode = 3

        D, I = index.search_defer_id_decoding(ds.get_queries(), 10, decode_1by1=True)

        np.testing.assert_array_equal(I, Iref)
        np.testing.assert_array_equal(D, Dref)


if __name__ == "__main__":
    unittest.main()
