import unittest
import warnings
from functools import reduce
from itertools import combinations_with_replacement, product

import numpy as np
from test_types import Test_f32, Test_f64

from arches.drivers import integral_category
from arches.integral_indexing_utils import (
    canonical_idx4,
    compound_idx2,
    compound_idx4,
)
from arches.integrals import JChunk, JChunkFactory


class FakeComm:
    # to avoid initializing MPI for just tests
    def __init__(self, rank, size):
        self.rank = rank
        self.size = size

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size


class FakeReader:
    # to isolate testing of IO
    def __init__(self):
        pass

    def __getitem__(self, idx):
        return 0.0

    @property
    def dtype(self):
        return np.float32


class Test_Chunks(unittest.TestCase):
    __test__ = False


class Test_Chunks_f32(Test_f32, Test_Chunks):
    __test__ = True


class Test_Chunks_f64(Test_f64, Test_Chunks):
    __test__ = True


class Test_ChunkFactory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        N_mos = 16
        cls.N_mos = N_mos

        orb_list = tuple([x for x in range(N_mos)])
        cls.canon_order = dict()
        for i, j, k, l in product(orb_list, orb_list, orb_list, orb_list):  # noqa: E741
            canon_idx = canonical_idx4(i, j, k, l)
            cls.canon_order[canon_idx] = {
                "idx": compound_idx4(*canon_idx),
                "category": integral_category(*canon_idx),
            }

        for i, j in combinations_with_replacement(orb_list, 2):
            cls.canon_order[(i, j)] = {"idx": compound_idx2(i, j), "category": "OE"}

        cls.categories = ("A", "B", "C", "D", "E", "F", "G", "OE")

    def test_iters(self):
        def check_category(cat, idx):
            ref_ind = set([v["idx"] for v in self.canon_order.values() if v["category"] == cat])
            self.assertEqual(ref_ind, set(idx), msg=f"Failed to get all indices for category {cat}")
            self.assertEqual(
                len(ref_ind), len(idx), msg=f"Some indices double counted in category {cat}"
            )

        check_category("A", list(JChunkFactory.A_idx_iter(self.N_mos)))
        check_category("B", list(JChunkFactory.B_idx_iter(self.N_mos)))
        check_category("C", list(JChunkFactory.C_idx_iter(self.N_mos)))
        check_category("D", list(JChunkFactory.D_idx_iter(self.N_mos)))
        check_category("E", list(JChunkFactory.E_idx_iter(self.N_mos)))
        check_category("F", list(JChunkFactory.F_idx_iter(self.N_mos)))
        check_category("G", list(JChunkFactory.G_idx_iter(self.N_mos)))
        check_category("OE", list(JChunkFactory.OE_idx_iter(self.N_mos)))

    def test_idx(self):
        def check_category(cat):
            factory = JChunkFactory(self.N_mos, cat, FakeReader())
            chunk = factory.get_chunks()[0]

            ref_ind = set([v["idx"] for v in self.canon_order.values() if v["category"] == cat])
            test_set = set(chunk.idx.np_arr)
            test_ind = list(chunk.idx.np_arr)

            self.assertEqual(ref_ind, test_set, msg=f"Failed to get all indices for category {cat}")
            self.assertEqual(
                len(ref_ind),
                len(test_ind),
                msg=f"Some indices double counted in category {cat}",
            )

        for cat in self.categories:
            check_category(cat)

    def test_batched_idx(self):
        def check_category(cat):
            factory = JChunkFactory(self.N_mos, cat, FakeReader(), chunk_size=256)
            chunks = factory.get_chunks()

            ref_ind = set([v["idx"] for v in self.canon_order.values() if v["category"] == cat])
            test_sets = [set(chunk.idx.np_arr) for chunk in chunks]
            test_ind = reduce(lambda x, y: x.union(y), test_sets)

            N_ind = 0
            for t in test_sets:
                N_ind += len(t)

            self.assertEqual(ref_ind, test_ind, msg=f"Failed to get all indices for category {cat}")
            self.assertEqual(
                len(ref_ind),
                N_ind,
                msg=f"Some indices double counted in category {cat}",
            )

        for cat in self.categories:
            check_category(cat)

    def test_dist_batched_idx(self):
        def check_category(cat, comm_size):
            local_inds = []
            for rank in range(comm_size):
                comm = FakeComm(rank, comm_size)
                fact = JChunkFactory(self.N_mos, cat, FakeReader(), comm=comm, chunk_size=256)
                chunks = fact.get_chunks()
                local_sets = [set(chunk.idx.np_arr) for chunk in chunks]
                print(cat, rank, comm_size, len(local_sets), len(chunks))
                local_ind = reduce(lambda x, y: x.union(y), local_sets)
                local_inds.append(local_ind)

            ref_ind = set([v["idx"] for v in self.canon_order.values() if v["category"] == cat])
            test_ind = reduce(lambda x, y: x.union(y), local_inds)

            N_ind = 0
            for t in local_inds:
                N_ind += len(t)

            self.assertEqual(
                ref_ind,
                test_ind,
                msg=f"Failed to get all indices for category {cat} with comm size {comm_size}",
            )
            self.assertEqual(
                len(ref_ind),
                N_ind,
                msg=f"Some indices double counted in category {cat} with comm size {comm_size}",
            )

        with warnings.catch_warnings(record=False):
            warnings.simplefilter("ignore")
            for comm_size in [1, 2, 8, 17, 39]:
                for cat in self.categories:
                    check_category(cat, comm_size)

    def test_dist_batched_warning(self):
        def check_category(cat):
            comm = FakeComm(1, 2)
            ref_ind = set([v["idx"] for v in self.canon_order.values() if v["category"] == cat])
            fact = JChunkFactory(self.N_mos, cat, FakeReader(), comm=comm, chunk_size=len(ref_ind))
            chunks = fact.get_chunks()

        with self.assertWarns(Warning):
            for cat in self.categories:
                check_category(cat)
