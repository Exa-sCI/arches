import pathlib
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
    compound_idx2_reverse,
    compound_idx4,
    compound_idx4_reverse,
)
from arches.integrals import JChunk, JChunkFactory, load_integrals_into_chunks
from arches.io import load_integrals

seed = 79123
rng = np.random.default_rng(seed=seed)


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

    def setUp(self):
        self.size = 256
        self.N_trials = 50
        self.rng = rng

    def test_constructor(self):
        def check_chunk():
            J_ind = rng.integers(0, 1000000, size=(self.size)).astype(np.int64)
            J_vals = rng.normal(0, 1, size=(self.size)).astype(self.dtype)
            chunk = JChunk("A", self.size, J_ind, J_vals, dtype=self.dtype)

            self.assertTrue(np.allclose(chunk.idx.np_arr, J_ind))
            self.assertTrue(np.allclose(chunk.J.np_arr, J_vals))

        for _ in range(self.N_trials):
            check_chunk()

    def test_bad_category(self):
        J_ind = np.zeros(self.size, dtype=np.int64)
        J_vals = np.zeros(self.size, dtype=self.dtype)
        for cat in ["", "AB", "dog", "OEEE"]:
            self.assertRaises(
                ValueError, lambda: JChunk(cat, self.size, J_ind, J_vals, dtype=self.dtype)
            )


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


class Test_IO(unittest.TestCase):
    __test__ = False

    def test_IO(self):
        run_folder = pathlib.Path(__file__).parent.resolve()
        fp = run_folder.joinpath("../data/f2_631g.18det.fcidump")

        N_orb_ref, E0_ref, ref_J_oe, ref_J_te = load_integrals(str(fp))

        N_orb_test, E0_test, chunks = load_integrals_into_chunks(
            str(fp), FakeComm(0, 1), dtype=self.dtype
        )

        self.assertEqual(N_orb_ref, N_orb_test)
        self.assertEqual(E0_ref, E0_test)

        ref_J_oe_check = {k: False for k in ref_J_oe.keys()}
        ref_J_te_check = {k: False for k in ref_J_te.keys()}

        for chunk in chunks:
            if chunk.category == "OE":
                for k, v in zip(chunk.idx.np_arr, chunk.J.np_arr):
                    self.assertTrue(
                        np.isclose(v, ref_J_oe[k], rtol=self.rtol, atol=self.atol),
                        msg=f"Failed on category {chunk.category} for index {compound_idx2_reverse(k)}",
                    )
                    ref_J_oe_check[k] = True
            else:
                for k, v in zip(chunk.idx.np_arr, chunk.J.np_arr):
                    self.assertTrue(
                        np.isclose(v, ref_J_te[k], rtol=self.rtol, atol=self.atol),
                        msg=f"Failed on category {chunk.category} for index {compound_idx4_reverse(k)}",
                    )
                    ref_J_te_check[k] = True

        self.assertTrue(
            reduce(lambda x, y: x and y, ref_J_oe_check.values()),
            msg="Some OE indices not acccounted for.",
        )
        self.assertTrue(
            reduce(lambda x, y: x and y, ref_J_te_check.values()),
            msg="Some TE indices not acccounted for.",
        )

    def test_IO_batched(self):
        run_folder = pathlib.Path(__file__).parent.resolve()
        fp = run_folder.joinpath("../data/f2_631g.18det.fcidump")

        N_orb_ref, E0_ref, ref_J_oe, ref_J_te = load_integrals(str(fp))

        N_orb_test, E0_test, chunks = load_integrals_into_chunks(
            str(fp), FakeComm(0, 1), chunk_size=32
        )

        self.assertEqual(N_orb_ref, N_orb_test)
        self.assertEqual(E0_ref, E0_test)

        ref_J_oe_check = {k: False for k in ref_J_oe.keys()}
        ref_J_te_check = {k: False for k in ref_J_te.keys()}

        for chunk in chunks:
            if chunk.category == "OE":
                for k, v in zip(chunk.idx.np_arr, chunk.J.np_arr):
                    self.assertTrue(
                        np.isclose(v, ref_J_oe[k], rtol=self.rtol, atol=self.atol),
                        msg=f"Failed on category {chunk.category} for index {compound_idx2_reverse(k)}",
                    )
                    ref_J_oe_check[k] = True
            else:
                for k, v in zip(chunk.idx.np_arr, chunk.J.np_arr):
                    self.assertTrue(
                        np.isclose(v, ref_J_te[k], rtol=self.rtol, atol=self.atol),
                        msg=f"Failed on category {chunk.category} for index {compound_idx4_reverse(k)}",
                    )
                    ref_J_te_check[k] = True

        self.assertTrue(
            reduce(lambda x, y: x and y, ref_J_oe_check.values()),
            msg="Some OE indices not acccounted for.",
        )
        self.assertTrue(
            reduce(lambda x, y: x and y, ref_J_te_check.values()),
            msg="Some TE indices not acccounted for.",
        )


class Test_IO_f32(Test_f32, Test_IO):
    __test__ = True


class Test_IO_f64(Test_f64, Test_IO):
    __test__ = True


if __name__ == "__main__":
    unittest.main()