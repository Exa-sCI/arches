import unittest
from itertools import combinations_with_replacement, product

import numpy as np

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
            all_ind = set([v["idx"] for v in self.canon_order.values() if v["category"] == cat])
            self.assertTrue(
                all_ind == set(idx), msg=f"Failed to get all indices for category {cat}"
            )
            self.assertTrue(
                len(all_ind) == len(idx), msg=f"Some indices double counted in category {cat}"
            )

        check_category("A", list(JChunkFactory.A_idx_iter(self.N_mos)))
        check_category("B", list(JChunkFactory.B_idx_iter(self.N_mos)))
        check_category("C", list(JChunkFactory.C_idx_iter(self.N_mos)))
        check_category("D", list(JChunkFactory.D_idx_iter(self.N_mos)))
        check_category("E", list(JChunkFactory.E_idx_iter(self.N_mos)))
        check_category("F", list(JChunkFactory.F_idx_iter(self.N_mos)))
        check_category("G", list(JChunkFactory.G_idx_iter(self.N_mos)))
        check_category("OE", list(JChunkFactory.OE_idx_iter(self.N_mos)))
