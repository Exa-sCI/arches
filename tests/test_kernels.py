import pathlib
import unittest

import numpy as np
from test_types import Test_f32, Test_f64
from tqdm import tqdm

from arches.determinant import det_t, spin_det_t
from arches.drivers import Hamiltonian_one_electron, Hamiltonian_two_electrons_determinant_driven
from arches.fundamental_types import Determinant as det_ref
from arches.integral_indexing_utils import (
    canonical_idx4,
    compound_idx2_reverse,
    compound_idx4,
    compound_idx4_reverse,
)
from arches.integrals import JChunkFactory, load_integrals_into_chunks
from arches.io import load_integrals
from arches.linked_object import f32_p, idx_t
from arches.matrix import DMatrix

seed = 521392
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


class FilteredDict:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        try:
            return self._data[key]
        except KeyError:
            return 0.0

    def items(self):
        return self._data.items()


class Test_Kernel_Fixture(unittest.TestCase):
    __test__ = False

    @classmethod
    def setUpClass(cls):
        cls.rng = rng

        run_folder = pathlib.Path(__file__).parent.resolve()
        fp = run_folder.joinpath("../data/nh3.5det.fcidump")
        # fp = run_folder.joinpath("../data/f2_631g.18det.fcidump")
        # fp = run_folder.joinpath("../data/c2_eq_hf_dz.fcidump")
        cls.fp = fp

        _, _, J_oe, J_te = load_integrals(str(fp))

        N_mos, N_elec, E0, chunks = load_integrals_into_chunks(
            str(fp), FakeComm(0, 1), dtype=cls.dtype
        )

        _, _, _, batched_chunks = load_integrals_into_chunks(
            str(fp), FakeComm(0, 1), chunk_size=32, dtype=cls.dtype
        )

        cls.N_mos = N_mos
        cls.E0 = E0
        cls.chunks = chunks
        cls.batched_chunks = batched_chunks
        cls.ref_J_oe = J_oe
        cls.ref_J_te = J_te

        max_orb = N_elec // 2
        ground_state = det_t(
            N_mos,
            spin_det_t(N_mos, occ=True, max_orb=max_orb),
            spin_det_t(N_mos, occ=True, max_orb=max_orb),
        )

        cls.ground_state = ground_state
        cls.alpha_orbs = cls.ground_state.alpha.as_orb_list
        cls.beta_orbs = cls.ground_state.beta.as_orb_list
        cls.ref_ground_state = det_ref(cls.alpha_orbs, cls.beta_orbs)

    def filter_chunks(self, cat, batched=False):
        if batched:
            return [chunk for chunk in self.batched_chunks if chunk.category == cat]
        else:
            return [chunk for chunk in self.chunks if chunk.category == cat]

    def get_ref_driver(self, cat):
        fact = JChunkFactory(self.N_mos, cat, None)

        if cat == "OE":
            ref_dict = {compound_idx2_reverse(k): self.ref_J_oe[k] for k in fact.idx_iter}
            return Hamiltonian_one_electron(FilteredDict(ref_dict), 0.0)
        else:
            ref_dict = {k: self.ref_J_te[k] for k in fact.idx_iter}
            return Hamiltonian_two_electrons_determinant_driven(FilteredDict(ref_dict))


class Test_pt2_Kernels(Test_Kernel_Fixture):
    __test__ = False

    @classmethod
    def setUpClass(cls):
        super(Test_pt2_Kernels, cls).setUpClass()

        h_constraint = cls.alpha_orbs[-2:]
        p_constraint = tuple(set([x for x in range(cls.N_mos)]).difference(set(cls.alpha_orbs)))[:2]
        cls.dets_int = cls.ground_state.generate_connected_dets((h_constraint, p_constraint))

        h_constraint = cls.alpha_orbs[-3:]
        p_constraint = tuple(set([x for x in range(cls.N_mos)]).difference(set(cls.alpha_orbs)))[:4]
        cls.dets_ext = cls.dets_int.generate_connected_dets((h_constraint, p_constraint))

        cls.ref_int = [det_ref(d.alpha.as_orb_list, d.beta.as_orb_list) for d in cls.dets_int]
        cls.ref_ext = [det_ref(d.alpha.as_orb_list, d.beta.as_orb_list) for d in cls.dets_ext]

        print(set(cls.ref_int).intersection(set(cls.ref_ext)))
        cls.N_states = 1
        cls.N_int = cls.dets_int.N_dets
        cls.N_ext = cls.dets_ext.N_dets
        ref_psi_coef = rng.uniform(size=(cls.N_int, cls.N_states)).astype(cls.dtype)
        ref_psi_coef = ref_psi_coef / np.linalg.norm(ref_psi_coef, axis=0)
        cls.ref_psi_coef = ref_psi_coef
        cls.test_psi_coef = DMatrix(cls.N_int, cls.N_states, ref_psi_coef, dtype=cls.dtype)

    def launch_denom_kernel(self, kernel, chunk, ext_dets, res):
        kernel(
            chunk.J.p,
            chunk.idx.p,
            idx_t(chunk.chunk_size),
            idx_t(self.N_states),
            ext_dets.det_pointer,
            ext_dets.N_dets,
            res.arr.p,
        )

    def launch_num_kernel(self, kernel, chunk, int_dets, ext_dets, res):
        kernel(
            chunk.J.p,
            chunk.idx.p,
            idx_t(chunk.chunk_size),
            int_dets.det_pointer,
            self.test_psi_coef.arr.p,
            int_dets.N_dets,
            idx_t(self.N_states),
            ext_dets.det_pointer,
            ext_dets.N_dets,
            res.arr.p,
        )

    def run_denom_test(self, cat, verbose=False):
        chunk = self.filter_chunks(cat)[0]
        kernel = chunk.pt2_kernels[1]
        pt2_d = DMatrix(self.N_ext, self.N_states, dtype=self.dtype)

        ref_driver = self.get_ref_driver(cat)

        ref_pt2_d = np.array([ref_driver.H_ii(det_J) for det_J in self.ref_ext])

        self.launch_denom_kernel(kernel, chunk, self.dets_ext, pt2_d)
        for k in range(self.N_states):
            self.assertTrue(
                np.allclose(ref_pt2_d, pt2_d.np_arr[:, k], rtol=self.rtol, atol=self.atol)
            )

    def run_batched_denom_test(self, cat, custom_batches=None):
        if custom_batches is None:
            chunks = self.filter_chunks(cat, batched=True)
        else:
            chunks = custom_batches

        self.assertTrue(len(chunks) > 1)
        kernel = chunks[0].pt2_kernels[1]
        pt2_d = DMatrix(self.N_ext, self.N_states, dtype=self.dtype)

        ref_driver = self.get_ref_driver(cat)
        ref_pt2_d = np.array([ref_driver.H_ii(det_J) for det_J in self.ref_ext])

        for chunk in chunks:
            self.launch_denom_kernel(kernel, chunk, self.dets_ext, pt2_d)

        for k in range(self.N_states):
            self.assertTrue(
                np.allclose(ref_pt2_d, pt2_d.np_arr[:, k], rtol=self.rtol, atol=self.atol)
            )

    def run_num_test(self, cat):
        ref_driver = self.get_ref_driver(cat)
        ref_res = np.zeros((self.N_ext, self.N_states), dtype=self.dtype)

        if cat == "OE":
            for i, det_I in enumerate(self.ref_int):
                for j, det_J in enumerate(self.ref_ext):
                    for k in range(self.N_states):
                        ref_res[j, k] += self.ref_psi_coef[i, k] * ref_driver.H_ij(det_I, det_J)
        else:
            for i, det_I in enumerate(self.ref_int):
                for j, det_J in enumerate(self.ref_ext):
                    if det_I == det_J:
                        continue

                    for idx, phase in ref_driver.H_ij_indices(det_I, det_J):
                        for k in range(self.N_states):
                            ref_res[j, k] += (
                                phase * self.ref_psi_coef[i, k] * ref_driver.H_ijkl_orbital(*idx)
                            )

        chunk = self.filter_chunks(cat)[0]
        kernel = chunk.pt2_kernels[0]
        pt2_n = DMatrix(self.N_ext, self.N_states, dtype=self.dtype)

        self.launch_num_kernel(kernel, chunk, self.dets_int, self.dets_ext, pt2_n)

        # if cat == "F":
        #     test_arr = pt2_n.np_arr
        #     for i in range(self.N_int):
        #         for k in range(self.N_states):
        #             print(f"{(i,k)} : {ref_res[i,k]:0.4e}, {test_arr[i,k]:0.4e}")
        self.assertTrue(np.allclose(ref_res, pt2_n.np_arr, rtol=self.rtol, atol=self.atol))

    def run_batched_num_test(self, cat):
        ref_driver = self.get_ref_driver(cat)
        ref_res = np.zeros((self.N_ext, self.N_states), dtype=self.dtype)

        if cat == "OE":
            for i, det_I in enumerate(self.ref_int):
                for j, det_J in enumerate(self.ref_ext):
                    for k in range(self.N_states):
                        ref_res[j, k] += self.ref_psi_coef[i, k] * ref_driver.H_ij(det_I, det_J)
        else:
            for i, det_I in enumerate(self.ref_int):
                for j, det_J in enumerate(self.ref_ext):
                    for idx, phase in ref_driver.H_ij_indices(det_I, det_J):
                        for k in range(self.N_states):
                            ref_res[j, k] += (
                                phase * self.ref_psi_coef[i, k] * ref_driver.H_ijkl_orbital(*idx)
                            )

        chunks = self.filter_chunks(cat, batched=True)
        kernel = chunks[0].pt2_kernels[0]
        pt2_n = DMatrix(self.N_ext, self.N_states, dtype=self.dtype)

        for chunk in chunks:
            self.launch_num_kernel(kernel, chunk, self.dets_int, self.dets_ext, pt2_n)

        self.assertTrue(np.allclose(ref_res, pt2_n.np_arr, rtol=self.rtol, atol=self.atol))

    def test_cat_OE_denom(self):
        self.run_denom_test("OE")

    def test_cat_OE_denom_batched(self):
        self.run_batched_denom_test("OE")

    def test_cat_OE_num(self):
        self.run_num_test("OE")

    def test_cat_OE_num_batched(self):
        self.run_batched_num_test("OE")

    def test_cat_A(self):
        self.run_denom_test("A")

    def test_cat_A_batched(self):
        ## since test dump only has 18 orbs, need to have smaller batches to test batched chunks for A
        _, _, _, batched_chunks = load_integrals_into_chunks(
            str(self.fp), FakeComm(0, 1), chunk_size=8, dtype=self.dtype
        )
        A_chunks = [chunk for chunk in batched_chunks if chunk.category == "A"]
        self.run_batched_denom_test("A", A_chunks)

    def test_cat_B(self):
        self.run_denom_test("B")

    def test_cat_B_batched(self):
        self.run_batched_denom_test("B")

    def test_cat_C(self):
        self.run_num_test("C")

    def test_cat_C_batched(self):
        self.run_batched_num_test("C")

    def test_cat_D(self):
        self.run_num_test("D")

    def test_cat_D_batched(self):
        self.run_batched_num_test("D")

    def test_cat_E(self):
        self.run_num_test("E")

    def test_cat_E_batched(self):
        self.run_batched_num_test("E")

    def test_cat_F_num(self):
        self.run_num_test("F")

    def test_cat_F_num_batched(self):
        self.run_batched_num_test("F")

    def test_cat_F_denom(self):
        self.run_denom_test("F")

    def test_cat_F_denom_batched(self):
        self.run_batched_denom_test("F")

    def test_cat_G(self):
        self.run_num_test("G")

    def test_cat_G_batched(self):
        self.run_batched_num_test("G")


class Test_pt2_Kernels_f32(Test_f32, Test_pt2_Kernels):
    __test__ = True


class Test_pt2_Kernels_f64(Test_f64, Test_pt2_Kernels):
    __test__ = True


class Test_H_Kernels(Test_Kernel_Fixture):
    __test__ = False

    @classmethod
    def setUpClass(cls):
        super(Test_H_Kernels, cls).setUpClass()

        h_constraint = cls.alpha_orbs[-3:]
        p_constraint = tuple(set([x for x in range(cls.N_mos)]).difference(set(h_constraint)))[-4:]

        constraint = (h_constraint, p_constraint)
        cls.connected = cls.ground_state.generate_connected_dets(constraint)
        cls.ref_dets = [det_ref(d.alpha.as_orb_list, d.beta.as_orb_list) for d in cls.connected]

        ref_dets = cls.ref_dets
        N_ref_dets = len(ref_dets)
        ref_entries = []
        for i in range(N_ref_dets):
            for j in range(i, N_ref_dets):
                exc = ref_dets[i].exc_degree(ref_dets[j])
                if exc[0] + exc[1] <= 2:
                    ref_entries.append((i, j))

        cls.ref_entries = ref_entries

        cls.H_structure = cls.connected.get_H_structure(cls.dtype)

    def test_H_structure(self):
        ref_set = set(self.ref_entries)
        test_entries = []
        for i in range(self.H_structure.m):
            row = i
            row_start = self.H_structure.A_p[row]
            row_end = self.H_structure.A_p[row + 1]
            test_entries.extend(
                [(row, self.H_structure.A_c[idx]) for idx in range(row_start, row_end)]
            )

        test_set = set(test_entries)
        self.assertEqual(ref_set, test_set)

    def launch_H_ii_kernel(self, kernel, chunk):
        kernel(
            chunk.J.p,
            chunk.idx.p,
            idx_t(chunk.chunk_size),
            self.connected.det_pointer,
            idx_t(self.connected.N_dets),
            self.H_structure.A_p.p,
            self.H_structure.A_v.p,
        )

    def launch_H_ij_kernel(self, kernel, chunk):
        kernel(
            chunk.J.p,
            chunk.idx.p,
            idx_t(chunk.chunk_size),
            self.connected.det_pointer,
            idx_t(self.connected.N_dets),
            self.H_structure.A_p.p,
            self.H_structure.A_c.p,
            self.H_structure.A_v.p,
        )

    def run_ii_test(self, cat):
        chunk = self.filter_chunks(cat)[0]
        kernel = chunk.H_kernels[1]

        ref_driver = self.get_ref_driver(cat)
        ref_H_ii = np.array([ref_driver.H_ii(det_J) for det_J in self.ref_dets])

        # Clear out result since H_structure is cached
        for i in range(self.H_structure.m):
            row_start = self.H_structure.A_p[i]
            self.H_structure.A_v[row_start] = 0.0

        self.launch_H_ii_kernel(kernel, chunk)
        test_H_ii = np.zeros(self.H_structure.m, dtype=self.dtype)
        for i in range(self.H_structure.m):
            row_start = self.H_structure.A_p[i]
            test_H_ii[i] = self.H_structure.A_v[row_start]

        self.assertTrue(np.allclose(ref_H_ii, test_H_ii, atol=self.atol, rtol=self.rtol))

    def run_batched_ii_test(self, cat, custom_batches=None):
        if custom_batches is None:
            chunks = self.filter_chunks(cat, batched=True)
        else:
            chunks = custom_batches

        kernel = chunks[0].H_kernels[1]
        self.assertTrue(len(chunks) > 1)

        ref_driver = self.get_ref_driver(cat)
        ref_H_ii = np.array([ref_driver.H_ii(det_J) for det_J in self.ref_dets])

        # Clear out result since H_structure is cached
        for i in range(self.H_structure.m):
            row_start = self.H_structure.A_p[i]
            self.H_structure.A_v[row_start] = 0.0

        for chunk in chunks:
            self.launch_H_ii_kernel(kernel, chunk)

        test_H_ii = np.zeros(self.H_structure.m, dtype=self.dtype)
        for i in range(self.H_structure.m):
            row_start = self.H_structure.A_p[i]
            test_H_ii[i] = self.H_structure.A_v[row_start]

        self.assertTrue(np.allclose(ref_H_ii, test_H_ii, atol=self.atol, rtol=self.rtol))

    def run_ij_test(self, cat):
        chunk = self.filter_chunks(cat)[0]
        kernel = chunk.H_kernels[0]
        ref_driver = self.get_ref_driver(cat)

        ref_H_ij = np.zeros((self.H_structure.m, self.H_structure.m), dtype=self.dtype)
        if cat == "OE":
            for i, j in self.ref_entries:
                if i == j:
                    continue
                det_I = self.ref_dets[i]
                det_J = self.ref_dets[j]
                ref_H_ij[i, j] += ref_driver.H_ij(det_I, det_J)
        else:
            for i, j in self.ref_entries:
                if i == j:
                    continue
                det_I = self.ref_dets[i]
                det_J = self.ref_dets[j]
                for idx, phase in ref_driver.H_ij_indices(det_I, det_J):
                    ref_H_ij[i, j] += phase * ref_driver.H_ijkl_orbital(*idx)

        # Clear out result since H_structure is cached
        for i in range(self.H_structure.N_entries):
            self.H_structure.A_v[i] = 0.0

        self.launch_H_ij_kernel(kernel, chunk)
        test_H_ij = np.zeros_like(ref_H_ij)
        for i in range(self.H_structure.m):
            row_start = self.H_structure.A_p[i]
            row_end = self.H_structure.A_p[i + 1]
            for idx in range(row_start, row_end):
                j = self.H_structure.A_c[idx]
                val = self.H_structure.A_v[idx]
                test_H_ij[i, j] = val

        self.assertTrue(np.allclose(ref_H_ij, test_H_ij, atol=self.atol, rtol=self.rtol))

    def run_batched_ij_test(self, cat):
        chunks = self.filter_chunks(cat)
        kernel = chunks[0].H_kernels[0]
        ref_driver = self.get_ref_driver(cat)

        ref_H_ij = np.zeros((self.H_structure.m, self.H_structure.m), dtype=self.dtype)
        if cat == "OE":
            for i, j in self.ref_entries:
                if i == j:
                    continue
                det_I = self.ref_dets[i]
                det_J = self.ref_dets[j]
                ref_H_ij[i, j] += ref_driver.H_ij(det_I, det_J)
        else:
            for i, j in self.ref_entries:
                if i == j:
                    continue
                det_I = self.ref_dets[i]
                det_J = self.ref_dets[j]
                for idx, phase in ref_driver.H_ij_indices(det_I, det_J):
                    ref_H_ij[i, j] += phase * ref_driver.H_ijkl_orbital(*idx)

        # Clear out result since H_structure is cached
        for i in range(self.H_structure.N_entries):
            self.H_structure.A_v[i] = 0.0

        for chunk in chunks:
            self.launch_H_ij_kernel(kernel, chunk)

        test_H_ij = np.zeros_like(ref_H_ij)
        for i in range(self.H_structure.m):
            row_start = self.H_structure.A_p[i]
            row_end = self.H_structure.A_p[i + 1]
            for idx in range(row_start, row_end):
                j = self.H_structure.A_c[idx]
                val = self.H_structure.A_v[idx]
                test_H_ij[i, j] = val

        self.assertTrue(np.allclose(ref_H_ij, test_H_ij, atol=self.atol, rtol=self.rtol))

    def test_cat_OE_ii(self):
        self.run_ii_test("OE")

    def test_cat_OE_ii_batched(self):
        self.run_batched_ii_test("OE")

    def test_cat_OE_ij(self):
        self.run_ij_test("OE")

    def test_cat_OE_ij_batched(self):
        self.run_batched_ij_test("OE")

    def test_cat_A(self):
        self.run_ii_test("A")

    def test_cat_A_batched(self):
        ## since test dump only has 18 orbs, need to have smaller batches to test batched chunks for A
        _, _, _, batched_chunks = load_integrals_into_chunks(
            str(self.fp), FakeComm(0, 1), chunk_size=8, dtype=self.dtype
        )
        A_chunks = [chunk for chunk in batched_chunks if chunk.category == "A"]
        self.run_batched_ii_test("A", A_chunks)

    def test_cat_B(self):
        self.run_ii_test("B")

    def test_cat_B_batched(self):
        self.run_batched_ii_test("B")

    def test_cat_C(self):
        self.run_ij_test("C")

    def test_cat_C_batched(self):
        self.run_batched_ij_test("C")

    def test_cat_D(self):
        self.run_ij_test("D")

    def test_cat_D_batched(self):
        self.run_batched_ij_test("D")

    def test_cat_E(self):
        self.run_ij_test("E")

    def test_cat_E_batched(self):
        self.run_batched_ij_test("E")

    def test_cat_F_ii(self):
        self.run_ii_test("F")

    def test_cat_F_ii_batched(self):
        self.run_batched_ii_test("F")

    def test_cat_F_ij(self):
        self.run_ij_test("F")

    def test_cat_F_ij_batched(self):
        self.run_batched_ij_test("F")

    def test_cat_G(self):
        self.run_ij_test("G")

    def test_cat_G_batched(self):
        self.run_batched_ij_test("G")


class Test_H_Kernels_f32(Test_f32, Test_H_Kernels):
    __test__ = True


class Test_H_Kernels_f64(Test_f64, Test_H_Kernels):
    __test__ = True
