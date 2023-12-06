import pathlib
import unittest

import numpy as np
from test_types import Test_f32, Test_f64

from arches.determinant import det_t, spin_det_t
from arches.drivers import Hamiltonian_one_electron, Hamiltonian_two_electrons_integral_driven
from arches.fundamental_types import Determinant as det_ref
from arches.integrals import JChunkFactory, load_integrals_into_chunks
from arches.io import load_integrals
from arches.linked_object import idx_t


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


class Test_pt2_Kernels(unittest.TestCase):
    __test__ = False

    @classmethod
    def setUpClass(cls):
        run_folder = pathlib.Path(__file__).parent.resolve()
        fp = run_folder.joinpath("../data/f2_631g.18det.fcidump")

        _, _, J_oe, J_te = load_integrals(str(fp))

        N_orbs, N_elec, E0, chunks = load_integrals_into_chunks(
            str(fp), FakeComm(0, 1), dtype=cls.dtype
        )

        _, _, _, batched_chunks = load_integrals_into_chunks(
            str(fp), FakeComm(0, 1), chunk_size=32, dtype=cls.dtype
        )

        cls.N_mos = N_orbs
        cls.E0 = E0
        cls.chunks = chunks
        cls.batched_chunks = batched_chunks
        cls.ref_J_oe = J_oe
        cls.ref_J_te = J_te

        max_orb = N_elec // 2
        ground_state = det_t(
            N_orbs,
            spin_det_t(N_orbs, occ=True, max_orb=max_orb),
            spin_det_t(N_orbs, occ=True, max_orb=max_orb),
        )

        cls.ground_state = ground_state
        cls.ground_connected = ground_state.generate_connected_dets()

        cls.ref_dets = [
            det_ref(d.alpha.as_orb_list, d.beta.as_orb_list) for d in cls.ground_connected
        ]

    def filter_chunks(self, cat, batched=False):
        if batched:
            return [chunk for chunk in self.batched_chunks if chunk.category == cat]
        else:
            return [chunk for chunk in self.chunks if chunk.category == cat]

    def launch_denom_kernel(self, kernel, chunk, ext_dets, res):
        kernel(
            chunk.J.p,
            chunk.idx.p,
            chunk.chunk_size,
            idx_t(1),
            ext_dets.det_pointer,
            ext_dets.N_dets,
            res.arr.p,
        )

    def test_cat_A(self):
        A_chunks = self.filter_chunks("A")
        kernel = A_chunks[0].pt2_kernels[1]  # A only has denominator contribution
        pt2_d = self.LArray(N=self.ground_connected.N_dets, fill=0.0)

        A_ref_dict = {k: self.ref_J_te[k] for k in JChunkFactory.A_idx_iter(self.N_mos)}
        ref_driver = Hamiltonian_two_electrons_integral_driven(FilteredDict(A_ref_dict))
        ref_pt2_d = np.array([ref_driver.H_ii(det_J) for det_J in self.ref_dets])

        for chunk in A_chunks:
            self.launch_denom_kernel(kernel, chunk, self.ground_connected, pt2_d)

        self.assertTrue(np.allclose(ref_pt2_d, pt2_d.arr.np_arr, rtol=self.rtol, atol=self.atol))


class Test_pt2_Kernels_f32(Test_f32, Test_pt2_Kernels):
    __test__ = True


class Test_pt2_Kernels_f64(Test_f64, Test_pt2_Kernels):
    __test__ = True
