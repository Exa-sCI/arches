import unittest

import numpy as np

from arches.determinant import DetArray, det_t, double_exc, single_exc, spin_det_t
from arches.fundamental_types import Determinant as det_ref
from arches.fundamental_types import Spin_determinant_tuple as spin_det_ref
from arches.linked_object import LinkedArray_f32, LinkedArray_idx_t, get_indices_by_threshold

rng = np.random.default_rng(seed=6329)


class Test_SpinDet(unittest.TestCase):
    def setUp(self):
        self.rng = rng
        self.N_trials = 64
        self.N_orbs = 72  # larger than both ui32 and ui64 blocks

    def test_constructor(self):
        N_orbs = 8
        N_filled = 4
        orb_list = (0, 1, 2, 4, 6)

        a = spin_det_t(N_orbs)
        b = spin_det_t(N_orbs, occ=True, max_orb=N_filled)
        c = spin_det_t(N_orbs, occ=orb_list)

        a_ref_tuple = tuple([0 for _ in range(N_orbs)])
        b_ref_tuple = tuple([int(x < N_filled) for x in range(N_orbs)])
        c_ref_tuple = tuple([int(x in orb_list) for x in range(N_orbs)])

        self.assertEqual(a.as_bit_tuple, a_ref_tuple)
        self.assertEqual(b.as_bit_tuple, b_ref_tuple)
        self.assertEqual(c.as_bit_tuple, c_ref_tuple)

    def test_multi_block_constructor(self):
        N_filled = 20
        orb_list = (0, 3, 17, 20, 32, 48, 60, 68)

        a = spin_det_t(self.N_orbs)
        b = spin_det_t(self.N_orbs, occ=True, max_orb=N_filled)
        c = spin_det_t(self.N_orbs, occ=orb_list)

        a_ref_tuple = tuple([0 for _ in range(self.N_orbs)])
        b_ref_tuple = tuple([int(x < N_filled) for x in range(self.N_orbs)])
        c_ref_tuple = tuple([int(x in orb_list) for x in range(self.N_orbs)])

        self.assertEqual(a.as_bit_tuple, a_ref_tuple)
        self.assertEqual(b.as_bit_tuple, b_ref_tuple)
        self.assertEqual(c.as_bit_tuple, c_ref_tuple)

    def test_setitem(self):
        def check_setsingle(orb):
            a = spin_det_t(self.N_orbs)
            a[orb] = 1

            ref_arr = np.zeros(self.N_orbs, dtype=int)
            ref_arr[orb] = 1
            self.assertEqual(a.as_bit_tuple, tuple(ref_arr), msg=f"Failed with index {orb}")

        def check_setrange(start_orb, end_orb):
            a = spin_det_t(self.N_orbs)
            a[start_orb:end_orb] = 1

            ref_tuple = tuple([int(x >= start_orb and x < end_orb) for x in range(self.N_orbs)])
            self.assertEqual(
                a.as_bit_tuple, ref_tuple, msg=f"Failed with range {start_orb}:{end_orb}"
            )

        for _ in range(self.N_trials):
            orbs = rng.integers(0, self.N_orbs, 2)
            while len(np.unique(orbs)) < 2:
                orbs = rng.integers(0, self.N_orbs, 2)  # guarantee two unique orbs

            start_orb, end_orb = min(orbs), max(orbs)
            check_setsingle(start_orb)
            check_setrange(start_orb, end_orb)

    def test_getitem(self):
        ref_orbs = (rng.normal(0, 1, size=(self.N_orbs,)) > 0.0).astype(np.int64)
        orb_list = [int(x) for x in np.nonzero(ref_orbs)[0]]

        a = spin_det_t(self.N_orbs, orb_list)

        def check_getsingle(orb):
            self.assertEqual(a[orb], ref_orbs[orb], msg=f"Failed with index {orb}")

        def check_getrange(start_orb, end_orb):
            self.assertEqual(
                a[start_orb:end_orb],
                tuple(ref_orbs[start_orb:end_orb]),
                msg=f"Failed with range {start_orb}:{end_orb}",
            )

        for _ in range(self.N_trials):
            orbs = rng.integers(0, self.N_orbs, 2)
            while len(np.unique(orbs)) < 2:
                orbs = rng.integers(0, self.N_orbs, 2)  # guarantee two unique orbs

            start_orb, end_orb = min(orbs), max(orbs)
            check_getsingle(start_orb)
            check_getrange(start_orb, end_orb)

    def test_bitflip(self):
        def check_bitflip():
            ref_orbs = (rng.normal(0, 1, size=(self.N_orbs,)) > 0.0).astype(np.int64)
            orb_list = [int(x) for x in np.nonzero(ref_orbs)[0]]

            a = spin_det_t(self.N_orbs, orb_list)
            b = ~a

            self.assertFalse(np.any(np.array(a.as_bit_tuple) == np.array(b.as_bit_tuple)))

        for _ in range(self.N_trials):
            check_bitflip()

    def test_xor(self):
        d = spin_det_t(self.N_orbs, True, max_orb=self.N_orbs)

        def check_xor():
            ref_orbs = (rng.normal(0, 1, size=(self.N_orbs,)) > 0.0).astype(np.int64)
            orb_list = [int(x) for x in np.nonzero(ref_orbs)[0]]

            a = spin_det_t(self.N_orbs, orb_list)
            b = ~a
            c = a ^ b

            self.assertTrue(np.all(c.as_bit_tuple))
            self.assertEqual(a.as_bit_tuple, (b ^ d).as_bit_tuple)

        for _ in range(self.N_trials):
            check_xor()

    def test_and(self):
        d = spin_det_t(self.N_orbs, True, max_orb=self.N_orbs)

        def check_and():
            ref_orbs = (rng.normal(0, 1, size=(self.N_orbs,)) > 0.0).astype(np.int64)
            orb_list = [int(x) for x in np.nonzero(ref_orbs)[0]]

            a = spin_det_t(self.N_orbs, orb_list)
            b = ~a
            c = a & b

            self.assertFalse(np.any(c.as_bit_tuple))
            self.assertEqual(a.as_bit_tuple, (a & d).as_bit_tuple)

        for _ in range(self.N_trials):
            check_and()

    def test_popcount(self):
        def check_popcount():
            ref_orbs = (rng.normal(0, 1, size=(self.N_orbs,)) > 0.0).astype(np.int64)
            orb_list = [int(x) for x in np.nonzero(ref_orbs)[0]]

            a = spin_det_t(self.N_orbs, orb_list)

            self.assertEqual(len(orb_list), a.popcount())

        for _ in range(self.N_trials):
            check_popcount()

    def test_compute_single_phase(self):
        def check_phase(orb_list, h, p):
            ref_phase = spin_det_ref(orb_list).single_phase(h, p)
            test_phase = spin_det_t(self.N_orbs, orb_list).compute_phase_single_exc(h, p)

            self.assertEqual(
                ref_phase,
                test_phase,
                msg=f"Incorrect phase for occupied orbitals {orb_list} and excitation {h}->{p}",
            )

        for _ in range(self.N_trials):
            orb_list = rng.integers(0, self.N_orbs, 8)
            orb_list = tuple(np.unique(orb_list))
            h = rng.choice(orb_list, replace=False)
            p = rng.integers(0, self.N_orbs, 1)[0]
            while p in orb_list:
                p = rng.integers(0, self.N_orbs, 1)[0]
            check_phase(orb_list, h, p)

    def test_compute_double_phase(self):
        def check_phase(orb_list, h1, h2, p1, p2):
            ref_phase = spin_det_ref(orb_list).double_phase(h1, p1, h2, p2)
            test_phase = spin_det_t(self.N_orbs, orb_list).compute_phase_double_exc(h1, h2, p1, p2)

            self.assertEqual(
                ref_phase,
                test_phase,
                msg=f"Incorrect phase for occupied orbitals {orb_list} and excitation {(h1, h2)}->{(p1, p2)}",
            )

        for _ in range(self.N_trials):
            orb_list = rng.integers(0, self.N_orbs, 16)
            orb_list = tuple(np.unique(orb_list))
            h1, h2 = rng.choice(orb_list, 2, replace=False)
            p1, p2 = rng.integers(0, self.N_orbs, 2)
            while (p1 in orb_list) or (p2 in orb_list):
                p1, p2 = rng.integers(0, self.N_orbs, 2)
            check_phase(orb_list, h1, h2, p1, p2)

    def test_apply_single_exc(self):
        def check_single_exc(orb_list, h, p):
            det = single_exc(h, p, None) @ spin_det_t(self.N_orbs, orb_list)
            orbs = det.as_orb_list

            self.assertTrue(set(orb_list).symmetric_difference(set(orbs)) == set([h, p]))

        for _ in range(self.N_trials):
            orb_list = rng.integers(0, self.N_orbs, 8)
            orb_list = tuple(np.unique(orb_list))
            h = rng.choice(orb_list, replace=False)
            p = rng.integers(0, self.N_orbs, 1)[0]
            while p in orb_list:
                p = rng.integers(0, self.N_orbs, 1)[0]
            check_single_exc(orb_list, h, p)

    def test_apply_double_exc(self):
        def check_double_exc(orb_list, h1, h2, p1, p2):
            det = double_exc((h1, h2), (p1, p2), None) @ spin_det_t(self.N_orbs, orb_list)
            orbs = det.as_orb_list

            self.assertTrue(set(orb_list).symmetric_difference(set(orbs)) == set([h1, h2, p1, p2]))

        def check_chained_singles(orb_list, h1, h2, p1, p2):
            exc = single_exc(h1, p1) @ single_exc(h2, p2)
            det = exc @ spin_det_t(self.N_orbs, orb_list)

            orbs = det.as_orb_list

            self.assertTrue(set(orb_list).symmetric_difference(set(orbs)) == set([h1, h2, p1, p2]))

        for _ in range(self.N_trials):
            orb_list = rng.integers(0, self.N_orbs, 8)
            orb_list = tuple(np.unique(orb_list))
            h1, h2 = rng.choice(orb_list, 2, replace=False)
            p1, p2 = rng.integers(0, self.N_orbs, 2)
            while p1 in orb_list or p2 in orb_list:
                p1, p2 = rng.integers(0, self.N_orbs, 2)

            check_double_exc(orb_list, h1, h2, p1, p2)
            check_chained_singles(orb_list, h1, h2, p1, p2)


class Test_Det(unittest.TestCase):
    def setUp(self):
        self.rng = rng
        self.N_trials = 64
        self.N_orbs = 72  # larger than both ui32 and ui64 blocks

    def test_constructor(self):
        N_orbs = 8
        N_filled = 4
        orb_list = (0, 1, 2, 4, 6)

        a = det_t(N_orbs)
        b_s = spin_det_t(N_orbs, occ=True, max_orb=N_filled)
        b_s_2 = spin_det_t(N_orbs, occ=True, max_orb=N_filled)
        b = det_t(alpha=b_s, beta=b_s_2)
        c = det_t(alpha=b_s, beta=spin_det_t(N_orbs, occ=orb_list))
        c_b = det_t(alpha=spin_det_t(N_orbs, occ=orb_list), beta=b_s)

        a_ref_tuple = tuple([0 for _ in range(N_orbs)])
        b_ref_tuple = tuple([int(x < N_filled) for x in range(N_orbs)])
        c_ref_tuple = tuple([int(x in orb_list) for x in range(N_orbs)])

        self.assertEqual(a.alpha.as_bit_tuple, a_ref_tuple)
        self.assertEqual(a.beta.as_bit_tuple, a_ref_tuple)

        self.assertEqual(b.alpha.as_bit_tuple, b_ref_tuple)
        self.assertEqual(b.beta.as_bit_tuple, b_ref_tuple)

        self.assertEqual(c[0].as_bit_tuple, b_ref_tuple)
        self.assertEqual(c[1].as_bit_tuple, c_ref_tuple)

        self.assertEqual(c_b[0].as_bit_tuple, c_ref_tuple)
        self.assertEqual(c_b[1].as_bit_tuple, b_ref_tuple)

    def test_multi_block_constructor(self):
        N_filled = 20
        orb_list = (0, 3, 17, 20, 32, 48, 60, 68)

        a = det_t(self.N_orbs)
        b_s = spin_det_t(self.N_orbs, occ=True, max_orb=N_filled)
        b = det_t(alpha=b_s, beta=b_s)
        c = det_t(alpha=b_s, beta=spin_det_t(self.N_orbs, occ=orb_list))
        c_b = det_t(alpha=spin_det_t(self.N_orbs, occ=orb_list), beta=b_s)

        a_ref_tuple = tuple([0 for _ in range(self.N_orbs)])
        b_ref_tuple = tuple([int(x < N_filled) for x in range(self.N_orbs)])
        c_ref_tuple = tuple([int(x in orb_list) for x in range(self.N_orbs)])

        self.assertEqual(a.alpha.as_bit_tuple, a_ref_tuple)
        self.assertEqual(a.beta.as_bit_tuple, a_ref_tuple)

        self.assertEqual(b.alpha.as_bit_tuple, b_ref_tuple)
        self.assertEqual(b.beta.as_bit_tuple, b_ref_tuple)

        self.assertEqual(c[0].as_bit_tuple, b_ref_tuple)
        self.assertEqual(c[1].as_bit_tuple, c_ref_tuple)

        self.assertEqual(c_b[0].as_bit_tuple, c_ref_tuple)
        self.assertEqual(c_b[1].as_bit_tuple, b_ref_tuple)

    def test_compute_opp_spin_double_phase(self):
        def check_phase(alpha_orb_list, beta_orb_list, h1, h2, p1, p2):
            alpha = spin_det_t(self.N_orbs, alpha_orb_list)
            beta = spin_det_t(self.N_orbs, beta_orb_list)
            det = det_t(alpha=alpha, beta=beta)
            aphase = alpha.compute_phase_single_exc(h1, p1)
            bphase = beta.compute_phase_single_exc(h2, p2)
            ref_phase = aphase * bphase

            test_phase = det.compute_phase_opp_spin_double_exc(h1, h2, p1, p2)

            self.assertEqual(
                ref_phase,
                test_phase,
                msg=f"Incorrect phase for occupied orbitals {alpha_orb_list, beta_orb_list} and excitation {(h1, h2)}->{(p1, p2)}",
            )

        for _ in range(self.N_trials):
            alpha_orb_list = rng.integers(0, self.N_orbs, 8)
            alpha_orb_list = tuple(np.unique(alpha_orb_list))
            beta_orb_list = rng.integers(0, self.N_orbs, 8)
            beta_orb_list = tuple(np.unique(beta_orb_list))

            h1 = rng.choice(alpha_orb_list, 1)[0]
            h2 = rng.choice(beta_orb_list, 1)[0]

            p1, p2 = rng.integers(0, self.N_orbs, 2)
            while (p1 in alpha_orb_list) or (p2 in beta_orb_list):
                p1, p2 = rng.integers(0, self.N_orbs, 2)

            check_phase(alpha_orb_list, beta_orb_list, h1, h2, p1, p2)

    def test_exc_det(self):
        def check_exc(a1, b1, a2, b2):
            ref_1 = det_ref(a1, b1)
            ref_2 = det_ref(a2, b2)
            test_1 = det_t(alpha=spin_det_t(self.N_orbs, a1), beta=spin_det_t(self.N_orbs, b1))
            test_2 = det_t(alpha=spin_det_t(self.N_orbs, a2), beta=spin_det_t(self.N_orbs, b2))

            ref_exc = det_ref(ref_1.alpha ^ ref_2.alpha, ref_1.beta ^ ref_2.beta)
            test_exc = test_1.get_exc_det(test_2)

            alpha_orbs = test_exc.alpha.as_orb_list
            beta_orbs = test_exc.beta.as_orb_list

            self.assertEqual(ref_exc.alpha, alpha_orbs)
            self.assertEqual(ref_exc.beta, beta_orbs)

        for _ in range(self.N_trials):
            a1 = rng.integers(0, self.N_orbs, 8)
            a1 = tuple(np.unique(a1))
            b1 = rng.integers(0, self.N_orbs, 8)
            b1 = tuple(np.unique(b1))

            a2 = rng.integers(0, self.N_orbs, 8)
            a2 = tuple(np.unique(a2))
            b2 = rng.integers(0, self.N_orbs, 8)
            b2 = tuple(np.unique(b2))

            check_exc(a1, b1, a2, b2)

    def test_apply_single_exc(self):
        def check_single_exc(alpha_orbs, beta_orbs, h, p, spin):
            exc = single_exc(h, p, spin)
            det = det_t(
                alpha=spin_det_t(self.N_orbs, alpha_orbs),
                beta=spin_det_t(self.N_orbs, beta_orbs),
            )

            res = exc @ det
            orbs = res[spin].as_orb_list
            orb_list = beta_orbs if spin else alpha_orbs
            self.assertTrue(set(orb_list).symmetric_difference(set(orbs)) == set([h, p]))

        for _ in range(self.N_trials):
            alpha_orb_list = rng.integers(0, self.N_orbs, 8)
            alpha_orb_list = tuple(np.unique(alpha_orb_list))
            beta_orb_list = rng.integers(0, self.N_orbs, 8)
            beta_orb_list = tuple(np.unique(beta_orb_list))

            h1 = rng.choice(alpha_orb_list, 1)[0]
            h2 = rng.choice(beta_orb_list, 1)[0]

            p1, p2 = rng.integers(0, self.N_orbs, 2)
            while (p1 in alpha_orb_list) or (p2 in beta_orb_list):
                p1, p2 = rng.integers(0, self.N_orbs, 2)

            check_single_exc(alpha_orb_list, beta_orb_list, h1, p1, 0)
            check_single_exc(alpha_orb_list, beta_orb_list, h2, p2, 1)

    def test_apply_double_exc(self):
        def check_same_spin_double(alpha_orbs, beta_orbs, h1, h2, p1, p2, spin):
            exc = double_exc((h1, h2), (p1, p2), (spin, spin))
            det = det_t(
                alpha=spin_det_t(self.N_orbs, alpha_orbs),
                beta=spin_det_t(self.N_orbs, beta_orbs),
            )

            res = exc @ det
            orbs = res[spin].as_orb_list
            orb_list = beta_orbs if spin else alpha_orbs
            self.assertTrue(set(orb_list).symmetric_difference(set(orbs)) == set([h1, h2, p1, p2]))

        def check_same_spin_chained_single(alpha_orbs, beta_orbs, h1, h2, p1, p2, spin):
            exc = single_exc(h1, p1, spin) @ single_exc(h2, p2, spin)
            det = det_t(
                alpha=spin_det_t(self.N_orbs, alpha_orbs),
                beta=spin_det_t(self.N_orbs, beta_orbs),
            )

            res = exc @ det
            orbs = res[spin].as_orb_list
            orb_list = beta_orbs if spin else alpha_orbs
            self.assertTrue(set(orb_list).symmetric_difference(set(orbs)) == set([h1, h2, p1, p2]))

        def check_opp_spin_double(alpha_orbs, beta_orbs, h1, h2, p1, p2, s1, s2):
            exc = double_exc((h1, h2), (p1, p2), (s1, s2))
            det = det_t(
                alpha=spin_det_t(self.N_orbs, alpha_orbs),
                beta=spin_det_t(self.N_orbs, beta_orbs),
            )
            res = exc @ det

            orbs1 = res[s1].as_orb_list
            orbs2 = res[s2].as_orb_list

            orb_list1 = beta_orbs if s1 else alpha_orbs
            orb_list2 = beta_orbs if s2 else alpha_orbs

            self.assertTrue(set(orb_list1).symmetric_difference(set(orbs1)) == set([h1, p1]))
            self.assertTrue(set(orb_list2).symmetric_difference(set(orbs2)) == set([h2, p2]))

        def check_opp_spin_chained_single(alpha_orbs, beta_orbs, h1, h2, p1, p2, s1, s2):
            exc = single_exc(h1, p1, s1) @ single_exc(h2, p2, s2)
            det = det_t(
                alpha=spin_det_t(self.N_orbs, alpha_orbs),
                beta=spin_det_t(self.N_orbs, beta_orbs),
            )
            res = exc @ det

            orbs1 = res[s1].as_orb_list
            orbs2 = res[s2].as_orb_list

            orb_list1 = beta_orbs if s1 else alpha_orbs
            orb_list2 = beta_orbs if s2 else alpha_orbs

            self.assertTrue(set(orb_list1).symmetric_difference(set(orbs1)) == set([h1, p1]))
            self.assertTrue(set(orb_list2).symmetric_difference(set(orbs2)) == set([h2, p2]))

        for _ in range(self.N_trials):
            alpha_orb_list = rng.integers(0, self.N_orbs, 8)
            alpha_orb_list = tuple(np.unique(alpha_orb_list))
            beta_orb_list = rng.integers(0, self.N_orbs, 8)
            beta_orb_list = tuple(np.unique(beta_orb_list))

            ha1, ha2 = rng.choice(alpha_orb_list, 2, replace=False)
            hb1, hb2 = rng.choice(beta_orb_list, 2, replace=False)

            pa1, pa2 = rng.integers(0, self.N_orbs, 2)
            while pa1 in alpha_orb_list or pa2 in alpha_orb_list:
                pa1, pa2 = rng.integers(0, self.N_orbs, 2)

            pb1, pb2 = rng.integers(0, self.N_orbs, 2)
            while pb1 in beta_orb_list or pb2 in beta_orb_list:
                pb1, pb2 = rng.integers(0, self.N_orbs, 2)

            check_same_spin_double(alpha_orb_list, beta_orb_list, ha1, ha2, pa1, pa2, 0)
            check_same_spin_chained_single(alpha_orb_list, beta_orb_list, ha1, ha2, pa1, pa2, 0)
            check_same_spin_double(alpha_orb_list, beta_orb_list, hb1, hb2, pb1, pb2, 1)
            check_same_spin_chained_single(alpha_orb_list, beta_orb_list, hb1, hb2, pb1, pb2, 1)

            check_opp_spin_double(alpha_orb_list, beta_orb_list, ha1, hb1, pa1, pb1, 0, 1)
            check_opp_spin_double(alpha_orb_list, beta_orb_list, hb2, ha2, pb2, pa2, 1, 0)
            check_opp_spin_chained_single(alpha_orb_list, beta_orb_list, ha1, hb1, pa1, pb1, 0, 1)
            check_opp_spin_chained_single(alpha_orb_list, beta_orb_list, hb2, ha2, pb2, pa2, 1, 0)


class Test_DetArray(unittest.TestCase):
    def setUp(self):
        self.rng = rng
        self.N_trials = 32
        self.N_orbs = 72  # larger than both ui32 and ui64 blocks
        self.N_dets = 1024

    def test_constructor(self):
        N_orbs = 8
        N_dets = 32
        arr = DetArray(N_dets, N_orbs)
        ref_tuple = tuple([0 for _ in range(N_orbs)])
        for i in range(N_dets):
            self.assertEqual(arr[i][0].as_bit_tuple, ref_tuple)
            self.assertEqual(arr[i][1].as_bit_tuple, ref_tuple)

    def test_set_get_element(self):
        dets = DetArray(self.N_dets, self.N_orbs)

        ref_orb_lists = []
        for i in range(self.N_dets):
            alpha_orb_list = rng.integers(0, self.N_orbs, 8)
            alpha_orb_list = tuple(np.unique(alpha_orb_list))
            beta_orb_list = rng.integers(0, self.N_orbs, 8)
            beta_orb_list = tuple(np.unique(beta_orb_list))

            ref_orb_lists.append((alpha_orb_list, beta_orb_list))

            dets[i] = det_t(
                alpha=spin_det_t(self.N_orbs, alpha_orb_list),
                beta=spin_det_t(self.N_orbs, beta_orb_list),
            )

        for i, det in enumerate(dets):
            self.assertEqual(ref_orb_lists[i][0], det[0].as_orb_list)
            self.assertEqual(ref_orb_lists[i][1], det[1].as_orb_list)

    def test_get_connected_singles(self):
        N_orbs = 16

        def check_connected_singles(alpha_orb_list, beta_orb_list):
            source_det = det_t(
                alpha=spin_det_t(N_orbs, alpha_orb_list), beta=spin_det_t(N_orbs, beta_orb_list)
            )
            ref_source = det_ref(alpha_orb_list, beta_orb_list)
            connected_dets = ref_source.gen_all_connected_det(N_orbs)

            def check_single(x, y):
                exc_degree = x.exc_degree(y)
                return (exc_degree == (1, 0)) or (exc_degree == (0, 1))

            connected_singles = [d for d in connected_dets if check_single(ref_source, d)]
            ref_set = set([(d.alpha, d.beta) for d in connected_singles])

            test_singles = source_det.get_connected_singles()
            test_set = set((d[0].as_orb_list, d[1].as_orb_list) for d in test_singles)

            self.assertEqual(
                ref_set, test_set, msg=f"Failed for {alpha_orb_list} X {beta_orb_list}"
            )

        for _ in range(self.N_trials):
            alpha_orb_list = rng.integers(0, N_orbs, 4)
            alpha_orb_list = tuple(np.unique(alpha_orb_list))
            beta_orb_list = rng.integers(0, N_orbs, 4)
            beta_orb_list = tuple(np.unique(beta_orb_list))

            check_connected_singles(alpha_orb_list, beta_orb_list)

    def test_get_constrained_singles(self):
        N_orbs = 16

        def check_common_constraint(alpha_orb_list, beta_orb_list, constraint):
            source_det = det_t(
                alpha=spin_det_t(N_orbs, alpha_orb_list), beta=spin_det_t(N_orbs, beta_orb_list)
            )
            ref_source = det_ref(alpha_orb_list, beta_orb_list)
            connected_dets = ref_source.gen_all_connected_det(N_orbs)

            def check_single(x, y):
                match x.exc_degree(y):
                    case (1, 0):
                        h, p = det_ref.single_exc_no_phase(x.alpha, y.alpha)
                    case (0, 1):
                        h, p = det_ref.single_exc_no_phase(x.beta, y.beta)
                    case _:
                        return False

                return (h in constraint[0]) and (p in constraint[1])

            connected_singles = [d for d in connected_dets if check_single(ref_source, d)]
            ref_set = set([(d.alpha, d.beta) for d in connected_singles])

            test_singles = source_det.get_connected_singles(constraint)
            test_set = set((d[0].as_orb_list, d[1].as_orb_list) for d in test_singles)

            self.assertEqual(
                ref_set, test_set, msg=f"Failed for {alpha_orb_list} X {beta_orb_list}"
            )

        def check_different_constraint(alpha_orb_list, beta_orb_list, constraint):
            source_det = det_t(
                alpha=spin_det_t(N_orbs, alpha_orb_list), beta=spin_det_t(N_orbs, beta_orb_list)
            )
            ref_source = det_ref(alpha_orb_list, beta_orb_list)
            connected_dets = ref_source.gen_all_connected_det(N_orbs)

            def check_single(x, y):
                match x.exc_degree(y):
                    case (1, 0):
                        h, p = det_ref.single_exc_no_phase(x.alpha, y.alpha)
                        return (h in constraint[0]) and (p in constraint[2])
                    case (0, 1):
                        h, p = det_ref.single_exc_no_phase(x.beta, y.beta)
                        return (h in constraint[1]) and (p in constraint[3])
                    case _:
                        return False

            connected_singles = [d for d in connected_dets if check_single(ref_source, d)]
            ref_set = set([(d.alpha, d.beta) for d in connected_singles])

            test_singles = source_det.get_connected_singles(constraint)
            test_set = set((d[0].as_orb_list, d[1].as_orb_list) for d in test_singles)

            self.assertEqual(
                ref_set, test_set, msg=f"Failed for {alpha_orb_list} X {beta_orb_list}"
            )

        for _ in range(self.N_trials):
            common_h, common_p = tuple(), tuple()
            while len(common_h) < 2 or len(common_p) < 2:
                alpha_orb_list = rng.integers(0, N_orbs, 8)
                alpha_orb_list = tuple(np.unique(alpha_orb_list))
                beta_orb_list = rng.integers(0, N_orbs, 8)
                beta_orb_list = tuple(np.unique(beta_orb_list))

                ah = set(alpha_orb_list)
                ap = set(tuple(range(N_orbs))).difference(ah)
                bh = set(beta_orb_list)
                bp = set(tuple(range(N_orbs))).difference(bh)
                common_h = tuple(ah.intersection(bh))
                common_p = tuple(ap.intersection(bp))

            common_constraint = (
                tuple(rng.choice(common_h, 2, replace=False)),
                tuple(rng.choice(common_p, 2, replace=False)),
            )
            check_common_constraint(alpha_orb_list, beta_orb_list, common_constraint)

            diff_constraint = tuple(
                tuple(rng.choice(tuple(x), 2, replace=False)) for x in (ah, bh, ap, bp)
            )
            check_different_constraint(alpha_orb_list, beta_orb_list, diff_constraint)

    def test_get_connected_ss_doubles(self):
        N_orbs = 16

        def check_connected_ss_doubles(alpha_orb_list, beta_orb_list):
            source_det = det_t(
                alpha=spin_det_t(N_orbs, alpha_orb_list), beta=spin_det_t(N_orbs, beta_orb_list)
            )
            ref_source = det_ref(alpha_orb_list, beta_orb_list)
            connected_dets = ref_source.gen_all_connected_det(N_orbs)

            def check_ss_double(x, y):
                exc_degree = x.exc_degree(y)
                return (exc_degree == (2, 0)) or (exc_degree == (0, 2))

            connected_ss_doubles = [d for d in connected_dets if check_ss_double(ref_source, d)]
            ref_set = set([(d.alpha, d.beta) for d in connected_ss_doubles])

            test_doubles = source_det.get_connected_ss_doubles()
            test_set = set((d[0].as_orb_list, d[1].as_orb_list) for d in test_doubles)

            self.assertEqual(
                ref_set, test_set, msg=f"Failed for {alpha_orb_list} X {beta_orb_list}"
            )

        for _ in range(self.N_trials):
            alpha_orb_list = rng.integers(0, N_orbs, 4)
            alpha_orb_list = tuple(np.unique(alpha_orb_list))
            beta_orb_list = rng.integers(0, N_orbs, 4)
            beta_orb_list = tuple(np.unique(beta_orb_list))

            check_connected_ss_doubles(alpha_orb_list, beta_orb_list)

    def test_get_constrained_ss_doubles(self):
        N_orbs = 16

        def check_common_constraint(alpha_orb_list, beta_orb_list, constraint):
            source_det = det_t(
                alpha=spin_det_t(N_orbs, alpha_orb_list), beta=spin_det_t(N_orbs, beta_orb_list)
            )
            ref_source = det_ref(alpha_orb_list, beta_orb_list)
            connected_dets = ref_source.gen_all_connected_det(N_orbs)

            def check_double(x, y):
                holes = constraint[0]
                parts = constraint[1]
                match x.exc_degree(y):
                    case (2, 0):
                        h1, h2, p1, p2 = det_ref.double_exc_no_phase(x.alpha, y.alpha)
                    case (0, 2):
                        h1, h2, p1, p2 = det_ref.double_exc_no_phase(x.beta, y.beta)
                    case _:
                        return False
                return h1 in holes and h2 in holes and p1 in parts and p2 in parts

            constrained_ss_doubles = [d for d in connected_dets if check_double(ref_source, d)]
            ref_set = set([(d.alpha, d.beta) for d in constrained_ss_doubles])

            test_doubles = source_det.get_connected_ss_doubles(constraint)
            test_set = set((d[0].as_orb_list, d[1].as_orb_list) for d in test_doubles)

            self.assertEqual(
                ref_set, test_set, msg=f"Failed for {alpha_orb_list} X {beta_orb_list}"
            )
            return ref_set, test_set

        def check_different_constraint(alpha_orb_list, beta_orb_list, constraint):
            source_det = det_t(
                alpha=spin_det_t(N_orbs, alpha_orb_list), beta=spin_det_t(N_orbs, beta_orb_list)
            )
            ref_source = det_ref(alpha_orb_list, beta_orb_list)
            connected_dets = ref_source.gen_all_connected_det(N_orbs)

            def check_double(x, y):
                match x.exc_degree(y):
                    case (2, 0):
                        h1, h2, p1, p2 = det_ref.double_exc_no_phase(x.alpha, y.alpha)
                        holes = constraint[0]
                        parts = constraint[2]
                    case (0, 2):
                        h1, h2, p1, p2 = det_ref.double_exc_no_phase(x.beta, y.beta)
                        holes = constraint[1]
                        parts = constraint[3]
                    case _:
                        return False

                return h1 in holes and h2 in holes and p1 in parts and p2 in parts

            constrained_ss_doubles = [d for d in connected_dets if check_double(ref_source, d)]
            ref_set = set([(d.alpha, d.beta) for d in constrained_ss_doubles])
            test_doubles = source_det.get_connected_ss_doubles(constraint)
            test_set = set((d[0].as_orb_list, d[1].as_orb_list) for d in test_doubles)

            self.assertEqual(
                ref_set, test_set, msg=f"Failed for {alpha_orb_list} X {beta_orb_list}"
            )

        for _ in range(self.N_trials):
            common_h, common_p = tuple(), tuple()
            while len(common_h) < 4 or len(common_p) < 4:
                alpha_orb_list = tuple(rng.choice(np.arange(N_orbs), 8, replace=False))
                beta_orb_list = tuple(rng.choice(np.arange(N_orbs), 8, replace=False))

                ah = set(alpha_orb_list)
                ap = set(tuple(range(N_orbs))).difference(ah)
                bh = set(beta_orb_list)
                bp = set(tuple(range(N_orbs))).difference(bh)
                common_h = tuple(ah.intersection(bh))
                common_p = tuple(ap.intersection(bp))

            common_constraint = (
                tuple(rng.choice(common_h, 4, replace=False)),
                tuple(rng.choice(common_p, 4, replace=False)),
            )
            check_common_constraint(alpha_orb_list, beta_orb_list, common_constraint)

            diff_constraint = tuple(
                tuple(rng.choice(tuple(x), 4, replace=False)) for x in (ah, bh, ap, bp)
            )
            check_different_constraint(alpha_orb_list, beta_orb_list, diff_constraint)

    def test_get_connected_os_doubles(self):
        N_orbs = 16

        def check_connected_os_doubles(alpha_orb_list, beta_orb_list):
            source_det = det_t(
                alpha=spin_det_t(N_orbs, alpha_orb_list), beta=spin_det_t(N_orbs, beta_orb_list)
            )
            ref_source = det_ref(alpha_orb_list, beta_orb_list)
            connected_dets = ref_source.gen_all_connected_det(N_orbs)

            def check_os_double(x, y):
                exc_degree = x.exc_degree(y)
                return exc_degree == (1, 1)

            connected_os_doubles = [d for d in connected_dets if check_os_double(ref_source, d)]
            ref_set = set([(d.alpha, d.beta) for d in connected_os_doubles])

            test_doubles = source_det.get_connected_os_doubles()
            test_set = set((d[0].as_orb_list, d[1].as_orb_list) for d in test_doubles)

            self.assertEqual(
                ref_set, test_set, msg=f"Failed for {alpha_orb_list} X {beta_orb_list}"
            )

        for _ in range(self.N_trials):
            alpha_orb_list = rng.integers(0, N_orbs, 4)
            alpha_orb_list = tuple(np.unique(alpha_orb_list))
            beta_orb_list = rng.integers(0, N_orbs, 4)
            beta_orb_list = tuple(np.unique(beta_orb_list))

            check_connected_os_doubles(alpha_orb_list, beta_orb_list)

    def test_get_constrained_os_doubles(self):
        N_orbs = 16

        def check_common_constraint(alpha_orb_list, beta_orb_list, constraint):
            source_det = det_t(
                alpha=spin_det_t(N_orbs, alpha_orb_list), beta=spin_det_t(N_orbs, beta_orb_list)
            )
            ref_source = det_ref(alpha_orb_list, beta_orb_list)
            connected_dets = ref_source.gen_all_connected_det(N_orbs)

            def check_os_double(x, y):
                exc_degree = x.exc_degree(y)
                holes = constraint[0]
                parts = constraint[1]
                if exc_degree == (1, 1):
                    h1, p1 = det_ref.single_exc_no_phase(x.alpha, y.alpha)
                    h2, p2 = det_ref.single_exc_no_phase(x.beta, y.beta)
                    return h1 in holes and h2 in holes and p1 in parts and p2 in parts
                else:
                    return False

            connected_os_doubles = [d for d in connected_dets if check_os_double(ref_source, d)]
            ref_set = set([(d.alpha, d.beta) for d in connected_os_doubles])

            test_doubles = source_det.get_connected_os_doubles(constraint)
            test_set = set((d[0].as_orb_list, d[1].as_orb_list) for d in test_doubles)

            self.assertEqual(
                ref_set, test_set, msg=f"Failed for {alpha_orb_list} X {beta_orb_list}"
            )

        def check_different_constraint(alpha_orb_list, beta_orb_list, constraint):
            source_det = det_t(
                alpha=spin_det_t(N_orbs, alpha_orb_list), beta=spin_det_t(N_orbs, beta_orb_list)
            )
            ref_source = det_ref(alpha_orb_list, beta_orb_list)
            connected_dets = ref_source.gen_all_connected_det(N_orbs)

            def check_os_double(x, y):
                exc_degree = x.exc_degree(y)
                ah, bh, ap, bp = constraint
                if exc_degree == (1, 1):
                    h1, p1 = det_ref.single_exc_no_phase(x.alpha, y.alpha)
                    h2, p2 = det_ref.single_exc_no_phase(x.beta, y.beta)
                    return h1 in ah and p1 in ap and h2 in bh and p2 in bp
                else:
                    return False

            connected_os_doubles = [d for d in connected_dets if check_os_double(ref_source, d)]
            ref_set = set([(d.alpha, d.beta) for d in connected_os_doubles])

            test_doubles = source_det.get_connected_os_doubles(constraint)
            test_set = set((d[0].as_orb_list, d[1].as_orb_list) for d in test_doubles)

            self.assertEqual(
                ref_set, test_set, msg=f"Failed for {alpha_orb_list} X {beta_orb_list}"
            )

        for _ in range(self.N_trials):
            common_h, common_p = tuple(), tuple()
            while len(common_h) < 2 or len(common_p) < 2:
                alpha_orb_list = rng.integers(0, N_orbs, 8)
                alpha_orb_list = tuple(np.unique(alpha_orb_list))
                beta_orb_list = rng.integers(0, N_orbs, 8)
                beta_orb_list = tuple(np.unique(beta_orb_list))

                ah = set(alpha_orb_list)
                ap = set(tuple(range(N_orbs))).difference(ah)
                bh = set(beta_orb_list)
                bp = set(tuple(range(N_orbs))).difference(bh)
                common_h = tuple(ah.intersection(bh))
                common_p = tuple(ap.intersection(bp))

            common_constraint = (
                tuple(rng.choice(common_h, 2, replace=False)),
                tuple(rng.choice(common_p, 2, replace=False)),
            )

            check_common_constraint(alpha_orb_list, beta_orb_list, common_constraint)

            diff_constraint = tuple(
                tuple(rng.choice(tuple(x), 2, replace=False)) for x in (ah, bh, ap, bp)
            )
            check_different_constraint(alpha_orb_list, beta_orb_list, diff_constraint)

    def test_generate_connected_dets(self):
        N_orbs = 72

        def check_generate_dets(alpha_orb_list, beta_orb_list):
            source_det = det_t(
                alpha=spin_det_t(N_orbs, alpha_orb_list), beta=spin_det_t(N_orbs, beta_orb_list)
            )
            ref_source = det_ref(alpha_orb_list, beta_orb_list)
            connected_dets = ref_source.gen_all_connected_det(N_orbs)
            ref_set = set([(d.alpha, d.beta) for d in connected_dets])

            test_dets = source_det.generate_connected_dets()
            test_set = set((d[0].as_orb_list, d[1].as_orb_list) for d in test_dets)

            self.assertEqual(
                ref_set, test_set, msg=f"Failed for {alpha_orb_list} X {beta_orb_list}"
            )

        for N_occupied in [4, 6, 7, 17, 24]:
            alpha_orb_list = rng.integers(0, N_orbs, N_occupied)
            alpha_orb_list = tuple(np.unique(alpha_orb_list))
            beta_orb_list = rng.integers(0, N_orbs, N_occupied)
            beta_orb_list = tuple(np.unique(beta_orb_list))

            check_generate_dets(alpha_orb_list, beta_orb_list)

    def test_generate_constrained_dets(self):
        N_orbs = 16

        def check_common_constraint(alpha_orb_list, beta_orb_list, constraint):
            source_det = det_t(
                alpha=spin_det_t(N_orbs, alpha_orb_list), beta=spin_det_t(N_orbs, beta_orb_list)
            )
            ref_source = det_ref(alpha_orb_list, beta_orb_list)
            connected_dets = ref_source.gen_all_connected_det(N_orbs)

            def check_exc(x, y):
                exc_degree = x.exc_degree(y)
                holes, parts = constraint
                match exc_degree:
                    case (1, 0) | (0, 1):
                        sdets = (x.alpha, y.alpha) if exc_degree[0] else (x.beta, y.beta)
                        h1, p1 = det_ref.single_exc_no_phase(*sdets)
                        h2, p2 = h1, p1
                    case (2, 0) | (0, 2):
                        sdets = (x.alpha, y.alpha) if exc_degree[0] else (x.beta, y.beta)
                        h1, h2, p1, p2 = det_ref.double_exc_no_phase(*sdets)
                    case (1, 1):
                        h1, p1 = det_ref.single_exc_no_phase(x.alpha, y.alpha)
                        h2, p2 = det_ref.single_exc_no_phase(x.beta, y.beta)
                    case _:
                        raise ValueError
                return h1 in holes and h2 in holes and p1 in parts and p2 in parts

            ref_set = set([(d.alpha, d.beta) for d in connected_dets if check_exc(ref_source, d)])

            test_dets = source_det.generate_connected_dets(constraint)
            test_set = set((d[0].as_orb_list, d[1].as_orb_list) for d in test_dets)

            self.assertEqual(
                ref_set, test_set, msg=f"Failed for {alpha_orb_list} X {beta_orb_list}"
            )

        def check_different_constraint(alpha_orb_list, beta_orb_list, constraint):
            source_det = det_t(
                alpha=spin_det_t(N_orbs, alpha_orb_list), beta=spin_det_t(N_orbs, beta_orb_list)
            )
            ref_source = det_ref(alpha_orb_list, beta_orb_list)
            connected_dets = ref_source.gen_all_connected_det(N_orbs)

            def check_exc(x, y):
                exc_degree = x.exc_degree(y)
                ah, bh, ap, bp = constraint
                holes, parts = (ah, ap) if exc_degree[0] else (bh, bp)
                sdets = (x.alpha, y.alpha) if exc_degree[0] else (x.beta, y.beta)
                match exc_degree:
                    case (1, 0) | (0, 1):
                        h, p = det_ref.single_exc_no_phase(*sdets)
                        return h in holes and p in parts
                    case (2, 0) | (0, 2):
                        sdets = (x.alpha, y.alpha) if exc_degree[0] else (x.beta, y.beta)
                        h1, h2, p1, p2 = det_ref.double_exc_no_phase(*sdets)
                        return h1 in holes and h2 in holes and p1 in parts and p2 in parts
                    case (1, 1):
                        h1, p1 = det_ref.single_exc_no_phase(x.alpha, y.alpha)
                        h2, p2 = det_ref.single_exc_no_phase(x.beta, y.beta)
                        return h1 in ah and h2 in bh and p1 in ap and p2 in bp
                    case _:
                        raise ValueError

            ref_set = set([(d.alpha, d.beta) for d in connected_dets if check_exc(ref_source, d)])

            test_dets = source_det.generate_connected_dets(constraint)
            test_set = set((d[0].as_orb_list, d[1].as_orb_list) for d in test_dets)

            self.assertEqual(
                ref_set, test_set, msg=f"Failed for {alpha_orb_list} X {beta_orb_list}"
            )

        for _ in range(self.N_trials):
            common_h, common_p = tuple(), tuple()
            while len(common_h) < 4 or len(common_p) < 4:
                alpha_orb_list = tuple(rng.choice(np.arange(N_orbs), 8, replace=False))
                beta_orb_list = tuple(rng.choice(np.arange(N_orbs), 8, replace=False))

                ah = set(alpha_orb_list)
                ap = set(tuple(range(N_orbs))).difference(ah)
                bh = set(beta_orb_list)
                bp = set(tuple(range(N_orbs))).difference(bh)
                common_h = tuple(ah.intersection(bh))
                common_p = tuple(ap.intersection(bp))

            common_constraint = (
                tuple(rng.choice(common_h, 4, replace=False)),
                tuple(rng.choice(common_p, 4, replace=False)),
            )
            check_common_constraint(alpha_orb_list, beta_orb_list, common_constraint)

            diff_constraint = tuple(
                tuple(rng.choice(tuple(x), 4, replace=False)) for x in (ah, bh, ap, bp)
            )
            check_different_constraint(alpha_orb_list, beta_orb_list, diff_constraint)

    def test_generate_array_connected(self):
        N_orbs = 32

        def check_generate_array(alpha_orb_sets, beta_orb_sets):
            ref_set = set()
            for alpha_orbs, beta_orbs in zip(alpha_orb_sets, beta_orb_sets):
                ref_source = det_ref(alpha_orbs, beta_orbs)
                connected_dets = ref_source.gen_all_connected_det(N_orbs)
                ref_set = ref_set.union(set([(d.alpha, d.beta) for d in connected_dets]))

            source_arr = DetArray(len(alpha_orb_sets), N_orbs)
            source_set = []
            for i, (alpha_orbs, beta_orbs) in enumerate(zip(alpha_orb_sets, beta_orb_sets)):
                source_arr[i] = det_t(
                    alpha=spin_det_t(N_orbs, alpha_orbs), beta=spin_det_t(N_orbs, beta_orbs)
                )
                source_set.append((alpha_orbs, beta_orbs))

            test_dets = source_arr.generate_connected_dets()
            source_set = set(source_set)
            test_set = set((d[0].as_orb_list, d[1].as_orb_list) for d in test_dets)
            self.assertEqual(ref_set, test_set)
            self.assertEqual(set(), test_set.intersection(source_set))

        for _ in range(1):
            for N_occupied in [4, 7, 17]:
                alpha_orb_sets = []
                beta_orb_sets = []
                for _ in range(8):
                    alpha_orb_sets.append(tuple(rng.choice(N_orbs, N_occupied, False)))
                    beta_orb_sets.append(tuple(rng.choice(N_orbs, N_occupied, False)))

                check_generate_array(alpha_orb_sets, beta_orb_sets)

    def test_generate_array_constrained(self):
        N_orbs = 16

        def check_common_constraint(alpha_orb_sets, beta_orb_sets, constraint):
            def check_exc(x, y):
                exc_degree = x.exc_degree(y)
                holes, parts = constraint
                match exc_degree:
                    case (1, 0) | (0, 1):
                        sdets = (x.alpha, y.alpha) if exc_degree[0] else (x.beta, y.beta)
                        h1, p1 = det_ref.single_exc_no_phase(*sdets)
                        h2, p2 = h1, p1
                    case (2, 0) | (0, 2):
                        sdets = (x.alpha, y.alpha) if exc_degree[0] else (x.beta, y.beta)
                        h1, h2, p1, p2 = det_ref.double_exc_no_phase(*sdets)
                    case (1, 1):
                        h1, p1 = det_ref.single_exc_no_phase(x.alpha, y.alpha)
                        h2, p2 = det_ref.single_exc_no_phase(x.beta, y.beta)
                    case _:
                        raise ValueError
                return h1 in holes and h2 in holes and p1 in parts and p2 in parts

            ref_set = set()
            for alpha_orbs, beta_orbs in zip(alpha_orb_sets, beta_orb_sets):
                ref_source = det_ref(alpha_orbs, beta_orbs)
                connected_dets = ref_source.gen_all_connected_det(N_orbs)
                ref_set = ref_set.union(
                    set([(d.alpha, d.beta) for d in connected_dets if check_exc(ref_source, d)])
                )

            source_arr = DetArray(len(alpha_orb_sets), N_orbs)
            source_set = []
            for i, (alpha_orbs, beta_orbs) in enumerate(zip(alpha_orb_sets, beta_orb_sets)):
                source_arr[i] = det_t(
                    alpha=spin_det_t(N_orbs, alpha_orbs), beta=spin_det_t(N_orbs, beta_orbs)
                )
                source_set.append((alpha_orbs, beta_orbs))

            test_dets = source_arr.generate_connected_dets(constraint)
            source_set = set(source_set)
            test_set = set((d[0].as_orb_list, d[1].as_orb_list) for d in test_dets)

            self.assertEqual(ref_set, test_set)
            self.assertEqual(set(), test_set.intersection(source_set))

        def check_different_constraint(alpha_orb_sets, beta_orb_sets, constraint):
            def check_exc(x, y):
                exc_degree = x.exc_degree(y)
                ah, bh, ap, bp = constraint
                holes, parts = (ah, ap) if exc_degree[0] else (bh, bp)
                sdets = (x.alpha, y.alpha) if exc_degree[0] else (x.beta, y.beta)
                match exc_degree:
                    case (1, 0) | (0, 1):
                        h, p = det_ref.single_exc_no_phase(*sdets)
                        return h in holes and p in parts
                    case (2, 0) | (0, 2):
                        sdets = (x.alpha, y.alpha) if exc_degree[0] else (x.beta, y.beta)
                        h1, h2, p1, p2 = det_ref.double_exc_no_phase(*sdets)
                        return h1 in holes and h2 in holes and p1 in parts and p2 in parts
                    case (1, 1):
                        h1, p1 = det_ref.single_exc_no_phase(x.alpha, y.alpha)
                        h2, p2 = det_ref.single_exc_no_phase(x.beta, y.beta)
                        return h1 in ah and h2 in bh and p1 in ap and p2 in bp
                    case _:
                        raise ValueError

            ref_set = set()
            for alpha_orbs, beta_orbs in zip(alpha_orb_sets, beta_orb_sets):
                ref_source = det_ref(alpha_orbs, beta_orbs)
                connected_dets = ref_source.gen_all_connected_det(N_orbs)
                ref_set = ref_set.union(
                    set([(d.alpha, d.beta) for d in connected_dets if check_exc(ref_source, d)])
                )

            source_arr = DetArray(len(alpha_orb_sets), N_orbs)
            source_set = []
            for i, (alpha_orbs, beta_orbs) in enumerate(zip(alpha_orb_sets, beta_orb_sets)):
                source_arr[i] = det_t(
                    alpha=spin_det_t(N_orbs, alpha_orbs), beta=spin_det_t(N_orbs, beta_orbs)
                )
                source_set.append((alpha_orbs, beta_orbs))

            test_dets = source_arr.generate_connected_dets(constraint)
            source_set = set(source_set)
            test_set = set((d[0].as_orb_list, d[1].as_orb_list) for d in test_dets)

            self.assertEqual(ref_set, test_set)
            self.assertEqual(set(), test_set.intersection(source_set))

        for _ in range(8):
            common_constraint = (
                tuple(rng.choice(N_orbs, 4, replace=False)),
                tuple(rng.choice(N_orbs, 4, replace=False)),
            )
            diff_constraint = (
                tuple(rng.choice(N_orbs, 4, replace=False)),
                tuple(rng.choice(N_orbs, 4, replace=False)),
                tuple(rng.choice(N_orbs, 4, replace=False)),
                tuple(rng.choice(N_orbs, 4, replace=False)),
            )
            for N_occupied in [4, 7, 9]:
                alpha_orb_sets = []
                beta_orb_sets = []
                for _ in range(8):
                    alpha_orb_sets.append(tuple(rng.choice(N_orbs, N_occupied, False)))
                    beta_orb_sets.append(tuple(rng.choice(N_orbs, N_occupied, False)))

                check_common_constraint(alpha_orb_sets, beta_orb_sets, common_constraint)
                check_different_constraint(alpha_orb_sets, beta_orb_sets, diff_constraint)

    def test_connected_recursion(self):
        N_orbs = 4
        ground_state = det_t(
            N_orbs, spin_det_t(N_orbs, occ=True, max_orb=2), spin_det_t(N_orbs, occ=True, max_orb=2)
        )
        c1 = ground_state.generate_connected_dets()
        c1_set = set((d[0].as_orb_list, d[1].as_orb_list) for d in c1)
        c2 = c1.generate_connected_dets()
        c2_set = set((d[0].as_orb_list, d[1].as_orb_list) for d in c2)
        self.assertEqual(c1_set.intersection(c2_set), set())

    def test_add_filtered_list(self):
        N_orbs = 12
        max_orb = 4
        ground_state = det_t(
            N_orbs,
            spin_det_t(N_orbs, occ=True, max_orb=max_orb),
            spin_det_t(N_orbs, occ=True, max_orb=max_orb),
        )
        c1 = ground_state.generate_connected_dets()

        c2 = c1.generate_connected_dets()
        threshold = 0.95
        pt2_arr = rng.uniform(size=(c2.N_dets)).astype(np.float32)
        idx_arr = np.where(pt2_arr > threshold)[0]

        pt2 = LinkedArray_f32(c2.N_dets, pt2_arr)
        idx = get_indices_by_threshold(pt2, threshold)
        # idx = LinkedArray_idx_t(100, idx_arr)

        ref_set = set((d[0].as_orb_list, d[1].as_orb_list) for d in c1)
        c1.extend_with_filter(c2, idx)
        add_set = set((c2[i][0].as_orb_list, c2[i][1].as_orb_list) for i in idx_arr)
        ref_set = ref_set.union(add_set)
        test_set = set((d[0].as_orb_list, d[1].as_orb_list) for d in c1)

        self.assertEqual(ref_set, test_set)


if __name__ == "__main__":
    unittest.main()
