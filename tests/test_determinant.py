import unittest

import numpy as np

from arches.determinant import spin_det_t

rng = np.random.default_rng(seed=6329)


class Test_SpinDet(unittest.TestCase):
    def setUp(self):
        self.rng = rng
        self.N_trials = 1024
        self.N_orbs = 72  # larger than both ui32 and ui64 blocks

    def test_constuctor(self):
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


if __name__ == "__main__":
    unittest.main()
