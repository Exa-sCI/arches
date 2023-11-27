import unittest

import numpy as np

from arches.determinant import spin_det_t


class Test_SpinDet(unittest.TestCase):
    def test_constuctor(self):
        N_orbs = 8
        N_filled = 4
        orb_list = (0, 1, 2, 4, 6)

        a = spin_det_t(N_orbs)
        b = spin_det_t(N_orbs, occ=True, max_orb=N_filled)
        c = spin_det_t(N_orbs, occ=orb_list)
