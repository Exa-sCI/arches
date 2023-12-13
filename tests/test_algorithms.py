import unittest

import numpy as np
from test_types import Test_f32, Test_f64

from arches.algorithms import bmgs_h, davidson, diagonalize
from arches.matrix import DMatrix

seed = 9180237
rng = np.random.default_rng(seed=seed)


class Test_BMGS(unittest.TestCase):
    __test__ = False

    def setUp(self):
        self.N_trials = 32
        self.rng = rng
        self.m = 128
        self.n = 64

    def check_orthogonality(self, A, m):
        for i in range(m - 1):
            for j in range(i + 1, m):
                col_i = A[:, i]
                col_j = A[:, j]
                proj = col_i.T @ col_j
                if proj > self.atol:
                    n_proj = proj / (np.linalg.norm(col_i) * np.linalg.norm(col_j))
                    print(f"Vector {i} not orthogonal to vector {j}: {n_proj} ")
                    return False

        for i in range(m):
            col_i = A[:, i]
            proj = col_i.T @ col_i
            if not np.isclose(proj, 1.0):
                return False

        return True

    def test_bmgs_h_from_scratch(self):
        for _ in range(self.N_trials):
            temp = rng.normal(0, 1, size=(self.m, self.n)).astype(self.dtype)
            X = DMatrix(self.m, self.n, temp, dtype=self.dtype)
            for block_size in [1, 2, 4, 8, 16]:
                Q, R, T = bmgs_h(X, block_size)
                self.assertTrue(
                    np.allclose((Q @ R).np_arr, X.np_arr, atol=self.atol, rtol=self.rtol)
                )
                self.assertTrue(
                    np.allclose(
                        T.np_arr,
                        np.linalg.inv(np.triu((Q.T @ Q).np_arr)),
                        atol=self.atol,
                        rtol=self.rtol,
                    )
                )
                self.assertTrue(self.check_orthogonality(Q.np_arr, self.n))

    def test_bmgs_h_with_restart(self):
        for _ in range(self.N_trials):
            temp_0 = rng.normal(0, 1, size=(self.m, self.n)).astype(self.dtype)
            X0 = DMatrix(self.m, self.n, temp_0, dtype=self.dtype)
            for block_size in [1, 2, 4, 8, 16]:
                Q0, R0, T0 = bmgs_h(X0, block_size)

                temp_1 = rng.normal(0, 1, size=(self.m, self.n)).astype(self.dtype)
                ref_res = np.hstack([temp_0, temp_1]).astype(self.dtype)
                temp_1 = np.hstack([Q0.np_arr, temp_1]).astype(self.dtype)

                X1 = DMatrix(self.m, self.n * 2, temp_1, dtype=self.dtype)
                Q, R, T = bmgs_h(X1, block_size, Q0, R0, T0)
                self.assertTrue(
                    np.allclose(
                        (Q @ R).np_arr,
                        ref_res,
                        atol=self.atol,
                        rtol=self.rtol,
                    )
                )
                self.assertTrue(self.check_orthogonality(Q.np_arr, self.n))
                self.assertTrue(
                    np.allclose(
                        T.np_arr,
                        np.linalg.inv(np.triu((Q.T @ Q).np_arr)),
                        atol=self.atol,
                        rtol=self.rtol,
                    )
                )


class Test_BMGS_f32(Test_f32, Test_BMGS):
    __test__ = True


class Test_BMGS_f64(Test_f64, Test_BMGS):
    __test__ = True


class Test_Davidson(unittest.TestCase):
    __test__ = False

    def setUp(self):
        self.N_trials = 8
        self.rng = rng
        self.m = 1024
        self.block_size = 32

    def check_orthogonality(self, A, m):
        for i in range(m - 1):
            for j in range(i + 1, m):
                col_i = A[:, i]
                col_j = A[:, j]
                proj = col_i.T @ col_j
                if proj > self.atol:
                    n_proj = proj / (np.linalg.norm(col_i) * np.linalg.norm(col_j))
                    print(f"Vector {i} not orthogonal to vector {j}: {n_proj} ")
                    return False

        return True

    def get_random_spd_matrix(self):
        Q = self.rng.normal(size=(self.m, self.m)).astype(self.dtype)
        evals = np.sort(self.rng.uniform(0.5, 10, size=(self.m)))
        # L = np.diag(np.arange(self.m) + 1)
        L = np.diag(evals)

        QD = DMatrix(self.m, self.m, Q, dtype=self.dtype)
        Q, _, _ = bmgs_h(QD, self.block_size)
        Q_np = Q.np_arr

        res = Q_np.T @ L @ Q_np
        return res.astype(self.dtype)

    def test_diagonalize(self):
        for _ in range(self.N_trials):
            H = self.get_random_spd_matrix()
            evals, _ = np.linalg.eigh(H)
            self.assertTrue(np.allclose(H, H.T, atol=self.atol, rtol=self.rtol))
            self.assertTrue(np.all(evals > 0))

            Q = DMatrix(*H.shape, H, dtype=self.dtype)
            test_evals = diagonalize(Q)
            Qt = Q.np_arr

            self.assertTrue(
                np.allclose(evals, test_evals.arr.np_arr, atol=self.atol, rtol=self.rtol)
            )
            self.assertTrue(
                np.allclose(
                    Qt @ np.diag(test_evals.arr.np_arr) @ Qt.T, H, atol=self.atol, rtol=self.rtol
                )
            )

    def test_diag_preconditioner(self):
        for n in range(self.N_trials):
            H = self.get_random_spd_matrix()
            evals, _ = np.linalg.eigh(H)
            self.assertTrue(np.allclose(H, H.T, atol=self.atol, rtol=self.rtol))
            self.assertTrue(np.all(evals > 0))

            H = DMatrix(self.m, self.m, H, dtype=self.dtype)
            L, Q = davidson(H, l=8, max_subspace_rank=128)

            self.assertTrue(self.check_orthogonality(Q.np_arr, Q.n))
            self.assertTrue(
                np.isclose(L.arr[0], np.min(evals)), msg=f"{L.arr[0]} : {np.min(evals)}"
            )

    def test_tridiag_preconditioner(self):
        pass

    def test_multi_state(self):
        pass


class Test_Davidson_f32(Test_f32, Test_Davidson):
    __test__ = True


class Test_Davidson_f64(Test_f64, Test_Davidson):
    __test__ = True
