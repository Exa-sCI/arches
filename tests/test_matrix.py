import unittest

import numpy as np

from arches.matrix import DMatrix

seed = 21872
rng = np.random.default_rng(seed=seed)


class Test_DMatrix(unittest.TestCase):
    def setUp(self):
        self.N_trials = 50
        self.rng = rng

    def test_square_matmul(self):
        def check_square_matmul(rank):
            A_ref = self.rng.normal(size=(rank, rank))
            B_ref = self.rng.normal(size=(rank, rank))

            A_test = DMatrix(rank, rank, A_ref)
            B_test = DMatrix(rank, rank, B_ref)

            self.assertTrue(np.allclose(A_ref @ B_ref, (A_test @ B_test).np_arr))

        rank = 32
        for _ in range(self.N_trials):
            check_square_matmul(rank)

    def test_matmul_shape_error(self):
        def check_mamtul_shape_error():
            A = DMatrix(4, 4)
            B = DMatrix(3, 4)
            A @ B

        self.assertRaises(ValueError, check_mamtul_shape_error)

    def test_rect_matmul(self):
        def check_rect_matmul(shape_A, shape_B):
            A_ref = self.rng.normal(size=shape_A)
            B_ref = self.rng.normal(size=shape_B)

            A_test = DMatrix(shape_A[0], shape_A[1], A_ref)
            B_test = DMatrix(shape_B[0], shape_B[1], B_ref)

            self.assertTrue(np.allclose(A_ref @ B_ref, (A_test @ B_test).np_arr))

        for _ in range(self.N_trials):
            ranks = self.rng.integers(8, 32, 3)
            shape_A = (ranks[0], ranks[1])
            shape_B = (ranks[1], ranks[2])
            check_rect_matmul(shape_A, shape_B)

    def test_transpose_matmul(self):
        def check_transpose_matmul_lt(shape_A, shape_B):
            A_ref = self.rng.normal(size=shape_A)
            B_ref = self.rng.normal(size=shape_B)

            A_test = DMatrix(shape_A[0], shape_A[1], A_ref)
            B_test = DMatrix(shape_B[0], shape_B[1], B_ref)

            ref_res = A_ref.T @ B_ref
            test_res = A_test.T @ B_test
            self.assertEqual(ref_res.shape, (test_res.m, test_res.n))
            self.assertTrue(np.allclose(ref_res, test_res.np_arr))

        def check_transpose_matmul_rt(shape_A, shape_B):
            A_ref = self.rng.normal(size=shape_A)
            B_ref = self.rng.normal(size=shape_B)

            A_test = DMatrix(shape_A[0], shape_A[1], A_ref)
            B_test = DMatrix(shape_B[0], shape_B[1], B_ref)

            self.assertTrue(np.allclose(A_ref @ B_ref.T, (A_test @ B_test.T).np_arr))

        def check_transpose_matmul_bt(shape_A, shape_B):
            A_ref = self.rng.normal(size=shape_A)
            B_ref = self.rng.normal(size=shape_B)

            A_test = DMatrix(shape_A[0], shape_A[1], A_ref)
            B_test = DMatrix(shape_B[0], shape_B[1], B_ref)

            self.assertTrue(np.allclose(B_ref.T @ A_ref.T, (B_test.T @ A_test.T).np_arr))
            self.assertTrue(np.allclose((B_test.T @ A_test.T).np_arr, (A_test @ B_test).np_arr.T))

        for _ in range(self.N_trials):
            ranks_lt = self.rng.integers(8, 32, 3)
            shape_A_lt = (ranks_lt[1], ranks_lt[0])
            shape_B_lt = (ranks_lt[1], ranks_lt[2])
            check_transpose_matmul_lt(shape_A_lt, shape_B_lt)

            ranks_rt = self.rng.integers(8, 32, 3)
            shape_A_rt = (ranks_rt[0], ranks_rt[1])
            shape_B_rt = (ranks_rt[2], ranks_rt[1])
            check_transpose_matmul_rt(shape_A_rt, shape_B_rt)

            ranks_bt = self.rng.integers(8, 32, 3)
            shape_A_bt = (ranks_bt[0], ranks_bt[1])
            shape_B_bt = (ranks_bt[1], ranks_bt[2])
            check_transpose_matmul_bt(shape_A_bt, shape_B_bt)

    def test_elementwise_shape_error(self):
        A = DMatrix(10, 7)
        B = DMatrix(10, 8)
        C = DMatrix(9, 7)

        def _iadd_AB(A=A):
            A += B

        def _iadd_AC(A=A):
            A += C

        def _isub_AB(A=A):
            A -= B

        def _isub_AC(A=A):
            A -= C

        self.assertRaises(ValueError, lambda: A + B)
        self.assertRaises(ValueError, lambda: A + C)
        self.assertRaises(ValueError, lambda: A - B)
        self.assertRaises(ValueError, lambda: A - C)
        self.assertRaises(ValueError, _iadd_AB)
        self.assertRaises(ValueError, _iadd_AC)
        self.assertRaises(ValueError, _isub_AB)
        self.assertRaises(ValueError, _isub_AC)

    def test_add(self):
        def check_add(shape):
            A_ref = rng.normal(size=shape)
            B_ref = rng.normal(size=shape)

            A_test = DMatrix(shape[0], shape[1], A_ref)
            B_test = DMatrix(shape[0], shape[1], B_ref)

            self.assertTrue(np.allclose(A_ref + B_ref, (A_test + B_test).np_arr))

        for _ in range(self.N_trials):
            shape = self.rng.integers(8, 32, 2)
            check_add(shape)

    def test_iadd(self):
        def check_iadd(shape):
            A_ref = rng.normal(size=shape)
            B_ref = rng.normal(size=shape)

            A_test = DMatrix(shape[0], shape[1], A_ref)
            B_test = DMatrix(shape[0], shape[1], B_ref)

            A_ref += B_ref
            A_test += B_test

            self.assertTrue(np.allclose(A_ref, A_test.np_arr))

        for _ in range(self.N_trials):
            shape = self.rng.integers(8, 32, 2)
            check_iadd(shape)

    def test_sub(self):
        def check_sub(shape):
            A_ref = rng.normal(size=shape)
            B_ref = rng.normal(size=shape)

            A_test = DMatrix(shape[0], shape[1], A_ref)
            B_test = DMatrix(shape[0], shape[1], B_ref)

            self.assertTrue(np.allclose(A_ref - B_ref, (A_test - B_test).np_arr))

        for _ in range(self.N_trials):
            shape = self.rng.integers(8, 32, 2)
            check_sub(shape)

    def test_isub(self):
        def check_isub(shape):
            A_ref = rng.normal(size=shape)
            B_ref = rng.normal(size=shape)

            A_test = DMatrix(shape[0], shape[1], A_ref)
            B_test = DMatrix(shape[0], shape[1], B_ref)

            A_ref -= B_ref
            A_test -= B_test

            self.assertTrue(np.allclose(A_ref, A_test.np_arr))

        for _ in range(self.N_trials):
            shape = self.rng.integers(8, 32, 2)
            check_isub(shape)

    def test_subslice_matmul(self):
        rank = 32
        A_ref = rng.normal(size=(rank, rank))
        B_ref = rng.normal(size=(rank, rank))
        A_test = DMatrix(rank, rank, A_ref)
        B_test = DMatrix(rank, rank, B_ref)

        def check_subslice_matmul(slice_A, slice_B):
            self.assertTrue(
                np.allclose(
                    A_ref[slice_A] @ B_ref[slice_B], (A_test[slice_A] @ B_test[slice_B]).np_arr
                )
            )

        for _ in range(self.N_trials):
            ranks = self.rng.integers(8, 16, 3)
            row_offset_A = self.rng.integers(0, 16)
            col_offset_A = self.rng.integers(0, 16)
            row_offset_B = self.rng.integers(0, 16)
            col_offset_B = self.rng.integers(0, 16)
            slice_A = (
                slice(row_offset_A, row_offset_A + ranks[0]),
                slice(col_offset_A, col_offset_A + ranks[1]),
            )

            slice_B = (
                slice(row_offset_B, row_offset_B + ranks[1]),
                slice(col_offset_B, col_offset_B + ranks[2]),
            )

            check_subslice_matmul(slice_A, slice_B)

    def test_subslice_transpose_matmul(self):
        pass


if __name__ == "__main__":
    unittest.main()
