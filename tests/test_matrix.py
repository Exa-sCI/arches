import unittest

import numpy as np
from test_types import Test_f32, Test_f64

from arches.matrix import DMatrix

seed = 21872
rng = np.random.default_rng(seed=seed)


class Test_DMatrix(unittest.TestCase):
    __test__ = False

    def setUp(self):
        self.N_trials = 50
        self.rng = rng

    def test_square_matmul(self):
        def check_square_matmul(rank):
            A_ref = self.rng.normal(size=(rank, rank)).astype(self.dtype)
            B_ref = self.rng.normal(size=(rank, rank)).astype(self.dtype)

            A_test = DMatrix(rank, rank, A_ref, dtype=self.dtype)
            B_test = DMatrix(rank, rank, B_ref, dtype=self.dtype)

            self.assertTrue(np.allclose(A_ref @ B_ref, (A_test @ B_test).np_arr))

        rank = 32
        for _ in range(self.N_trials):
            check_square_matmul(rank)

    def test_matmul_shape_error(self):
        def check_mamtul_shape_error():
            A = DMatrix(4, 4, dtype=self.dtype)
            B = DMatrix(3, 4, dtype=self.dtype)
            A @ B

        self.assertRaises(ValueError, check_mamtul_shape_error)

    def test_rect_matmul(self):
        def check_rect_matmul(shape_A, shape_B):
            A_ref = self.rng.normal(size=shape_A).astype(self.dtype)
            B_ref = self.rng.normal(size=shape_B).astype(self.dtype)

            A_test = DMatrix(shape_A[0], shape_A[1], A_ref, dtype=self.dtype)
            B_test = DMatrix(shape_B[0], shape_B[1], B_ref, dtype=self.dtype)

            self.assertTrue(np.allclose(A_ref @ B_ref, (A_test @ B_test).np_arr))

        for _ in range(self.N_trials):
            ranks = self.rng.integers(8, 32, 3)
            shape_A = (ranks[0], ranks[1])
            shape_B = (ranks[1], ranks[2])
            check_rect_matmul(shape_A, shape_B)

    def test_transpose_matmul(self):
        def check_transpose_matmul_lt(shape_A, shape_B):
            A_ref = self.rng.normal(size=shape_A).astype(self.dtype)
            B_ref = self.rng.normal(size=shape_B).astype(self.dtype)

            A_test = DMatrix(shape_A[0], shape_A[1], A_ref)
            B_test = DMatrix(shape_B[0], shape_B[1], B_ref)

            self.assertTrue(
                np.allclose(
                    A_ref.T @ B_ref, (A_test.T @ B_test).np_arr, rtol=self.rtol, atol=self.atol
                )
            )

        def check_transpose_matmul_rt(shape_A, shape_B):
            A_ref = self.rng.normal(size=shape_A).astype(self.dtype)
            B_ref = self.rng.normal(size=shape_B).astype(self.dtype)

            A_test = DMatrix(shape_A[0], shape_A[1], A_ref)
            B_test = DMatrix(shape_B[0], shape_B[1], B_ref)

            self.assertTrue(
                np.allclose(
                    A_ref @ B_ref.T, (A_test @ B_test.T).np_arr, rtol=self.rtol, atol=self.atol
                )
            )

        def check_transpose_matmul_bt(shape_A, shape_B):
            A_ref = self.rng.normal(size=shape_A).astype(self.dtype)
            B_ref = self.rng.normal(size=shape_B).astype(self.dtype)

            A_test = DMatrix(shape_A[0], shape_A[1], A_ref, dtype=self.dtype)
            B_test = DMatrix(shape_B[0], shape_B[1], B_ref, dtype=self.dtype)

            self.assertTrue(
                np.allclose(
                    B_ref.T @ A_ref.T, (B_test.T @ A_test.T).np_arr, rtol=self.rtol, atol=self.atol
                )
            )
            self.assertTrue(
                np.allclose(
                    (B_test.T @ A_test.T).np_arr,
                    (A_test @ B_test).np_arr.T,
                    rtol=self.rtol,
                    atol=self.atol,
                )
            )

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
        A = DMatrix(10, 7, dtype=self.dtype)
        B = DMatrix(10, 8, dtype=self.dtype)
        C = DMatrix(9, 7, dtype=self.dtype)

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
            A_ref = rng.normal(size=shape).astype(self.dtype)
            B_ref = rng.normal(size=shape).astype(self.dtype)

            A_test = DMatrix(shape[0], shape[1], A_ref, dtype=self.dtype)
            B_test = DMatrix(shape[0], shape[1], B_ref, dtype=self.dtype)

            self.assertTrue(np.allclose(A_ref + B_ref, (A_test + B_test).np_arr))

        for _ in range(self.N_trials):
            shape = self.rng.integers(8, 32, 2)
            check_add(shape)

    def test_iadd(self):
        def check_iadd(shape):
            A_ref = rng.normal(size=shape).astype(self.dtype)
            B_ref = rng.normal(size=shape).astype(self.dtype)

            A_test = DMatrix(shape[0], shape[1], A_ref, dtype=self.dtype)
            B_test = DMatrix(shape[0], shape[1], B_ref, dtype=self.dtype)

            A_ref += B_ref
            A_test += B_test

            self.assertTrue(np.allclose(A_ref, A_test.np_arr))

        for _ in range(self.N_trials):
            shape = self.rng.integers(8, 32, 2)
            check_iadd(shape)

    def test_sub(self):
        def check_sub(shape):
            A_ref = rng.normal(size=shape).astype(self.dtype)
            B_ref = rng.normal(size=shape).astype(self.dtype)

            A_test = DMatrix(shape[0], shape[1], A_ref)
            B_test = DMatrix(shape[0], shape[1], B_ref)

            self.assertTrue(np.allclose(A_ref - B_ref, (A_test - B_test).np_arr))

        for _ in range(self.N_trials):
            shape = self.rng.integers(8, 32, 2)
            check_sub(shape)

    def test_isub(self):
        def check_isub(shape):
            A_ref = rng.normal(size=shape).astype(self.dtype)
            B_ref = rng.normal(size=shape).astype(self.dtype)

            A_test = DMatrix(shape[0], shape[1], A_ref, dtype=self.dtype)
            B_test = DMatrix(shape[0], shape[1], B_ref, dtype=self.dtype)

            A_ref -= B_ref
            A_test -= B_test

            self.assertTrue(np.allclose(A_ref, A_test.np_arr))

        for _ in range(self.N_trials):
            shape = self.rng.integers(8, 32, 2)
            check_isub(shape)

    def test_mul(self):
        def check_mul(shape):
            A_ref = rng.normal(size=shape).astype(self.dtype)
            B_ref = rng.normal(size=shape).astype(self.dtype)

            A_test = DMatrix(shape[0], shape[1], A_ref, dtype=self.dtype)
            B_test = DMatrix(shape[0], shape[1], B_ref, dtype=self.dtype)

            self.assertTrue(np.allclose(A_ref * B_ref, (A_test * B_test).np_arr))

        for _ in range(self.N_trials):
            shape = self.rng.integers(8, 32, 2)
            check_mul(shape)

    def test_imul(self):
        def check_imul(shape):
            A_ref = rng.normal(size=shape).astype(self.dtype)
            B_ref = rng.normal(size=shape).astype(self.dtype)

            A_test = DMatrix(shape[0], shape[1], A_ref, dtype=self.dtype)
            B_test = DMatrix(shape[0], shape[1], B_ref, dtype=self.dtype)

            A_ref *= B_ref
            A_test *= B_test

            self.assertTrue(np.allclose(A_ref, A_test.np_arr))

        for _ in range(self.N_trials):
            shape = self.rng.integers(8, 32, 2)
            check_imul(shape)

    def test_truediv(self):
        def check_truediv(shape):
            A_ref = rng.normal(size=shape).astype(self.dtype)
            B_ref = rng.normal(size=shape).astype(self.dtype)

            A_test = DMatrix(shape[0], shape[1], A_ref)
            B_test = DMatrix(shape[0], shape[1], B_ref)

            self.assertTrue(np.allclose(A_ref / B_ref, (A_test / B_test).np_arr))

        for _ in range(self.N_trials):
            shape = self.rng.integers(8, 32, 2)
            check_truediv(shape)

    def test_itruediv(self):
        def check_itruediv(shape):
            A_ref = rng.normal(size=shape).astype(self.dtype)
            B_ref = rng.normal(size=shape).astype(self.dtype)

            A_test = DMatrix(shape[0], shape[1], A_ref, dtype=self.dtype)
            B_test = DMatrix(shape[0], shape[1], B_ref, dtype=self.dtype)

            A_ref /= B_ref
            A_test /= B_test

            self.assertTrue(np.allclose(A_ref, A_test.np_arr))

        for _ in range(self.N_trials):
            shape = self.rng.integers(8, 32, 2)
            check_itruediv(shape)

    def test_subslice_get_shape_error(self):
        row_rank = 64
        col_rank = 32
        A = DMatrix(row_rank, col_rank, dtype=self.dtype)
        B = DMatrix(row_rank, 1, dtype=self.dtype)
        C = DMatrix(1, col_rank, dtype=self.dtype)

        # invalid ranges
        self.assertRaises(ValueError, lambda: A[slice(-1, 7), slice(0, 8)])
        self.assertRaises(ValueError, lambda: A[slice(0, 8), slice(-1, 7)])
        self.assertRaises(ValueError, lambda: A[slice(0, 65), slice(0, 8)])
        self.assertRaises(ValueError, lambda: A[slice(0, 8), slice(0, 33)])
        self.assertRaises(ValueError, lambda: B[slice(-1, 7)])
        self.assertRaises(ValueError, lambda: B[slice(0, 65)])
        self.assertRaises(ValueError, lambda: C[slice(-1, 7)])
        self.assertRaises(ValueError, lambda: C[slice(0, 33)])

        # bad number of dims
        self.assertRaises(ValueError, lambda: A[:, :, :])
        self.assertRaises(ValueError, lambda: A[:])

        # no striding
        self.assertRaises(NotImplementedError, lambda: A[0:4:2, :])
        self.assertRaises(NotImplementedError, lambda: A[:, 0:4:2])
        self.assertRaises(NotImplementedError, lambda: B[0:4:2])
        self.assertRaises(NotImplementedError, lambda: C[0:4:2])

    def test_subslice_matmul(self):
        row_rank = 64
        col_rank = 32
        A_ref = rng.normal(size=(row_rank, col_rank)).astype(self.dtype)
        B_ref = rng.normal(size=(row_rank, col_rank)).astype(self.dtype)
        A_test = DMatrix(row_rank, col_rank, A_ref, dtype=self.dtype)
        B_test = DMatrix(row_rank, col_rank, B_ref, dtype=self.dtype)

        def check_subslice_matmul(slice_A, slice_B):
            self.assertTrue(
                np.allclose(
                    A_ref[slice_A] @ B_ref[slice_B], (A_test[slice_A] @ B_test[slice_B]).np_arr
                )
            )

        def check_partial_slices(ranks):
            self.assertTrue(
                np.allclose(
                    A_ref[:, :] @ B_ref[:col_rank, :], (A_test[:, :] @ B_test[:col_rank, :]).np_arr
                )
            )

            # TODO: the formatting of the notation below is awful..., configure ruff to make look like above?
            self.assertTrue(
                np.allclose(
                    A_ref[:, : ranks[1]] @ B_ref[: ranks[1], :],
                    (A_test[:, : ranks[1]] @ B_test[: ranks[1], :]).np_arr,
                )
            )

            self.assertTrue(
                np.allclose(
                    A_ref[ranks[0] :, : ranks[1]] @ B_ref[: ranks[1], ranks[2] :],
                    (A_test[ranks[0] :, : ranks[1]] @ B_test[: ranks[1], ranks[2] :]).np_arr,
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
            check_partial_slices(ranks)

    def test_subslice_transpose_matmul(self):
        row_rank = 64
        col_rank = 32
        A_ref = rng.normal(size=(row_rank, col_rank)).astype(self.dtype)
        B_ref = rng.normal(size=(row_rank, col_rank)).astype(self.dtype)
        A_test = DMatrix(row_rank, col_rank, A_ref, dtype=self.dtype)
        B_test = DMatrix(row_rank, col_rank, B_ref, dtype=self.dtype)

        def check_subslice_transpose_matmul_lt(slice_A, slice_B):
            #  check A[s].T @ B[s]
            self.assertTrue(
                np.allclose(
                    A_ref[slice_A].T @ B_ref[slice_B], (A_test[slice_A].T @ B_test[slice_B]).np_arr
                )
            )

        def check_subslice_transpose_matmul_rt(slice_A, slice_B):
            #  check A[s] @ B[s].T
            self.assertTrue(
                np.allclose(
                    A_ref[slice_A] @ B_ref[slice_B].T, (A_test[slice_A] @ B_test[slice_B].T).np_arr
                )
            )

        def check_subslice_transpose_matmul_bt(slice_A, slice_B):
            # check A[s].T @ B[s].T
            self.assertTrue(
                np.allclose(
                    A_ref[slice_A].T @ B_ref[slice_B].T,
                    (A_test[slice_A].T @ B_test[slice_B].T).np_arr,
                )
            )

        for _ in range(self.N_trials):
            ranks = self.rng.integers(8, 16, 3)
            row_offset_A, col_offset_A, row_offset_B, col_offset_B = self.rng.integers(0, 16, 4)

            slice_A_lt = (
                slice(row_offset_A, row_offset_A + ranks[1]),
                slice(col_offset_A, col_offset_A + ranks[0]),
            )
            slice_B_lt = (
                slice(row_offset_B, row_offset_B + ranks[1]),
                slice(col_offset_B, col_offset_B + ranks[2]),
            )
            slice_A_rt = (
                slice(row_offset_A, row_offset_A + ranks[0]),
                slice(col_offset_A, col_offset_A + ranks[1]),
            )
            slice_B_rt = (
                slice(row_offset_B, row_offset_B + ranks[2]),
                slice(col_offset_B, col_offset_B + ranks[1]),
            )
            slice_A_bt = (
                slice(row_offset_A, row_offset_A + ranks[1]),
                slice(col_offset_A, col_offset_A + ranks[0]),
            )
            slice_B_bt = (
                slice(row_offset_B, row_offset_B + ranks[2]),
                slice(col_offset_B, col_offset_B + ranks[1]),
            )

            check_subslice_transpose_matmul_lt(slice_A_lt, slice_B_lt)
            check_subslice_transpose_matmul_rt(slice_A_rt, slice_B_rt)
            check_subslice_transpose_matmul_bt(slice_A_bt, slice_B_bt)

    def test_transpose_subslice_matmul(self):
        row_rank = 64
        col_rank = 32
        A_ref = rng.normal(size=(row_rank, col_rank)).astype(self.dtype)
        B_ref = rng.normal(size=(row_rank, col_rank)).astype(self.dtype)
        A_test = DMatrix(row_rank, col_rank, A_ref, dtype=self.dtype)
        B_test = DMatrix(row_rank, col_rank, B_ref, dtype=self.dtype)

        def check_transpose_subslice_matmul_lt(slice_A, slice_B):
            #  check A.T[s] @ B[s]
            self.assertTrue(
                np.allclose(
                    A_ref.T[slice_A] @ B_ref[slice_B], (A_test.T[slice_A] @ B_test[slice_B]).np_arr
                )
            )

        def check_transpose_subslice_matmul_rt(slice_A, slice_B):
            #  check A[s] @ B.T[s]
            self.assertTrue(
                np.allclose(
                    A_ref[slice_A] @ B_ref.T[slice_B], (A_test[slice_A] @ B_test.T[slice_B]).np_arr
                )
            )

        def check_transpose_subslice_matmul_bt(slice_A, slice_B):
            # check A.T[s] @ B.T[s]
            self.assertTrue(
                np.allclose(
                    A_ref.T[slice_A] @ B_ref.T[slice_B],
                    (A_test.T[slice_A] @ B_test.T[slice_B]).np_arr,
                )
            )

        for _ in range(self.N_trials):
            ranks = self.rng.integers(8, 16, 3)
            row_offset_A, col_offset_A, row_offset_B, col_offset_B = self.rng.integers(0, 16, 4)

            slice_A = (
                slice(row_offset_A, row_offset_A + ranks[0]),
                slice(col_offset_A, col_offset_A + ranks[1]),
            )
            slice_B = (
                slice(row_offset_B, row_offset_B + ranks[1]),
                slice(col_offset_B, col_offset_B + ranks[2]),
            )
            check_transpose_subslice_matmul_lt(slice_A, slice_B)
            check_transpose_subslice_matmul_rt(slice_A, slice_B)
            check_transpose_subslice_matmul_bt(slice_A, slice_B)

    def test_mixed_transpose_subslice_matmul(self):
        row_rank = 64
        col_rank = 32
        A_ref = rng.normal(size=(row_rank, col_rank)).astype(self.dtype)
        B_ref = rng.normal(size=(row_rank, col_rank)).astype(self.dtype)
        A_test = DMatrix(row_rank, col_rank, A_ref, dtype=self.dtype)
        B_test = DMatrix(row_rank, col_rank, B_ref, dtype=self.dtype)

        def check_ts_by_st(slice_A, slice_B):
            # check A.T[s] @ B[s].T
            self.assertTrue(
                np.allclose(
                    A_ref.T[slice_A] @ B_ref[slice_B].T,
                    (A_test.T[slice_A] @ B_test[slice_B].T).np_arr,
                )
            )

        def check_st_by_ts(slice_A, slice_B):
            # check A[s].T @ B.T[s]
            self.assertTrue(
                np.allclose(
                    A_ref[slice_A].T @ B_ref.T[slice_B],
                    (A_test[slice_A].T @ B_test.T[slice_B]).np_arr,
                )
            )

        for _ in range(self.N_trials):
            ranks = self.rng.integers(8, 16, 3)
            row_offset_A, col_offset_A, row_offset_B, col_offset_B = self.rng.integers(0, 16, 4)

            slice_A_tsst = (
                slice(row_offset_A, row_offset_A + ranks[0]),
                slice(col_offset_A, col_offset_A + ranks[1]),
            )
            slice_B_tsst = (
                slice(row_offset_B, row_offset_B + ranks[2]),
                slice(col_offset_B, col_offset_B + ranks[1]),
            )
            slice_A_stts = (
                slice(row_offset_A, row_offset_A + ranks[1]),
                slice(col_offset_A, col_offset_A + ranks[0]),
            )
            slice_B_stts = (
                slice(row_offset_B, row_offset_B + ranks[1]),
                slice(col_offset_B, col_offset_B + ranks[2]),
            )

            check_ts_by_st(slice_A_tsst, slice_B_tsst)
            check_st_by_ts(slice_A_stts, slice_B_stts)

    def test_subslice_set_shape_error(self):
        row_rank = 64
        col_rank = 32
        A = DMatrix(row_rank, col_rank, dtype=self.dtype)
        B = DMatrix(row_rank, col_rank, dtype=self.dtype)
        C = DMatrix(row_rank, 1, dtype=self.dtype)
        D = DMatrix(1, col_rank, dtype=self.dtype)

        def check_subslice_set_ranges():
            N_errors = 0
            try:
                A[40:80, :] = None
            except ValueError:
                N_errors += 1

            try:
                A[:, 20:40] = None
            except ValueError:
                N_errors += 1

            try:
                A[-10:50, :] = None
            except ValueError:
                N_errors += 1

            try:
                A[:, -10:10] = None
            except ValueError:
                N_errors += 1

            try:
                C[-10:50] = None
            except ValueError:
                N_errors += 1

            try:
                C[40:80] = None
            except ValueError:
                N_errors += 1

            try:
                D[-10:50] = None
            except ValueError:
                N_errors += 1

            try:
                D[40:80] = None
            except ValueError:
                N_errors += 1

            return N_errors

        def check_subslice_set_shapes():
            N_errors = 0
            try:
                A[10:20, :] = B[0:5, :]
            except ValueError:
                N_errors += 1

            try:
                A[:, 10:20] = B[:, 0:5]
            except ValueError:
                N_errors += 1

            return N_errors

        def check_subslice_set_dims():
            N_errors = 0
            try:
                A[:] = None
            except ValueError:
                N_errors += 1

            try:
                A[:, :, :] = None
            except ValueError:
                N_errors += 1

            return N_errors

        def check_subslice_set_strides():
            N_errors = 0
            try:
                A[0:4:2, :] = None
            except NotImplementedError:
                N_errors += 1

            try:
                A[:, 0:4:2] = None
            except NotImplementedError:
                N_errors += 1

            try:
                C[0:4:2] = None
            except NotImplementedError:
                N_errors += 1

            try:
                D[0:4:2] = None
            except NotImplementedError:
                N_errors += 1

            return N_errors

        self.assertEqual(check_subslice_set_ranges(), 8)
        self.assertEqual(check_subslice_set_dims(), 2)
        self.assertEqual(check_subslice_set_strides(), 4)
        self.assertEqual(check_subslice_set_shapes(), 2)

    def test_subslice_set(self):
        srow_rank = 64
        scol_rank = 32
        drow_rank = 67
        dcol_rank = 37

        source_ref = np.zeros((srow_rank, scol_rank), dtype=self.dtype)
        source_ref.ravel()[:] = np.arange(0, srow_rank * scol_rank, dtype=self.dtype)
        source_test = DMatrix(srow_rank, scol_rank, source_ref, dtype=self.dtype)

        def check_nt(source_slice, dest_slice):
            dest = DMatrix(drow_rank, dcol_rank, dtype=self.dtype)
            dest[dest_slice] = source_test[source_slice]

            self.assertTrue(np.allclose(source_ref[source_slice], dest[dest_slice].np_arr))

        def check_lt(source_slice, dest_slice):
            dest = DMatrix(drow_rank, dcol_rank, dtype=self.dtype)
            dest.T[dest_slice] = source_test[source_slice]

            self.assertTrue(np.allclose(source_ref[source_slice], dest.T[dest_slice].np_arr))

        def check_rt(source_slice, dest_slice):
            dest = DMatrix(drow_rank, dcol_rank, dtype=self.dtype)
            dest[dest_slice] = source_test.T[source_slice]

            self.assertTrue(np.allclose(source_ref.T[source_slice], dest[dest_slice].np_arr))

        def check_bt(source_slice, dest_slice):
            dest = DMatrix(drow_rank, dcol_rank, dtype=self.dtype)
            dest.T[dest_slice] = source_test.T[source_slice]

            self.assertTrue(np.allclose(source_ref.T[source_slice], dest.T[dest_slice].np_arr))

        for _ in range(self.N_trials):
            ranks = self.rng.integers(8, 16, 2)
            drow_offset, dcol_offset, srow_offset, scol_offset = self.rng.integers(0, 16, 4)
            sslice = (
                slice(srow_offset, srow_offset + ranks[0]),
                slice(scol_offset, scol_offset + ranks[1]),
            )

            dslice = (
                slice(drow_offset, drow_offset + ranks[0]),
                slice(dcol_offset, dcol_offset + ranks[1]),
            )

            check_nt(sslice, dslice)
            check_lt(sslice, dslice)
            check_rt(sslice, dslice)
            check_bt(sslice, dslice)

    def test_neg(self):
        row_rank = 64
        col_rank = 32
        A_ref = rng.normal(size=(row_rank, col_rank)).astype(self.dtype)
        A_test = DMatrix(row_rank, col_rank, A_ref, dtype=self.dtype)
        B_test = -A_test
        C_test = A_test + B_test

        self.assertTrue(
            np.allclose(np.zeros((row_rank, col_rank), dtype=self.dtype), C_test.np_arr)
        )

    def test_column2norm(self):
        row_rank = 64
        col_rank = 32
        for _ in range(self.N_trials):
            A_ref = rng.normal(size=(row_rank, col_rank)).astype(self.dtype)
            A_test = DMatrix(row_rank, col_rank, A_ref, dtype=self.dtype)
            test_norms = A_test.column_2norm()
            test_norms = test_norms.arr.np_arr
            ref_norms = np.linalg.norm(A_ref, axis=0)
            self.assertTrue(np.allclose(test_norms, ref_norms, atol=self.atol, rtol=self.rtol))


class Test_DMatrix_f32(Test_f32, Test_DMatrix):
    __test__ = True


class Test_DMatrix_f64(Test_f64, Test_DMatrix):
    __test__ = True


if __name__ == "__main__":
    unittest.main()
