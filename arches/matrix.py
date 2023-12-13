import pathlib
from abc import abstractmethod
from ctypes import CDLL, c_char

import numpy as np
from mpi4py import MPI
from scipy.linalg import lapack as la

from arches.linked_object import (
    LinkedArray,
    LinkedArray_f32,
    LinkedArray_f64,
    LinkedHandle,
    ManagedArray,
    ManagedArray_f32,
    ManagedArray_f64,
    ManagedArray_idx_t,
    f32,
    f32_p,
    f64,
    f64_p,
    handle_t,
    idx_t,
    idx_t_p,
    np_type_map,
    type_dict,
)

run_folder = pathlib.Path(__file__).parent.resolve()
lib_matrix = CDLL(run_folder.joinpath("build/libmatrix.so"))

"""
Matrix classes
"""


# TODO: Settle whether or not to continue using the match : case or move to singledispatchmethods
class AMatrix(LinkedHandle):
    """Abstract matrix class."""

    def __init__(self, m, n, dtype, ctype, **kwargs):
        self.dtype = dtype
        self._ctype = ctype
        super().__init__(m=m, n=n, **kwargs)  # m, n get eaten by init args
        self._registry = {}
        self.m = m
        self.n = n

    def __hash__(self):
        return hash(repr(self))

    @property
    def _f_ctor(self):
        match self.dtype:
            case np.float32:
                return self._constructor_f32
            case np.float64:
                return self._constructor_f64
            case _:
                raise NotImplementedError

    @property
    def _d_tor(self):
        match self.dtype:
            case np.float32:
                return self._destructor_f32
            case np.float64:
                return self._destructor_f64
            case _:
                raise NotImplementedError

    def destructor(self, handle):
        self._d_tor(handle)

    @property
    def _M_array_type(self):
        match self.dtype:
            case np.float32:
                return ManagedArray_f32
            case np.float64:
                return ManagedArray_f64

    @abstractmethod
    def __matmul__(self, b):
        pass

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, val):
        self._dtype = val

    @property
    def ctype(self):
        return self._ctype

    @property
    def m(self):
        """Number of rows."""
        return self._m

    @m.setter
    def m(self, val):
        if not np.issubdtype(type(val), np.integer):
            raise TypeError
        if val < 0:
            raise ValueError

        self._m = val

    @property
    def n(self):
        """Number of columns."""
        return self._n

    @n.setter
    def n(self, val):
        if not np.issubdtype(type(val), np.integer):
            raise TypeError
        if val < 0:
            raise ValueError

        self._n = val

    def MM_shape(self, B):
        A_m, A_n = self.m, self.n
        B_m, B_n = B.m, B.n

        if A_n != B_m:
            raise ValueError

        return (A_m, B_m, B_n)

    @property
    def registry(self):
        return self._registry

    def register_res(self, B, C):
        if B not in self._registry.keys():
            self._registry[B] = C


### Register C++ library functions for all DMatrix utilies
# For copy constructor, use direct pointer, to be able to construct with both data as np.array or as managed matrix
for k in [f32, f64]:
    ## Handle management
    pfix = "DMatrix_"
    sfix = "_" + type_dict[k][0]  # key : value is (c_type : (name, pointer, np ndpointer))
    k_p = type_dict[k][1]
    fill_ctor = getattr(lib_matrix, pfix + "ctor_c" + sfix)
    copy_ctor = getattr(lib_matrix, pfix + "ctor_a" + sfix)
    ptr_return = getattr(lib_matrix, pfix + "get_arr_ptr" + sfix)
    dtor = getattr(lib_matrix, pfix + "dtor" + sfix)

    fill_ctor.argtypes = [idx_t, idx_t, k]
    copy_ctor.argtypes = [idx_t, idx_t, k_p]
    ptr_return.argtypes = [handle_t, idx_t, idx_t]
    dtor.argtypes = [handle_t]

    fill_ctor.restype = handle_t
    copy_ctor.restype = handle_t
    ptr_return.restype = k_p
    dtor.restype = None

    ## Matrix operations
    sd = {f32: "s", f64: "d"}
    for op in ["ApB", "AmB", "AtB", "AdB"]:  # elementwise +, -, *, /
        c_op = getattr(lib_matrix, pfix + sd[k] + op)
        c_op.argtypes = [c_char, c_char, idx_t, idx_t, k_p, idx_t, k_p, idx_t, k_p, idx_t]
        c_op.restype = None

    submat_assign = getattr(lib_matrix, pfix + "set_submatrix" + sfix)
    submat_assign.argtypes = [c_char, c_char, idx_t, idx_t, k_p, idx_t, k_p, idx_t]
    submat_assign.restype = None

    fill_diagonal = getattr(lib_matrix, pfix + "fill_diagonal" + sfix)
    fill_diagonal.argtypes = [idx_t, handle_t, idx_t, k_p]
    fill_diagonal.restype = None

    extract_diagonal = getattr(lib_matrix, pfix + "extract_diagonal" + sfix)
    extract_diagonal.argtypes = [idx_t, handle_t, idx_t, k_p]
    extract_diagonal.restype = None

    column_2norm = getattr(lib_matrix, pfix + "column_2norm" + sfix)
    column_2norm.argtypes = [idx_t, idx_t, handle_t, idx_t, k_p]
    column_2norm.restype = None

    for cfig in ["mkl"]:
        gemm = getattr(lib_matrix, sd[k] + "gemm_" + cfig)
        gemm.argtypes = [  # C = alpha * A @ B + beta * C
            c_char,  # op A
            c_char,  # op B
            idx_t,  # m
            idx_t,  # n
            idx_t,  # k
            k,  # alpha
            k_p,  # *A
            idx_t,  # lda
            k_p,  # *B
            idx_t,  # ldb
            k,  # beta
            k_p,  # *C
            idx_t,  # ldc
        ]

        gemm.restype = None

        spgemm = getattr(lib_matrix, "sym_csr_" + sd[k] + "_MM_" + cfig)
        spgemm.argtypes = [
            k,  # alpha
            idx_t_p,  # A_rows
            idx_t_p,  # A_cols
            k_p,  # A_vals
            k_p,  # B
            k,  # beta
            k_p,  # C
            idx_t,  # M
            idx_t,  # K
            idx_t,  # N
        ]
        spgemm.restype = None


class DMatrix(AMatrix):
    """Dense matrix stored in row-major order."""

    _constructor_f32 = {
        "fill": lib_matrix.DMatrix_ctor_c_f32,
        "copy": lib_matrix.DMatrix_ctor_a_f32,
    }
    _constructor_f64 = {
        "fill": lib_matrix.DMatrix_ctor_c_f64,
        "copy": lib_matrix.DMatrix_ctor_a_f64,
    }
    _get_arr_ptr_f32 = lib_matrix.DMatrix_get_arr_ptr_f32
    _get_arr_ptr_f64 = lib_matrix.DMatrix_get_arr_ptr_f64
    _destructor_f32 = lib_matrix.DMatrix_dtor_f32
    _destructor_f64 = lib_matrix.DMatrix_dtor_f64

    _set_submatrix_f32 = lib_matrix.DMatrix_set_submatrix_f32
    _set_submatrix_f64 = lib_matrix.DMatrix_set_submatrix_f64

    _fill_diagonal_f32 = lib_matrix.DMatrix_fill_diagonal_f32
    _fill_diagonal_f64 = lib_matrix.DMatrix_fill_diagonal_f64

    _extract_diagonal_f32 = lib_matrix.DMatrix_extract_diagonal_f32
    _extract_diagonal_f64 = lib_matrix.DMatrix_extract_diagonal_f64

    _column_2norm_f32 = lib_matrix.DMatrix_column_2norm_f32
    _column_2norm_f64 = lib_matrix.DMatrix_column_2norm_f64

    _sgemm = lib_matrix.sgemm_mkl
    _dgemm = lib_matrix.dgemm_mkl
    _sApB = lib_matrix.DMatrix_sApB
    _dApB = lib_matrix.DMatrix_dApB
    _sAmB = lib_matrix.DMatrix_sAmB
    _dAmB = lib_matrix.DMatrix_dAmB
    _sAtB = lib_matrix.DMatrix_sAtB
    _dAtB = lib_matrix.DMatrix_dAtB
    _sAdB = lib_matrix.DMatrix_sAdB
    _dAdB = lib_matrix.DMatrix_dAdB

    def __init__(
        self,
        m,
        n,
        arr=0.0,
        dtype=np.float64,
        t=False,
        handle=None,
        max_row_rank=None,  # The following are used for subslice handling
        max_col_rank=None,
        row_offset=0,
        col_offset=0,
        **kwargs,
    ):
        """
        Args:
            m : row rank of A
            n : col rank of A
            A : initializing array or fill value
            dtype : data type, overrides A if A type does not match and is array
            t : whether or not A is currently transposed

        """
        if isinstance(arr, np.ndarray):
            arr = arr.astype(dtype)  # cast type

        # call constructor via super to initialize handles and typing
        # anything that needs to be passed to the constructor needs to be passed as a kwarg
        super().__init__(
            handle=handle, m=m, n=n, dtype=dtype, ctype=np_type_map[dtype], arr=arr, **kwargs
        )

        if max_row_rank is None:
            self.max_row_rank = m
        else:
            self.max_row_rank = max_row_rank

        if max_col_rank is None:
            self.max_col_rank = n
        else:
            self.max_col_rank = max_col_rank

        # TODO: when an offset is provided, all of the array access operations break on the composed ManagedArray
        # This can cause unexpected behavior and possibly seg faults if an object is assigned to a subslice of a DMatrix,
        # and the managed array of said object is then directly modified,
        # neither of which are currently used in the ARCHES code.

        # data is initialized by constructor, no need to set it here
        self.row_offset = row_offset
        self.col_offset = col_offset
        self.arr = self._M_array_type(
            self.get_arr_ptr(self.handle, row_offset, col_offset), self.m * self.n, None
        )
        self.transposed = t

    def __repr__(self):
        return f"Dense {self.m} by {self.n} matrix with pointer: {self.arr.p}"

    def constructor(self, m, n, arr=0.0, **kwargs):
        if isinstance(arr, np.ndarray):
            if m * n != arr.size:
                raise ValueError

            # use copy constructor
            return self._f_ctor["copy"](
                idx_t(m), idx_t(n), arr.ctypes.data_as(type_dict[self.ctype][1])
            )
        elif isinstance(arr, DMatrix):
            if m != arr.m or n != arr.n:
                raise ValueError
            return self._f_ctor["copy"](idx_t(m), idx_t(n), arr.arr.p)
        else:
            # use constant fill constructor
            return self._f_ctor["fill"](idx_t(m), idx_t(n), self.ctype(arr))

    @property
    def max_row_rank(self):
        return self._max_row_rank

    @max_row_rank.setter
    def max_row_rank(self, val):
        if not np.issubdtype(type(val), np.integer):
            raise TypeError

        if val < self.m:
            raise ValueError

        self._max_row_rank = val

    @property
    def max_col_rank(self):
        return self._max_col_rank

    @max_col_rank.setter
    def max_col_rank(self, val):
        if not np.issubdtype(type(val), np.integer):
            raise TypeError

        if val < self.n:
            raise ValueError

        self._max_col_rank = val

    # TODO: decide if @singledispatch might be better suited/cleaner
    # or, perhaps, these should all be set at initialization instead of being runtime
    @property
    def get_arr_ptr(self):
        match self.dtype:
            case np.float32:
                return self._get_arr_ptr_f32
            case np.float64:
                return self._get_arr_ptr_f64
            case _:
                raise NotImplementedError

    @property
    def set_submatrix(self):
        match self.dtype:
            case np.float32:
                return self._set_submatrix_f32
            case np.float64:
                return self._set_submatrix_f64
            case _:
                raise NotImplementedError

    def fill_diagonal(self, fill):
        if not isinstance(fill, LinkedArray):
            raise TypeError

        if fill.arr.size != self.m:
            raise ValueError

        lda = self.max_col_rank
        match self.dtype:
            case np.float32:
                self._fill_diagonal_f32(idx_t(self.m), self.handle, idx_t(lda), fill.arr.p)
            case np.float64:
                self._fill_diagonal_f64(idx_t(self.m), self.handle, idx_t(lda), fill.arr.p)
            case _:
                raise NotImplementedError

    def extract_diagonal(self, res):
        if not isinstance(res, LinkedArray):
            raise TypeError

        if res.arr.size != self.m:
            raise ValueError

        lda = self.max_col_rank
        match self.dtype:
            case np.float32:
                self._extract_diagonal_f32(idx_t(self.m), self.handle, idx_t(lda), res.arr.p)
            case np.float64:
                self._extract_diagonal_f64(idx_t(self.m), self.handle, idx_t(lda), res.arr.p)
            case _:
                raise NotImplementedError

    def column_2norm(self):
        lda = self.max_col_rank
        match self.dtype:
            case np.float32:
                res = LinkedArray_f32(self.n)
                self._column_2norm_f32(
                    idx_t(self.m), idx_t(self.n), self.handle, idx_t(lda), res.arr.p
                )
            case np.float64:
                res = LinkedArray_f64(self.n)
                self._column_2norm_f64(
                    idx_t(self.m), idx_t(self.n), self.handle, idx_t(lda), res.arr.p
                )
            case _:
                raise NotImplementedError

        return res

    @property
    def gemm(self):
        match self.dtype:
            case np.float32:
                return self._sgemm
            case np.float64:
                return self._dgemm
            case _:
                raise NotImplementedError

    @property
    def ApB(self):
        match self.dtype:
            case np.float32:
                return self._sApB
            case np.float64:
                return self._dApB
            case _:
                raise NotImplementedError

    @property
    def AmB(self):
        match self.dtype:
            case np.float32:
                return self._sAmB
            case np.float64:
                return self._dAmB
            case _:
                raise NotImplementedError

    @property
    def AtB(self):
        match self.dtype:
            case np.float32:
                return self._sAtB
            case np.float64:
                return self._dAtB
            case _:
                raise NotImplementedError

    @property
    def AdB(self):
        match self.dtype:
            case np.float32:
                return self._sAdB
            case np.float64:
                return self._dAdB
            case _:
                raise NotImplementedError

    @property
    def arr(self):
        return self._arr

    @arr.setter
    def arr(self, val):
        if isinstance(val, ManagedArray):
            self._arr = val
        else:
            raise TypeError

    @property
    def np_arr(self):
        if self.m == self.max_row_rank and self.n == self.max_col_rank:
            arr = np.reshape(
                np.fromiter(self.arr.p, dtype=self.dtype, count=self.arr.size), (self.m, self.n)
            )
        elif self.transposed:
            temp = DMatrix(self.m, self.n, dtype=self.dtype)

            # TODO: something gets switched around twice, so swapping the offsets
            # on the pass off to the temp matrix is necessary when the assignee
            # matrix is a transposed view. Figure out if that's okay? Tests pass for now.
            ro, co = self.row_offset, self.col_offset
            self.row_offset, self.col_offset = co, ro

            temp[:, :] = self[:, :]
            arr = temp.np_arr

            self.row_offset, self.col_offset = ro, co
        else:  # Matrix is subsliced
            # TODO: Consider making a faster option. For now, copy into a dense DMatrix
            # and return its np arr
            temp = DMatrix(self.m, self.n, dtype=self.dtype)
            temp[:, :] = self[:, :]
            arr = temp.np_arr

        return arr

    @property
    def T(self):
        # Return a view of the matrix
        return DMatrix(
            handle=self.handle,
            m=self.n,
            n=self.m,
            max_row_rank=self.max_col_rank,
            max_col_rank=self.max_row_rank,
            row_offset=self.row_offset,
            col_offset=self.col_offset,
            dtype=self.dtype,
            t=not self.transposed,
        )

    def _parse_slice(self, t):
        match t:
            case slice():
                if t.step is not None:
                    raise NotImplementedError("Strided slicing of matrices not supported.")
                if self.m == 1:
                    row_start = 0
                    row_stop = 1
                    col_start = 0 if t.start is None else t.start
                    col_stop = self.n if t.stop is None else t.stop
                elif self.n == 1:
                    row_start = 0 if t.start is None else t.start
                    row_stop = self.n if t.stop is None else t.stop
                    col_start = 0
                    col_stop = 1
                else:
                    raise ValueError(
                        "Single slice indexing not supported for matrices without singleton dimension."
                    )
            case tuple():
                if len(t) != 2:
                    raise ValueError("Wrong number of dimensions indexed.")

                match t[0]:
                    case slice():
                        if t[0].step is not None:
                            raise NotImplementedError("Strided slicing of matrices not supported.")

                        row_start = 0 if t[0].start is None else t[0].start
                        row_stop = self.m if t[0].stop is None else t[0].stop
                    case int():
                        row_start = t[0]
                        row_stop = t[0] + 1

                match t[1]:
                    case slice():
                        if t[1].step is not None:
                            raise NotImplementedError("Strided slicing of matrices not supported.")

                        col_start = 0 if t[1].start is None else t[1].start
                        col_stop = self.n if t[1].stop is None else t[1].stop
                    case int():
                        col_start = t[1]
                        col_stop = t[1] + 1
            case _:
                raise ValueError

        if row_start < 0 or row_stop > self.m:
            raise ValueError(
                f"Requested slice ({row_start}, {row_stop}) extends beyond row rank {self.m}."
            )

        if col_start < 0 or col_stop > self.n:
            raise ValueError(
                f"Requested slice ({col_start}, {col_stop}) extends beyond col rank {self.n}."
            )

        row_offset = self.col_offset + col_start if self.transposed else self.row_offset + row_start
        col_offset = self.row_offset + row_start if self.transposed else self.col_offset + col_start

        return row_start, row_stop, col_start, col_stop, row_offset, col_offset

    def __getitem__(self, t):
        row_start, row_stop, col_start, col_stop, row_offset, col_offset = self._parse_slice(t)

        # TODO: What happens with nested subslices?
        # Do we need to revert back to global coordinates always or can the
        # array location in original memory be determined elsewise?
        return DMatrix(
            handle=self.handle,
            m=row_stop - row_start,
            n=col_stop - col_start,
            dtype=self.dtype,
            t=self.transposed,
            max_row_rank=self.max_row_rank,
            max_col_rank=self.max_col_rank,
            row_offset=row_offset,
            col_offset=col_offset,
        )

    def __setitem__(self, t, B):
        row_start, row_stop, col_start, col_stop, row_offset, col_offset = self._parse_slice(t)

        if not isinstance(B, DMatrix):
            raise NotImplementedError

        if B.m != (row_stop - row_start) or B.n != (col_stop - col_start):
            raise ValueError(
                f"Source shape {B.m, B.n} and destination shape {row_stop-row_start, col_stop-col_start} do not match."
            )

        op_A = "t" if self.transposed else "n"
        op_B = "t" if B.transposed else "n"
        dest_p = self.get_arr_ptr(self.handle, row_offset, col_offset)
        lda = self.max_row_rank if self.transposed else self.max_col_rank
        ldb = B.max_row_rank if B.transposed else B.max_col_rank

        self.set_submatrix(
            c_char(op_A.encode("utf-8")),
            c_char(op_B.encode("utf-8")),
            B.m,
            B.n,
            dest_p,
            lda,
            B.arr.p,
            ldb,
        )

    def _parse_op_args(self, B):
        op_A = "t" if self.transposed else "n"
        lda = self.max_row_rank if self.transposed else self.max_col_rank
        match B:
            case DMatrix():
                if self.transposed or B.transposed:
                    raise NotImplementedError

                # check shape compatibility
                if (self.m != B.m) or (self.n != B.n):
                    raise ValueError

                if self.dtype != B.dtype:
                    raise TypeError

                op_B = "t" if B.transposed else "n"

                # TODO: switch to m, k, m for col-ordered
                ldb = B.max_row_rank if B.transposed else B.max_col_rank
            case LinkedArray():
                if self.dtype != B.arr.dtype:
                    raise TypeError
                if self.m == 1:
                    if not (self.n == B.N):
                        raise ValueError
                elif self.n == 1:
                    if not (self.m == B.N):
                        raise ValueError
                else:
                    raise ValueError
                op_B = "n"
                ldb = 1
            case _:
                raise NotImplementedError

        return op_A, op_B, lda, ldb

    def __add__(self, B):
        op_A, op_B, lda, ldb = self._parse_op_args(B)

        C = DMatrix(self.m, self.n, dtype=self.dtype)
        ldc = C.max_col_rank

        self.ApB(
            c_char(op_A.encode("utf-8")),
            c_char(op_B.encode("utf-8")),
            self.m,
            self.n,
            self.arr.p,
            lda,
            B.arr.p,
            ldb,
            C.arr.p,
            ldc,
        )
        return C

    def __iadd__(self, B):
        op_A, op_B, lda, ldb = self._parse_op_args(B)

        self.ApB(
            c_char(op_A.encode("utf-8")),
            c_char(op_B.encode("utf-8")),
            self.m,
            self.n,
            self.arr.p,
            lda,
            B.arr.p,
            ldb,
            self.arr.p,
            lda,
        )
        return self

    def __sub__(self, B):
        op_A, op_B, lda, ldb = self._parse_op_args(B)

        C = DMatrix(self.m, self.n, dtype=self.dtype)
        ldc = C.max_col_rank

        self.AmB(
            c_char(op_A.encode("utf-8")),
            c_char(op_B.encode("utf-8")),
            self.m,
            self.n,
            self.arr.p,
            lda,
            B.arr.p,
            ldb,
            C.arr.p,
            ldc,
        )

        return C

    def __isub__(self, B):
        op_A, op_B, lda, ldb = self._parse_op_args(B)

        self.AmB(
            c_char(op_A.encode("utf-8")),
            c_char(op_B.encode("utf-8")),
            self.m,
            self.n,
            self.arr.p,
            lda,
            B.arr.p,
            ldb,
            self.arr.p,
            lda,
        )
        return self

    def __mul__(self, B):
        op_A, op_B, lda, ldb = self._parse_op_args(B)

        C = DMatrix(self.m, self.n, dtype=self.dtype)
        ldc = C.max_col_rank

        self.AtB(
            c_char(op_A.encode("utf-8")),
            c_char(op_B.encode("utf-8")),
            self.m,
            self.n,
            self.arr.p,
            lda,
            B.arr.p,
            ldb,
            C.arr.p,
            ldc,
        )

        return C

    def __imul__(self, B):
        op_A, op_B, lda, ldb = self._parse_op_args(B)

        self.AtB(
            c_char(op_A.encode("utf-8")),
            c_char(op_B.encode("utf-8")),
            self.m,
            self.n,
            self.arr.p,
            lda,
            B.arr.p,
            ldb,
            self.arr.p,
            lda,
        )
        return self

    def __truediv__(self, B):
        op_A, op_B, lda, ldb = self._parse_op_args(B)

        C = DMatrix(self.m, self.n, dtype=self.dtype)
        ldc = C.max_col_rank

        self.AdB(
            c_char(op_A.encode("utf-8")),
            c_char(op_B.encode("utf-8")),
            self.m,
            self.n,
            self.arr.p,
            lda,
            B.arr.p,
            ldb,
            C.arr.p,
            ldc,
        )

        return C

    def __itruediv__(self, B):
        op_A, op_B, lda, ldb = self._parse_op_args(B)

        self.AdB(
            c_char(op_A.encode("utf-8")),
            c_char(op_B.encode("utf-8")),
            self.m,
            self.n,
            self.arr.p,
            lda,
            B.arr.p,
            ldb,
            self.arr.p,
            lda,
        )
        return self

    def __neg__(self):
        res = DMatrix(self.m, self.n, dtype=self.dtype)
        return res - self

    def __matmul__(self, B):
        """Left-multiply against matrix B."""
        if self.dtype != B.dtype:
            raise TypeError

        # Maybe need good slicing capability to make this work neatly?
        m, k, n = self.MM_shape(B)
        if B in self.registry.keys():
            # TODO: implement better shape checking - want to be compatible with pre-registered result
            # TODO: enforce that C is never transposed
            C = self.registry[B]
        else:
            C = DMatrix(m, n, dtype=self.dtype)

        op_A = "t" if self.transposed else "n"
        op_B = "t" if B.transposed else "n"
        # TODO: switch to m, k, m for col-ordered
        lda = self.max_row_rank if self.transposed else self.max_col_rank
        ldb = B.max_row_rank if B.transposed else B.max_col_rank
        ldc = C.max_col_rank  # if C in registry is transposed, this breaks
        self.gemm(
            c_char(op_A.encode("utf-8")),
            c_char(op_B.encode("utf-8")),
            m,
            n,
            k,
            self.ctype(1.0),
            self.arr.p,
            lda,
            B.arr.p,
            ldb,
            self.ctype(1.0),
            C.arr.p,
            ldc,
        )

        return C

    @classmethod
    def eye(cls, l, dtype=np.float64):
        res = DMatrix(l, l, dtype=dtype)
        match dtype:
            case np.float32:
                temp = LinkedArray_f32(l, fill=1.0)
            case np.float64:
                temp = LinkedArray_f64(l, fill=1.0)
        res.fill_diagonal(temp)
        return res


### Register C++ library functions for all SymCSRMatrix utilies
for k in [f32, f64]:
    ## Handle management
    pfix = "SymCSRMatrix_"
    sfix = "_" + type_dict[k][0]  # key : value is (c_type : (name, pointer, np ndpointer))
    k_p = type_dict[k][1]
    ctor = getattr(lib_matrix, pfix + "ctor" + sfix)
    dtor = getattr(lib_matrix, pfix + "dtor" + sfix)
    ap_ptr_return = getattr(lib_matrix, pfix + "get_ap_ptr" + sfix)
    ac_ptr_return = getattr(lib_matrix, pfix + "get_ac_ptr" + sfix)
    av_ptr_return = getattr(lib_matrix, pfix + "get_av_ptr" + sfix)
    n_entries_return = getattr(lib_matrix, pfix + "get_n_entries" + sfix)

    ctor.argtypes = [idx_t, idx_t, idx_t_p, idx_t_p, k_p]
    dtor.argtypes = [handle_t]
    ap_ptr_return.argtypes = [handle_t]
    ac_ptr_return.argtypes = [handle_t]
    av_ptr_return.argtypes = [handle_t]
    n_entries_return.argtypes = [handle_t]

    ctor.restype = handle_t
    dtor.restype = None
    ap_ptr_return.restype = idx_t_p
    ac_ptr_return.restype = idx_t_p
    av_ptr_return.restype = k_p
    n_entries_return.restype = idx_t


class SymCSRMatrix(AMatrix):
    """Symmetric matrix stored in CSR format."""

    _constructor_f32 = lib_matrix.SymCSRMatrix_ctor_f32
    _constructor_f64 = lib_matrix.SymCSRMatrix_ctor_f64
    _destructor_f32 = lib_matrix.SymCSRMatrix_dtor_f32
    _destructor_f64 = lib_matrix.SymCSRMatrix_dtor_f64

    _get_A_p_ptr_f32 = lib_matrix.SymCSRMatrix_get_ap_ptr_f32
    _get_A_p_ptr_f64 = lib_matrix.SymCSRMatrix_get_ap_ptr_f64

    _get_A_c_ptr_f32 = lib_matrix.SymCSRMatrix_get_ac_ptr_f32
    _get_A_c_ptr_f64 = lib_matrix.SymCSRMatrix_get_ac_ptr_f64

    _get_A_v_ptr_f32 = lib_matrix.SymCSRMatrix_get_av_ptr_f32
    _get_A_v_ptr_f64 = lib_matrix.SymCSRMatrix_get_av_ptr_f64

    _get_n_entries_f32 = lib_matrix.SymCSRMatrix_get_n_entries_f32
    _get_n_entries_f64 = lib_matrix.SymCSRMatrix_get_n_entries_f64

    _s_spgemm = lib_matrix.sym_csr_s_MM_mkl
    _d_spgemm = lib_matrix.sym_csr_d_MM_mkl

    def __init__(self, m, n, dtype, A_p=None, A_c=None, A_v=None, handle=None, **kwargs):
        """
        Args:
            A_p : row starts s.t. A_i lies in [A_p[i], A_i[i+1])
            A_c : col indices
            A_v : matrix values
        """
        super().__init__(
            handle=handle,
            m=m,
            n=n,
            dtype=dtype,
            ctype=np_type_map[dtype],
            A_p=A_p,
            A_c=A_c,
            A_v=A_v,
            **kwargs,
        )

        self.N_entries = self.get_N_entries(self.handle)
        self.A_p = ManagedArray_idx_t(self.get_A_p_ptr(self.handle), self.m + 1, None)
        self.A_c = ManagedArray_idx_t(self.get_A_c_ptr(self.handle), self.N_entries, None)
        self.A_v = self._M_array_type(self.get_A_v_ptr(self.handle), self.N_entries, None)

    # Since SymCSR really only applies to the Hamiltonian, and always left-multiplies, don't need nearly as much
    # functionality to create new arrays and manage memory

    # Just assume that we have access to the underlying buffers somewhere

    def __repr__(self):
        return f"Symmetric sparse {self.m} by {self.n} matrix in CSR format with pointers: {self.A_p.p, self.A_c.p, self.A_v.p}"

    def constructor(self, m, n, A_p, A_c, A_v, **kwargs):
        if A_p.size != m + 1:
            raise ValueError

        if A_c.size != A_v.size != A_p[m]:
            raise ValueError

        return self._f_ctor(idx_t(m), idx_t(n), A_p.p, A_c.p, A_v.p)

    @property
    def A_p(self):
        return self._A_p

    @A_p.setter
    def A_p(self, val):
        if isinstance(val, ManagedArray_idx_t):
            self._A_p = val
        else:
            raise ValueError

    @property
    def A_c(self):
        return self._A_c

    @A_c.setter
    def A_c(self, val):
        if isinstance(val, ManagedArray_idx_t):
            self._A_c = val
        else:
            raise ValueError

    @property
    def A_v(self):
        return self._A_v

    @A_v.setter
    def A_v(self, val):
        if isinstance(val, ManagedArray):
            self._A_v = val
        else:
            raise ValueError

    @property
    def get_A_p_ptr(self):
        match self.dtype:
            case np.float32:
                return self._get_A_p_ptr_f32
            case np.float64:
                return self._get_A_p_ptr_f64
            case _:
                raise NotImplementedError

    @property
    def get_A_c_ptr(self):
        match self.dtype:
            case np.float32:
                return self._get_A_c_ptr_f32
            case np.float64:
                return self._get_A_c_ptr_f64
            case _:
                raise NotImplementedError

    @property
    def get_A_v_ptr(self):
        match self.dtype:
            case np.float32:
                return self._get_A_v_ptr_f32
            case np.float64:
                return self._get_A_v_ptr_f64
            case _:
                raise NotImplementedError

    @property
    def get_N_entries(self):
        match self.dtype:
            case np.float32:
                return self._get_n_entries_f32
            case np.float64:
                return self._get_n_entries_f64
            case _:
                raise NotImplementedError

    @property
    def spgemm(self, op_A, op_B, alpha, A, B, beta, C):
        match self.dtype:
            case np.float32:
                return self._s_spgemm
            case np.float64:
                return self._d_spgemm
            case _:
                raise NotImplementedError

    def __matmul__(self, B):
        """Implementation of DMatrix = SymCSRMatrix @ DMatrix"""
        m, k, n = self.MM_shape(B)
        if B in self.registry.keys():
            res = self.registry[B]
        else:
            res = DMatrix(m, n, dtype=self.dtype)

        # TODO: consider interface for op B? op A is not so useful since A is symmetric
        # and/or consider passing arguments for B storage being col. major instead
        op_A = "t" if self.transposed else "n"
        op_B = "t" if B.transpose else "n"
        lda = m
        ldb = k
        ldc = m
        self.spgemm(
            self.ctype(1.0),
            self.A_p.p,
            self.A_c.p,
            self.A_v.p,
            B.arr.p,
            self.ctype(1.0),
            res.arr.p,
            m,
            k,
            n,
        )

        return res


class DistMatrix(AMatrix):
    """Abstract class for distributed matrices."""

    def __init__(self, comm, A, m, n, dtype, mpi_type):
        super().init(m, n, dtype)
        self.lm = A.m
        self.ln = A.n
        self.A = A
        self.comm = comm
        self.mpi_type = mpi_type  # TODO: perhaps better to automatically set this rather than input it multipe times

    @property
    def lm(self):
        """Local row rank."""
        return self._lm

    @lm.setter
    def lm(self, val):
        if not isinstance(val, int):
            raise TypeError
        if val < 0:
            raise ValueError

        self._lm = val

    @property
    def ln(self):
        """Local col rank."""
        return self._ln

    @ln.setter
    def ln(self, val):
        if not isinstance(val, int):
            raise TypeError
        if val < 0:
            raise ValueError

        self._ln = val

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, val):
        if not isinstance(val, AMatrix):
            raise TypeError

        self._A = val


class PartialSumMatrix(DistMatrix):
    """Matrix distributed as a sum of matrices, i.e., A = \\sum_i A_i"""

    def __init__(self, comm, A, m, n, dtype, mpi_type):
        assert A.m == m  # local size needs to be the same as global
        assert A.n == n
        super().__init__(comm, A, m, n, dtype, mpi_type)

    def __repr__(self):
        return (
            f"Partial sum matrix distributed over comm {repr(self.comm)}\n"
            + f"Local {repr(self.A)}"
        )

    def __matmul__(self, B):
        ### Direct implementation for DMatrix = PartialSum @ DMatrix
        assert isinstance(B, DMatrix)
        m, k, n = self.MM_shape(B)
        if B in self.registry.keys():
            C = self.registry[B]
        else:
            C = DMatrix(m, n, dtype=self.dtype)

        C = self.A @ B
        # self.comm.Allreduce(MPI.IN_PLACE, [C.arr.p, self.mpi_type])

        return C


class RowDistMatrix(DistMatrix):
    def __init__(self, comm, A, m, n, dtype, mpi_type):
        super().__init__(comm, A, m, n, dtype, mpi_type)

    def __repr__(self):
        return (
            f"Row distributed matrix distributed over comm {repr(self.comm)}\n"
            + f"Local {repr(self.A)}"
        )

    def __matmul__(self, B):
        out_shape = self.MM_shape(B)

        # might be better to have more sophisticated result structure determination
        # if we want these to be composable on multiple levels
        if B in self.registry.keys():
            g_res = self.registry[B]
        elif isinstance(self.A, SymCSRMatrix):
            # if local matrix is SymCSR, then will still need to reduce over result
            # if local matrix is GenCSR (not implemented), can return as RowDist
            pass
        elif isinstance(self.A, DMatrix):
            # if local matrix is Dense, can return as RowDist on the same comm
            l_res = DMatrix(self.A.m, B.n, dtype=self.dtype)
            g_res = RowDistMatrix(
                self.comm, l_res, out_shape[0], out_shape[1], self.dtype, self.mpi_type
            )

            l_res = self.A @ B

            # no reduction needs to be done here
            return g_res


"""
Linear algebra routines
"""

for k in [f32, f64]:
    sd = {f32: "s", f64: "d"}
    k_p = type_dict[k][1]
    f_qr = getattr(lib_matrix, sd[k] + "qr_mkl")
    f_qr.argtypes = [idx_t, idx_t, k_p, idx_t]
    f_qr.restype = handle_t

    f_syevd = getattr(lib_matrix, sd[k] + "syevd_mkl")
    f_syevd.argtypes = [idx_t, k_p, idx_t, k_p]
    f_syevd.restype = None


def qr_factorization(X):
    """Factorize X into Q @ R

    X will be overwritten with Q

    """
    lda = X.max_col_rank
    match X.dtype:
        case np.float32:
            res_handle = lib_matrix.sqr_mkl(X.m, X.n, X.arr.p, lda)
        case np.float64:
            res_handle = lib_matrix.dqr_mkl(X.m, X.n, X.arr.p, lda)
        case _:
            raise NotImplementedError

    return DMatrix(X.n, X.n, dtype=X.dtype, handle=res_handle, override_original=True)


def diagonalize(X):
    """Diagonalize matrix and return eigenbasis and corresponding eigenvalues

        diagonalize s.t. Z @ L @ Z.T = A
    Args:
        X (array) : (symmetric) matrix to be diagonalized

    Returns:
        L (array) : eigenvalues of X
        Z (array) : eigenbasis of X


    """
    lda = X.max_col_rank
    if X.m != X.n:
        raise ValueError
    match X.dtype:
        case np.float32:
            w = LinkedArray_f32(X.n)
            lib_matrix.ssyevd_mkl(idx_t(X.n), X.arr.p, idx_t(lda), w.arr.p)
        case np.float64:
            w = LinkedArray_f64(X.n)
            lib_matrix.dsyevd_mkl(idx_t(X.n), X.arr.p, idx_t(lda), w.arr.p)
    return w
