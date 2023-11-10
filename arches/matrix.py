import pathlib
from abc import abstractmethod
from ctypes import CDLL, c_char

import numpy as np
from mpi4py import MPI
from scipy.linalg import lapack as la

from arches.linked_object import (
    LinkedHandle,
    ManagedArray,
    ManagedArray_f32,
    ManagedArray_f64,
    ManagedArray_idx_t,
    f32,
    f64,
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


# TODO: Settle whether or not AMatrix derives from LinkedHandle, or if it is itself
# an ABC and the derived matrix types use multiple inheritance
# TODO: Settle whether or not to continue using the match : case or move to singledispatchmethods
class AMatrix(LinkedHandle):
    """Abstract matrix class."""

    def __init__(self, m, n, dtype, ctype, **kwargs):
        self.dtype = dtype
        self._ctype = ctype
        super().__init__(**kwargs)
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

    def destructor(self):
        match self.dtype:
            case np.float32:
                return self._destructor_f32
            case np.float64:
                return self._destructor_f64
            case _:
                raise NotImplementedError

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
        if not isinstance(val, int):
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
        if not isinstance(val, int):
            raise TypeError
        if val < 0:
            raise ValueError

        self._n = val

    def MM_shape(self, B):
        if self.transposed:
            A_m, A_n = self.n, self.m
        else:
            A_m, A_n = self.m, self.n

        if B.transposed:
            B_m, B_n = B.n, B.m
        else:
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
    ptr_return.argtypes = [handle_t]
    dtor.argtypes = [handle_t]

    fill_ctor.restype = handle_t
    copy_ctor.restype = handle_t
    ptr_return.restype = k_p
    dtor.restype = None

    ## Matrix operations
    sd = {f32: "s", f64: "d"}
    gemm = getattr(lib_matrix, pfix + sd[k] + "gemm")
    ApB = getattr(lib_matrix, pfix + sd[k] + "ApB")
    AmB = getattr(lib_matrix, pfix + sd[k] + "AmB")

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

    ApB.argtypes = [k_p, k_p, k_p, idx_t, idx_t]
    AmB.argtypes = [k_p, k_p, k_p, idx_t, idx_t]

    gemm.restype = None
    ApB.restype = None
    AmB.restype = None


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

    _sgemm = lib_matrix.DMatrix_sgemm
    _dgemm = lib_matrix.DMatrix_dgemm
    _sApB = lib_matrix.DMatrix_sApB
    _dApB = lib_matrix.DMatrix_dApB
    _sAmB = lib_matrix.DMatrix_sAmB
    _dAmB = lib_matrix.DMatrix_dAmB

    def __init__(self, m, n, A=0.0, dtype=np.float64, t=False, handle=None):
        """
        Args:
            m : row rank of A
            n : col rank of A
            A : initializing array or fill value
            dtype : data type, overrides A if A type does not match and is array
            t : whether or not A is currently transposed

        """
        if isinstance(A, np.ndarray):
            A = A.astype(dtype)  # cast type

        # call constructor via super to initialize handles and typing
        # anything that needs to be passed to the constructor needs to be passed as a kwarg
        super().__init__(handle=handle, m=m, n=n, dtype=dtype, ctype=np_type_map[dtype], arr=A)

        # data is initialized by constructor, no need to set it here
        self.arr = self._M_array_type(self.get_arr_ptr(self.handle), self.m * self.n, None)
        self.transposed = t

    def __repr__(self):
        return f"Dense {self.m} by {self.n} matrix with pointer: {self.arr.p}"

    def constructor(self, m, n, arr=0.0, **kwargs):
        if isinstance(arr, np.ndarray):
            if m * n != arr.size:
                raise ValueError

            # use copy constructor
            return self._f_ctor["copy"](idx_t(m), idx_t(n), arr)
        else:
            # use constant fill constructor
            return self._f_ctor["fill"](idx_t(m), idx_t(n), self.ctype(arr))

    # TODO: decide if @singledispatch might be better suited/cleaner
    # or, perhaps, these should all be set at initialization instead of being runtime
    def get_arr_ptr(self):
        match self.dtype:
            case np.float32:
                return self._get_arr_ptr_f32
            case np.float64:
                return self._get_arr_ptr_f64
            case _:
                raise NotImplementedError

    def gemm(self):
        match self.dtype:
            case np.float32:
                return self._sgemm
            case np.float64:
                return self._dgemm
            case _:
                raise NotImplementedError

    def ApB(self):
        match self.dtype:
            case np.float32:
                return self._sApB
            case np.float64:
                return self._dApB
            case _:
                raise NotImplementedError

    def AmB(self):
        match self.dtype:
            case np.float32:
                return self._sAmB
            case np.float64:
                return self._dAmB
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
        return np.reshape(
            np.fromiter(self.arr.p, dtype=self.dtype, count=self.arr.size), (self.m, self.n)
        )

    @property
    def T(self):
        # Return a view of the matrix
        return DMatrix(
            handle=self.handle, m=self.m, n=self.n, dtype=self.dtype, t=not self.transposed
        )

    def __add__(self, B):
        # check shape compatibility
        # TODO: adding a transposed matrix will break, decide if we want to deal with that
        if (self.m != B.m) or (self.n != B.n):
            raise ValueError

        if self.dtype != B.dtype:
            raise TypeError

        C = DMatrix(self.m, self.n, dtype=self.dtype)
        self.ApB(self.arr.p, B.arr.p, C.arr.p, self.m, self.n)
        return C

    def __iadd__(self, B):
        if (self.m != B.m) or (self.n != B.n):
            raise ValueError

        if self.dtype != B.dtype:
            raise TypeError

        self.ApB(self.arr.p, B.arr.p, self.arr.p, self.m, self.n)
        return self

    def __sub__(self, B):
        if (self.m != B.m) or (self.n != B.n):
            raise ValueError

        if self.dtype != B.dtype:
            raise TypeError

        C = DMatrix(self.m, self.n, dtype=self.dtype)
        self.AmB(self.arr.p, B.arr.p, C.arr.p, self.m, self.n)
        return C

    def __isub__(self, B):
        if (self.m != B.m) or (self.n != B.n):
            raise ValueError

        if self.dtype != B.dtype:
            raise TypeError

        self.AmB(self.arr.p, B.arr.p, self.arr.p, self.m, self.n)
        return self

    def __matmul__(self, B):
        """Left-multiply against matrix B."""
        # TODO: Handle LDA, LDB, LDC

        if self.dtype != B.dtype:
            raise TypeError

        # Maybe need good slicing capability to make this work neatly?
        m, k, n = self.MM_shape(B)
        if B in self.registry.keys():
            # TODO: implement better shape checking - want to be compatible with pre-registered result
            C = self.registry[B]
        else:
            C = DMatrix(m, n, dtype=self.dtype)

        op_A = "t" if self.transposed else "n"
        op_B = "t" if B.transposed else "n"
        lda = m
        ldb = k
        ldc = m
        self.gemm(
            c_char(op_A),
            c_char(op_B),
            m,
            n,
            lda,
            self.ctype(1.0),
            self.arr.p,
            m,
            B.arr.p,
            ldb,
            self.ctype(1.0),
            C.arr.p,
            ldc,
        )

        return C


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

    ctor.argtypes = [idx_t, idx_t, idx_t_p, idx_t_p, k_p]
    dtor.argtypes = [handle_t]
    ap_ptr_return.argtypes = [handle_t]
    ac_ptr_return.argtypes = [handle_t]
    av_ptr_return.argtypes = [handle_t]

    ctor.restype = handle_t
    dtor.restype = None
    ap_ptr_return.restype = idx_t_p
    ac_ptr_return.restype = idx_t_p
    av_ptr_return.restype = k_p


class SymCSRMatrix(AMatrix):
    """Symmetric matrix stored in CSR format."""

    _constructor_f32 = lib_matrix.SymCSRMatrix_ctor_f32
    _constructor_f64 = lib_matrix.SymCSRMatrix_ctor_f64
    _destructor_f32 = lib_matrix.SymCSRMatrix_dtor_f32
    _destructor_f64 = lib_matrix.SymCSRMatrix_dtor_f64

    _get_ap_ptr_f32 = lib_matrix.SymCSRMatrix_get_ap_ptr_f32
    _get_ac_ptr_f32 = lib_matrix.SymCSRMatrix_get_ac_ptr_f32
    _get_av_ptr_f32 = lib_matrix.SymCSRMatrix_get_av_ptr_f32
    _get_ap_ptr_f32 = lib_matrix.SymCSRMatrix_get_ap_ptr_f64
    _get_ac_ptr_f32 = lib_matrix.SymCSRMatrix_get_ac_ptr_f64
    _get_av_ptr_f32 = lib_matrix.SymCSRMatrix_get_av_ptr_f64

    def __init__(self, m, n, dtype, A_p, A_c, A_v, handle=None):
        """
        Args:
            A_p : row starts s.t. A_i lies in [A_p[i], A_i[i+1])
            A_c : col indices
            A_v : matrix values
        """
        self.A_p = A_p
        self.A_c = A_c
        self.A_v = A_v
        super().__init__(
            handle=handle,
            m=m,
            n=n,
            dtype=dtype,
            ctype=np_type_map[dtype],
            A_p=self.A_p,
            A_c=self.A_c,
            A_v=self.A_v,
        )

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

    def get_A_p_ptr(self):
        match self.dtype:
            case np.float32:
                return self._get_A_p_ptr_f32
            case np.float64:
                return self._get_A_p_ptr_f64
            case _:
                raise NotImplementedError

    def get_A_c_ptr(self):
        match self.dtype:
            case np.float32:
                return self._get_A_p_ptr_f32
            case np.float64:
                return self._get_A_p_ptr_f64
            case _:
                raise NotImplementedError

    def get_A_v_ptr(self):
        match self.dtype:
            case np.float32:
                return self._get_A_p_ptr_f32
            case np.float64:
                return self._get_A_p_ptr_f64
            case _:
                raise NotImplementedError

    @staticmethod
    def SpGEMM(op_A, op_B, alpha, A, B, beta, C):
        # For testing purposes
        # This is obviously slow but I want to loosely mimic the call interface of cuSPARSE SpMM /mkl SpGEMM

        # since this is Sym, op_A is ignored
        bA_p, bA_c, bA_v = A.buffers
        A_p = np.frombuffer(bA_p, dtype=np.int32)
        A_c = np.frombuffer(bA_c, dtype=np.int32)
        A_v = np.frombuffer(bA_v, dtype=A.dtype)

        # Iterate over rows of A
        for i in range(A.m):
            # Iterate over cols of B
            for j in range(B.n):
                # Iterate over cols of A
                for idx in range(A_p[i], A_p[i + 1]):
                    k = A_c[idx]
                    b_idx = (j, k) if op_B else (k, j)
                    # Calculate A[i, k] * B[k, j]
                    C.arr[i, j] += A_v[idx] * B.arr[b_idx]

                    if k > i:
                        # Calculate A[k, i] * B[i, j]
                        b_idx = (j, i) if op_B else (i, j)
                        C.arr[k, j] += A_v[idx] * B.arr[b_idx]

    def __matmul__(self, B):
        """Implementation of DMatrix = SymCSRMatrix @ DMatrix"""
        out_shape = self.MM_shape(B)
        if B in self.registry.keys():
            res = self.registry[B]
        else:
            res = DMatrix.allocate_new(out_shape, self.dtype, self._p)

        SymCSRMatrix.SpGEMM(None, B.transposed, 1.0, self, B, 1.0, res)
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
    """Matrix distributed as a sum of matrices, i.e., A = \sum_i A_i"""

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
            C = DMatrix(m, n, self.dtype, self._p)

        C = self.A @ B
        self.comm.Allreduce(MPI.IN_PLACE, [C.arr.p, self.mpi_type])

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
            l_res = DMatrix.allocate_new(self.A.m, B.n, self.dtype, self._p)
            g_res = RowDistMatrix(
                self.comm, l_res, out_shape[0], out_shape[1], self.dtype, self.mpi_type
            )

            l_res = self.A @ B

            # no reduction needs to be done here
            return g_res


"""
Linear algebra routines
"""


def diagonalize(X):
    """Diagonalize matrix and return eigenbasis and corresponding eigenvalues

        diagonalize s.t. Z @ L @ Z.T = A
    Args:
        X (array) : (symmetric) matrix to be diagonalized

    Returns:
        L (array) : eigenvalues of X
        Z (array) : eigenbasis of X


    """
    syevr = la.get_lapack_funcs("syevr", dtype=X.dtype)
    L, Z, _, _, _ = syevr(X)
    return L, Z
