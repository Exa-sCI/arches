import numpy as np
from scipy.linalg import lapack as la
from scipy.linalg import blas
from mpi4py import MPI
from abc import ABC


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


class PointerStorage:
    """A fake pointer interface."""

    def __init__(self):
        self.pointers = {}

    def __getitem__(self, k):
        return self.pointers[k]

    def __setitem__(self, k, v):
        self.pointers[k] = v


class AMatrix(ABC):
    """Abstract matrix class."""

    def __init__(self, m, n, dtype):
        self._registry = {}
        self.m = m
        self.n = n
        self.dtype = dtype

    def __hash__(self):
        return hash(repr(self))

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
        if self.n != B.m:
            raise ValueError

        return (self.m, B.n)

    @property
    def registry(self):
        return self._registry

    def register_res(self, B, C):
        if not B in self._registry.keys():
            self._registry[B] = C


class DMatrix(AMatrix):
    """Dense matrix stored in col-major order."""

    def __init__(self, p, A, m, n, dtype, t=False):
        """
        Args:
            p : pointer/buffer mapping dictionary (for testing only)
            A : pointer to buffer
            m : row rank of A
            n : col rank of A
            dtype : data type
            t : whether or not A is currently transposed

        """
        # TODO: type should be associated with A/buffer
        super().__init__(m, n, dtype)
        self._p = p
        self.A = A
        self.transposed = t

    def __repr__(self):
        return f"Dense {self.m} by {self.n} matrix with pointer: {self.A}"

    @classmethod
    def allocate_new(cls, shape, dtype, p):
        m, n = shape
        buffer = bytearray(m * n * dtype.itemsize)
        p[id(buffer)] = buffer
        return cls(p, id(buffer), m, n, dtype)

    @property
    def arr(self):
        return np.reshape(np.frombuffer(self.buffer, dtype=dtype), (self.m, self.n))

    @arr.setter
    def arr(self, v):
        self._p[self.A] = v

    @property
    def buffer(self):
        return self._p[self.A]

    @property
    def p(self):
        return self._p

    @property
    def T(self):
        return DMatrix(
            self.p, self.A, self.m, self.n, self.dtype, t=not self.transposed
        )

    def __add__(self, B):
        return self.arr + B.arr

    def __sub__(self, B):
        return self.arr - B.arr

    def __matmul__(self, B):
        """Left-multiply against a col-major matrix B."""
        out_shape = self.MM_shape(B)
        if self.dtype != B.dtype:
            raise TypeError

        if B in self.registry.keys():
            res = self.registry[B]
            # TODO: implement shape checking - want to be compatible
            # Likely need good slicing capability to make this work neatly
        else:
            res = DMatrix.allocate_new(out_shape, self.dtype, self._p)

        # Scipy's BLAS won't actually overwrite C, even if requested... so this is a very fake pointer illusion
        res.arr = bytearray(
            blas.dgemm(
                1.0,
                self.arr,
                B.arr,
                c=res.arr,
                trans_a=self.transposed,
                trans_b=B.transposed,
            ).tobytes()
        )

        return res


class SymCSRMatrix(AMatrix):
    """Symmetric matrix stored in CSR format."""

    def __init__(self, p, A_p, A_c, A_v, m, n, dtype):
        """
        Args:
            A_p : row starts s.t. A_i lies in [A_p[i], A_i[i+1])
            A_c : col indices
            A_v : matrix values
        """
        super().__init__(m, n, dtype)
        self._p = p
        self.A_p = A_p
        self.A_c = A_c
        self.A_v = A_v

    # Since SymCSR really only applies to the Hamiltonian, and always left-multiplies, don't need nearly as much
    # functionality to create new arrays and manage memory

    # Just assume that we have access to the underlying buffers somewhere

    def __repr__(self):
        return f"Symmetric sparse {self.m} by {self.n} matrix in CSR format with pointers: {self.A_p, self.A_c, self.A_v}"

    @property
    def buffers(self):
        return self._p[self.A_p], self._p[self.A_c], self._p[self.A_v]

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

        SymCSRMatrix.SpGEMM(A.transposed, B.transposed, 1.0, self, B, 1.0, res)
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
        if not isinstance(A, AMatrix):
            raise TypeError

        self._A = A


class PartialSumMatrix(DistMatrix):
    """Matrix distributed as a sum of matrices, i.e., A = \sum_i A_i"""

    def __init__(self, comm, A, m, n, dtype, mpi_type):
        assert A.m == m  # local size needs to be the same as global
        assert A.n == n
        super().__init__(comm, A, m, n, dtype, mpi_type)

    def __repr__(self):
        return (
            f"Partial sum matrix distributed over comm {repr(comm)}\n"
            + f"Local {repr(self.A)}"
        )

    def __matmul__(self, B):
        ### Direct implementation for DMatrix = PartialSum @ DMatrix
        assert isinstance(B, DMatrix)
        out_shape = self.MM_shape(B)
        if B in self.registry.keys():
            res = self.registry[B]
        else:
            res = DMatrix.allocate_new(out_shape, self.dtype, self._p)

        res = self.A @ B
        self.comm.Allreduce(MPI.IN_PLACE, [res.arr, self.mpi_type])

        return res

    def local_dense_to_dense(self):


class RowDistMatrix(DistMatrix):

    def __init__(self, comm, A, m, n, dtype, mpi_type):
        super().__init__(comm, A, m, n, dtype, mpi_type)

    def __repr__(self):
        return (
            f"Row distributed matrix distributed over comm {repr(comm)}\n"
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
            g_res = RowDistMatrix(comm, l_res, out_shape[0], out_shape[1], self.dtype, self.mpi_type)

            l_res = self.A @ B

            # no reduction needs to be done here
            return g_res


