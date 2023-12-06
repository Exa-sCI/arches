import pathlib
import warnings
from ctypes import CDLL
from itertools import combinations, combinations_with_replacement, islice

import numpy as np

from arches.integral_indexing_utils import (
    compound_idx2,
    compound_idx4,
)
from arches.io import load_integrals
from arches.kernels import dispatch_H_kernel, dispatch_pt2_kernel
from arches.linked_object import (
    LinkedHandle,
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
lib_integrals = CDLL(run_folder.joinpath("build/libintegrals.so"))


# TODO: check if python version is at least 3.12, if so, import from itertools instead
def batched(iterable, n):
    ## Lifted from 3.12 documentation
    while batch := tuple(islice(iterable, n)):
        yield batch


class IntegralDictReader:
    def __init__(self, d, dtype=np.float32):
        self.d = d
        self._dtype = dtype

    def __getitem__(self, idx):
        try:
            return self.d[idx]
        except KeyError:
            return 0.0

    @property
    def dtype(self):
        return self._dtype


### Register handles for JChunks
for k in [f32, f64]:
    pfix = "JChunk_"
    sfix = "_" + type_dict[k][0]  # key : value is (c_type : (name, pointer, np ndpointer))
    k_p = type_dict[k][1]

    ctor = getattr(lib_integrals, pfix + "ctor" + sfix)
    dtor = getattr(lib_integrals, pfix + "dtor" + sfix)
    idx_ptr_return = getattr(lib_integrals, pfix + "get_idx_ptr" + sfix)
    J_ptr_return = getattr(lib_integrals, pfix + "get_J_ptr" + sfix)

    ctor.argtypes = [idx_t, idx_t_p, k_p]
    dtor.argtypes = [handle_t]
    idx_ptr_return.argtyps = [handle_t]
    J_ptr_return.argtyps = [handle_t]

    ctor.restype = handle_t
    dtor.restype = None
    idx_ptr_return.restype = idx_t_p
    J_ptr_return.restype = k_p


class JChunk(LinkedHandle):
    _constructor_f32 = lib_integrals.JChunk_ctor_f32
    _constructor_f64 = lib_integrals.JChunk_ctor_f64
    _destructor_f32 = lib_integrals.JChunk_dtor_f32
    _destructor_f64 = lib_integrals.JChunk_dtor_f64

    _get_idx_ptr_f32 = lib_integrals.JChunk_get_idx_ptr_f32
    _get_idx_ptr_f64 = lib_integrals.JChunk_get_idx_ptr_f64
    _get_J_ptr_f32 = lib_integrals.JChunk_get_J_ptr_f32
    _get_J_ptr_f64 = lib_integrals.JChunk_get_J_ptr_f64

    def __init__(self, category, chunk_size, J_ind, J, dtype=np.float64, handle=None, **kwargs):
        self.dtype = dtype
        self.chunk_size = chunk_size

        super().__init__(handle=handle, chunk_size=chunk_size, J_ind=J_ind, J=J)
        self.category = category
        self._pt2_kernels = dispatch_pt2_kernel(category)
        self._H_kernels = dispatch_H_kernel(category)
        self.idx = ManagedArray_idx_t(self.get_idx_ptr(self.handle), self.chunk_size, None)
        self.J = self._M_array_type(self.get_J_ptr(self.handle), self.chunk_size, None)

    @property
    def _f_ctor(self):
        match self.dtype:
            case np.float32:
                return self._constructor_f32
            case np.float64:
                return self._constructor_f64
            case _:
                raise NotImplementedError

    def constructor(self, chunk_size, J_ind, J, **kwargs):
        # TODO: make this cleaner and reusable
        if isinstance(J_ind, np.ndarray):
            J_ind_d = J_ind.ctypes.data_as(idx_t_p)
        else:
            J_ind_d = J_ind

        if isinstance(J, np.ndarray):
            J_d = J.ctypes.data_as(self.ptype)
        else:
            J_d = J

        return self._f_ctor(idx_t(chunk_size), J_ind_d, J_d)

    @property
    def _M_array_type(self):
        match self.dtype:
            case np.float32:
                return ManagedArray_f32
            case np.float64:
                return ManagedArray_f64

    @property
    def get_J_ptr(self):
        match self.dtype:
            case np.float32:
                return self._get_J_ptr_f32
            case np.float64:
                return self._get_J_ptr_f64
            case _:
                raise NotImplementedError

    @property
    def get_idx_ptr(self):
        match self.dtype:
            case np.float32:
                return self._get_idx_ptr_f32
            case np.float64:
                return self._get_idx_ptr_f64
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
    def category(self):
        return self._category

    @category.setter
    def category(self, val):
        if val not in ("A", "B", "C", "D", "E", "F", "G", "OE"):
            raise ValueError

        self._category = val

    @property
    def chunk_size(self):
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, val):
        if not np.issubdtype(type(val), np.integer):
            raise TypeError
        if val < 0:
            raise ValueError

        self._chunk_size = val

    @property
    def idx(self):
        return self._idx

    @idx.setter
    def idx(self, val):
        if isinstance(val, ManagedArray_idx_t):
            self._idx = val
        else:
            raise TypeError

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, val):
        self._dtype = val
        self._ctype = np_type_map[val]
        self._ptype = type_dict[self.ctype][1]

    @property
    def ctype(self):
        return self._ctype

    @property
    def ptype(self):
        return self._ptype

    @property
    def J(self):
        return self._J

    @J.setter
    def J(self, val):
        if isinstance(val, (ManagedArray_f32, ManagedArray_f64)):
            self._J = val
        else:
            raise TypeError

    @property
    def pt2_kernels(self):
        return self._pt2_kernels

    @property
    def H_kernels(self):
        return self._H_kernels

    def __getitem__(self, idx):
        if idx > 0 and idx < self.chunk_size:
            return self.J[idx], self.idx[idx]
        else:
            raise ValueError


# TODO: Create a paired iter that yields pairs of idx/values conditional on value,
#  to be able to implement screening on read. Perhaps would be good as a subclass?
#  Would be useful to be able to provide custom hook f: (idx, val) -> Bool
#  and/or flexibility to be used in repacking of chunks when using adaptive integral pruning
class JChunkFactory:
    def __init__(self, N_mo, category, src_data, chunk_size=-1, comm=None):
        if comm is None:
            self.comm_rank = 0
            self.comm_size = None
        else:
            self.comm_rank = comm.Get_rank()
            self.comm_size = comm.Get_size()

        self.N_mo = N_mo
        self.chunk_size = chunk_size
        self.src_data = src_data
        self.category = category

    @property
    def category(self):
        return self._category

    @category.setter
    def category(self, val):
        if val not in ("OE", "A", "B", "C", "D", "E", "F", "G"):
            raise ValueError

        self._category = val

        match val:
            case "OE":
                f = JChunkFactory.OE_idx_iter
            case "A":
                f = JChunkFactory.A_idx_iter
            case "B":
                f = JChunkFactory.B_idx_iter
            case "C":
                f = JChunkFactory.C_idx_iter
            case "D":
                f = JChunkFactory.D_idx_iter
            case "E":
                f = JChunkFactory.E_idx_iter
            case "F":
                f = JChunkFactory.F_idx_iter
            case "G":
                f = JChunkFactory.G_idx_iter

        # Yield indices in batches and distribute batches over communicator
        if self.chunk_size < 1:
            self.batched = False

            # cast to list: batched version caches the iterator anyway, so the interface is
            # more consistent. w/o some form of caching, would need to have two synced iterators
            # or similar to be able to yield indices for the index arrays and value arrays without
            # exhausing the index generator
            self._idx_iter = list(islice(f(self.N_mo), self.comm_rank, None, self.comm_size))
        else:
            self.batched = True
            self._idx_iter = islice(
                batched(f(self.N_mo), self.chunk_size), self.comm_rank, None, self.comm_size
            )
            self._advance_batch()  # initialize first batch

    def _advance_batch(self):
        self._batch_iter = next(self._idx_iter, False)

    @property
    def val_iter(self):
        for idx in self.idx_iter:
            yield self.src_data[idx]

    @property
    def idx_iter(self):
        if self.batched:
            return self._batch_iter
        else:
            return self._idx_iter

    @staticmethod
    def OE_idx_iter(N_mo):
        for i, j in combinations_with_replacement(range(N_mo), 2):
            yield compound_idx2(i, j)

    @staticmethod
    def A_idx_iter(N_mo):
        for i in range(N_mo):
            yield compound_idx4(i, i, i, i)

    @staticmethod
    def B_idx_iter(N_mo):
        for i, j in combinations(range(N_mo), 2):
            yield compound_idx4(i, j, i, j)

    @staticmethod
    def C_idx_iter(N_mo):
        for i, j, k in combinations(range(N_mo), 3):
            yield compound_idx4(i, j, i, k)
            yield compound_idx4(i, k, j, k)
            yield compound_idx4(j, i, j, k)

    @staticmethod
    def D_idx_iter(N_mo):
        for i, j in combinations(range(N_mo), 2):
            yield compound_idx4(i, i, i, j)
            yield compound_idx4(i, j, j, j)

    @staticmethod
    def E_idx_iter(N_mo):
        for i, j, k in combinations(range(N_mo), 3):
            yield compound_idx4(i, i, j, k)
            yield compound_idx4(i, j, j, k)
            yield compound_idx4(i, j, k, k)

    @staticmethod
    def F_idx_iter(N_mo):
        for i, j in combinations(range(N_mo), 2):
            yield compound_idx4(i, i, j, j)

    @staticmethod
    def G_idx_iter(N_mo):
        for i, j, k, l in combinations(range(N_mo), 4):  # noqa: E741
            yield compound_idx4(i, j, k, l)
            yield compound_idx4(i, k, j, l)
            yield compound_idx4(j, i, k, l)

    def get_chunks(self):
        if self.batched:  # this batching procedure is a little ugly but it works for now
            chunks = []
            empty = True
            while self.idx_iter:
                empty = False
                # TODO: Would really like to be able to tie the count to the chunk size for performance
                # but with this current design it is hard to see if this is the last batch
                # TODO: Change np.int64 to map from idx_t
                J_ind = np.fromiter(self.idx_iter, count=-1, dtype=np.int64)
                J_vals = np.fromiter(self.val_iter, count=-1, dtype=self.src_data.dtype)
                chunk_size = J_ind.shape[0]
                new_chunk = JChunk(
                    self.category, chunk_size, J_ind, J_vals, dtype=self.src_data.dtype
                )
                chunks.append(new_chunk)
                self._advance_batch()

            if empty:
                warnings.warn(
                    f"Current chunking configuration with batch size of {self.chunk_size}"
                    f"and comm size of {self.comm_size} leaves rank {self.comm_rank} with an empty chunk"
                )
                empty_J_ind = np.empty(shape=0, dtype=np.int64)
                empty_J = np.empty(shape=0, dtype=self.src_data.dtype)
                return [JChunk(self.category, 0, empty_J_ind, empty_J, dtype=self.src_data.dtype)]

            else:
                return chunks

        else:
            J_ind = np.fromiter(self.idx_iter, count=-1, dtype=np.int64)
            J_vals = np.fromiter(self.val_iter, count=-1, dtype=self.src_data.dtype)
            chunk_size = J_ind.shape[0]

            return [JChunk(self.category, chunk_size, J_ind, J_vals, dtype=self.src_data.dtype)]


def default_comm_cat_map(rank):
    # Only distribute G chunks over communicator
    if rank > 0:
        cats = ("G",)
    else:
        cats = ("A", "B", "C", "D", "E", "F", "G", "OE")

    return cats


def load_integrals_into_chunks(
    fp, comm, comm_cat_map=default_comm_cat_map, chunk_size=-1, dtype=np.float64
):
    # Set-up IO
    n_orb, n_elec, E0, J_oe, J_te = load_integrals(fp, return_N_elec=True)
    J_oe_reader = IntegralDictReader(J_oe, dtype=dtype)
    J_te_reader = IntegralDictReader(J_te, dtype=dtype)

    # Get chunks as defined by comm, batch size, and category mapping onto ranks
    cats = comm_cat_map(comm.Get_rank())

    chunks = []
    for cat in cats:
        if cat == "OE":
            fact = JChunkFactory(n_orb, cat, J_oe_reader, chunk_size, comm)
        else:
            fact = JChunkFactory(n_orb, cat, J_te_reader, chunk_size, comm)

        # add new chunks and prune any empty chunks coming from bad distribution of integrals
        chunks += [chunk for chunk in fact.get_chunks() if chunk.chunk_size > 0]

    return n_orb, n_elec, E0, chunks
