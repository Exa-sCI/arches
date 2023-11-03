import pathlib
from ctypes import CDLL
from functools import reduce
from itertools import combinations, islice, product

import numpy as np

from arches.drivers import integral_category
from arches.integral_indexing_utils import (
    canonical_idx4,
    compound_idx4,
)
from arches.kernels import dispatch_kernel
from arches.linked_object import (
    LinkedArray_f32,
    LinkedArray_f64,
    LinkedArray_idx_t,
    LinkedHandle,
    det_t_p,
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
lib_integrals = CDLL(run_folder.joinpath("build/libintegrals.so"))


def batched(iterable, n):
    ## Lifted from 3.12 documentation
    while batch := tuple(islice(iterable, n)):
        yield batch


class IntegralReader:
    def __init__(self, d=None):
        self.d = d

    def __getitem__(self, idx):
        if self.d is None:
            return 0.0
        else:
            return self.d[idx]

    @property
    def dtype(self):
        return np.float32


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
    _destructor_f32 = lib_integrals.JChunk_dtor_f64

    _get_idx_ptr_f32 = lib_integrals.JChunk_get_idx_ptr_f32
    _get_idx_ptr_f64 = lib_integrals.JChunk_get_idx_ptr_f64
    _get_J_ptr_f32 = lib_integrals.JChunk_get_J_ptr_f32
    _get_J_ptr_f64 = lib_integrals.JChunk_get_J_ptr_f64

    def __init__(self, category, chunk_size, J_ind, J, dtype=np.float64, handle=None, **kwargs):
        self.dtype = dtype
        super().__init__(handle=handle, chunk_size=chunk_size, J_ind=J_ind, J=J)
        self.category = category
        self._kernels = dispatch_kernel(category)

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
    def category(self):
        return self._category

    @category.setter
    def category(self, val):
        if (val not in "ABCDEFG") or (len(val) != 1):
            raise ValueError

        self._category = val

    @property
    def chunk_size(self):
        return self.J.size

    @property
    def idx(self):
        return self._idx

    @idx.setter
    def idx(self, val):
        if isinstance(val, LinkedArray_idx_t):
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
        if isinstance(val, (LinkedArray_f32, LinkedArray_f64)):
            self._J = val
        else:
            raise TypeError

    @property
    def kernels(self):
        return self._kernels

    def __getitem__(self, idx):
        if idx > 0 and idx < self.chunk_size:
            return self.J[idx], self.idx[idx]
        else:
            raise ValueError


# TODO: Create a paired iter that yields pairs of idx/values conditional on value,
#  to be able to implement screening on read. Perhaps would be good as a subclass?
#  Would be useful to be able to provide custom hook f: (idx, val) -> Bool
#  and/or flexibility to be used in repacking of chunks when using adaptive integral pruning
# TODO: Let the chunk factory make one electron chunks!
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
        if val not in "ABCDEFG":
            raise ValueError
        if len(val) != 1:
            raise ValueError

        self._category = val

        match val:
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
            self._idx_iter = islice(f(N_mo), self.comm_rank, None, self.comm_size)
        else:
            self.batched = True
            self._idx_iter = islice(
                batched(f(N_mo), self.chunk_size), self.comm_rank, None, self.comm_size
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
            while self.idx_iter:
                # TODO: Would really like to be able to tie the count to the chunk size for performance
                # but with this current design it is hard to see if this is the last batch
                J_ind = np.fromiter(self.idx_iter, count=-1, dtype=np.int32)
                J_vals = np.fromiter(self.val_iter, count=-1, dtype=self.src_data.dtype)
                chunk_size = J_ind.shape[0]

                new_chunk = JChunk(chunk_size, J_vals, J_ind)
                chunks.append(new_chunk)
                self._advance_batch()

            return chunks

        else:
            J_ind = np.fromiter(self.idx_iter, count=-1, dtype=np.int32)
            J_vals = np.fromiter(self.val_iter, count=-1, dtype=self.src_data.dtype)
            chunk_size = J_ind.shape[0]

            return JChunk(chunk_size, J_vals, J_ind)


if __name__ == "__main__":

    class FakeComm:
        def __init__(self, rank, size):
            self.rank = rank
            self.size = size

        def Get_rank(self):
            return self.rank

        def Get_size(self):
            return self.size

    def test_chunk(N_mo, cat, ref_data, src_data=IntegralReader()):
        fact = JChunkFactory(N_mo, cat, src_data)
        chunk = fact.get_chunks()
        all_ind = set([v["idx"] for k, v in ref_data.items() if v["category"] == cat])

        assert all_ind == set(chunk.idx)
        assert (len(all_ind)) == chunk.chunk_size

    def test_chunk_batched(N_mo, cat, ref_data, src_data=IntegralReader()):
        fact = JChunkFactory(N_mo, cat, src_data, chunk_size=2048)
        chunks = fact.get_chunks()
        all_ind = set([v["idx"] for k, v in ref_data.items() if v["category"] == cat])

        assert all_ind == set(
            reduce(lambda x, y: set(x).union(set(y)), [chunk.idx for chunk in chunks])
        )

    def test_chunk_batched_dist(N_mo, cat, ref_data, comm_size, src_data=IntegralReader()):
        all_ind = set([v["idx"] for k, v in ref_data.items() if v["category"] == cat])

        local_chunks = []
        for rank in range(comm_size):
            comm = FakeComm(rank, comm_size)
            fact = JChunkFactory(N_mo, cat, src_data, comm=comm, chunk_size=2048)
            chunks = fact.get_chunks()
            local_chunk = set(reduce(lambda x, y: x.union(y), [set(chunk.idx) for chunk in chunks]))
            local_chunks.append(local_chunk)

        assert all_ind == set(reduce(lambda x, y: x.union(y), [chunk for chunk in local_chunks]))

    def get_canon_order(N_mo):
        orb_list = tuple([x for x in range(N_mo)])
        canon_order = dict()
        for i, j, k, l in product(orb_list, orb_list, orb_list, orb_list):  # noqa: E741
            canon_idx = canonical_idx4(i, j, k, l)
            canon_order[canon_idx] = {
                "idx": compound_idx4(*canon_idx),
                "category": integral_category(*canon_idx),
            }
        return canon_order

    N_mo = 32
    canon_order = get_canon_order(N_mo=N_mo)

    for cat in "ABCDEFG":
        try:
            test_chunk(N_mo, cat, canon_order)
            test_chunk_batched(N_mo, cat, canon_order)
        except AssertionError:
            print(f"Failed on category {cat}")

    comm_size = 8
    for cat in "G":
        test_chunk_batched_dist(N_mo, cat, canon_order, comm_size)
