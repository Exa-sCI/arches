from functools import reduce
from dataclasses import dataclass
from itertools import islice, combinations, product
import numpy as np
import numpy.typing as npt
from qe.integral_indexing_utils import (
    compound_idx4,
    canonical_idx4,
)
from qe.drivers import integral_category


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


@dataclass
class JChunk:
    chunk_size: int
    J: npt.NDArray[[np.float32, np.float64]]
    idx: np.ndarray[[np.int32, np.int64]]
    category: str

    def __getitem__(self, idx):
        if idx > 0 and idx < self.chunk_size:
            return self.J[idx], self.idx[idx]
        else:
            raise ValueError


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
        if (
            self.batched
        ):  # this batching procedure is a little ugly but it works for now
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

    def test_chunk_batched_dist(
        N_mo, cat, ref_data, comm_size, src_data=IntegralReader()
    ):
        all_ind = set([v["idx"] for k, v in ref_data.items() if v["category"] == cat])

        local_chunks = []
        for rank in range(comm_size):
            comm = FakeComm(rank, comm_size)
            fact = JChunkFactory(N_mo, cat, src_data, comm=comm, chunk_size=2048)
            chunks = fact.get_chunks()
            local_chunk = set(
                reduce(lambda x, y: x.union(y), [set(chunk.idx) for chunk in chunks])
            )
            local_chunks.append(local_chunk)

        assert all_ind == set(
            reduce(lambda x, y: x.union(y), [chunk for chunk in local_chunks])
        )

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
