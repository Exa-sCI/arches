# ruff : noqa : E741
import math
import pathlib
from ctypes import CDLL, Structure, c_char
from ctypes import c_longlong as idx_t

from arches.func_decorators import offload, return_str, return_tuple

run_folder = pathlib.Path(__file__).parent.resolve()


class ij_tuple(Structure):
    _fields_ = [
        ("i", idx_t),
        ("j", idx_t),
    ]

    @property
    def t(self):
        return (self.i, self.j)


class ijkl_tuple(Structure):
    _fields_ = [
        ("i", idx_t),
        ("j", idx_t),
        ("k", idx_t),
        ("l", idx_t),
    ]

    @property
    def t(self):
        return (self.i, self.j, self.k, self.l)


class ijkl_perms(Structure):
    _fields_ = [
        ("ijkl", ijkl_tuple),
        ("jilk", ijkl_tuple),
        ("klij", ijkl_tuple),
        ("lkji", ijkl_tuple),
        ("ilkj", ijkl_tuple),
        ("lijk", ijkl_tuple),
        ("kjil", ijkl_tuple),
        ("jkli", ijkl_tuple),
    ]

    @property
    def t(self):
        return (
            self.ijkl.t,
            self.jilk.t,
            self.klij.t,
            self.lkji.t,
            self.ilkj.t,
            self.lijk.t,
            self.kjil.t,
            self.jkli.t,
        )


# _____          _           _               _   _ _   _ _
# |_  _|        | |         (_)             | | | | | (_) |
#  | | _ __   __| | _____  ___ _ __   __ _  | | | | |_ _| |___
#  | || '_ \ / _` |/ _ \ \/ / | '_ \ / _` | | | | | __| | / __|
# _| || | | | (_| |  __/>  <| | | | | (_| | | |_| | |_| | \__ \
# \___/_| |_|\__,_|\___/_/\_\_|_| |_|\__, |  \___/ \__|_|_|___/
#                                     __/ |
#                                    |___/

lib_iu = CDLL(run_folder.joinpath("build/libintegral_indexing_utils.so"))

lib_iu.compound_idx2.restype = idx_t
lib_iu.compound_idx2.argtypes = [idx_t, idx_t]

lib_iu.compound_idx4.restype = idx_t
lib_iu.compound_idx4.argtypes = [idx_t, idx_t, idx_t, idx_t]

lib_iu.compound_idx2_reverse.restype = ij_tuple
lib_iu.compound_idx2_reverse.argtypes = [idx_t]

lib_iu.compound_idx4_reverse.restype = ijkl_tuple
lib_iu.compound_idx4_reverse.argtypes = [idx_t]

lib_iu.canonical_idx4.restype = ijkl_tuple
lib_iu.canonical_idx4.argtypes = [idx_t, idx_t, idx_t, idx_t]

lib_iu.compound_idx4_reverse_all.restype = ijkl_perms
lib_iu.compound_idx4_reverse_all.argtypes = [idx_t]

# lib_iu.get_unique_idx4.restype = c_int
# lib_iu.get_unique_idx4.argtypes = [POINTER(ijkl_tuple), ijkl_perms]

lib_it = CDLL(run_folder.joinpath("build/libintegral_types.so"))

lib_it.integral_category.restype = c_char
lib_it.integral_category.argtypes = [idx_t, idx_t, idx_t, idx_t]


@offload(return_str(lib_it.integral_category))
def integral_category(i, j, k, l):
    """
    +-------+-------------------+---------------+-----------------+-------------------------------+------------------------------+-----------------------------+
    | label |                   | ik/jl i/k j/l | i/j j/k k/l i/l | singles                       | doubles                      | diagonal                    |
    +-------+-------------------+---------------+-----------------+-------------------------------+------------------------------+-----------------------------+
    |   A   | i=j=k=l (1,1,1,1) |   =    =   =  |  =   =   =   =  |                               |                              | coul. (1 occ. both spins?)  |
    +-------+-------------------+---------------+-----------------+-------------------------------+------------------------------+-----------------------------+
    |   B   | i=k<j=l (1,2,1,2) |   <    =   =  |  <   >   <   <  |                               |                              | coul. (1,2 any spin occ.?)  |
    +-------+-------------------+---------------+-----------------+-------------------------------+------------------------------+-----------------------------+
    |       | i=k<j<l (1,2,1,3) |   <    =   <  |  <   >   <   <  | 2<->3, 1 occ. (any spin)      |                              |                             |
    |   C   | i<k<j=l (1,3,2,3) |   <    <   =  |  <   >   <   <  | 1<->2, 3 occ. (any spin)      |                              |                             |
    |       | j<i=k<l (2,1,2,3) |   <    =   <  |  >   <   <   <  | 1<->3, 2 occ. (any spin)      |                              |                             |
    +-------+-------------------+---------------+-----------------+-------------------------------+------------------------------+-----------------------------+
    |   D   | i=j=k<l (1,1,1,2) |   <    =   <  |  =   =   <   <  | 1<->2, 1 occ. (opposite spin) |                              |                             |
    |       | i<j=k=l (1,2,2,2) |   <    <   =  |  <   =   =   <  | 1<->2, 2 occ. (opposite spin) |                              |                             |
    +-------+-------------------+---------------+-----------------+-------------------------------+------------------------------+-----------------------------+
    |       | i=j<k<l (1,1,2,3) |   <    <   <  |  =   <   <   <  | 2<->3, 1 occ. (same spin)     | 1a<->2a x 1b<->3b, (and a/b) |                             |
    |   E   | i<j=k<l (1,2,2,3) |   <    <   <  |  <   =   <   <  | 1<->3, 2 occ. (same spin)     | 1a<->2a x 2b<->3b, (and a/b) |                             |
    |       | i<j<k=l (1,2,3,3) |   <    <   <  |  <   <   =   <  | 1<->2, 3 occ. (same spin)     | 1a<->3a x 2b<->3b, (and a/b) |                             |
    +-------+-------------------+---------------+-----------------+-------------------------------+------------------------------+-----------------------------+
    |   F   | i=j<k=l (1,1,2,2) |   =    <   <  |  =   <   =   <  |                               | 1a<->2a x 1b<->2b            | exch. (1,2 same spin occ.?) |
    +-------+-------------------+---------------+-----------------+-------------------------------+------------------------------+-----------------------------+
    |       | i<j<k<l (1,2,3,4) |   <    <   <  |  <   <   <   <  |                               | 1<->3 x 2<->4                |                             |
    |   G   | i<k<j<l (1,3,2,4) |   <    <   <  |  <   >   <   <  |                               | 1<->2 x 3<->4                |                             |
    |       | j<i<k<l (2,1,3,4) |   <    <   <  |  >   <   <   <  |                               | 1<->4 x 2<->3                |                             |
    +-------+-------------------+---------------+-----------------+-------------------------------+------------------------------+-----------------------------+
    """
    if (i, j, k, l) != canonical_idx4(i, j, k, l):
        raise ValueError("Integral indices are not cannonical.")
    if i == l:
        return "A"
    elif (i == k) and (j == l):
        return "B"
    elif (i == k) or (j == l):
        if j == k:
            return "D"
        else:
            return "C"
    elif j == k:
        return "E"
    elif (i == j) and (k == l):
        return "F"
    elif (i == j) or (k == l):
        return "E"
    else:
        return "G"


@offload(lib_iu.compound_idx2)
def compound_idx2(i, j):
    """
    get compound (triangular) index from (i,j)

    (first few elements of lower triangle shown below)
          j
        │ 0   1   2   3
     ───┼───────────────
    i 0 │ 0
      1 │ 1   2
      2 │ 3   4   5
      3 │ 6   7   8   9

    position of i,j in flattened triangle

    >>> compound_idx2(0,0)
    0
    >>> compound_idx2(0,1)
    1
    >>> compound_idx2(1,0)
    1
    >>> compound_idx2(1,1)
    2
    >>> compound_idx2(1,2)
    4
    >>> compound_idx2(2,1)
    4
    """
    p, q = min(i, j), max(i, j)
    return (q * (q + 1)) // 2 + p


@offload(lib_iu.compound_idx4)
def compound_idx4(i, j, k, l):
    """
    nested calls to compound_idx2
    >>> compound_idx4(0,0,0,0)
    0
    >>> compound_idx4(0,1,0,0)
    1
    >>> compound_idx4(1,1,0,0)
    2
    >>> compound_idx4(1,0,1,0)
    3
    >>> compound_idx4(1,0,1,1)
    4
    """
    return compound_idx2(compound_idx2(i, k), compound_idx2(j, l))


@offload(return_tuple(lib_iu.compound_idx2_reverse))
def compound_idx2_reverse(ij):
    """
    inverse of compound_idx2
    returns (i, j) with i <= j
    >>> compound_idx2_reverse(0)
    (0, 0)
    >>> compound_idx2_reverse(1)
    (0, 1)
    >>> compound_idx2_reverse(2)
    (1, 1)
    >>> compound_idx2_reverse(3)
    (0, 2)
    """
    assert (1 + 8 * ij) >= 0
    j = (math.isqrt(1 + 8 * ij) - 1) // 2
    i = ij - (j * (j + 1) // 2)
    return i, j


@offload(return_tuple(lib_iu.compound_idx4_reverse))
def compound_idx4_reverse(ijkl):
    """
    inverse of compound_idx4
    returns (i, j, k, l) with ik <= jl, i <= k, and j <= l (i.e. canonical ordering)
    where ik == compound_idx2(i, k) and jl == compound_idx2(j, l)
    >>> compound_idx4_reverse(0)
    (0, 0, 0, 0)
    >>> compound_idx4_reverse(1)
    (0, 0, 0, 1)
    >>> compound_idx4_reverse(2)
    (0, 0, 1, 1)
    >>> compound_idx4_reverse(3)
    (0, 1, 0, 1)
    >>> compound_idx4_reverse(37)
    (0, 2, 1, 3)
    """
    ik, jl = compound_idx2_reverse(ijkl)
    i, k = compound_idx2_reverse(ik)
    j, l = compound_idx2_reverse(jl)
    return i, j, k, l


@offload(return_tuple(lib_iu.compound_idx4_reverse_all))
def compound_idx4_reverse_all(ijkl):
    """
    return all 8 permutations that are equivalent for real orbitals
    returns 8 4-tuples, even when there are duplicates
    for complex orbitals, they are ordered as:
    v, v, v*, v*, u, u, u*, u*
    where v == <ij|kl>, u == <ij|lk>, and * denotes the complex conjugate
    >>> compound_idx4_reverse_all(0)
    ((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
    >>> compound_idx4_reverse_all(1)
    ((0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0))
    >>> compound_idx4_reverse_all(37)
    ((0, 2, 1, 3), (2, 0, 3, 1), (1, 3, 0, 2), (3, 1, 2, 0), (0, 3, 1, 2), (3, 0, 2, 1), (1, 2, 0, 3), (2, 1, 3, 0))
    """
    i, j, k, l = compound_idx4_reverse(ijkl)
    return (
        (i, j, k, l),
        (j, i, l, k),
        (k, l, i, j),
        (l, k, j, i),
        (i, l, k, j),
        (l, i, j, k),
        (k, j, i, l),
        (j, k, l, i),
    )


# TODO: Still debugging implementation, but not used anywhere, so moving on for now.
def compound_idx4_reverse_all_unique(ijkl):
    """
    return only the unique 4-tuples from compound_idx4_reverse_all
    """
    return tuple(set(compound_idx4_reverse_all(ijkl)))
    # perms = indexing_utils_lib.compound_idx4_reverse_all(ijkl)
    # u_set = (ijkl_tuple*8)()

    # N = indexing_utils_lib.get_unique_idx4(byref(u_set), perms)
    # ref_t = perms.t
    # print("/n")
    # for i in range(N):
    #     print(ref_t[i], u_set[i].t)
    # print(f"N: {N}")
    # return tuple([u_set[i].t for i in range(N)])


@offload(return_tuple(lib_iu.canonical_idx4))
def canonical_idx4(i, j, k, l):
    """
    for real orbitals, return same 4-tuple for all equivalent integrals
    returned (i,j,k,l) should satisfy the following:
        i <= k
        j <= l
        (k < l) or (k==l and i <= j)
    the last of these is equivalent to (compound_idx2(i,k) <= compound_idx2(j,l))
    >>> canonical_idx4(1, 0, 0, 0)
    (0, 0, 0, 1)
    >>> canonical_idx4(4, 2, 3, 1)
    (1, 3, 2, 4)
    >>> canonical_idx4(3, 2, 1, 4)
    (1, 2, 3, 4)
    >>> canonical_idx4(1, 3, 4, 2)
    (2, 1, 3, 4)
    """
    i, k = min(i, k), max(i, k)
    ik = compound_idx2(i, k)
    j, l = min(j, l), max(j, l)
    jl = compound_idx2(j, l)
    if ik <= jl:
        return i, j, k, l
    else:
        return j, i, l, k
