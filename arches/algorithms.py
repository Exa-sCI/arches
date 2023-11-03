# ruff : noqa : E741

from typing import Iterable

import numpy as np
from mpi4py import MPI
from scipy.linalg import lapack

from arches.fundamental_types import Determinant
from arches.integrals import JChunk
from arches.matrix import AMatrix, diagonalize


def bmgs_h(X, p, Q_0=None, R_0=None, T_0=None):
    """
    Block version of Modified Gram-Schmidt orthogonalization
    as described in Barlow, 2019 (https://doi.org/10.1137/18M1197400)

    Relates MGS via Househoulder QR to pose MGS
    via MM operations instead of VV or MV products

    TODO: Handle case when n // p != n / p, i.e., number of columns not divisible by block size
    Should be able to cleanly pad by zeros

    TODO: Possible to write in-place version?

    Args:
        X (array)   : m by n matrix to orthogonalize
        p (int)     : size of block
        Q_0 (array) : m by (s*p) matrix, initial s*p orthogonal vectors
        R_0 (array) : (s*p) by (s*p) matrix, initial R matrix s.t. X = Q @ R
        T_0 (array) : (s*p) by (s*p) matrix, initial T matrix s.t. T = (triu(Q.T @ Q))^-1
    """

    m, n = X.shape
    s = n // p

    # preallocate Q, R, and T
    Q = np.zeros_like(X)
    R = np.zeros((n, n))
    T = np.zeros((n, n))

    if (Q_0 is None) or (R_0 is None) or (T_0 is None):
        Q_0, R_0 = np.linalg.qr(X[:, :p], mode="reduced")
        T_0 = np.eye(p)
        Q[:, :p] = Q_0
        R[:p, :p] = R_0
        T[:p, :p] = T_0
        s_idx = 1
    else:
        # infer start step from size of input guesses
        s_idx = Q_0.shape[1] // p
        ss = slice(int(s_idx * p))
        Q[:, ss] = Q_0
        R[ss, ss] = R_0
        T[ss, ss] = T_0

    for k in range(s_idx, s):
        a = slice(k * p)
        b = slice(k * p, (k + 1) * p)

        H_k = Q[:, a].T @ X[:, b]
        H_k = T[a, a].T @ H_k

        Y_k = X[:, b] - Q[:, a] @ H_k
        Q_kk, R_kk = np.linalg.qr(Y_k, mode="reduced")  # any fast, stable QR can replace this
        F_k = Q[:, a].T @ Q_kk
        G_k = -T[a, a] @ F_k

        Q[:, b] = Q_kk

        R[a, b] = H_k
        R[b, b] = R_kk

        T[a, b] = G_k
        T[b, b] = np.eye(p)

    return Q, R, T


def davidson(H, *args, V_0=None, N_states=1, l=32, pc="D", max_iter=100, tol=1e-6, **kwargs):  # noqa: E741
    """
    Implementation of the Davidson diagonalization algorithm in serial,
    with blocked eigenvector prediction and BMGS.
    """
    if V_0 is None:
        V_k = np.zeros((H.shape[0], l))
        V_k[:l, :l] = np.eye(l)
    else:
        V_k = V_0

    R_k = np.eye(l)
    T_k = np.eye(l)
    N = H.shape[0]
    C_d = np.diag(H)
    if pc == "T":
        C_sd = np.zeros(N - 1)
        for i in range(N - 1):
            C_sd[i] = H[i, i + 1]

        ptsvx = lapack.get_lapack_funcs("ptsvx", dtype=H.dtype)

    for k in range(max_iter):
        ### Project Hamiltonian onto subspace spanned by V_k
        W_k = H @ V_k
        S_k = V_k.T @ W_k

        ### diagonalize S_k to find (projected) eigenbasis
        L_k, Q_k = diagonalize(S_k)  # L_k is sorted smallest -> largest

        ### Form Ritz vectors (in place) and calculate residuals
        R_k = V_k @ Q_k[:, :l] @ L_k[:l, :l] - W_k @ Q_k[:, :l]

        r_norms = np.linalg.norm(R_k, axis=0)

        # process early exits
        early_exit = False
        if np.all(r_norms[:N_states] < tol):
            early_exit = True
            print(f"Davidson has converged with tolerance {tol:0.4e}")
            for i in range(N_states):
                print(f"Energy of state {i}:{L_k[i]} Ha")

        elif k == max_iter - 1:
            early_exit = True
            print("Davidson has hit max number of iterations. Exiting early.")
            for i in range(N_states):
                print(f"Energy of state {i}:{L_k[i]} Ha. Residual {r_norms[i]}")

        if early_exit:
            return L_k[:N_states], Q_k[:, :N_states]

        ### Calculate next batch of trial vectors
        # TODO: get rid of allocatons in loop
        V_kk = np.zeros((H.shape[0], l))

        for i in range(l):
            if pc == "D":  # diagonal preconditioner
                C_ki = np.ones(N) * L_k[i] - C_d
                C_ki = 1.0 / C_ki
                V_kk[:, i] = C_ki * R_k[:, i]
            elif pc == "T":  # tridiagonal preconditioner
                C_d_i = np.ones(N) * L_k[i] - C_d
                res = ptsvx(C_d_i, C_sd, R_k[:, i])
                V_kk[:, i] = res[0]

        ### Form new orthonormal subspace
        V_k, R_k, T_k = bmgs_h(np.hstack([V_k, V_kk]), l // 2, V_k, R_k, T_k)


def davidson_par_0(
    H, comm, *args, V_0=None, N_states=1, l=32, pc="D", max_iter=100, tol=1e-6, **kwargs
):  # noqa: E741
    """
    Implementation of the Davidson diagonalization algorithm in parallel.

    Assumes each worker i has partial component of H s.t. H = \sum_i H_i

    Assume that subspace is small enough that diagonalization/BMGS are fast enough locally
    to outspeed communicating
    """

    if V_0 is None:
        V_k = np.zeros((H.shape[0], l))
        V_k[:l, :l] = np.eye(l)
    else:
        V_k = V_0

    R_k = np.eye(l)
    T_k = np.eye(l)
    N = H.shape[0]

    ## TODO: Diagonal and subdiagonal need to be communicated
    C_d = np.diag(H)
    if pc == "T":
        C_sd = np.zeros(N - 1)
        for i in range(N - 1):
            C_sd[i] = H[i, i + 1]

        ptsvx = lapack.get_lapack_funcs("ptsvx", dtype=H.dtype)

    rank = comm.Get_rank()
    assert l == comm.Get_size()

    for k in range(max_iter):
        ### Project Hamiltonian onto subspace spanned by V_k
        W_k = H @ V_k
        comm.Allreduce(MPI.IN_PLACE, [W_k, MPI.DOUBLE])  ## Synchronization

        S_k = V_k.T @ W_k

        ### diagonalize S_k to find (projected) eigenbasis
        # assume S_k \in R^(kl, kl) is small enough to diagonalize locally
        L_k, Q_k = diagonalize(S_k)  # L_k is sorted smallest -> largest

        ### Calculate residual of owned Ritz vector
        R_k = L_k[rank] * V_k @ Q_k[:, rank] - W_k @ Q_k[:, rank]

        r_norm = np.linalg.norm(R_k)
        r_norms = np.zeros(l)
        comm.Allgather([r_norm, MPI.DOUBLE], [r_norms, MPI.DOUBLE])  ## Synchronization

        # process early exits
        early_exit = False
        if np.all(r_norms[:N_states] < tol):
            early_exit = True
            if rank == 0:
                print(f"Davidson has converged with tolerance {tol:0.4e}")
                for i in range(N_states):
                    print(f"Energy of state {i}:{L_k[i]} Ha")
        elif k == max_iter - 1:
            early_exit = True
            if rank == 0:
                print("Davidson has hit max number of iterations. Exiting early.")
                for i in range(N_states):
                    print(f"Energy of state {i}:{L_k[i]} Ha. Residual {r_norms[i]}")

        if early_exit:
            return L_k[:N_states], Q_k[:, :N_states]

        ### Calculate next batch of trial vectors
        # TODO: get rid of allocatons in loop
        V_kk = np.zeros((H.shape[0], l))

        if pc == "D":  # diagonal preconditioner
            C_ki = np.ones(N) * L_k[rank] - C_d
            C_ki = 1.0 / C_ki
            V_kk[:, rank] = C_ki * R_k
        elif pc == "T":  # tridiagonal preconditioner
            C_d_i = np.ones(N) * L_k[rank] - C_d
            res = ptsvx(C_d_i, C_sd, R_k)
            V_kk[:, rank] = res[0]

        comm.Allgather(MPI.IN_PLACE, [V_kk, MPI.DOUBLE])

        ### Form new orthonormal subspace
        V_k, R_k, T_k = bmgs_h(np.hstack([V_k, V_kk]), l // 2, V_k, R_k, T_k)


def get_connected_dets(int_dets: Iterable[Determinant]) -> Iterable[Determinant]:
    raise NotImplementedError


def cipsi(
    E0,
    int_dets,
    psi_coef,
    J_chunks: Iterable[JChunk],
    *args,
    pt2_threshold=1e-8,
    exc_constraints=None,
    **kwargs,
):
    ext_dets = get_connected_dets(int_dets, exc_constraints)

    e_pt2_n = np.array(len(ext_dets))  # TODO: convert these arrays to LinkedArrays
    e_pt2_d = np.array(len(ext_dets))  # TODO: initialize with E0

    for chunk in J_chunks:
        kernels = (
            chunk.pt2_kernels
        )  # kernels[0] is numerator contrib., kernels[1] is denom. contrib.
        match chunk.category:
            case "OE" | "F":  # both num and denominator
                kernels[0](chunk, int_dets, psi_coef, ext_dets, e_pt2_n)
                kernels[1](chunk, ext_dets, e_pt2_d, E0)
            case "A" | "B":  # only denominator
                kernels[1](chunk, ext_dets, e_pt2_d, E0)
            case _:  # only numerator
                kernels[0](chunk, int_dets, psi_coef, ext_dets, e_pt2_n)

    # reduce e_pt2_n,d over processes
    e_pt2 = e_pt2_n / e_pt2_d  # TODO: implement +, -, /, * for LinkedArrays
    pt2_filter = e_pt2 > pt2_threshold  # TODO: pass this off to customizable selection function
    e_pt2_total = np.sum(e_pt2)

    # TODO: sort ext dets so that they are in weight order? Or otherwise, pass in information if N_max_dets will be hit
    return int_dets + ext_dets[pt2_filter], e_pt2_total


def prepare_hamiltonian(psi_dets, H_partial: AMatrix = None) -> AMatrix:
    raise NotImplementedError


def s_CI(
    E0,
    dets,
    psi_coef,
    J_chunks,
    *args,
    N_states=1,
    N_max_dets=1e6,
    pt2_conv=1e-4,
    **kwargs,
):
    est_pt2_correction = np.inf

    while len(dets) < N_max_dets and est_pt2_correction > pt2_conv:
        # expand wavefunction in space of Slater determinants
        dets, est_pt2_correction = cipsi(E0, dets, psi_coef, J_chunks)

        # find wavefunctions and energies of first N states
        H = prepare_hamiltonian(dets)
        E0, psi_coef = davidson(H)

    return E0, psi_coef, dets
