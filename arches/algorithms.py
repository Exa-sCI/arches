# ruff : noqa : E741

from typing import Iterable

import numpy as np
from mpi4py import MPI

from arches.integrals import JChunk
from arches.kernels import (
    launch_denom_kernel,
    launch_H_ii_kernel,
    launch_H_ij_kernel,
    launch_num_kernel,
)
from arches.linked_object import get_indices_by_threshold, get_LArray
from arches.log_timer import ScopedTimer
from arches.matrix import DMatrix, diagonalize, gtsv, qr_factorization


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

    m = X.m
    n = X.n
    s = n // p

    # preallocate Q, R, and T
    Q = DMatrix(m, n, dtype=X.dtype)
    R = DMatrix(n, n, dtype=X.dtype)
    T = DMatrix(n, n, dtype=X.dtype)

    T_block = DMatrix(p, p, np.eye(p, dtype=X.dtype), dtype=X.dtype)
    if (Q_0 is None) or (R_0 is None) or (T_0 is None):
        Q_0 = DMatrix(X.m, p, dtype=X.dtype)
        Q_0[:, :] = X[:, :p]
        R_0 = qr_factorization(Q_0)

        Q[:, :p] = Q_0
        R[:p, :p] = R_0
        T[:p, :p] = T_block
        s_idx = 1
    else:
        # infer start step from size of input guesses
        s_idx = Q_0.n // p
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
        R_k = qr_factorization(Y_k)
        F_k = Q[:, a].T @ Y_k
        G_k = -T[a, a] @ F_k

        Q[:, b] = Y_k

        R[a, b] = H_k
        R[b, b] = R_k

        T[a, b] = G_k
        T[b, b] = T_block

    return Q, R, T


def davidson(
    H,
    *args,
    V_0=None,
    N_states=1,
    l=32,
    pc="D",
    max_subspace_rank=256,
    max_iter=100,
    tol=1e-6,
    rtol=1e-4,
    atol=1e-6,
    **kwargs,
):  # noqa: E741
    """
    Implementation of the Davidson diagonalization algorithm in serial,
    with blocked eigenvector prediction and BMGS.
    """
    V_k = DMatrix(H.m, l, dtype=H.dtype)
    if V_0 is None:
        V_k[:l, :l] = DMatrix.eye(l, dtype=H.dtype)
    else:
        V_k[: V_0.m, :l] = V_0[:, :l]
        V_k[V_0.m : V_0.m + l, :l] = DMatrix.eye(l, dtype=H.dtype)

    bmgs_R_k = DMatrix.eye(l, dtype=H.dtype)
    bmgs_T_k = DMatrix.eye(l, dtype=H.dtype)
    N = H.m
    C_d = H.extract_diagonal()
    if pc == "T":
        C_sd = H.extract_superdiagonal()

    max_subspace_rank = min(max_subspace_rank, H.m)
    for k in range(max_iter):
        ### Project Hamiltonian onto subspace spanned by V_k
        # S_k -> Q_k is diagonalized in place since S_k is not needed again afterwards
        W_k = H @ V_k
        Q_k = V_k.T @ W_k

        ### diagonalize S_k to find (projected) eigenbasis
        w = diagonalize(Q_k)  # L_k is sorted smallest -> largest

        L_k = DMatrix(V_k.n, V_k.n, dtype=H.dtype)
        L_k.fill_diagonal(w)

        ### Form Ritz vectors (in place) and calculate residuals
        R_k = V_k @ Q_k[:, :l] @ L_k[:l, :l] - W_k @ Q_k[:, :l]

        r_norms = R_k.column_2norm()
        r_norms = r_norms.arr.np_arr

        # process early exits
        early_exit = False
        if np.all(r_norms[:N_states] < tol):
            early_exit = True
            print(f"Davidson has converged with tolerance {tol:0.2e} in {k+1} iterations")
            for i in range(N_states):
                print(f"Energy of state {i} : {w.arr[i]:0.6e} Ha. Residual {r_norms[i]:0.6e}")

        elif k == max_iter - 1:
            early_exit = True
            print("Davidson has hit max number of iterations. Exiting early.")
            for i in range(N_states):
                print(f"Energy of state {i} : {w.arr[i]:0.6e} Ha. Residual {r_norms[i]:0.6e}")

        if early_exit:
            # TODO: This works okay in this case because __setitem__ does the bounds checks,
            # but should really implement __getitem__ on LinkedArrays
            # and make sure desired ranges are compatible
            E = w.allocate_result(N_states)
            E.arr[:N_states] = w.arr

            psi = DMatrix(H.m, N_states, dtype=H.dtype)
            psi[:, :] = V_k @ Q_k[:, :N_states]

            V_0 = DMatrix(H.m, l, dtype=H.dtype)
            V_0[: H.m, :] = V_k @ Q_k[:, :l]

            return E, psi, V_0

        ### Calculate next batch of trial vectors
        V_trial = DMatrix(H.m, l, dtype=H.dtype)

        for i in range(l):
            if pc == "D":  # diagonal preconditioner
                # If eigenvalue collides with diagonal value, this can propagate NaNs
                # So replace preconditioner entry and accept a bad vector
                C_ki = get_LArray(C_d.arr.dtype)(N, fill=w.arr[i]) - C_d
                C_ki.reset_near_zeros(np.finfo(H.dtype).eps, 1.0)
                V_trial[:, i] = R_k[:, i] / C_ki
            elif pc == "T":  # tridiagonal preconditioner
                C_d_i = get_LArray(C_d.arr.dtype)(N, fill=w.arr[i]) - C_d
                C_d_i.reset_near_zeros(np.finfo(H.dtype).eps, 1.0)
                C_sd_i = get_LArray(C_d.arr.dtype)(
                    N - 1, fill=C_sd
                )  # need a copy since gtsv overwrites array
                # TODO: I believe the tridiagonal preconditioner should get -C_sd,
                # but it seems to fare better with +C_sd on random matrices...
                # C_sd_i = -C_sd
                gtsv(C_d_i, C_sd_i, R_k[:, i])  # solution written into R_k
                V_trial[:, i] = R_k[:, i]

        ### Form new orthonormal subspace
        reset_subspace = V_k.n + l >= max_subspace_rank
        if reset_subspace:
            new_rank = 2 * l
            V_kk = DMatrix(V_k.m, new_rank, dtype=H.dtype)
            V_kk[:, :l] = V_k @ Q_k[:, :l]
            V_kk[:, l:] = V_trial
            V_k = None
        else:
            new_rank = V_k.n + l
            V_kk = DMatrix(V_k.m, new_rank, dtype=H.dtype)
            V_kk[:, : V_k.n] = V_k
            V_kk[:, V_k.n :] = V_trial

        V_k, bmgs_R_k, bmgs_T_k = bmgs_h(V_kk, l // 2, V_k, bmgs_R_k, bmgs_T_k)

def cipsi(
    E0,
    V_nn,
    int_dets,
    psi_coef,
    J_chunks: Iterable[JChunk],
    *args,
    N_states=1,
    pt2_threshold=1e-4,
    exc_constraint=None,
    **kwargs,
):
    with ScopedTimer(
        log_message=f"Generating connected space for {int_dets.N_dets} internal determinants"
    ):
        ext_dets = int_dets.generate_connected_dets(exc_constraint)

    e_pt2_n = get_LArray(psi_coef.dtype)(ext_dets.N_dets * N_states, fill=0.0)
    e_pt2_d = get_LArray(psi_coef.dtype)(ext_dets.N_dets * N_states, fill=0.0)

    with ScopedTimer(
        log_message=f"Calculating pt2 contribution for {ext_dets.N_dets} external determinants"
    ):
        for chunk in J_chunks:
            kernels = (
                chunk.pt2_kernels
            )  # kernels[0] is numerator contrib., kernels[1] is denom. contrib.
            match chunk.category:
                case "OE" | "F":  # both num and denominator
                    launch_num_kernel(
                        kernels[0], chunk, int_dets, psi_coef, ext_dets, N_states, e_pt2_n
                    )
                    launch_denom_kernel(kernels[1], chunk, ext_dets, N_states, e_pt2_d)
                case "A" | "B":  # only denominator
                    launch_denom_kernel(kernels[1], chunk, ext_dets, N_states, e_pt2_d)
                case _:  # only numerator
                    launch_num_kernel(
                        kernels[0], chunk, int_dets, psi_coef, ext_dets, N_states, e_pt2_n
                    )

    E0_arr = get_LArray(psi_coef.dtype)(ext_dets.N_dets * N_states, E0.arr[0])
    V_nn_arr = get_LArray(psi_coef.dtype)(ext_dets.N_dets * N_states, V_nn)

    e_pt2_d = E0_arr - (V_nn_arr + e_pt2_d)
    e_pt2 = e_pt2_n / e_pt2_d
    e_pt2_total = np.sum(e_pt2.arr.np_arr)

    pt2_filter = get_indices_by_threshold(e_pt2, pt2_threshold)
    int_dets = int_dets.extend_with_filter(ext_dets, pt2_filter)
    # TODO: sort ext dets so that they are in weight order? Or otherwise, pass in information if N_max_dets will be hit

    print(f"Internal det space is now rank {int_dets.N_dets}")
    return int_dets, e_pt2_total


def evaluate_H_entries(H, psi_dets, chunks, E0):
    H.add_to_diagonal(E0)

    for chunk in chunks:
        kernels = chunk.H_kernels
        match chunk.category:
            case "OE" | "F":
                launch_H_ij_kernel(kernels[0], chunk, psi_dets, H)
                launch_H_ii_kernel(kernels[1], chunk, psi_dets, H)
            case "A" | "B":
                launch_H_ii_kernel(kernels[1], chunk, psi_dets, H)
            case _:
                launch_H_ij_kernel(kernels[0], chunk, psi_dets, H)


def prepare_hamiltonian(psi_dets, chunks, E0):
    with ScopedTimer(log_message="Finding non-zero Hamiltonian entries"):
        H = psi_dets.get_H_structure(chunks[0].dtype)

    with ScopedTimer(log_message="Calculating Hamiltonian entries"):
        evaluate_H_entries(H, psi_dets, chunks, E0)

    return H


def get_davidson_work(rank):
    if rank < 64:
        return 8, 2
    elif rank < 512:
        return 32, 8
    else:
        return 256, 8


def s_CI(
    V_nn,
    E0,
    dets,
    psi_coef,
    J_chunks,
    *args,
    constraint=None,
    N_states=1,
    N_max_dets=1e6,
    pt2_conv=1e-6,
    pt2_threshold=1e-7,
    **kwargs,
):
    est_pt2_correction = np.inf
    print(f"Starting selected CI routine with {dets.N_dets} determinants. SCF energy {E0:0.4f} Ha")
    E0 = get_LArray(J_chunks[0].dtype)(1, E0)
    N_rounds = 0
    while dets.N_dets < N_max_dets and np.abs(est_pt2_correction) > pt2_conv:
        print("\n###################\n")
        with ScopedTimer(log_message=f"CIPSI with {dets.N_dets} internal determinants"):
            # expand wavefunction in space of Slater determinants
            dets, est_pt2_correction = cipsi(
                E0,
                V_nn,
                dets,
                psi_coef,
                J_chunks,
                pt2_threshold=pt2_threshold,
                exc_constraint=constraint,
            )

        print(f"Estimated pt2 contribution {est_pt2_correction:0.4e} Ha")
        # find wavefunctions and energies of first N states
        with ScopedTimer(log_message=f"Generating Hamiltonian with rank {dets.N_dets}"):
            H = prepare_hamiltonian(dets, J_chunks, V_nn)

        max_subspace_rank, N_trial_vectors = get_davidson_work(H.m)
        with ScopedTimer(log_message=f"Davidson for sparse H with {H.N_entries} entries"):
            E0, psi_coef, _ = davidson(H, l=N_trial_vectors, max_subspace_rank=max_subspace_rank)

        N_rounds += 1
        print("\n## sCI Iteration Summary")
        for i in range(N_states):
            Evar_with_pt2 = E0.arr[i] + est_pt2_correction
            print(
                f"E{i} after {N_rounds} rounds (Ha): E{i}_var : {E0.arr[i]:0.4f} | E_pt2 : {est_pt2_correction:0.4e}  | E_var + pt2 : {Evar_with_pt2:0.4f}"
            )

    return E0, psi_coef, dets
