#pragma once
#include <array>
#include <determinant.h>
#include <integral_indexing_utils.h>
#include <tuple>

// Abstract storage object for 1 and 2 electron integrals
template <class T> class JChunk {

  public:
    std::vector<T> integrals;           // this will break, how to create constructor?
    spin_det_t orbital_mask;            // cheap bit mask filter for owned orbitals
    std::vector<idx_T> active_orbitals; // list of owned orbitals
    idx_t N_orb;                        // number of orbitals
    T null_J = 0;

    bool det_in_chunk(const spin_det_t &s) { return (active_orbitals & s).any(); }

    bool owns_index(const idx_t i){return (i >= sidx) && (i < eidx)};
}

// one electron integrals
template <class T>
JChunk : OEJ {
  public:
    T &operator[](idx_t ij) { return integrals[ij - internal_offset]; }
    const T &operator[](const idx_t ij) const { return integrals[ij - internal_offset]; }

    T &operator()(idx_t i, idx_t j) {
        idx_t ij = compound_idx2(i, j);
        return owns_idx(ij) ? integrals[ij] : null_J;
    }
    const T &operator()(const idx_t i, const idx_t j) const {
        idx_t ij = compound_idx2(i, j);
        return owns_idx(ij) ? integrals[ij] : null_J;
    }

    // in practice, likely that one electron integrals will be owned by a single worker
    bool owns_index(const idx_t i, const idx_t j) {
        idx_t ij = compound_idx2(i, j);
        return owns_index(ij);
    }
}

// two electron integrals
// integrals are packaged
template <class T> JChunk : TEJ {

  public:
    std::tuple<j_category> owned_categories; // identities of chunk for dispatching

    T &operator[](idx_t ijkl) { return integrals[ijkl - internal_offset]; }
    const T &operator[](const idx_t ijkl) const { return integrals[ijkl - internal_offset]; }

    T &operator()(idx_t i, idx_t j, idx_t k, idx_t l) {
        idx_t ijkl = compound_idx4(i, j, k, l);
        return owns_idx(ijkl) ? integrals[ijkl] : null_J;
    }
    const T &operator()(const idx_t i, const idx_t j, const idx_t k, const idx_t l) const {
        idx_t ijkl = compound_idx4(i, j, k, l);
        return owns_idx(ijkl) ? integrals[ijkl] : null_J;
    }

    bool owns_index(const idx_t i, const idx_t j, const idx_t k, const idx_t l) {
        struct ijkl_tuple ijkl = canonical_idx(i, j, k, l);

        return owns_index(compound_idx4(ijkl));
    }
}

/*
 Functions for performing integral driven calculations
 // TODO: look into unique pointers and evaluate how
*/

//// Dispatch kernel
// void dispatch_chunk()

// one electron contributions
// TODO: pass in result pointer instead of returning
template <class T>
std::vector<T> e_pt2_ii_OE(const OEJ<T> &J, const T E0, const std::vector<det_t> &psi_ext) {
    // Contribution to denominator from one electron integrals
    std::vector<T> res(psi_ext.size(), E0);
    auto N = psi_ext.size();

    // TODO: benchmark ordering of loops
    for (auto &orb : J.active_orbitals) { // loop over all active orbitals
        for (auto det_j = 0; det_j < N; det_j++) {
            auto &ext_det = psi_ext[det_j]; // TODO: make sure this doesn't copy
            res[det_j] += (ext_det.alpha[orb] + ext_det.beta[orb]) *
                          J(orb, orb); // TODO: wasted computation if dets not prefiltered
        }
    }

    return res;
}

// TODO: maybe convert double loops to single loop over 2_idx block
template <class T>
std::vector<T> e_pt2_ij_OE(const OEJ<T> &J, const std::vector<det_t> &psi_int,
                           const std::vector<det_t> &psi_ext) {
    // Contribution to numerator from one electron integrals
    std::vector<T> res(psi_ext.size());

    auto N = psi_ext.size();
    for (auto &orb_i : J.active_orbitals) { // loop over all active orbitals in integrals
        for (auto &orb_j : J_active_orbitals) {

            // loop over internal determinants and check if orb_i is occupied in either spin
            for (auto &int_det : psi_int) {
                // i must be occupied only in int_det; j only in det_j
                bool i_alpha = int_det.alpha[orb_i] && ~int_det.alpha[orb_j];
                bool i_beta = int_det.beta[orb_i] && ~int_det.beta[orb_j];
                if (i_alpha || i_beta) {
                    // loop over external determinants
                    for (auto j = 0; j < N; j++) {
                        auto &ext_det = psi_ext[j];
                        std::array<int, N_species> exc_degree = int_det.exc_degree(ext_det);
                        if ((exc_degree[0] + exc_degree[1]) != 1)
                            continue; // determinants not related by single exc
                        bool j_check = exc_degree[0]
                                           ? (~ext_det.alpha[orb_i] && ext_det.alpha[orb_j])
                                           : (~ext_det.beta[orb_i] && ext_det.beta[orb_j]);
                        if (~j_check)
                            continue; // integral doesn't apply

                        int phase =
                            compute_phase_single_excitation(int_det[exc_degree[1]], orb_i, orb_j);
                        res[j] += phase * J(orb_i, orb_j);
                    }
                }
            }
        }
    }

    return res;
};

// two electron contributions
// TODO: convert double loops to single loop over 4_idx block
template <class T>
std::vector<T> e_pt2_ii_TE(const TEJ<T> &J, const T E0, const std::vector<det_t> &psi_ext) {
    // Contribution to denominator from two electron integrals
    std::vector<T> res(psi_ext.size(), 0);
    auto N = psi_ext.size();

    // TODO: benchmark ordering of loops, which locality wins?
    // one loop for internal combinations of alpha (beta)
    for (auto i = 0; i < J.N_active_orb; i++) { // loop over pairs of active orbitals
        auto orb_i = J.active_orbitals[orb_i];
        // start at i + 1 to restrict to N_orb choose 2
        for (auto j = i + 1; j <= J.N_active_orb; j++) {
            auto orb_j = J.active_orbitals[orb_j];

            // loop over determinants
            // TODO: wasted computation if dets not prefiltered
            for (auto det_j = 0; det_j < N; det_j++) {
                auto &ext_det = psi_ext[det_j];
                // alpha contributions
                res[det_j] += ext_det.alpha[orb_i] * ext_det.alpha[orb_j] *
                              J(orb_i, orb_j, orb_i,
                                orb_j); // multiply by occupancy to ensure combination exists
                res[det_j] -= ext_det.alpha[orb_i] * ext_det.alpha[orb_j] *
                              J(orb_i, orb_j, orb_j, orb_i); // phase implicit in -=/+=

                // beta contributions
                res[det_j] +=
                    ext_det.beta[orb_i] * ext_det.beta[orb_j] * J(orb_i, orb_j, orb_i, orb_j);
                res[det_j] -=
                    ext_det.beta[orb_i] * ext_det.beta[orb_j] * J(orb_i, orb_j, orb_j, orb_i);
            }
        }
    }

    // one loop for product of alpha X beta
    for (auto &orb_i : J.active_orbitals) { // loop over all active orbitals in integrals
        for (auto &orb_j : J_active_orbitals) {
            for (auto det_j = 0; det_j < N; det_j++) {
                auto &ext_det = psi_ext[det_j];
                res[det_j] +=
                    ext_det.alpha[orb_i] * ext_det.beta[orb_j] * J(orb_i, orb_j, orb_i, orb_j);
            }
        }
    }
}

// TODO determine level of const needed

/*
General kernel workflow:
Iterate over all integrals
{
    For ea. J, convert to standard form
    then,
    iterate over all psi_int
    {
        check if integral applies,
        then,
        iterate over all psi_ext
        {
            check if integral applies and check if correct exc. degree/type
            apply integral to result
        }
    }
}

when needed, iterate over spin types in most nested loop
*/

void map_idx_C(const ijkl_tuple idx, idx_t &q, idx_t &r, idx_t &s) {
    if (idx.i == idx.k) {
        q = idx.i;
        r = idx.j;
        s = idx.l;
    } else {
        q = idx.j;
        r = idx.i;
        s = idx.k;
    }
}

/*
C: J_qrqs has the following contributions, singles of form hipi:
    C1s) r_a -> s_a; q occupied in a
    C1o) r_a -> s_a; q occupied in b
    C2s) r_b -> s_b; q occupied in a
    C2o) r_b -> s_b; q occupied in b
    C3s) s_a -> r_a; q occupied in a
    C3o) s_a -> r_a; q occupied in b
    C4s) s_b -> r_b; q occupied in a
    C4o) s_b -> r_b; q occupied in b
*/
template <class T>
void C_pt2_kernel(T *J, idx_t *J_ind, idx_t N, det_t *psi_int, idx_t N_int, det_t *psi_ext,
                  idx_t N_ext, T *res) {
    /*
    J is array of integral values
    J_ind is array of integral compound indices
    N is number of integrals in chunk
    psi_int is array of internal determinants
    N_int is number of internal determinants
    psi_ext is array of external determinants for which we are evaluating pt2 contribution
    N_ext is number of external determinants
    res is output array for pt2 storage
    */

    // Iterate over all integrals in chunk
    for (auto i = 0; i < N; i++) {

        // by construction of the chunks, this should be the canonical index
        struct ijkl_tuple c_idx = compound_idx4_reverse(J_ind[i]);

        // map index to standard form: J_ijil, J_ijkj -> J_qrqs
        idx_t q, r, s;
        map_idx_C(c_idx, q, r, s);

        // iterate over internal determinants
        for (auto d_i = 0; d_i < N_int; d_i++) {
            bool c13, c24, q_ai, q_bi;
            det_t &d_int = psi_int[d_i];
            q_ai = d_int[0][q];
            q_bi = d_int[1][q];
            c13 = (d_int[0][r] != d_int[0][s]) && (q_a || q_b);
            c24 = (d_int[1][r] != d_int[1][s]) && (q_a || q_b);

            if (!(c13 || c24)) // J[i] has no contribution for this internal det
                continue;
            // iterate over external determinants
            for (auto d_e = 0; d_e < N_ext; d_e++) {
                det_t &d_ext = psi_ext[d_e];
                bool q_a = d_ext[0][q] && q_ai;
                bool q_b = d_ext[1][q] && q_bi;
                if (!(q_a || q_b)) // q must be occupied in beta/alpha simultaneously
                    continue;

                // Can probably optimize for early exit: will need to check many things overall
                // But for now, will need to get excitations for every contribution anyway
                // and it is a definite exit point
                det_t exc = exc_det(d_int, d_ext);
                auto degree = (exc[0].count() + exc[1].count()) / 2;
                if (degree != 1) // |d_i> and |d_e> not related by a single excitation
                    continue;

                // TODO: consider branchless? Profile and see
                int phase;
                if (c13 && exc[0][r] && exc[0][s]) {
                    phase = compute_phase_single_excitation(det_i[0], r, s);
                    res[d_e] += q_a * J[i] * phase;
                    res[d_e] += q_b * J[i] * phase;
                }

                if (c24 && exc[1][r] && exc[1][s]) {
                    phase = compute_phase_single_excitation(det_i[1], r, s);
                    res[d_e] += q_a * J[i] * phase;
                    res[d_e] += q_b * J[i] * phase;
                }
            }
        }
    }

    return res;
};

void map_idx_D(const ijkl_tuple idx, idx_t &q, idx_t &r) {
    if (idx.i == idx.j) {
        q = idx.i;
        r = idx.l;
    } else {
        q = idx.j;
        r = idx.i;
    }
}

/*
D: J_qqqr has the following contributions, singles of form hipi(o):
    D_1) q_a -> r_a; q occupied in b
    D_2) q_b -> r_b; q occupied in a
    D_3) r_a -> q_a; q_occupied in b
    D_4) r_b -> q_b; q occupied in a
*/
template <class T>
void D_pt2_kernel(T *J, idx_t *J_ind, idx_t N, det_t *psi_int, idx_t N_int, det_t *psi_ext,
                  idx_t N_ext, T *res) {

    // Iterate over all integrals in chunk
    for (auto i = 0; i < N; i++) {

        // by construction of the chunks, this should be the canonical index
        struct ijkl_tuple c_idx = compound_idx4_reverse(J_ind[i]);

        // map index to standard form: J_ijil, J_ijkj -> J_qrqs
        idx_t q, r;
        map_idx_D(c_idx, q, r);

        // iterate over internal determinants
        for (auto d_i = 0; d_i < N_int; d_i++) {
            det_t &d_int = psi_int[d_i];
            bool d13, d24, q_ai, q_bi;
            q_ai = d_int[0][q];
            q_bi = d_int[1][q];
            d13 = (d_int[0][q] != d_int[0][r]) && q_bi;
            d24 = (d_int[1][q] != d_int[1][r]) && q_ai;

            if (!(d13 || d24)) // J[i] has no contribution
                continue;

            // iterate over external determinants
            for (auto d_e = 0; d_e < N_ext; d_e++) {
                det_t &d_ext = psi_ext[d_e];
                bool q_a = d_ext[0][q] && q_ai;
                bool q_b = d_ext[1][q] && q_bi;
                if (!(q_a || q_b)) // q must be occupied in beta/alpha simultaneously
                    continue;

                det_t exc = exc_det(d_int, d_ext);
                auto degree = (exc[0].count() + exc[1].count()) / 2;
                if (degree != 1) // |d_i> and |d_e> not related by a single excitation
                    continue;

                int phase;
                // include q_(a/b) in check since there is only one contribution
                if (d13 && q_b && exc[0][q] && exc[0][r]) {
                    phase = compute_phase_single_excitation(det_i[0], q, r);
                    res[d_e] += J[i] * phase;
                }

                if (d24 && q_a && exc[1][q] && exc[1][r]) {
                    phase = compute_phase_single_excitation(det_i[1], q, r);
                    res[d_e] += J[i] * phase;
                }
            }
        }
    }
}

template <class T>
void pt2_kernel(T *J, idx_t *J_ind, idx_t N, det_t *psi_int, idx_t N_int, det_t *psi_ext,
                idx_t N_ext, T *res) {

    // Iterate over all integrals in chunk
    for (auto i = 0; i < N; i++) {

        // by construction of the chunks, this should be the canonical index
        struct ijkl_tuple c_idx = compound_idx4_reverse(J_ind[i]);

        // map index to standard form: J_ijil, J_ijkj -> J_qrqs
        idx_t q, r, s, t;
        map_idx(c_idx, q, r, s, t);

        // iterate over internal determinants
        for (auto d_i = 0; d_i < N_int; d_i++) {
            det_t &d_int = psi_int[d_i];

            // iterate over external determinants
            for (auto d_e = 0; d_e < N_ext; d_e++) {
                det_t &d_ext = psi_ext[d_e];
            }
        }
    }
}