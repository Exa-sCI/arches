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
    T &operator[](idx_t ij) { return integrals[ij]; }
    const T &operator[](const idx_t ij) const { return integrals[ij]; }

    T &operator()(idx_t i, idx_t j) {
        idx_t ij = compound_idx2(i, j);
        return integrals[ij];
        // return owns_idx(ij) ? integrals[ij] : null_J;
    }
    const T &operator()(const idx_t i, const idx_t j) const {
        idx_t ij = compound_idx2(i, j);
        return integrals[ij];
        // return owns_idx(ij) ? integrals[ij] : null_J;
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
 Kernels for performing integral driven calculations

 Intended strategy: since G >> (A,B,C,D,E,F, and OE), root worker will get all non-G and small G
 chunk; every other worker distributes G evenly.

 root worker will iterate through all of its work and everyone else will just dispatch to G kernel
*/

// One electron contributions
template <class T>
void e_pt2_ii_OE(T *J, idx_t N_orb, const T E0, det_t *psi_ext, idx_t N_ext, T *res) {
    // Contribution to denominator from one electron integrals

    for (auto det_j = 0; det_j < N_ext; det_j++) {
        res[det_j] = E0;                         // TODO: perhaps initialize denom with E0 outside
        for (auto orb = 0; orb < N_orb; orb++) { // loop over all active orbitals
            auto &ext_det = psi_ext[det_j];
            idx_t idx = compound_idx2(orb, orb);
            res[det_j] += (ext_det.alpha[orb] + ext_det.beta[orb]) * J[idx];
        }
    }
}

template <class T>
void e_pt2_ij_OE(T *J, idx_t N, det_t *psi_int, idx_t N_int, det_t *psi_ext, idx_t N_ext, T *res) {
    // Contribution to numerator from one electron integrals
    for (auto i = 0; i < N; i++) { // loop over all integrals

        struct ij_tuple ij = compound_idx2_reverse(i);
        // loop over internal determinants and check if orb_i is occupied in either spin
        for (auto det_i = 0; det_i < N_int; det_i++) {

            // i must be occupied only in int_det; j only in det_j
            bool i_alpha = int_det.alpha[ij.i] && ~int_det.alpha[ij.j];
            bool i_beta = int_det.beta[ij.i] && ~int_det.beta[ij.j];
            if (!(i_alpha || i_beta))
                continue;

            // loop over external determinants
            for (auto det_j = 0; det_j < N; det_j++) {
                auto &ext_det = psi_ext[det_j];
                std::array<int, N_species> exc_degree = int_det.exc_degree(ext_det);
                if ((exc_degree[0] + exc_degree[1]) != 1)
                    continue; // determinants not related by single exc

                bool j_check = exc_degree[0] ? (~ext_det.alpha[ij.i] && ext_det.alpha[ij.j])
                                             : (~ext_det.beta[ij.i] && ext_det.beta[ij.j]);
                if (~j_check)
                    continue; // integral doesn't apply

                int phase = compute_phase_single_excitation(int_det[exc_degree[1]], ij.i, ij.j);
                res[det_j] += phase * J[i];
            }
        }
    }

    return res;
};

// Two electron contributions
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

TODO: determine level of const needed
*/

/*
A: J_qqqq only has one contribution (to the denominator), when q is occupied in both spins
Contribution is part of product terms
*/
template <class T> void A_pt2_kernel(T *J, idx_t N, det_t *psi_ext, idx_t N_ext, T *res) {
    // Contributes to denominator of pt2 energy

    // iterate over external determinants first since A chunk is almost always smaller
    for (auto d_e = 0; d_e < N_ext; d_e++) {
        auto &ext_det = psi_ext[d_e];

        // Iterate over all integrals in chunk (should be N_orb integrals)
        // order of integrals in chunk is known by construction
        for (auto i = 0; i < N; i++) {
            res[d_e] += ext_det[0][i] * ext_det[1][i] * J[i];
        }
    }
}

/*
B: J_qqrr has the following contributions (to the denominator):
    B_1) q_a -> q_a, r_b -> r_b
    B_2) r_a -> r_a, q_b -> q_b

Contributions are part of combination terms
*/
temp template <class T>
void B_pt2_kernel(T *J, idx_t *J_ind, idx_t N, det_t *psi_ext, idx_t N_ext, T *res) {
    // Contributes to denominator of pt2 energy
    for (auto i = 0; i < N; i++) {
        struct ijkl_tuple c_idx = compound_idx4_reverse(J_ind[i]);

        // index already in standard form: J_ijij -> J_qrqr
        idx_t q, r;
        q = c_idx.i;
        r = c_idx.j;

        // iterate over external determinants
        for (auto d_e = 0; d_e < N_ext; d_e++) {
            auto &ext_det = psi_ext[d_e];
            res[d_e] += ext_det[0][q] * ext_det[0][r] * J[i];
            res[d_e] += ext_det[1][q] * ext_det[1][r] * J[i];
        }
    }
}

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
            det_t &d_int = psi_int[d_i];
            bool c13, c24, q_ai, q_bi;
            q_ai = d_int[0][q];
            q_bi = d_int[1][q];
            c13 = (d_int[0][r] != d_int[0][s]) && (q_ai || q_bi);
            c24 = (d_int[1][r] != d_int[1][s]) && (q_ai || q_bi);

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

        // map index to standard form: J_iiil, J_ijjj -> J_qqqr
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

void map_idx_E(const ijkl_tuple idx, idx_t &q, idx_t &r, idx_t &s) {
    if (idx.i == idx.j) {
        q = idx.i;
        r = idx.k;
        s = idx.l;
    } else if (idx.j == idx_k) {
        q = idx.j;
        r = idx.i;
        s = idx.l;
    } else {
        q = idx.k;
        r = idx.i;
        s = idx.j;
    }
}

/*
E: J_qqrs has the following contributions,
    Singles of form hiip:
        E_1) r_a -> s_a; q occupied in a
        E_2) r_b -> s_b; q occupied in b
        E_3) s_a -> r_a; q_occupied in a
        E_4) s_b -> r_b; q occupied in b

    Oposite spin doubles:
        E_a) q_a -> r_a | q_b -> s_b
        E_d) r_a -> q_a | s_b -> q_b
        E_e) q_a -> r_a | s_b -> q_b
        E_g) r_a -> q_a | q_b -> s_b

        E_b) q_a -> s_a | q_b -> r_b
        E_c) s_a -> q_a | r_b -> q_b
        E_f) s_a -> q_a | q_b -> r_b
        E_h) q_a -> s_a | r_b -> q_b

*/
template <class T>
void E_pt2_kernel(T *J, idx_t *J_ind, idx_t N, det_t *psi_int, idx_t N_int, det_t *psi_ext,
                  idx_t N_ext, T *res) {

    // Iterate over all integrals in chunk
    for (auto i = 0; i < N; i++) {

        // by construction of the chunks, this should be the canonical index
        struct ijkl_tuple c_idx = compound_idx4_reverse(J_ind[i]);

        // map index to standard form: J_iikl, J_ijjl, J_ijkk -> J_qqrs
        idx_t q, r, s;
        map_idx_E(c_idx, q, r, s);

        // iterate over internal determinants
        for (auto d_i = 0; d_i < N_int; d_i++) {
            det_t &d_int = psi_int[d_i];
            bool e13, e24, e_adeg, e_bcfh, q_ai, q_bi;
            q_ai = d_int[0][q];
            q_bi = d_int[1][q];
            e_13 = (d_int[0][r] != d_int[0][s]) && q_ai;
            e_24 = (d_int[1][r] != d_int[1][s]) && q_bi;

            e_adeg = (d_int[0][q] != d_int[0][r]) && (d_int[1][q] != d_int[1][s]);
            e_bcfh = (d_int[0][q] != d_int[0][r]) && (d_int[1][q] != d_int[1][s]);

            if (!(e_13 || e_24 || e_adeg || e_bcfh)) // J[i] has no contribution
                continue;

            // iterate over external determinants
            for (auto d_e = 0; d_e < N_ext; d_e++) {
                det_t &d_ext = psi_ext[d_e];

                // early exit on exc degree is probably simplest
                det_t exc = exc_det(d_int, d_ext);
                auto degree = (exc[0].count() + exc[1].count()) / 2;
                if (degree > 2) // |d_i> and |d_e> not connnected, assumes d_i != d_e
                    continue;

                int phase;
                if (degree == 1) {
                    bool q_a = d_ext[0][q] && q_ai;
                    bool q_b = d_ext[1][q] && q_bi;
                    if (!(q_a || q_b)) // q must be occupied in beta/alpha simultaneously
                        continue;

                    if (e13 && q_a && exc[0][r] && exc[0][s]) {
                        phase = compute_phase_single_excitation(det_i[0], r, s);
                        res[d_e] += J[i] * phase * -1;
                    }

                    if (e24 && q_b && exc[1][r] && exc[1][s]) {
                        phase = compute_phase_single_excitation(det_i[0], r, s);
                        res[d_e] += J[i] * phase * -1;
                    }
                } else if (degree == 2) {

                    if (exc[0].count() == 0 || exc[1].count() == 0) // must be opp. spin double
                        continue;

                    if (!(exc[0][q] && exc[1][q])) // q must be involved in both spins
                        continue;

                    // adeg
                    if (exc[0][r] && exc[1][s]) {
                        phase = compute_phase_double_excitation(d_int, q, q, r, s);
                        res[d_e] += J[i] * phase;
                    }

                    // bcfh
                    if (exc[0][s] && exc[1][r]) {
                        phase = compute_phase_double_excitation(d_int, q, q, s, r);
                        res[d_e] += J[i] * phase;
                    }
                }
            }
        }
    }
}

/*
F: J_qqrr has the following (off diagonal) contributions, all opposite spin doubles:
    F_1) q_a -> r_a | q_b -> r_b
    F_2) r_a -> q_a | r_b -> q_b
    F_3) q_a -> r_a | r_b -> q_b
    F_4) r_a -> q_a | q_b -> r_b
*/
template <class T>
void F_pt2_kernel(T *J, idx_t *J_ind, idx_t N, det_t *psi_int, idx_t N_int, det_t *psi_ext,
                  idx_t N_ext, T *res) {

    // Iterate over all integrals in chunk
    for (auto i = 0; i < N; i++) {

        // by construction of the chunks, this should be the canonical index
        struct ijkl_tuple c_idx = compound_idx4_reverse(J_ind[i]);

        // index already in standard form: J_iikk -> J_qqrr
        idx_t q, r;
        q = c_idx.i;
        r = c_idx.k;

        // iterate over internal determinants
        for (auto d_i = 0; d_i < N_int; d_i++) {
            det_t &d_int = psi_int[d_i];

            bool i_check = (d_int[0][q] != d_int[0][r]) && (d_int[1][q] != d_int[1][r]);

            if (!i_check) // J[i] has no contribution
                continue;

            // iterate over external determinants
            for (auto d_e = 0; d_e < N_ext; d_e++) {
                det_t &d_ext = psi_ext[d_e];
                det_t exc = exc_det(d_int, d_ext);
                auto a_degree = exc[1].count() / 2;
                auto b_degree = exc[0].count() / 2;
                if (!((a_degree == 1) && (b_degree == 1))) // must be opp. spin double
                    continue;

                if (exc[0][q] && exc[0][r] && exc[1][q] && exc[1][r]) {
                    int phase = compute_phase_double_excitation(d_int, q, q, r, r);
                    res[d_e] += J[i] * phase;
                }
            }
        }
    }
}

template <class T>
void F_pt2_kernel_denom(T *J, idx_t *J_ind, idx_t N, det_t *psi_ext, idx_t N_ext, T *res) {
    // Contributions to combination terms in denominator

    // Iterate over all integrals in chunk
    for (auto i = 0; i < N; i++) {

        // by construction of the chunks, this should be the canonical index
        struct ijkl_tuple c_idx = compound_idx4_reverse(J_ind[i]);

        // index already in standard form: J_iikk -> J_qqrr
        idx_t q, r;
        q = c_idx.i;
        r = c_idx.k;

        // iterate over external determinants
        for (auto d_e = 0; d_e < N_ext; d_e++) {
            det_t &d_ext = psi_ext[d_e];
            res[d_e] -= d_ext[0][q] * d_ext[0][r] * J[i]; // phase implicit in -=
            res[d_e] -= d_ext[1][q] * d_ext[1][r] * J[i];
        }
    }
}

/*
This is the most expensive kernel, and by far where most compute time will be spent.
At 64 MOs, G integrals comprise ~88% of all unique integrals (C, E ~5.8% ea.)
At 128 MOs, G integrals comprise ~94% of all unique integrals (C, E ~3% ea.)

G:
    G_A := q < r < s < t
    G_B := q < s < r < t
    G_C := r < q < s < t

    J_qrst has the following contributions:
    Opposite spin doubles: (h1, h2, p1, p2)*(phase_a * phase_b)
        G_a) q_a -> s_a | r_b -> t_b
        G_c) s_a -> q_a | t_b -> r_b
        G_e) q_a -> s_a | t_b -> r_b
        G_g) s_a -> q_a | r_b -> t_b

        G_b) r_a -> t_a | q_b -> s_b
        G_d) t_a -> r_a | s_b -> q_b
        G_f) t_a -> r_a | q_b -> s_b
        G_h) r_a -> t_a | s_b -> q_b

    Same spin doubles: (h1, h2, p1, p2)*(phase) - (h1, h2, p2, p1)*(phase)
        G_11) q,r -> s,t | A, B    || q r !s !t, C is complement integral
              r,q -> t,s | C

        G_22) s,t -> q,r | A, B, C || s t !q !r, C is complement integral

        G_33) q,t -> s,r | A, B, C || q t !s !r, B is complement integral

        G_44) r,s -> t,q | A, C    || r s !q !t, B is complement integral
              s,r -> q,t | B

*/
template <class T>
void G_pt2_kernel(T *J, idx_t *J_ind, idx_t N, det_t *psi_int, idx_t N_int, det_t *psi_ext,
                  idx_t N_ext, T *res) {

    // Iterate over all integrals in chunk
    for (auto i = 0; i < N; i++) {

        // by construction of the chunks, this should be the canonical index
        struct ijkl_tuple c_idx = compound_idx4_reverse(J_ind[i]);

        // index already in standard form: J_ijkl -> J_qrst
        idx_t q, r, s, t;
        q = c_idx.i;
        r = c_idx.j;
        s = c_idx.k;
        t = c_idx.l;

        // iterate over internal determinants
        for (auto d_i = 0; d_i < N_int; d_i++) {
            det_t &d_int = psi_int[d_i];

            // checks for opposite spin doubles
            bool g_aceg, g_bdfh;
            g_aceg = (d_int[0][q] != d_int[0][s]) && (d_int[1][r] != d_int[1][t]);
            g_bdfh = (d_int[0][r] != d_int[0][t]) && (d_int[1][q] != d_int[1][s]);

            // checks for same spin doubles
            // g_ii in (0,1,2,3) :
            // 0 - none, 1 - only alpha 2- only beta 3 - both alpha and beta
            int g_11, g_22, g_33, g_44 = 0;
            for (auto spin = 0; spin < N_SPIN_SPECIES; spin++) {
                g_11 = (spin + 1) *
                       ((d_int[spin][q] && d_int[spin][r]) && (!d_int[spin][s] && !d_int[spin][t]));
                g_22 = (spin + 1) *
                       ((d_int[spin][s] && d_int[spin][t]) && (!d_int[spin][q] && !d_int[spin][r]));
                g_33 = (spin + 1) *
                       ((d_int[spin][q] && d_int[spin][t]) && (!d_int[spin][s] && !d_int[spin][r]));
                g_44 = (spin + 1) *
                       ((d_int[spin][r] && d_int[spin][s]) && (!d_int[spin][q] && !d_int[spin][t]));
            }

            if (!(g_aceg || g_bdfh || ((g_11 + g_22 + g_33 + g44) > 0))) // J[i] has no contribution
                continue;

            // iterate over external determinants
            for (auto d_e = 0; d_e < N_ext; d_e++) {
                det_t &d_ext = psi_ext[d_e];

                det_t exc = exc_det(d_int, d_ext);
                auto a_degree = exc[0].count() / 2;
                auto degree = a_degree + exc[1].count() / 2;
                if (degree != 2) // |d_i> and |d_e> not connnected by double exc.
                    continue;

                // TODO: profile branching
                // e.g., default phase = 0 and then always add into result?
                // using checks to make phase nonzero?

                // TODO: profile some form of inlining for phase computations
                int phase;
                if (a_degree == 1) {
                    // aceg
                    if (exc[0][q] && exc[0][s] && exc[1][r] && exc[1][t]) {
                        phase = compute_phase_double_excitation(d_int, q, r, s, t);
                        res[d_e] += J[i] * phase;
                    }

                    // bdfh
                    if (exc[0][r] && exc[0][t] && exc[1][q] && exc[1][s]) {
                        phase = compute_phase_double_excitation(d_int, r, q, t, s);
                        res[d_e] += J[i] * phase;
                    }
                } else if (a_degree == 0) {
                    // (0,2) : g_ii >= 2 is criterion for acceptance
                    bool aa_check = exc[0][q] && exc[0][r] && exc[0][s] && exc[0][t];
                    if (!aa_check)
                        continue;

                    // now must be one of g_11, g_22, g_33, g_44
                    if (g_11 >= 2) {
                        phase = compute_phase_double_excitation(d_int[0], q, r, s, t);
                    } else if (g_22 >= 2) {
                        phase = compute_phase_double_excitation(d_int[0], s, t, q, r);
                    } else if (g_33 >= 2) {
                        phase = compute_phase_double_excitation(d_int[0], q, t, s, r);
                    } else {
                        phase = compute_phase_double_excitation(d_int[0], r, s, q, t);
                    }
                    res[d_e] += J[i] * phase;
                } else {
                    // (2,0) : g_ii % 2 is criterion for acceptance
                    bool bb_check = exc[1][q] && exc[1][r] && exc[1][s] && exc[1][t];
                    if (!bb_check)
                        continue;

                    // now must be one of g_11, g_22, g_33, g_44
                    if (g_11 % 2) {
                        phase = compute_phase_double_excitation(d_int[1], q, r, s, t);
                    } else if (g_22 % 2) {
                        phase = compute_phase_double_excitation(d_int[1], s, t, q, r);
                    } else if (g_33 % 2) {
                        phase = compute_phase_double_excitation(d_int[1], q, t, s, r);
                    } else {
                        phase = compute_phase_double_excitation(d_int[1], r, s, q, t);
                    }
                    res[d_e] += J[i] * phase;
                }
            }
        }
    }
}