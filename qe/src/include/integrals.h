#pragma once
#include <array>
#include <determinant.h>
#include <integral_indexing_utils.h>
#include <tuple>

// Abstract storage object for 1 and 2 electron integrals
template <class T> class JChunk {

  public:
    std::array<T> integrals;            // this will break, how to create constructor?
    spin_det_t orbital_mask;            // cheap bit mask filter for owned orbitals
    std::vector<idx_T> active_orbitals; // list of owned orbitals
    idx_t N_orb;                        // number of orbitals
    idx_t N_active_orb;                 // number of active orbitals
    idx_t interal_offset;               // integrals are packed towards zero
    idx_t sidx;                         // canonical starting index
    idt_t eidx;                         // canonical ending index
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

template <class T>
std::vector<T> ept2_C(const TEJ<T> &J, const std::vector<det_t> &psi_int,
                      const std::vector<det_t> &psi_ext) {

    std::vector<T> res(psi_ext.size());

    return res;
};

template <class T>
std::vector<T> ept2_D(const TEJ<T> &J, const std::vector<det_t> &psi_int,
                      const std::vector<det_t> &psi_ext) {

    std::vector<T> res(psi_ext.size());

    return res;
};
template <class T>
std::vector<T> ept2_E(const TEJ<T> &J, const std::vector<det_t> &psi_int,
                      const std::vector<det_t> &psi_ext) {

    std::vector<T> res(psi_ext.size());

    return res;
};
template <class T>
std::vector<T> ept2_F(const TEJ<T> &J, const std::vector<det_t> &psi_int,
                      const std::vector<det_t> &psi_ext) {

    std::vector<T> res(psi_ext.size());

    return res;
};

template <class T>
std::vector<T> ept2_G(const TEJ<T> &J, const std::vector<det_t> &psi_int,
                      const std::vector<det_t> &psi_ext) {

    std::vector<T> res(psi_ext.size());

    return res;
};
