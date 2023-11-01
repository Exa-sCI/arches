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
};

// one electron integrals
template <class T> JChunk : OEJ{public : T & operator[](idx_t ij){return integrals[ij];
}
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
;

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
};