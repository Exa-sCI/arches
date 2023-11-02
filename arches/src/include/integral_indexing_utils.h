#pragma once
#include <cmath>

typedef long int idx_t;

enum j_category { IC_A, IC_B, IC_C, IC_D, IC_E, IC_F, IC_G };

idx_t isqrt(const idx_t i) { return (idx_t)std::sqrt(i); }

extern "C" idx_t compound_idx2(const idx_t i, const idx_t j);

extern "C" idx_t compound_idx4(const idx_t i, const idx_t j, const idx_t k, const idx_t l);

struct ij_tuple {
    idx_t i;
    idx_t j;
};

inline bool operator==(const ij_tuple &lhs, const ij_tuple &rhs) {
    return (lhs.i == rhs.i) && (lhs.j == rhs.j);
}

inline bool operator<(const ij_tuple &lhs, const ij_tuple &rhs) {
    return (lhs.i < rhs.i) || (lhs.j < rhs.j);
}

struct ijkl_tuple {
    idx_t i;
    idx_t j;
    idx_t k;
    idx_t l;
};
inline bool operator==(const ijkl_tuple &lhs, const ijkl_tuple &rhs) {
    return (lhs.i == rhs.i) && (lhs.j == rhs.j) && (lhs.k == rhs.k) && (lhs.l == rhs.l);
}

inline bool operator<(const ijkl_tuple &lhs, const ijkl_tuple &rhs) {
    return (lhs.i < rhs.i) || (lhs.j < rhs.j) || (lhs.k < rhs.k) || (lhs.l < rhs.l);
}

struct ijkl_perms {
    struct ijkl_tuple ijkl;
    struct ijkl_tuple jilk;
    struct ijkl_tuple klij;
    struct ijkl_tuple lkji;
    struct ijkl_tuple ilkj;
    struct ijkl_tuple lijk;
    struct ijkl_tuple kjil;
    struct ijkl_tuple jkli;
};

// Would like to use std::tuple here but not handled in ctypes. Easier to use
// structs for now.
extern "C" struct ijkl_tuple canonical_idx4(const idx_t i, const idx_t j, const idx_t k,
                                            const idx_t l);

extern "C" struct ij_tuple compound_idx2_reverse(const idx_t ij);

extern "C" struct ijkl_tuple compound_idx4_reverse(const idx_t ijkl);

extern "C" struct ijkl_perms compound_idx4_reverse_all(const idx_t ijkl);

idx_t compound_idx4(const ijkl_tuple ijkl);
// extern "C" int get_unique_idx4(ijkl_tuple* u_idx, const ijkl_perms all_idx);
