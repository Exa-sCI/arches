#include "integral_indexing_utils.h"
#include <algorithm>
#include <iostream>
#include <set>

// TODO: profile to see if it would be useful to implement branchless min/max
// and canonical idx
// TODO: create array forms with offloading/SIMD ?
extern "C" idx_t compound_idx2(const idx_t i, const idx_t j) {
    // idx_t p = std::min(i, j);
    // idx_t q = std::max(i, j);
    const auto &[p, q] = std::minmax(i, j);
    idx_t res = (q * (q + 1)) / 2 + p;
    return res;
}

extern "C" idx_t compound_idx4(const idx_t i, const idx_t j, const idx_t k, const idx_t l) {
    idx_t a = compound_idx2(i, k);
    idx_t b = compound_idx2(j, l);
    return compound_idx2(a, b);
}

idx_t compound_idx4(const ijkl_tuple ijkl) { return compound_idx4(ijkl.i, ijkl.j, ijkl.k, ijkl.l); }

extern "C" struct ij_tuple compound_idx2_reverse(const idx_t ij) {
    idx_t j = (isqrt(1 + 8 * ij) - 1) / 2;
    idx_t i = ij - (j * (j + 1) / 2);
    struct ij_tuple res = {i, j};
    return res;
}

extern "C" struct ijkl_tuple compound_idx4_reverse(const idx_t ijkl) {
    struct ij_tuple ik_jl = compound_idx2_reverse(ijkl);
    struct ij_tuple ik = compound_idx2_reverse(ik_jl.i);
    struct ij_tuple jl = compound_idx2_reverse(ik_jl.j);
    struct ijkl_tuple res = {ik.i, jl.i, ik.j, jl.j};
    return res;
}

extern "C" struct ijkl_perms compound_idx4_reverse_all(const idx_t ijkl) {
    struct ijkl_tuple idx = compound_idx4_reverse(ijkl);
    struct ijkl_perms res = {
        idx.i, idx.j, idx.k, idx.l, idx.j, idx.i, idx.l, idx.k, idx.k, idx.l, idx.i,
        idx.j, idx.l, idx.k, idx.j, idx.i, idx.i, idx.l, idx.k, idx.j, idx.l, idx.i,
        idx.j, idx.k, idx.k, idx.j, idx.i, idx.l, idx.j, idx.k, idx.l, idx.i,
    };
    return res;
}

extern "C" struct ijkl_tuple canonical_idx4(const idx_t i, const idx_t j, const idx_t k,
                                            const idx_t l) {
    // idx_t ii = std::min(i, k);
    // idx_t kk = std::max(i, k);
    // idx_t jj = std::min(j, l);
    // idx_t ll = std::max(j, l);
    const auto &[ii, kk] = std::minmax(i, k);
    const auto &[jj, ll] = std::minmax(j, l);

    idx_t a = compound_idx2(ii, kk);
    idx_t b = compound_idx2(jj, ll);

    if (a <= b) {
        struct ijkl_tuple res = {ii, jj, kk, ll};
        return res;
    }

    struct ijkl_tuple res = {jj, ii, ll, kk};
    return res;
}

/*
extern "C" int get_unique_idx4(ijkl_tuple* u_idx, const ijkl_perms all_idx){
    // Construct std::set, iterate through items and assign to output pointer

    std::set<ijkl_tuple> u_set = {all_idx.ijkl, all_idx.jilk, all_idx.klij,
all_idx.lkji, all_idx.ilkj, all_idx.lijk, all_idx.kjil, all_idx.jkli};

    // for (const auto& i : u_set) std::cout << i.i << " " << i.j << " " << i.k
<< " " << i.l << std::endl;
    // for (const auto& i : u_set) *u_idx++ = i;
    int N = u_set.size();
    auto s = u_set.begin();
    for (auto i = 0; i < N; i++){
        u_idx[i] = *s++;
    }

    return N;
}
*/
