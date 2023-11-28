#pragma once

#include "integral_indexing_utils.h"
#include "matrix.h"
#include <array>
#include <bitset>
#include <functional>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// We could change this to short or long or whatever--but it should be fixed.
typedef unsigned int mo_block_t;

int popcount(unsigned int x) { return __builtin_popcount(x); }

int popcount(unsigned long x) { return __builtin_popcountl(x); }

int popcount(unsigned long long x) { return __builtin_popcountll(x); }

// define left most bit in 0th block to be the ground state

class spin_det_t {
  protected:
    std::unique_ptr<mo_block_t[]> block_arr_ptr;

  public:
    idx_t N_mos;
    idx_t N_blocks;
    mo_block_t *block_arr;
    mo_block_t read_mask;
    mo_block_t block_size;

    // needed for det_t to compile
    spin_det_t(){};

    // empty initialization
    spin_det_t(idx_t min_mos) {
        block_size = sizeof(mo_block_t) * 8;
        read_mask = ((mo_block_t)1) << (block_size - 1);

        N_mos = min_mos;
        N_blocks = N_mos / (block_size) + (N_mos % (block_size) != 0);

        std::unique_ptr<mo_block_t[]> p(new mo_block_t[N_blocks]());
        block_arr_ptr = std::move(p);
        block_arr = block_arr_ptr.get();
    }

    // fill initialization
    spin_det_t(idx_t min_mos, idx_t max_orb, bool fill) : spin_det_t(min_mos) {
        this->set(0, max_orb, fill);
    }

    // iter initialization
    spin_det_t(idx_t min_mos, const idx_t N_filled, const idx_t *orbs) : spin_det_t(min_mos) {
        for (auto i = 0; i < N_filled; i++) {
            this->set(orbs[i], true);
        }
    }

    // lvalue assignment
    spin_det_t &operator=(const spin_det_t &other) {
        N_mos = other.N_mos;
        N_blocks = other.N_blocks;
        block_size = other.block_size;
        read_mask = ((mo_block_t)1) << (block_size - 1);

        std::unique_ptr<mo_block_t[]> p(new mo_block_t[N_blocks]());
        block_arr_ptr = std::move(p);
        block_arr = block_arr_ptr.get();
        std::copy(other.block_arr, other.block_arr + N_blocks, block_arr);
        return *this;
    }

    // move operator for rvalue assignment, for temps
    spin_det_t &operator=(spin_det_t &&other) {
        N_mos = other.N_mos;
        N_blocks = other.N_blocks;
        block_size = other.block_size;
        read_mask = ((mo_block_t)1) << (block_size - 1);

        block_arr_ptr = std::move(other.block_arr_ptr);
        block_arr = block_arr_ptr.get();
        return *this;
    }

    // copy constructors, which were implicitly deleted via move operator
    spin_det_t(spin_det_t &other) { *this = other; }
    spin_det_t(const spin_det_t &other) { *this = other; }

    // default destructor
    ~spin_det_t() = default;

    /*
    Usage operators and methods
    */
    bool operator[](idx_t orb) const {
        // assert(orb >= 0);
        // assert(orb < N_mos);
        idx_t block = orb / block_size;
        idx_t offset = orb % block_size;
        return (block_arr[block] << offset) & read_mask;
    }

    void set(idx_t orb, bool val) {
        // assert(orb >= 0);
        // assert(orb < N_mos);
        idx_t block = orb / block_size;
        idx_t offset = orb % block_size;

        mo_block_t mask;
        if (val) {
            // Turning on bit at orb
            mask = ((mo_block_t)1) << (block_size - offset - 1);
            block_arr[block] |= mask;
        } else {
            // Turning off bit at orb
            mask = ~(((mo_block_t)1) << (block_size - offset - 1));
            block_arr[block] &= mask;
        }
    };

    void set(idx_t start_orb, idx_t end_orb, bool val) {
        // assert(start_orb >= 0 && start_orb <= end_orb);
        // assert(end_orb < N_mos);

        idx_t s_block = start_orb / block_size;
        idx_t e_block = end_orb / block_size;

        // for all blocks after first and before end block, can handle whole block
        mo_block_t block_mask = val ? ~((mo_block_t)0) : 0;
        for (auto i = s_block + 1; i < e_block; i++) {
            block_arr[i] = block_mask;
        }

        idx_t max_offset = end_orb % block_size;
        mo_block_t end_mask = max_offset ? block_mask << (block_size - max_offset) : 0;
        // if val, trailing orbs after end_orb should be zero
        // else, up to offset should be zero and after should be one
        end_mask = val ? end_mask : ~end_mask;

        idx_t min_offset = start_orb % block_size;
        mo_block_t start_mask = block_mask >> (min_offset);
        start_mask = val ? start_mask : ~start_mask;
        if (s_block == e_block) {
            start_mask = start_mask & end_mask;
        } else {
            if (val) {
                block_arr[e_block] |= end_mask;
            } else {
                block_arr[e_block] &= end_mask;
            }
        }

        if (val) {
            block_arr[s_block] |= start_mask;
        } else {
            block_arr[s_block] &= start_mask;
        }
    };

    // TODO: is it worth checking that this and other have the same size MO basis?
    bool operator<(const spin_det_t &other) const {
        // assert(N_blocks = other.N_blocks);
        idx_t i = 0;
        bool success;
        while ((success = this->block_arr[i] < other.block_arr[i]) && (i < N_blocks))
            i++;
        return success;
    }

    bool operator==(const spin_det_t &other) const {
        // assert(N_blocks = other.N_blocks);
        idx_t i = 0;
        bool success;
        while ((success = block_arr[i] == other.block_arr[i]) && (i < N_blocks))
            i++;
        return success;
    }

    spin_det_t operator~() {
        spin_det_t res(N_mos);
        for (auto i = 0; i < N_blocks; i++) {
            res.block_arr[i] = ~block_arr[i];
        }
        res.set(N_mos, N_blocks * block_size, 0); // clear out excess orbitals
        return res;
    }

    spin_det_t operator^(const spin_det_t &other) {
        spin_det_t res(N_mos);
        for (auto i = 0; i < N_blocks; i++) {
            res.block_arr[i] = block_arr[i] ^ other.block_arr[i];
        }
        return res; // no need to clear out excess orbitals
    }

    spin_det_t operator&(const spin_det_t &other) {
        spin_det_t res(N_mos);
        for (auto i = 0; i < N_blocks; i++) {
            res.block_arr[i] = block_arr[i] & other.block_arr[i];
        }
        return res; // no need to clear out excess orbitals
    }

    int count() {
        // both Clang and GCC have __builtin_popcount support, but I have no idea how this would
        // port to device
        int res = 0;
        for (auto i = 0; i < N_blocks; i++) {
            // use a wrapper for overloading against the builtins, in case mo_block_t is adjusted
            res += popcount(block_arr[i]);
        }
        return res;
    }
};

class det_t {
  public:
    spin_det_t alpha;
    spin_det_t beta;
    idx_t N_mos;

    det_t(){};

    det_t(idx_t min_mos) {
        N_mos = min_mos;
        alpha = spin_det_t(min_mos);
        beta = spin_det_t(min_mos);
    }

    det_t(spin_det_t _alpha, spin_det_t _beta) {
        // TODO: throw an exception if _alpha, _beta have different number of MOs
        N_mos = _alpha.N_mos;
        alpha = _alpha;
        beta = _beta;
    }

    det_t &operator=(const det_t &other) {
        N_mos = other.N_mos;
        alpha = other.alpha;
        beta = other.beta;
        return *this;
    }

    det_t &operator=(det_t &&other) {
        N_mos = other.N_mos;
        alpha = other.alpha;
        beta = other.beta;
        return *this;
    }

    // copy constructors
    det_t(det_t &other) { *this = other; }
    det_t(const det_t &other) { *this = other; }

    ~det_t() = default;

    /*
    Usage operators and methods
    */
    bool operator<(const det_t &b) const {
        if (alpha == b.alpha)
            return (beta < b.beta);
        return (alpha < b.alpha);
    }

    bool operator==(const det_t &b) const { return (alpha == b.alpha) && (beta == b.beta); }

    spin_det_t &operator[](idx_t i) {
        // assert(i < N_SPIN_SPECIES);
        switch (i) {
        case 0:
            return alpha;
        // Avoid `non-void function does not return a value in all control
        // paths`
        default:
            return beta;
        }
    }

    // https://stackoverflow.com/a/27830679/7674852 seem to recommand doing the
    // other way arround
    const spin_det_t &operator[](idx_t i) const {
        switch (i) {
        case 0:
            return alpha;
        // Avoid `non-void function does not return a value in all control
        // paths`
        default:
            return beta;
        }
    }
};

class DetArray {
  protected:
    // det_t cannot be default constructed, since we need to know the size of the MO basis at run
    // time therefore, it is easier to store in a vector here instead of wrapping the storage with a
    // unique ptr
    std::vector<det_t> storage;

  public:
    det_t *arr;
    idx_t size;
    idx_t N_mos;

    DetArray(idx_t N_dets, idx_t min_mos) {
        size = N_dets;
        N_mos = min_mos;

        std::vector<det_t> _temp(size);
        // storage.reserve(size);
        for (auto i = 0; i < size; i++) {
            _temp.emplace_back(N_mos);
        }

        storage = std::move(_temp);
        arr = &storage[0];
    }

    ~DetArray() = default;
};

std::hash<mo_block_t> block_hash;

template <> struct std::hash<spin_det_t> {
    // Implementing something quick a dirty for now along the lines of:
    // https://math.stackexchange.com/a/4146931
    // TODO: Profile this in particular! Or come up with another algorithm that is unique and fast
    // and has good dispersion.
    std::size_t operator()(spin_det_t const &s) const noexcept {
        std::size_t m = ~0 >> (sizeof(std::size_t) * 8 - 19); // should be the Mersenne prime 2^19-1
        std::size_t res = 0x402df854;                         // e
        for (auto i = 0; i < s.N_blocks; i++) {
            res = (block_hash(s.block_arr[i]) ^ res) * m;
        }
        return res;
    }
};

std::hash<spin_det_t> spin_det_hash;

template <> struct std::hash<det_t> {
    std::size_t operator()(det_t const &s) const noexcept {
        std::size_t h1 = spin_det_hash(s.alpha);
        std::size_t h2 = spin_det_hash(s.beta);
        return h1 ^ (h2 << 1);
    }
};

std::hash<det_t> det_hash;

// // Should be moved in the cpp of det
// inline std::ostream &operator<<(std::ostream &os, const det_t &obj) {
//     return os << "(" << obj.alpha << "," << obj.beta << ")";
// }

det_t exc_det(det_t &a, det_t &b);

int compute_phase_single_excitation(spin_det_t d, idx_t h, idx_t p);
int compute_phase_double_excitation(spin_det_t d, idx_t h1, idx_t h2, idx_t p1, idx_t p2);
int compute_phase_double_excitation(det_t d, idx_t h1, idx_t h2, idx_t p1, idx_t p2);

// overload phase compute for (1,1) excitations
det_t apply_single_excitation(det_t s, int spin, idx_t hole, idx_t particle);

spin_det_t apply_single_excitation(spin_det_t s, idx_t hole, idx_t particle);

// det_t apply_double_excitation(det_t s, std::pair<int, int> spin, idx_t h1, idx_t h2, idx_t p1,
//                               idx_t p2);

typedef std::vector<idx_t> spin_constraint_t;
typedef std::pair<spin_constraint_t, spin_constraint_t> exc_constraint_t;

std::string to_string(const spin_constraint_t &c, idx_t max_orb) {
    std::string s(max_orb, '0');

    for (const auto &i : c)
        s[i] = '1';
    return s;
}

spin_constraint_t to_constraint(const spin_det_t &c) {
    spin_constraint_t res;
    int count = 0;
    for (auto i = 0; i < c.N_mos; i++) {
        if (c[i]) {
            res.push_back(i);
            count++;
        }
    }
    return res;
}

// std::vector<det_t> get_constrained_determinants(det_t d, exc_constraint_t constraint,
//                                                 idx_t max_orb);

std::vector<det_t> get_constrained_singles(det_t d, exc_constraint_t constraint, idx_t max_orb);

std::vector<det_t> get_constrained_ss_doubles(det_t d, exc_constraint_t constraint, idx_t max_orb);

std::vector<det_t> get_constrained_os_doubles(det_t d, exc_constraint_t constraint, idx_t max_orb);

std::vector<det_t> get_singles_by_exc_mask(det_t d, int spin, spin_constraint_t h,
                                           spin_constraint_t p);

std::vector<spin_det_t> get_spin_singles_by_exc_mask(spin_det_t d, spin_constraint_t h,
                                                     spin_constraint_t p);

std::vector<det_t> get_ss_doubles_by_exc_mask(det_t d, int spin, spin_constraint_t h,
                                              spin_constraint_t p);

std::vector<det_t> get_all_singles(det_t d);

// void get_connected_dets(DetArray *dets_int, DetArray *dets_ext, idx_t *hc_alpha, idx_t *hc_beta,
//                         idx_t *pc_alpha, idx_t *pc_beta);

// This is way too slow for actual formation of explicit Hamiltonians, but it's easy to write!
// Should get bilinear mappings so that we can iterate over known determinants and find connections
// directly. Or, resort to on the fly generation of the Hamiltonian structure, which would need true
// expandable vectors inside the offloaded kernels
template <class T>
void get_H_structure_naive(DetArray *psi_det, SymCSRMatrix<T> *H, idx_t N_det, T dummy) {

    std::vector<std::vector<idx_t>> csr_rows;
    std::unique_ptr<idx_t[]> H_p(new idx_t[N_det + 1]);

    // find non-zero entries
    for (auto i = 0; i < N_det; i++) {
        det_t &d_row = psi_det->arr[i];

        H_p[i + 1] += 1; // add H_ii
        csr_rows[i].push_back(i);
        for (auto j = i + 1; j < N_det; j++) {
            det_t &d_col = psi_det->arr[j];

            det_t exc = exc_det(d_row, d_col);
            auto degree = (exc[0].count() + exc[1].count()) / 2;

            if (degree <= 2) { // add H_ij
                csr_rows[i].push_back(j);
                H_p[i + 1] += 1;
            }
        }

        H_p[i + 1] += H_p[i]; // adjust global row offset
    }

    // copy over from vector of vectors into single array
    std::unique_ptr<idx_t[]> H_c(new idx_t[H_p[N_det]]);
    idx_t *H_c_p = H_c.get();
    for (auto i = 0; i < N_det; i++) {
        std::copy(csr_rows[i].begin(), csr_rows[i].end(), H_c_p + H_p[i]);
    }

    // initialize values at 0
    std::unique_ptr<T[]> H_v(new T[H_p[N_det]]);
    T *H_v_p = H_v.get();
    std::fill(H_v_p, H_v_p + H_p[N_det], (T)0.0);

    *H = SymCSRMatrix<T>(N_det, N_det, H_p, H_c, H_v); // use the move constructor + move operator
}