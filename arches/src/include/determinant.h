#pragma once

#include "integral_indexing_utils.h"
#include <array>
#include <bitset>
#include <functional>
#include <iostream>
#include <memory>
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
        bool success = true;
        while (success && (i < this->N_blocks)) {
            success = this->block_arr[i] < other.block_arr[i];
            i++;
        }
        return success;
    }

    bool operator==(const spin_det_t &other) const {
        // assert(N_blocks = other.N_blocks);
        idx_t i = 0;
        bool success = true;
        while (success && (i < this->N_blocks)) {
            success = this->block_arr[i] == other.block_arr[i];
            i++;
        }
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
        det_t empty_det(N_mos);
        std::fill(_temp.begin(), _temp.end(), empty_det);
        storage = std::move(_temp);
        arr = &storage[0];
    }

    // copy constructor
    DetArray(const std::vector<det_t> &other) {
        size = other.size();
        N_mos = other[0].N_mos;
        storage = other;
        arr = &storage[0];
    }

    // move constructor
    DetArray(const std::vector<det_t> &&other) {
        size = other.size();
        N_mos = other[0].N_mos;
        storage = std::move(other);
        arr = &storage[0];
    }

    const det_t &operator[](idx_t i) const { return arr[i]; }

    void extend_with_filter(const DetArray &other, idx_t *filter, const idx_t N) {
        for (auto i = 0; i < N; i++) {
            storage.push_back(other[filter[i]]);
        }
        size += N;
        arr = &storage[0];
    }

    ~DetArray() = default;
};

template <> struct std::hash<spin_det_t> {
    // Implementing something quick a dirty for now along the lines of:
    // https://math.stackexchange.com/a/4146931
    // TODO: Profile this in particular! Or come up with another algorithm that is unique and fast
    // and has good dispersion.
    std::size_t operator()(spin_det_t const &s) const noexcept {
        // std::hash<mo_block_t> block_hash;
        // std::size_t m = ~((std::size_t)0) >>
        //                 (sizeof(std::size_t) * 8 - 19); // should be the Mersenne prime 2^19-1
        // std::size_t res = 0x5b174a16; // X0
        std::size_t res = 0x77123456; // X0
        std::size_t m = 0x402df854;   // e
        for (auto i = 0; i < s.N_blocks; i++) {
            res = (s.block_arr[i] ^ res) * m;
        }
        return res;
    }
};

template <> struct std::hash<det_t> {
    std::size_t operator()(det_t const &s) const noexcept {
        // std::hash<spin_det_t> spin_det_hash;
        // std::size_t h1 = spin_det_hash(s.alpha);
        // std::size_t h2 = spin_det_hash(s.beta);

        // std::size_t m = 0x402df854; // e
        // std::size_t n = 0x40490fdb; // pi
        // return (h1 * m) ^ (h2 * n);

        std::size_t res = 0x77123456; // X0
        std::size_t m = 0x402df854;   // e
        for (auto i = 0; i < s[0].N_blocks; i++) {
            res = (s[0].block_arr[i] ^ res) * m;
            res = (s[1].block_arr[i] ^ res) * m;
        }
        return res;
    }
};

class LinearUnorderedMap {
  protected:
    std::unordered_map<std::size_t, det_t> map;
    std::hash<det_t> hash_f;

  public:
    LinearUnorderedMap() = default;
    ~LinearUnorderedMap() = default;

    int add_det(const det_t &d) {
        // Hash is not perfect, but collisions are low
        // If collides, use linear probing to augment hash_val
        std::size_t hash_val = hash_f(d);
        while (map.count(hash_val) == 1) {
            if (map[hash_val] == d) {
                return 0;
            }
            hash_val++;
        }

        map[hash_val] = d;
        return 1;
    }
};

int compute_phase_single_excitation(spin_det_t d, idx_t h, idx_t p);
int compute_phase_double_excitation(spin_det_t d, idx_t h1, idx_t h2, idx_t p1, idx_t p2);
int compute_phase_double_excitation(det_t d, idx_t h1, idx_t h2, idx_t p1, idx_t p2);

det_t exc_det(det_t &a, det_t &b);

spin_det_t apply_single_excitation(spin_det_t det, idx_t h, idx_t p, bool &succcess);
det_t apply_single_excitation(det_t det, int spin, idx_t h, idx_t p, bool &succcess);

spin_det_t apply_double_excitation(spin_det_t det, idx_t h1, idx_t h2, idx_t p1, idx_t p2,
                                   bool &success);
det_t apply_double_excitation(det_t det, int s1, int s2, idx_t h1, idx_t h2, idx_t p1, idx_t p2,
                              bool &success);

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
    for (auto i = 0; i < c.N_mos; i++) {
        if (c[i]) {
            res.push_back(i);
        }
    }
    return res;
}

// std::vector<det_t> get_constrained_determinants(det_t d, exc_constraint_t constraint,
//                                                 idx_t max_orb);

std::vector<det_t> get_singles_by_exc_mask(det_t d, int spin, spin_constraint_t h,
                                           spin_constraint_t p);

std::vector<spin_det_t> get_spin_singles_by_exc_mask(spin_det_t d, spin_constraint_t h,
                                                     spin_constraint_t p);

std::vector<det_t> get_ss_doubles_by_exc_mask(det_t d, int spin, spin_constraint_t h,
                                              spin_constraint_t p);

std::vector<det_t> get_all_singles(det_t d);

std::vector<det_t> get_constrained_singles(det_t d, exc_constraint_t alpha_constraint,
                                           exc_constraint_t beta_constraint);

std::vector<det_t> get_os_doubles(det_t d, bool return_singles);

std::vector<det_t> get_constrained_os_doubles(det_t d, exc_constraint_t alpha_constraint,
                                              exc_constraint_t beta_constraint,
                                              bool return_singles);

std::vector<det_t> get_ss_doubles(det_t d);

std::vector<det_t> get_constrained_ss_doubles(det_t d, exc_constraint_t alpha_constraint,
                                              exc_constraint_t beta_constraint);

std::vector<det_t> get_all_connected_dets(det_t *d, idx_t N_dets);

std::vector<det_t> get_constrained_connected_dets(det_t d, exc_constraint_t alpha_constraint,
                                                  exc_constraint_t beta_constraint);