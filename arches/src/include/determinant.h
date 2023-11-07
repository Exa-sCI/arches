#pragma once

#include <array>
#include <functional>
#include <iostream>
#include <string>
#include <sul/dynamic_bitset.hpp>
#include <tuple>
#include <utility>
#include <vector>

typedef sul::dynamic_bitset<> spin_det_t;

template <> struct std::hash<spin_det_t> {
    std::size_t operator()(spin_det_t const &s) const noexcept {
        return std::hash<std::string>()(s.to_string());
    }
};

#define N_SPIN_SPECIES 2

class det_t { // The class

  public: // Access specifier
    spin_det_t alpha;
    spin_det_t beta;

    det_t(spin_det_t _alpha, spin_det_t _beta) {
        alpha = _alpha;
        beta = _beta;
    }

    bool operator<(const det_t &b) const {
        if (alpha == b.alpha)
            return (beta < b.beta);
        return (alpha < b.alpha);
    }

    bool operator==(const det_t &b) const { return (alpha == b.alpha) && (beta == b.beta); }

    spin_det_t &operator[](unsigned i) {
        assert(i < N_SPIN_SPECIES);
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
    const spin_det_t &operator[](unsigned i) const { return (*this)[i]; }

    // get excitation degree between self and other determinant
    // TODO: deal with narrowing conversion, and see if this should still be used anyway
    std::array<int, N_SPIN_SPECIES> exc_degree(const det_t &b) {
        auto ed_alpha = (this->alpha ^ b.alpha).count() / 2;
        auto ed_beta = (this->beta ^ b.beta).count() / 2;
        return std::array<int, N_SPIN_SPECIES>{ed_alpha, ed_beta};
    }
};

template <> struct std::hash<det_t> {
    std::size_t operator()(det_t const &s) const noexcept {
        std::size_t h1 = std::hash<spin_det_t>{}(s.alpha);
        std::size_t h2 = std::hash<spin_det_t>{}(s.beta);
        return h1 ^ (h2 << 1);
    }
};

// Should be moved in the cpp of det
inline std::ostream &operator<<(std::ostream &os, const det_t &obj) {
    return os << "(" << obj.alpha << "," << obj.beta << ")";
}

typedef sul::dynamic_bitset<> spin_occupancy_mask_t;
typedef std::array<spin_occupancy_mask_t, N_SPIN_SPECIES> occupancy_mask_t;

typedef sul::dynamic_bitset<> spin_unoccupancy_mask_t;
typedef std::array<spin_unoccupancy_mask_t, N_SPIN_SPECIES> unoccupancy_mask_t;

typedef std::array<uint64_t, 4> eri_4idx_t;

det_t exc_det(det_t &a, det_t &b);

int compute_phase_single_excitation(spin_det_t d, uint64_t h, uint64_t p);
int compute_phase_double_excitation(spin_det_t d, uint64_t h1, uint64_t h2, uint64_t p1,
                                    uint64_t p2);
int compute_phase_double_excitation(det_t d, uint64_t h1, uint64_t h2, uint64_t p1, uint64_t p2);

// overload phase compute for (1,1) excitations
det_t apply_single_excitation(det_t s, int spin, uint64_t hole, uint64_t particle);

spin_det_t apply_spin_single_excitation(spin_det_t s, uint64_t hole, uint64_t particle);

det_t apply_double_excitation(det_t s, std::pair<int, int> spin, uint64_t h1, uint64_t h2,
                              uint64_t p1, uint64_t p2);

typedef std::vector<uint64_t> spin_constraint_t;
typedef std::pair<spin_constraint_t, spin_constraint_t> exc_constraint_t;

std::string to_string(const spin_constraint_t &c, uint64_t max_orb) {
    std::string s(max_orb, '0');

    for (const auto &i : c)
        s[i] = '1';
    return s;
}

spin_constraint_t to_constraint(const spin_det_t &c) {
    spin_constraint_t res;
    auto npos = c.npos;
    auto c_pos = c.find_next(0);
    while (c_pos < npos) {
        res.push_back(c_pos);
        c_pos = c.find_next(0);
    }
    return res;
}

std::vector<det_t> get_constrained_determinants(det_t d, exc_constraint_t constraint,
                                                uint64_t max_orb);

std::vector<det_t> get_constrained_singles(det_t d, exc_constraint_t constraint, uint64_t max_orb);

std::vector<det_t> get_constrained_ss_doubles(det_t d, exc_constraint_t constraint,
                                              uint64_t max_orb);

std::vector<det_t> get_constrained_os_doubles(det_t d, exc_constraint_t constraint,
                                              uint64_t max_orb);

std::vector<det_t> get_singles_by_exc_mask(det_t d, int spin, spin_constraint_t h,
                                           spin_constraint_t p);

std::vector<spin_det_t> get_spin_singles_by_exc_mask(spin_det_t d, spin_constraint_t h,
                                                     spin_constraint_t p);

std::vector<det_t> get_ss_doubles_by_exc_mask(det_t d, int spin, spin_constraint_t h,
                                              spin_constraint_t p);

std::vector<det_t> get_all_singles(det_t d);
