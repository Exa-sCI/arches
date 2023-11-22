#if !defined(DOCTEST_CONFIG_DISABLE)
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#endif
#include <determinant.h>
#include <doctest/doctest.h>

int compute_phase_single_excitation(spin_det_t d, idx_t h, idx_t p) {
    const auto &[i, j] = std::minmax(h, p);
    spin_det_t hpmask(d.N_mos);
    hpmask.set(i + 1, j - i - 1, 1);
    const bool parity = (hpmask & d).count() % 2;
    return parity ? -1 : 1;
}

// TEST_CASE("testing get_phase_single") {
//     CHECK(compute_phase_single_excitation(spin_det_t{"11000"}, 4, 2) == -1);
//     CHECK(compute_phase_single_excitation(spin_det_t{"10001"}, 4, 2) == 1);
//     CHECK(compute_phase_single_excitation(spin_det_t{"01100"}, 2, 4) == -1);
//     CHECK(compute_phase_single_excitation(spin_det_t{"00100"}, 2, 4) == 1);
// }

int compute_phase_double_excitation(spin_det_t d, idx_t h1, idx_t h2, idx_t p1, idx_t p2) {
    // Single spin channel excitations, i.e., (2,0) or (0,2)
    int phase =
        compute_phase_single_excitation(d, h1, p1) * compute_phase_single_excitation(d, h2, p2);
    phase = (h2 < p1) ? phase * -1 : phase;
    phase = (p2 < h1) ? phase * -1 : phase;
    return phase;
}

int compute_phase_double_excitation(det_t d, idx_t h1, idx_t h2, idx_t p1, idx_t p2) {

    // Cross channel excitations, i.e., (1,1)
    // Assumes alpha are h1-p1, beta are h2-p2
    int phase = compute_phase_single_excitation(d[0], h1, p1) *
                compute_phase_single_excitation(d[1], h2, p2);
    return phase;
}

det_t exc_det(det_t &a, det_t &b) {
    // spin_det_t alpha = a[0] ^ b[0];
    // spin_det_t beta = a[1] ^ b[1];
    // return det_t(alpha, beta);

    return det_t(a[0] ^ b[0], a[1] ^ b[1]); // Does this work since I have the move defined?
}

det_t apply_single_excitation(det_t s, int spin, idx_t h, idx_t p) {
    // assert(s[spin][h] == 1);
    // assert(s[spin][p] == 0);

    auto s2 = det_t(s);
    // s2[spin][h] = 0;
    // s2[spin][p] = 1;
    s2[spin].set(h, 0);
    s2[spin].set(p, 1);
    return s2;
}

spin_det_t apply_single_excitation(spin_det_t s, idx_t h, idx_t p) {
    // assert(s[h] == 1);
    // assert(s[p] == 0);

    auto s2 = spin_det_t(s);
    s2.set(h, 0);
    s2.set(p, 1);
    return s2;
}

// TEST_CASE("testing apply_single_excitation") {
//     det_t s{spin_det_t{"11000"}, spin_det_t{"00001"}};
//     CHECK(apply_single_excitation(s, 0, 4, 1) == det_t{spin_det_t{"01010"},
//     spin_det_t{"00001"}}); CHECK(apply_single_excitation(s, 1, 0, 1) ==
//     det_t{spin_det_t{"11000"}, spin_det_t{"00010"}});
// }

det_t apply_double_excitation(det_t s, int spin_1, int spin_2, idx_t h1, idx_t h2, idx_t p1,
                              idx_t p2) {
    // Check if valid
    // assert(s[spin.first][h1] == 1);
    // assert(s[spin.second][h2] == 1);
    // assert(s[spin.first][p1] == 0);
    // assert(s[spin.second][p2] == 0);

    auto s2 = det_t(s);
    s2[spin_1].set(h1, 0);
    s2[spin_2].set(h2, 0);
    s2[spin_1].set(p1, 1);
    s2[spin_2].set(p2, 1);
    return s2;
}

// TEST_CASE("testing apply_double_excitation") {
//     det_t s{spin_det_t{"11000000"}, spin_det_t{"10001000"}};
//     CHECK(apply_double_excitation(s, std::pair<int>{0, 0}, 0, 1, 4, 5) ==
//           det_t(spin_det_t{"00001100"}, spin_det_t{"10001000"}));
//     CHECK(apply_double_excitation(s, std::pair<int>{1, 1}, 0, 4, 1, 7) ==
//           det_t(spin_det_t{"11000000"}, spin_det_t{"01000001"}));
//     CHECK(apply_double_excitation(s, std::pair<int>{0, 1}, 1, 0, 2, 2) ==
//           det_t(spin_det_t{"10100000"}, spin_det_t{"00101000"}));
// }

std::vector<det_t> get_singles_by_exc_mask(det_t d, int spin, spin_constraint_t h,
                                           spin_constraint_t p) {
    std::vector<det_t> res;
    for (auto &i : h) {
        for (auto &j : p) {
            res.push_back(apply_single_excitation(d, spin, i, j));
        }
    }
    return res;
}

std::vector<spin_det_t> get_spin_singles_by_exc_mask(spin_det_t d, spin_constraint_t h,
                                                     spin_constraint_t p) {
    std::vector<spin_det_t> res;
    for (auto &i : h) {
        for (auto &j : p) {
            res.push_back(apply_single_excitation(d, i, j));
        }
    }
    return res;
}

std::vector<det_t> get_ss_doubles_by_exc_mask(det_t d, int spin, spin_constraint_t h,
                                              spin_constraint_t p) {

    std::vector<det_t> res;
    // h, p are sorted so h1 < h2; p1 < p2 always
    for (auto h1 = 0; h1 < h.size() - 1; h1++) {
        for (auto h2 = h1 + 1; h2 < h.size(); h2++) {
            for (auto p1 = 0; p1 < p.size() - 1; p1++) {
                for (auto p2 = p1 + 1; p2 < p.size(); p2++) {
                    res.push_back(apply_double_excitation(d, spin, spin, h1, h2, p1, p2));
                }
            }
        }
    }
    return res;
}

std::vector<det_t> get_constrained_singles(det_t d, exc_constraint_t constraint, idx_t max_orb) {
    std::vector<det_t> singles;

    // convert constraints to bit masks
    // TODO: test if faster to create empty bit mask and set bits
    spin_det_t hole_mask(d.N_mos, constraint.first.size(),
                         &constraint.first[0]); // where holes can be created
    spin_det_t part_mask(d.N_mos, constraint.second.size(),
                         &constraint.second[0]); // where particles can be created

    // convert max orb to bit masks
    // TODO: as above, test if faster to create empty bit mask, now with
    // reserved size, and set bits
    spin_det_t max_orb_mask(d.N_mos, max_orb, 1);

    // apply bit masks and get final list
    spin_constraint_t alpha_holes = to_constraint((~d[0] & hole_mask) & max_orb_mask);
    spin_constraint_t alpha_parts = to_constraint((d[0] & part_mask) & max_orb_mask);

    spin_constraint_t beta_holes = to_constraint((~d[1] & hole_mask) & max_orb_mask);
    spin_constraint_t beta_parts = to_constraint((d[1] & part_mask) & max_orb_mask);

    // at this point, hole and particle bitsets are guaranteed to be disjoint
    // iterate over product list and add to return vector
    std::vector<det_t> alpha_singles = get_singles_by_exc_mask(d, 0, alpha_holes, alpha_parts);
    std::vector<det_t> beta_singles = get_singles_by_exc_mask(d, 1, beta_holes, beta_parts);
    singles.insert(singles.end(), alpha_singles.begin(), alpha_singles.end());
    singles.insert(singles.end(), beta_singles.begin(), beta_singles.end());

    return singles;
}

std::vector<det_t> get_all_singles(det_t d) {

    std::vector<det_t> singles;
    std::vector<det_t> alpha_singles =
        get_singles_by_exc_mask(d, 0, to_constraint(~d[0]), to_constraint(d[0]));
    std::vector<det_t> beta_singles =
        get_singles_by_exc_mask(d, 1, to_constraint(~d[1]), to_constraint(d[1]));
    singles.insert(singles.end(), alpha_singles.begin(), alpha_singles.end());
    singles.insert(singles.end(), beta_singles.begin(), beta_singles.end());

    return singles;
}

// TODO: refactor, a lot of code re-use
std::vector<det_t> get_constrained_os_doubles(det_t d, exc_constraint_t constraint, idx_t max_orb) {
    std::vector<det_t> os_doubles;

    // convert constraints to bit masks
    // TODO: test if faster to create empty bit mask and set bits
    spin_det_t hole_mask(d.N_mos, constraint.first.size(),
                         &constraint.first[0]); // where holes can be created
    spin_det_t part_mask(d.N_mos, constraint.second.size(),
                         &constraint.second[0]); // where particles can be created

    // convert max orb to bit masks
    // TODO: as above, test if faster to create empty bit mask, now with
    // reserved size, and set bits
    spin_det_t max_orb_mask(d.N_mos, max_orb, 1);

    // apply bit masks and get final list
    spin_constraint_t alpha_holes = to_constraint((~d[0] & hole_mask) & max_orb_mask);
    spin_constraint_t alpha_parts = to_constraint((d[0] & part_mask) & max_orb_mask);

    spin_constraint_t beta_holes = to_constraint((~d[1] & hole_mask) & max_orb_mask);
    spin_constraint_t beta_parts = to_constraint((d[1] & part_mask) & max_orb_mask);

    // get all singles and iterate over product of (1,0) X (0,1) to get (1,1)
    std::vector<spin_det_t> alpha_singles =
        get_spin_singles_by_exc_mask(d[0], alpha_holes, alpha_parts);
    std::vector<spin_det_t> beta_singles =
        get_spin_singles_by_exc_mask(d[1], beta_holes, beta_parts);

    for (auto &a : alpha_singles) {
        for (auto &b : beta_singles) {
            os_doubles.push_back(det_t(a, b));
        }
    }

    return os_doubles;
}

std::vector<det_t> get_constrained_ss_doubles(det_t d, exc_constraint_t constraint, idx_t max_orb) {
    std::vector<det_t> ss_doubles;

    // convert constraints to bit masks
    // TODO: test if faster to create empty bit mask and set bits
    spin_det_t hole_mask(d.N_mos, constraint.first.size(),
                         &constraint.first[0]); // where holes can be created
    spin_det_t part_mask(d.N_mos, constraint.second.size(),
                         &constraint.second[0]); // where particles can be created

    // convert max orb to bit masks
    // TODO: as above, test if faster to create empty bit mask, now with
    // reserved size, and set bits
    spin_det_t max_orb_mask(d.N_mos, max_orb, 1);

    // apply bit masks and get final list
    spin_constraint_t alpha_holes = to_constraint((~d[0] & hole_mask) & max_orb_mask);
    spin_constraint_t alpha_parts = to_constraint((d[0] & part_mask) & max_orb_mask);

    spin_constraint_t beta_holes = to_constraint((~d[1] & hole_mask) & max_orb_mask);
    spin_constraint_t beta_parts = to_constraint((d[1] & part_mask) & max_orb_mask);

    // at this point, hole and particle bitsets are guaranteed to be disjoint
    // iterate over product list and add to return vector
    std::vector<det_t> alpha_ss_doubles =
        get_ss_doubles_by_exc_mask(d, 0, alpha_holes, alpha_parts);
    std::vector<det_t> beta_ss_doubles = get_ss_doubles_by_exc_mask(d, 1, beta_holes, beta_parts);
    ss_doubles.insert(ss_doubles.end(), alpha_ss_doubles.begin(), alpha_ss_doubles.end());
    ss_doubles.insert(ss_doubles.end(), beta_ss_doubles.begin(), beta_ss_doubles.end());

    return ss_doubles;
}