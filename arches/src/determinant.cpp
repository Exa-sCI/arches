#if !defined(DOCTEST_CONFIG_DISABLE)
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#endif
#include <determinant.h>
#include <doctest/doctest.h>

int compute_phase_single_excitation(spin_det_t d, idx_t h, idx_t p) {
    const auto &[i, j] = std::minmax(h, p);
    spin_det_t hpmask(d.N_mos);
    hpmask.set(i + 1, j, 1);
    const bool parity = (hpmask & d).count() % 2;
    return parity ? -1 : 1;
}

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
    return det_t(a[0] ^ b[0], a[1] ^ b[1]); // Does this work since I have the move defined?
}

spin_det_t apply_single_excitation(spin_det_t det, idx_t h, idx_t p, bool &success) {
    // assert(s[h] == 1);
    // assert(s[p] == 0);
    success = det[h] && !det[p];

    auto s2 = spin_det_t(det);
    if (success) {
        s2.set(h, 0);
        s2.set(p, 1);
    }
    return s2;
}

det_t apply_single_excitation(det_t det, int spin, idx_t h, idx_t p, bool &success) {
    // assert(s[spin][h] == 1);
    // assert(s[spin][p] == 0);

    success = det[spin][h] && !det[spin][p];

    auto s2 = det_t(det);
    if (success) {
        s2[spin].set(h, 0);
        s2[spin].set(p, 1);
    }
    return s2;
}

spin_det_t apply_double_excitation(spin_det_t det, idx_t h1, idx_t h2, idx_t p1, idx_t p2,
                                   bool &success) {

    success = det[h1] && det[h2] && !det[p1] && !det[p2];
    auto res = spin_det_t(det);
    if (success) {
        res.set(h1, 0);
        res.set(h2, 0);
        res.set(p1, 1);
        res.set(p2, 1);
    }
    return res;
}

det_t apply_double_excitation(det_t det, int s1, int s2, idx_t h1, idx_t h2, idx_t p1, idx_t p2,
                              bool &success) {

    success = det[s1][h1] && det[s2][h2] && !det[s1][p1] && !det[s2][p2];

    auto res = det_t(det);
    if (success) {
        res[s1].set(h1, 0);
        res[s2].set(h2, 0);
        res[s1].set(p1, 1);
        res[s2].set(p2, 1);
    }
    return res;
}

std::vector<det_t> get_singles_by_exc_mask(det_t d, int spin, spin_constraint_t h,
                                           spin_constraint_t p) {
    std::vector<det_t> res;
    bool success;
    for (auto &i : h) {
        for (auto &j : p) {
            det_t temp_det = apply_single_excitation(d, spin, i, j, success);
            if (success)
                res.push_back(temp_det);
        }
    }
    return res;
}

std::vector<spin_det_t> get_spin_singles_by_exc_mask(spin_det_t d, spin_constraint_t h,
                                                     spin_constraint_t p) {
    std::vector<spin_det_t> res;
    bool success;
    for (auto &i : h) {
        for (auto &j : p) {
            spin_det_t temp_det = apply_single_excitation(d, i, j, success);
            if (success)
                res.push_back(temp_det);
        }
    }
    return res;
}

std::vector<det_t> get_ss_doubles_by_exc_mask(det_t d, int spin, spin_constraint_t h,
                                              spin_constraint_t p) {

    std::vector<det_t> res;
    // h, p are sorted so h1 < h2; p1 < p2 always
    bool success;
    int N_holes = h.size();
    int N_parts = p.size();
    for (auto h1 = 0; h1 < N_holes - 1; h1++) {
        for (auto h2 = h1 + 1; h2 < N_holes; h2++) {
            for (auto p1 = 0; p1 < N_parts - 1; p1++) {
                for (auto p2 = p1 + 1; p2 < N_parts; p2++) {
                    det_t temp_det =
                        apply_double_excitation(d, spin, spin, h[h1], h[h2], p[p1], p[p2], success);
                    if (success)
                        res.push_back(temp_det);
                }
            }
        }
    }
    return res;
}

std::vector<det_t> get_all_singles(det_t d) {

    std::vector<det_t> singles;

    std::vector<det_t> alpha_singles =
        get_singles_by_exc_mask(d, 0, to_constraint(d[0]), to_constraint(~d[0]));
    std::vector<det_t> beta_singles =
        get_singles_by_exc_mask(d, 1, to_constraint(d[1]), to_constraint(~d[1]));
    singles.insert(singles.end(), alpha_singles.begin(), alpha_singles.end());
    singles.insert(singles.end(), beta_singles.begin(), beta_singles.end());

    return singles;
}

std::vector<det_t> get_constrained_singles(det_t d, exc_constraint_t alpha_constraint,
                                           exc_constraint_t beta_constraint) {
    std::vector<det_t> singles;

    // convert constraints to bit masks
    spin_det_t ah_mask(d.N_mos, alpha_constraint.first.size(),
                       &alpha_constraint.first[0]); // where holes can be created
    spin_det_t ap_mask(d.N_mos, alpha_constraint.second.size(),
                       &alpha_constraint.second[0]); // where particles can be created

    spin_det_t bh_mask(d.N_mos, beta_constraint.first.size(),
                       &beta_constraint.first[0]); // where holes can be created
    spin_det_t bp_mask(d.N_mos, beta_constraint.second.size(),
                       &beta_constraint.second[0]); // where particles can be created

    // apply bit masks and get final list
    spin_constraint_t alpha_holes = to_constraint((d[0] & ah_mask));
    spin_constraint_t alpha_parts = to_constraint((~d[0] & ap_mask));

    spin_constraint_t beta_holes = to_constraint((d[1] & bh_mask));
    spin_constraint_t beta_parts = to_constraint((~d[1] & bp_mask));

    // at this point, hole and particle bitsets are guaranteed to be disjoint
    // iterate over product list and add to return vector
    std::vector<det_t> alpha_singles = get_singles_by_exc_mask(d, 0, alpha_holes, alpha_parts);
    std::vector<det_t> beta_singles = get_singles_by_exc_mask(d, 1, beta_holes, beta_parts);
    singles.insert(singles.end(), alpha_singles.begin(), alpha_singles.end());
    singles.insert(singles.end(), beta_singles.begin(), beta_singles.end());

    return singles;
}

std::vector<det_t> get_os_doubles(det_t d, bool return_singles) {
    std::vector<det_t> os_doubles;

    spin_constraint_t alpha_holes = to_constraint(d[0]);
    spin_constraint_t alpha_parts = to_constraint(~d[0]);

    spin_constraint_t beta_holes = to_constraint(d[1]);
    spin_constraint_t beta_parts = to_constraint(~d[1]);

    // get all singles and iterate over product of (1,0) X (0,1) to get (1,1)
    std::vector<spin_det_t> alpha_singles =
        get_spin_singles_by_exc_mask(d[0], alpha_holes, alpha_parts);
    std::vector<spin_det_t> beta_singles =
        get_spin_singles_by_exc_mask(d[1], beta_holes, beta_parts);

    if (return_singles) {
        for (auto &a : alpha_singles)
            os_doubles.push_back(det_t(a, d.beta));
        for (auto &b : beta_singles)
            os_doubles.push_back(det_t(d.alpha, b));
    }

    for (auto &a : alpha_singles) {
        for (auto &b : beta_singles) {
            os_doubles.push_back(det_t(a, b));
        }
    }

    return os_doubles;
}

// TODO: refactor, a lot of code re-use
std::vector<det_t> get_constrained_os_doubles(det_t d, exc_constraint_t alpha_constraint,
                                              exc_constraint_t beta_constraint,
                                              bool return_singles) {
    std::vector<det_t> os_doubles;

    // convert constraints to bit masks
    spin_det_t ah_mask(d.N_mos, alpha_constraint.first.size(),
                       &alpha_constraint.first[0]); // where holes can be created
    spin_det_t ap_mask(d.N_mos, alpha_constraint.second.size(),
                       &alpha_constraint.second[0]); // where particles can be created

    spin_det_t bh_mask(d.N_mos, beta_constraint.first.size(),
                       &beta_constraint.first[0]); // where holes can be created
    spin_det_t bp_mask(d.N_mos, beta_constraint.second.size(),
                       &beta_constraint.second[0]); // where particles can be created

    // apply bit masks and get final list
    spin_constraint_t alpha_holes = to_constraint((d[0] & ah_mask));
    spin_constraint_t alpha_parts = to_constraint((~d[0] & ap_mask));

    spin_constraint_t beta_holes = to_constraint((d[1] & bh_mask));
    spin_constraint_t beta_parts = to_constraint((~d[1] & bp_mask));

    // get all singles and iterate over product of (1,0) X (0,1) to get (1,1)
    std::vector<spin_det_t> alpha_singles =
        get_spin_singles_by_exc_mask(d[0], alpha_holes, alpha_parts);
    std::vector<spin_det_t> beta_singles =
        get_spin_singles_by_exc_mask(d[1], beta_holes, beta_parts);

    if (return_singles) {
        for (auto &a : alpha_singles)
            os_doubles.push_back(det_t(a, d.beta));
        for (auto &b : beta_singles)
            os_doubles.push_back(det_t(d.alpha, b));
    }

    for (auto &a : alpha_singles) {
        for (auto &b : beta_singles) {
            os_doubles.push_back(det_t(a, b));
        }
    }

    return os_doubles;
}

std::vector<det_t> get_ss_doubles(det_t d) {
    std::vector<det_t> ss_doubles;

    spin_constraint_t alpha_holes = to_constraint(d[0]);
    spin_constraint_t alpha_parts = to_constraint(~d[0]);

    spin_constraint_t beta_holes = to_constraint(d[1]);
    spin_constraint_t beta_parts = to_constraint(~d[1]);

    // at this point, hole and particle bitsets are guaranteed to be disjoint
    // iterate over product list and add to return vector
    std::vector<det_t> alpha_ss_doubles =
        get_ss_doubles_by_exc_mask(d, 0, alpha_holes, alpha_parts);
    std::vector<det_t> beta_ss_doubles = get_ss_doubles_by_exc_mask(d, 1, beta_holes, beta_parts);
    ss_doubles.insert(ss_doubles.end(), alpha_ss_doubles.begin(), alpha_ss_doubles.end());
    ss_doubles.insert(ss_doubles.end(), beta_ss_doubles.begin(), beta_ss_doubles.end());

    return ss_doubles;
}

std::vector<det_t> get_constrained_ss_doubles(det_t d, exc_constraint_t alpha_constraint,
                                              exc_constraint_t beta_constraint) {
    std::vector<det_t> ss_doubles;

    // convert constraints to bit masks
    spin_det_t ah_mask(d.N_mos, alpha_constraint.first.size(),
                       &alpha_constraint.first[0]); // where holes can be created
    spin_det_t ap_mask(d.N_mos, alpha_constraint.second.size(),
                       &alpha_constraint.second[0]); // where particles can be created

    spin_det_t bh_mask(d.N_mos, beta_constraint.first.size(),
                       &beta_constraint.first[0]); // where holes can be created
    spin_det_t bp_mask(d.N_mos, beta_constraint.second.size(),
                       &beta_constraint.second[0]); // where particles can be created

    // apply bit masks and get final list
    spin_constraint_t alpha_holes = to_constraint((d[0] & ah_mask));
    spin_constraint_t alpha_parts = to_constraint((~d[0] & ap_mask));

    spin_constraint_t beta_holes = to_constraint((d[1] & bh_mask));
    spin_constraint_t beta_parts = to_constraint((~d[1] & bp_mask));

    // at this point, hole and particle bitsets are guaranteed to be disjoint
    // iterate over product list and add to return vector
    std::vector<det_t> alpha_ss_doubles =
        get_ss_doubles_by_exc_mask(d, 0, alpha_holes, alpha_parts);
    std::vector<det_t> beta_ss_doubles = get_ss_doubles_by_exc_mask(d, 1, beta_holes, beta_parts);
    ss_doubles.insert(ss_doubles.end(), alpha_ss_doubles.begin(), alpha_ss_doubles.end());
    ss_doubles.insert(ss_doubles.end(), beta_ss_doubles.begin(), beta_ss_doubles.end());

    return ss_doubles;
}

void print(spin_det_t *det) {
    auto N_orbs = det->N_mos;
    for (auto i = 0; i < N_orbs; i++) {
        std::cout << det->operator[](i);
    }
    std::cout << std::endl;
}

std::vector<det_t> get_all_connected_dets(det_t *d, idx_t N_dets) {

    std::vector<det_t> connected;
    LinearUnorderedMap hash_map = LinearUnorderedMap();
    for (auto i = 0; i < N_dets; i++) {
        hash_map.add_det(d[i]);
    }

    for (auto i = 0; i < N_dets; i++) {
        auto &current_det = d[i];
        std::vector<det_t> singles_and_os_doubles = get_os_doubles(current_det, true);
        std::vector<det_t> ss_doubles = get_ss_doubles(current_det);

        for (auto &new_det : singles_and_os_doubles) {
            if (hash_map.add_det(new_det)) {
                connected.push_back(new_det);
            }
        }

        for (auto &new_det : ss_doubles) {
            if (hash_map.add_det(new_det)) {
                connected.push_back(new_det);
            }
        }
    }

    return connected;
}

std::vector<det_t> get_constrained_connected_dets(det_t *d, idx_t N_dets,
                                                  exc_constraint_t alpha_constraint,
                                                  exc_constraint_t beta_constraint) {

    std::vector<det_t> connected;
    LinearUnorderedMap hash_map = LinearUnorderedMap();

    for (auto i = 0; i < N_dets; i++) {
        hash_map.add_det(d[i]);
    }

    for (auto i = 0; i < N_dets; i++) {
        auto &current_det = d[i];
        std::vector<det_t> singles_and_os_doubles =
            get_constrained_os_doubles(current_det, alpha_constraint, beta_constraint, true);
        std::vector<det_t> ss_doubles =
            get_constrained_ss_doubles(current_det, alpha_constraint, beta_constraint);

        int count = 0;
        for (auto &new_det : singles_and_os_doubles) {
            if (hash_map.add_det(new_det))
                connected.push_back(new_det);
        }

        count = 0;
        if (ss_doubles.size()) {

            for (auto &new_det : ss_doubles) {

                if (hash_map.add_det(new_det))
                    connected.push_back(new_det);
            }
        }
    }

    return connected;
}

extern "C" {

//// spin_det_t
// constructors
spin_det_t *Dets_spin_det_t_empty_ctor(idx_t N_mos) { return new spin_det_t(N_mos); }

spin_det_t *Dets_spin_det_t_fill_ctor(idx_t N_mos, idx_t max_orb) {
    return new spin_det_t(N_mos, max_orb, true);
}

spin_det_t *Dets_spin_det_t_orb_list_ctor(idx_t N_mos, idx_t N_filled, idx_t *orbs) {
    return new spin_det_t(N_mos, N_filled, orbs);
}

// destructor
void Dets_spin_det_t_dtor(spin_det_t *sdet) { delete sdet; }

// utilities
void Dets_spin_det_t_print(spin_det_t *det) {
    auto N_orbs = det->N_mos;
    for (auto i = 0; i < N_orbs; i++) {
        std::cout << det->operator[](i);
    }
    std::cout << std::endl;
}

void Dets_spin_det_t_to_bit_tuple(spin_det_t *det, idx_t start_orb, idx_t end_orb, int *t) {
    auto j = 0;
    for (auto i = start_orb; i < end_orb; i++, j++) {
        t[j] = (int)det->operator[](i);
    }
}

// operations
void Dets_spin_det_t_set_orb(spin_det_t *det, idx_t orb, bool val) { det->set(orb, val); }

void Dets_spin_det_t_set_orb_range(spin_det_t *det, idx_t min_orb, idx_t max_orb, bool val) {
    det->set(min_orb, max_orb, val);
}
bool Dets_spin_det_t_get_orb(spin_det_t *det, idx_t orb) { return det->operator[](orb); }

spin_det_t *Dets_spin_det_t_bit_flip(spin_det_t *det) { return new spin_det_t(det->operator~()); }
spin_det_t *Dets_spin_det_t_xor(spin_det_t *det, spin_det_t *other) {
    return new spin_det_t(det->operator^(*other));
}
spin_det_t *Dets_spin_det_t_and(spin_det_t *det, spin_det_t *other) {
    return new spin_det_t(det->operator&(*other));
}

int Dets_spin_det_t_count(spin_det_t *det) { return det->count(); }

int Dets_spin_det_t_phase_single_exc(spin_det_t *det, idx_t h, idx_t p) {
    return compute_phase_single_excitation(*det, h, p);
}

int Dets_spin_det_t_phase_double_exc(spin_det_t *det, idx_t h1, idx_t h2, idx_t p1, idx_t p2) {
    return compute_phase_double_excitation(*det, h1, h2, p1, p2);
}

spin_det_t *Dets_spin_det_t_apply_single_exc(spin_det_t *det, idx_t h, idx_t p) {
    bool success;
    return new spin_det_t(apply_single_excitation(*det, h, p, success));
}

spin_det_t *Dets_spin_det_t_apply_double_exc(spin_det_t *det, idx_t h1, idx_t h2, idx_t p1,
                                             idx_t p2) {
    bool success;
    return new spin_det_t(apply_double_excitation(*det, h1, h2, p1, p2, success));
}

//// det_t

// constructors
det_t *Dets_det_t_empty_ctor(idx_t N_mos) { return new det_t(N_mos); }
det_t *Dets_det_t_copy_ctor(spin_det_t *alpha, spin_det_t *beta) {
    return new det_t(*alpha, *beta);
}

// destructor
void Dets_det_t_dtor(det_t *det) { delete det; }

// utilities
spin_det_t *Dets_det_t_get_spin_det_handle(det_t *det, bool spin) {
    return spin ? &det->beta : &det->alpha;
}

// operations
int Dets_det_t_phase_double_exc(det_t *det, idx_t h1, idx_t h2, idx_t p1, idx_t p2) {
    return compute_phase_double_excitation(*det, h1, h2, p1, p2);
}

det_t *Dets_det_t_exc_det(det_t *det_1, det_t *det_2) { return new det_t(exc_det(*det_1, *det_2)); }

det_t *Dets_det_t_apply_single_exc(det_t *det, idx_t spin, idx_t h, idx_t p) {
    bool success;
    return new det_t(apply_single_excitation(*det, spin, h, p, success));
}

det_t *Dets_det_t_apply_double_exc(det_t *det, idx_t s1, idx_t s2, idx_t h1, idx_t h2, idx_t p1,
                                   idx_t p2) {
    bool success;
    return new det_t(apply_double_excitation(*det, s1, s2, h1, h2, p1, p2, success));
}

//// DetArray

// constructor

DetArray *Dets_DetArray_empty_ctor(idx_t N_dets, idx_t N_orbs) {
    return new DetArray(N_dets, N_orbs);
}

// destructor

void Dets_DetArray_dtor(DetArray *arr) { delete arr; }

// member returns

idx_t Dets_DetArray_get_N_dets(DetArray *arr) { return arr->size; }
idx_t Dets_DetArray_get_N_mos(DetArray *arr) { return arr->N_mos; }
det_t *Dets_DetArray_get_arr_pointer(DetArray *arr) { return arr->arr; }

// utilities

det_t *Dets_DetArray_getitem(DetArray *arr, idx_t i) { return &arr->arr[i]; }
void Dets_DetArray_setitem(DetArray *arr, det_t *other, idx_t i) { arr->arr[i] = *other; }

//// Det generation routines

DetArray *Dets_get_connected_singles(det_t *source, idx_t N_dets) {
    return new DetArray(get_all_singles(*source));
}

DetArray *Dets_get_connected_same_spin_doubles(det_t *source, idx_t N_dets) {
    return new DetArray(get_ss_doubles(*source));
}

DetArray *Dets_get_connected_opp_spin_doubles(det_t *source, idx_t N_dets) {
    return new DetArray(get_os_doubles(*source, false));
}

DetArray *Dets_get_connected_dets(det_t *source, idx_t N_dets) {
    return new DetArray(get_all_connected_dets(source, N_dets));
}

DetArray *Dets_get_constrained_singles(det_t *source, idx_t N_dets, idx_t *h_a, idx_t N_h_a,
                                       idx_t *h_b, idx_t N_h_b, idx_t *p_a, idx_t N_p_a, idx_t *p_b,
                                       idx_t N_p_b) {
    spin_constraint_t ah(h_a, h_a + N_h_a);
    spin_constraint_t ap(p_a, p_a + N_p_a);
    spin_constraint_t bh(h_b, h_b + N_h_b);
    spin_constraint_t bp(p_b, p_b + N_p_b);

    const exc_constraint_t a(ah, ap);
    const exc_constraint_t b(bh, bp);

    return new DetArray(get_constrained_singles(*source, a, b));
}

DetArray *Dets_get_constrained_same_spin_doubles(det_t *source, idx_t N_dets, idx_t *h_a,
                                                 idx_t N_h_a, idx_t *h_b, idx_t N_h_b, idx_t *p_a,
                                                 idx_t N_p_a, idx_t *p_b, idx_t N_p_b) {
    spin_constraint_t ah(h_a, h_a + N_h_a);
    spin_constraint_t ap(p_a, p_a + N_p_a);
    spin_constraint_t bh(h_b, h_b + N_h_b);
    spin_constraint_t bp(p_b, p_b + N_p_b);

    const exc_constraint_t a(ah, ap);
    const exc_constraint_t b(bh, bp);

    return new DetArray(get_constrained_ss_doubles(*source, a, b));
}

DetArray *Dets_get_constrained_opp_spin_doubles(det_t *source, idx_t N_dets, idx_t *h_a,
                                                idx_t N_h_a, idx_t *h_b, idx_t N_h_b, idx_t *p_a,
                                                idx_t N_p_a, idx_t *p_b, idx_t N_p_b) {
    spin_constraint_t ah(h_a, h_a + N_h_a);
    spin_constraint_t ap(p_a, p_a + N_p_a);
    spin_constraint_t bh(h_b, h_b + N_h_b);
    spin_constraint_t bp(p_b, p_b + N_p_b);

    const exc_constraint_t a(ah, ap);
    const exc_constraint_t b(bh, bp);

    return new DetArray(get_constrained_os_doubles(*source, a, b, false));
}

DetArray *Dets_get_constrained_dets(det_t *source, idx_t N_dets, idx_t *h_a, idx_t N_h_a,
                                    idx_t *h_b, idx_t N_h_b, idx_t *p_a, idx_t N_p_a, idx_t *p_b,
                                    idx_t N_p_b) {
    spin_constraint_t ah(h_a, h_a + N_h_a);
    spin_constraint_t ap(p_a, p_a + N_p_a);
    spin_constraint_t bh(h_b, h_b + N_h_b);
    spin_constraint_t bp(p_b, p_b + N_p_b);

    const exc_constraint_t a(ah, ap);
    const exc_constraint_t b(bh, bp);

    return new DetArray(get_constrained_connected_dets(source, N_dets, a, b));
}
}