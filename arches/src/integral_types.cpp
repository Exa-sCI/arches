#include "integral_types.h"
#include <iostream>
#include <stdexcept>

extern "C" char integral_category(const idx_t i, const idx_t j, const idx_t k, const idx_t l) {

    struct ijkl_tuple in_idx = {i, j, k, l};
    struct ijkl_tuple can_idx = canonical_idx4(i, j, k, l);

    try {
        bool canonical = in_idx == can_idx;
        if (!canonical)
            throw std::invalid_argument("Input integral not canonical.");
    } catch (const std::invalid_argument &e) {
        std::cerr << e.what() << '\n';
    }

    // How fast does this need to be? How often are the integral categories
    // calculated?
    char res;

    if (i == l) {
        res = 'A';
    } else if ((i == k) && (j == l)) {
        res = 'B';
    } else if ((i == k) || (j == l)) {
        if (j == k) {
            res = 'D';
        } else {
            res = 'C';
        }
    } else if (j == k) {
        res = 'E';
    } else if ((i == j) && (k == l)) {
        res = 'F';
    } else if ((i == j) || (k == l)) {
        res = 'E';
    } else {
        res = 'G';
    }

    return res;
}
