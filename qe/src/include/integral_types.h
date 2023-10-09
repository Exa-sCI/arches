#include "integral_indexing_utils.h"

// Could make this a member function of the ijkl tuple, if subclassed into
// canonical vs non-canonical tuples
extern "C" char integral_category(idx_t i, idx_t j, idx_t k, idx_t l);

template <typename T> int sgn(T val) { return (T(0) < val) - (val < T(0)); }