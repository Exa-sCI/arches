#pragma once
#include <algorithm>
#include <memory>

// Abstract storage object for 1 and 2 electron integrals
template <class T> class JChunk {

  protected:
    std::unique_ptr<idx_t[]> idx_ptr;
    std::unique_ptr<T[]> J_ptr;

  public:
    idx_t size; // size of chunk
    idx_t *ind; // compound two- or four- indices
    T *J;       // integrals

    template <class Y> JChunk(idx_t N, idx_t *J_ind, Y *J_val) {
        size = N;
        std::unique_ptr<idx_t[]> p_ind(new idx_t[size]);
        std::unique_ptr<Y[]> p_val(new Y[size]);

        idx_ptr = std::move(p_ind);
        J_ptr = std::move(p_val);

        std::copy(J_ind, J_ind + size, ind);
        std::copy(J_val, J_val + size, J);
    }

    ~JChunk() = default;
};
