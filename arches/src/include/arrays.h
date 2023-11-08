#pragma once
#include "integral_indexing_utils.h"
#include <algorithm>
#include <memory>

template <class T> class LArray {
  protected:
    std::unique_ptr<T[]> ptr;

  public:
    idx_t size;
    T *arr;

    LArray(idx_t n) { // no initializaiton
        size = n;

        std::unique_ptr<Y[]> p(new Y[n]);
        ptr = std::move(p);
        arr = ptr.get();
    };

    // fill constructor
    LArray(idx_t n, T fill_val) : LArray(n) { std::fill(arr, arr + size, fill_val); };

    // copy constructor
    LArray(idx_t n, T *fill) : LArray(n) { std::copy(fill, fill + size, arr); };

    ~LArray() = default;
}