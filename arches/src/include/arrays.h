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

        std::unique_ptr<T[]> p(new T[n]);
        ptr = std::move(p);
        arr = ptr.get();
    };

    // fill constructor
    LArray(idx_t n, T fill_val) : LArray(n) { std::fill(arr, arr + size, fill_val); };

    // copy constructor
    LArray(idx_t n, T *fill) : LArray(n) { std::copy(fill, fill + size, arr); };

    ~LArray() = default;
};

// no const array, in case *a=*b=*c; for implementing ipow2
template <class T> void mul_LArray(T *a, T *b, T *c, const idx_t N) {
    for (auto i = 0; i < N; i++) {
        c[i] = a[i] * b[i];
    }
};

// *a != *b, *c != *b
template <class T> void add_LArray(T *a, const T *b, T *c, const idx_t N) {
    for (auto i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
};

template <class T> void sub_LArray(T *a, const T *b, T *c, const idx_t N) {
    for (auto i = 0; i < N; i++) {
        c[i] = a[i] - b[i];
    }
};

template <class T> void mul_LArray(T *a, const T *b, T *c, const idx_t N) {
    for (auto i = 0; i < N; i++) {
        c[i] = a[i] * b[i];
    }
};

template <class T> void div_LArray(T *a, const T *b, T *c, const idx_t N) {
    for (auto i = 0; i < N; i++) {
        c[i] = a[i] / b[i];
    }
};

// *c !=  *a, *b
template <class T> void add_LArray(const T *a, const T *b, T *c, const idx_t N) {
    for (auto i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
};

template <class T> void sub_LArray(const T *a, const T *b, T *c, const idx_t N) {
    for (auto i = 0; i < N; i++) {
        c[i] = a[i] - b[i];
    }
};

template <class T> void mul_LArray(const T *a, const T *b, T *c, const idx_t N) {
    for (auto i = 0; i < N; i++) {
        c[i] = a[i] * b[i];
    }
};

template <class T> void div_LArray(const T *a, const T *b, T *c, const idx_t N) {
    for (auto i = 0; i < N; i++) {
        c[i] = a[i] / b[i];
    }
};

// op(A, constant)
template <class T> void add_LArray(T *a, const T b, T *c, const idx_t N) {
    for (auto i = 0; i < N; i++) {
        c[i] = a[i] + b;
    }
};

template <class T> void sub_LArray(T *a, const T b, T *c, const idx_t N) {
    for (auto i = 0; i < N; i++) {
        c[i] = a[i] - b;
    }
};

template <class T> void mul_LArray(T *a, const T b, T *c, const idx_t N) {
    for (auto i = 0; i < N; i++) {
        c[i] = a[i] * b;
    }
};

template <class T> void div_LArray(T *a, const T b, T *c, const idx_t N) {
    for (auto i = 0; i < N; i++) {
        c[i] = a[i] / b;
    }
};

// op(A, constant), *a != *c
template <class T> void add_LArray(const T *a, const T b, T *c, const idx_t N) {
    for (auto i = 0; i < N; i++) {
        c[i] = a[i] + b;
    }
};

template <class T> void sub_LArray(const T *a, const T b, T *c, const idx_t N) {
    for (auto i = 0; i < N; i++) {
        c[i] = a[i] - b;
    }
};

template <class T> void mul_LArray(const T *a, const T b, T *c, const idx_t N) {
    for (auto i = 0; i < N; i++) {
        c[i] = a[i] * b;
    }
};

template <class T> void div_LArray(const T *a, const T b, T *c, const idx_t N) {
    for (auto i = 0; i < N; i++) {
        c[i] = a[i] / b;
    }
};
