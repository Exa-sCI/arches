#include "arrays.h"
#include "determinant.h"
#include "integral_indexing_utils.h"
#include <algorithm>
/*
Interface functions for reading and setting values of managed pointers
*/
extern "C" {

// int 32
int at_i32(int *a, idx_t i) { return a[i]; }

void set_i32(int *a, idx_t i, int val) { a[i] = val; }

void set_range_i32(int *a, idx_t start, idx_t stop, int *fill) {
    std::copy(fill, fill + (stop - start), a + start);
}

void set_strided_range_i32(int *a, idx_t start, idx_t stop, idx_t step, int *fill) {
    auto j = 0;
    for (auto i = 0; i < stop; i += step, j++) {
        a[i] = fill[j];
    }
}

// int 64
long int at_i64(long int *a, idx_t i) { return a[i]; }

void set_i64(long int *a, idx_t i, long int val) { a[i] = val; }

void set_range_i64(long int *a, idx_t start, idx_t stop, long int *fill) {
    std::copy(fill, fill + (stop - start), a + start);
}

void set_strided_range_i64(long int *a, idx_t start, idx_t stop, idx_t step, long int *fill) {
    auto j = 0;
    for (auto i = 0; i < stop; i += step, j++) {
        a[i] = fill[j];
    }
}

// uint 32
unsigned int at_ui32(unsigned int *a, idx_t i) { return a[i]; }

void set_ui32(unsigned int *a, idx_t i, unsigned int val) { a[i] = val; }

void set_range_ui32(unsigned int *a, idx_t start, idx_t stop, unsigned int *fill) {
    std::copy(fill, fill + (stop - start), a + start);
}

void set_strided_range_ui32(unsigned int *a, idx_t start, idx_t stop, idx_t step,
                            unsigned int *fill) {
    auto j = 0;
    for (auto i = 0; i < stop; i += step, j++) {
        a[i] = fill[j];
    }
}

// uint 64
unsigned long int at_ui64(unsigned long int *a, idx_t i) { return a[i]; }

void set_ui64(unsigned long int *a, idx_t i, unsigned long int val) { a[i] = val; }

void set_range_ui64(unsigned long int *a, idx_t start, idx_t stop, unsigned long int *fill) {
    std::copy(fill, fill + (stop - start), a + start);
}

void set_strided_range_ui64(unsigned long int *a, idx_t start, idx_t stop, idx_t step,
                            unsigned long int *fill) {
    auto j = 0;
    for (auto i = 0; i < stop; i += step, j++) {
        a[i] = fill[j];
    }
}

// float 32
float at_f32(float *a, idx_t i) { return a[i]; }

void set_f32(float *a, idx_t i, float val) { a[i] = val; }

void set_range_f32(float *a, idx_t start, idx_t stop, float *fill) {
    std::copy(fill, fill + (stop - start), a + start);
}

void set_strided_range_f32(float *a, idx_t start, idx_t stop, idx_t step, float *fill) {
    auto j = 0;
    for (auto i = 0; i < stop; i += step, j++) {
        a[i] = fill[j];
    }
}

// float 64
double at_f64(double *a, idx_t i) { return a[i]; }

void set_f64(double *a, idx_t i, double val) { a[i] = val; }

void set_range_f64(double *a, idx_t start, idx_t stop, double *fill) {
    std::copy(fill, fill + (stop - start), a + start);
}

void set_strided_range_f64(double *a, idx_t start, idx_t stop, idx_t step, double *fill) {
    auto j = 0;
    for (auto i = 0; i < stop; i += step, j++) {
        a[i] = fill[j];
    }
}

// complex 64

// complex 128
}

/*
Constructors and destructors for simple arrays for the following types
// i32, i64, ui32, ui64, f32, f64, idx_t, det_t
*/
extern "C" {

// Empty constructors
LArray<float> *LArray_ctor_e_f32(idx_t n) { return new LArray<float>(n); }
LArray<double> *LArray_ctor_e_f64(idx_t n) { return new LArray<double>(n); }
LArray<int> *LArray_ctor_e_i32(idx_t n) { return new LArray<int>(n); }
LArray<long int> *LArray_ctor_e_i64(idx_t n) { return new LArray<long int>(n); }
LArray<unsigned int> *LArray_ctor_e_ui32(idx_t n) { return new LArray<unsigned int>(n); }
LArray<unsigned long int> *LArray_ctor_e_ui64(idx_t n) { return new LArray<unsigned long int>(n); }
LArray<idx_t> *LArray_ctor_e_idx_t(idx_t n) { return new LArray<idx_t>(n); }

// Fill constructors
LArray<float> *LArray_ctor_c_f32(idx_t n, float fill_val) { return new LArray<float>(n, fill_val); }
LArray<double> *LArray_ctor_c_f64(idx_t n, double fill_val) {
    return new LArray<double>(n, fill_val);
}
LArray<int> *LArray_ctor_c_i32(idx_t n, int fill_val) { return new LArray<int>(n, fill_val); }
LArray<long int> *LArray_ctor_c_i64(idx_t n, long int fill_val) {
    return new LArray<long int>(n, fill_val);
}
LArray<unsigned int> *LArray_ctor_c_ui32(idx_t n, unsigned int fill_val) {
    return new LArray<unsigned int>(n, fill_val);
}
LArray<unsigned long int> *LArray_ctor_c_ui64(idx_t n, unsigned long int fill_val) {
    return new LArray<unsigned long int>(n, fill_val);
}
LArray<idx_t> *LArray_ctor_c_idx_t(idx_t n, idx_t fill_val) {
    return new LArray<idx_t>(n, fill_val);
}

// Copy constructors
LArray<float> *LArray_ctor_a_f32(idx_t n, float *fill_val) {
    return new LArray<float>(n, fill_val);
}
LArray<double> *LArray_ctor_a_f64(idx_t n, double *fill_val) {
    return new LArray<double>(n, fill_val);
}
LArray<int> *LArray_ctor_a_i32(idx_t n, int *fill_val) { return new LArray<int>(n, fill_val); }
LArray<long int> *LArray_ctor_a_i64(idx_t n, long int *fill_val) {
    return new LArray<long int>(n, fill_val);
}
LArray<unsigned int> *LArray_ctor_a_ui32(idx_t n, unsigned int *fill_val) {
    return new LArray<unsigned int>(n, fill_val);
}
LArray<unsigned long int> *LArray_ctor_a_ui64(idx_t n, unsigned long int *fill_val) {
    return new LArray<unsigned long int>(n, fill_val);
}
LArray<idx_t> *LArray_ctor_a_idx_t(idx_t n, idx_t *fill_val) {
    return new LArray<idx_t>(n, fill_val);
}

// Destructors
void LArray_dtor_f32(LArray<float> *X) { delete X; }
void LArray_dtor_f64(LArray<double> *X) { delete X; }
void LArray_dtor_i32(LArray<int> *X) { delete X; }
void LArray_dtor_i64(LArray<long int> *X) { delete X; }
void LArray_dtor_ui32(LArray<unsigned int> *X) { delete X; }
void LArray_dtor_ui64(LArray<unsigned long int> *X) { delete X; }
void LArray_dtor_idx_t(LArray<idx_t> *X) { delete X; }
void LArray_dtor_det_t(LArray<det_t> *X) { delete X; }
}