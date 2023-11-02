#include <algorithm>
/*
Interface functions for reading and setting values ot managed pointers
*/

typedef long int idx_t;

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