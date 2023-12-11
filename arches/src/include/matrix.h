#pragma once
#include "determinant.h"
#include "integral_indexing_utils.h"
#include <algorithm>
#include <memory>

// typedef long int idx_t;

// enum s_major { ROW, COL };
// TODO : figure out interface if we want both row and col major matrices
template <class T> class DMatrix {
  protected:
    std::unique_ptr<T[]> A_ptr;

  public:
    idx_t m; // row rank
    idx_t n; // col rank
    T *A;    // matrix entries
    // s_major storage;

    // TODO: adjust class for offload semantics
    // 1) In constructor, map A to target
    // 2) In constructor, store device pointer for A
    // 3) In destructor, make sure target is moved back/cleared

    // no initialization of values
    DMatrix(idx_t M, idx_t N) {
        m = M;
        n = N;

        std::unique_ptr<T[]> p(new T[m * n]);
        A_ptr = std::move(p);
        A = A_ptr.get();
    }

    // fill constructor
    DMatrix(idx_t M, idx_t N, T fill_val) : DMatrix(M, N) { std::fill(A, A + m * n, fill_val); };

    // copy constructor
    DMatrix(idx_t M, idx_t N, T *arr) : DMatrix(M, N) { std::copy(arr, arr + m * n, A); };

    // TODO: some form of slice assignment? In general, utilities for assigning, copying, and
    // returning submatrices

    ~DMatrix() = default;
};

template <class T> class SymCSRMatrix {
    // TODO: evaluate whether or not it would be useful to have list of list version, to convert
    // back and forth between

  protected:
    std::unique_ptr<idx_t[]> A_p_ptr;
    std::unique_ptr<idx_t[]> A_c_ptr;
    std::unique_ptr<T[]> A_v_ptr;

  public:
    idx_t m;    // row rank
    idx_t n;    // col rank
    idx_t *A_p; // row pointers
    idx_t *A_c; // col indices
    T *A_v;     // matrix entries

    // empty default constructor, to be able to get a handle
    SymCSRMatrix(){};

    // copy constructor
    SymCSRMatrix(idx_t M, idx_t N, idx_t *arr_p, idx_t *arr_c, T *arr_v) {
        m = M;
        n = N;

        // allocate row pointer array
        std::unique_ptr<idx_t[]> A_p_tmp(new idx_t[m + 1]);
        A_p_ptr = std::move(A_p_tmp);
        A_p = A_p_ptr.get();
        std::copy(arr_p, arr_p + m, A_p);

        // allocate col arr
        idx_t n_entries = A_p[m];
        std::unique_ptr<idx_t[]> A_c_tmp(new idx_t[n_entries]);
        A_c_ptr = std::move(A_c_tmp);
        A_c = A_c_ptr.get();
        std::copy(arr_c, arr_c + n_entries, A_c);

        // allocate values
        std::unique_ptr<T[]> A_v_tmp(new T[n_entries]);
        A_v_ptr = std::move(A_v_tmp);
        A_v = A_v_ptr.get();

        std::copy(arr_v, arr_v + n_entries, A_v);
    };

    // copy constructor from unique ptr, using move semantics
    SymCSRMatrix(idx_t M, idx_t N, std::unique_ptr<idx_t[]> &arr_p, std::unique_ptr<idx_t[]> &arr_c,
                 std::unique_ptr<T[]> &arr_v) {
        m = M;
        n = N;

        A_p_ptr = std::move(arr_p);
        A_c_ptr = std::move(arr_c);
        A_v_ptr = std::move(arr_v);

        A_p = A_p_ptr.get();
        A_c = A_c_ptr.get();
        A_v = A_v_ptr.get();
    }

    // This is way too slow for actual formation of explicit Hamiltonians, but it's easy to write!
    // Should get bilinear mappings so that we can iterate over known determinants and find
    // connections directly. Or, resort to on the fly generation of the Hamiltonian structure, which
    // would need true expandable vectors inside the offloaded kernels
    SymCSRMatrix(DetArray *psi_det) {
        idx_t N_dets = psi_det->size;
        m = N_dets;
        n = N_dets;

        std::vector<std::vector<idx_t>> csr_rows;
        std::unique_ptr<idx_t[]> H_p(new idx_t[N_dets + 1]);
        idx_t *H_p_p = H_p.get();
        std::fill(H_p_p, H_p_p + N_dets + 1, 0);
        // find non-zero entries
        for (auto i = 0; i < N_dets; i++) {
            det_t &d_row = psi_det->arr[i];

            H_p[i + 1] += 1; // add H_ii
            std::vector<idx_t> new_row(1, i);
            csr_rows.emplace_back(std::move(new_row));
            for (auto j = i + 1; j < N_dets; j++) {
                det_t &d_col = psi_det->arr[j];

                det_t exc = exc_det(d_row, d_col);
                auto degree = (exc[0].count() + exc[1].count()) / 2;

                if (degree <= 2) { // add H_ij
                    csr_rows[i].push_back(j);
                    H_p[i + 1] += 1;
                }
            }

            H_p[i + 1] += H_p[i]; // adjust global row offset
        }

        // copy over from vector of vectors into single array
        std::unique_ptr<idx_t[]> H_c(new idx_t[H_p[N_dets]]);
        idx_t *H_c_p = H_c.get();
        for (auto i = 0; i < N_dets; i++) {
            std::copy(csr_rows[i].begin(), csr_rows[i].end(), H_c_p + H_p[i]);
        }

        // initialize values at 0
        std::unique_ptr<T[]> H_v(new T[H_p[N_dets]]);
        T *H_v_p = H_v.get();
        std::fill(H_v_p, H_v_p + H_p[N_dets], (T)0.0);

        A_p_ptr = std::move(H_p);
        A_c_ptr = std::move(H_c);
        A_v_ptr = std::move(H_v);

        A_p = A_p_ptr.get();
        A_c = A_c_ptr.get();
        A_v = A_v_ptr.get();
    }

    // full move operator
    SymCSRMatrix &operator=(SymCSRMatrix &&other) {
        m = other.m;
        n = other.n;

        A_p_ptr = std::move(other.A_p_ptr);
        A_c_ptr = std::move(other.A_c_ptr);
        A_v_ptr = std::move(other.A_v_ptr);

        A_p = A_p_ptr.get();
        A_c = A_c_ptr.get();
        A_v = A_v_ptr.get();
        return *this;
    };
};

template <class T>
void gemm_kernel(char op_a, char op_b, idx_t m, idx_t n, idx_t k, T alpha, T *A, idx_t lda, T *B,
                 idx_t ldb, T beta, T *C, idx_t ldc) {
    // Ignore everything for now until I actually interface to blas
    // Assume row major storage
    std::unique_ptr<T[]> bT_p(new T[k * n]);
    T *bT = bT_p.get();

    for (auto i = 0; i < k; i++) {
        for (auto j = 0; j < n; j++) {
            bT[j * n + i] = B[i * k + j];
        }
    }

    for (auto ii = 0; ii < m; ii++) {
        for (auto jj = 0; jj < n; jj++) {
            for (auto kk = 0; kk < k; kk++) {
                C[ii * m + jj] = alpha * A[ii * m + kk] * bT[jj * k + kk] + beta * C[ii * m + jj];
            }
        }
    }
};

template <class T>
void ApB(const char op_A, const char op_B, const idx_t m, const idx_t n, T *A, const idx_t lda,
         T *B, const idx_t ldb, T *C, const idx_t ldc) {
    if (op_A == 'n') {
        if (op_B == 'n') {
            for (auto i = 0; i < m; i++) {
                for (auto j = 0; j < n; j++) {
                    C[i * ldc + j] = A[i * lda + j] + B[i * ldb + j];
                }
            }
        } else if (op_B == 't') {
            idx_t ib = 0;
            idx_t jb = 0;
            for (auto ia = 0; ia < m; ia++, jb++) {
                for (auto ja = 0; ja < n; ja++, ib++) {
                    C[ia * ldc + ja] = A[ia * lda + ja] + B[ib * ldb + jb];
                }
                ib -= n;
            }
        }

    } else if (op_A == 't') {
        if (op_B == 'n') {
            idx_t ia = 0;
            idx_t ja = 0;
            for (auto ib = 0; ib < m; ja++, ib++) {
                for (auto jb = 0; jb < n; ia++, jb++) {
                    C[ia * ldc + ja] = A[ia * lda + ja] + B[ib * ldb + jb];
                }
                ia -= n;
            }
        } else if (op_B == 't') {
            for (auto i = 0; i < n; i++) {
                for (auto j = 0; j < m; j++) {
                    C[i * ldc + j] = A[i * lda + j] + B[i * ldb + j];
                }
            }
        }
    }
};

template <class T>
void AmB(const char op_A, const char op_B, const idx_t m, const idx_t n, T *A, const idx_t lda,
         T *B, const idx_t ldb, T *C, const idx_t ldc) {
    if (op_A == 'n') {
        if (op_B == 'n') {
            for (auto i = 0; i < m; i++) {
                for (auto j = 0; j < n; j++) {
                    C[i * ldc + j] = A[i * lda + j] - B[i * ldb + j];
                }
            }
        } else if (op_B == 't') {
            idx_t ib = 0;
            idx_t jb = 0;
            for (auto ia = 0; ia < m; ia++, jb++) {
                for (auto ja = 0; ja < n; ja++, ib++) {
                    C[ia * ldc + ja] = A[ia * lda + ja] - B[ib * ldb + jb];
                }
                ib -= n;
            }
        }

    } else if (op_A == 't') {
        if (op_B == 'n') {
            idx_t ia = 0;
            idx_t ja = 0;
            for (auto ib = 0; ib < m; ja++, ib++) {
                for (auto jb = 0; jb < n; ia++, jb++) {
                    C[ia * ldc + ja] = A[ia * lda + ja] - B[ib * ldb + jb];
                }
                ia -= n;
            }
        } else if (op_B == 't') {
            for (auto i = 0; i < n; i++) {
                for (auto j = 0; j < m; j++) {
                    C[i * ldc + j] = A[i * lda + j] - B[i * ldb + j];
                }
            }
        }
    }
}

template <class T>
void set_submatrix(const char op_A, const char op_B, const idx_t m, const idx_t n, T *A,
                   const idx_t lda, const T *B, const idx_t ldb) {

    if (op_A == 'n') {
        if (op_B == 'n') {
            for (auto i = 0; i < m; i++) {
                for (auto j = 0; j < n; j++) {
                    A[i * lda + j] = B[i * ldb + j];
                }
            }
        } else if (op_B == 't') {
            idx_t ib = 0;
            idx_t jb = 0;
            for (auto ia = 0; ia < m; ia++, jb++) {
                for (auto ja = 0; ja < n; ja++, ib++) {
                    A[ia * lda + ja] = B[ib * ldb + jb];
                }
                ib -= n;
            }
        }

    } else if (op_A == 't') {
        if (op_B == 'n') {
            idx_t ia = 0;
            idx_t ja = 0;
            for (auto ib = 0; ib < m; ja++, ib++) {
                for (auto jb = 0; jb < n; ia++, jb++) {
                    A[ia * lda + ja] = B[ib * ldb + jb];
                }
                ia -= n;
            }
        } else if (op_B == 't') {
            for (auto i = 0; i < n; i++) {
                for (auto j = 0; j < m; j++) {
                    A[i * lda + j] = B[i * ldb + j];
                }
            }
        }
    }
}
