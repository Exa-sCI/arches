#include "matrix.h"
#include "mkl.h"
#include "mkl_spblas.h"

// ctypes matrix routine interfaces
extern "C" {

void sgemm_mkl(char op_a, char op_b, idx_t m, idx_t n, idx_t k, float alpha, const float *A,
               idx_t lda, const float *B, idx_t ldb, float beta, float *C, idx_t ldc) {

    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_TRANSPOSE trans_A = op_a == 't' ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE trans_B = op_b == 't' ? CblasTrans : CblasNoTrans;

    cblas_sgemm(layout, trans_A, trans_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void dgemm_mkl(char op_a, char op_b, idx_t m, idx_t n, idx_t k, double alpha, const double *A,
               idx_t lda, const double *B, idx_t ldb, double beta, double *C, idx_t ldc) {

    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_TRANSPOSE trans_A = op_a == 't' ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE trans_B = op_b == 't' ? CblasTrans : CblasNoTrans;

    cblas_dgemm(layout, trans_A, trans_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void sym_csr_s_MM_mkl(const float alpha, idx_t *A_rows, idx_t *A_cols, float *A_vals,
                      const float *B, const float beta, float *C, const idx_t M, const idx_t K,
                      const idx_t N) {
    const sparse_operation_t op_A = SPARSE_OPERATION_NON_TRANSPOSE;
    sparse_matrix_t csrA;
    sparse_status_t status;
    struct matrix_descr descr_A;
    descr_A.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
    descr_A.mode = SPARSE_FILL_MODE_UPPER;
    descr_A.diag = SPARSE_DIAG_NON_UNIT;

    mkl_sparse_s_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO, M, K, A_rows, A_rows + 1, A_cols,
                            A_vals);
    status =
        mkl_sparse_s_mm(op_A, alpha, csrA, descr_A, SPARSE_LAYOUT_ROW_MAJOR, B, N, K, beta, C, M);

    mkl_sparse_destroy(csrA);
}

void sym_csr_d_MM_mkl(const double alpha, idx_t *A_rows, idx_t *A_cols, double *A_vals,
                      const double *B, const double beta, double *C, const idx_t M, const idx_t K,
                      const idx_t N) {
    const sparse_operation_t op_A = SPARSE_OPERATION_NON_TRANSPOSE;
    sparse_matrix_t csrA;
    sparse_status_t status;
    struct matrix_descr descr_A;
    descr_A.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
    descr_A.mode = SPARSE_FILL_MODE_UPPER;
    descr_A.diag = SPARSE_DIAG_NON_UNIT;

    mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO, M, K, A_rows, A_rows + 1, A_cols,
                            A_vals);
    status =
        mkl_sparse_d_mm(op_A, alpha, csrA, descr_A, SPARSE_LAYOUT_ROW_MAJOR, B, N, K, beta, C, M);
    mkl_sparse_destroy(csrA);
}

void DMatrix_sApB(float *A, float *B, float *C, idx_t m, idx_t n) { ApB(A, B, C, m, n); }
void DMatrix_dApB(double *A, double *B, double *C, idx_t m, idx_t n) { ApB(A, B, C, m, n); }
void DMatrix_sAmB(float *A, float *B, float *C, idx_t m, idx_t n) { AmB(A, B, C, m, n); }
void DMatrix_dAmB(double *A, double *B, double *C, idx_t m, idx_t n) { AmB(A, B, C, m, n); }
}

// ctypes handler interfacing
extern "C" {

// DMatrix constructors
DMatrix<float> *DMatrix_ctor_c_f32(idx_t m, idx_t n, float fill) {
    return new DMatrix<float>(m, n, fill);
}

DMatrix<float> *DMatrix_ctor_a_f32(idx_t m, idx_t n, float *fill) {
    return new DMatrix<float>(m, n, fill);
}

DMatrix<double> *DMatrix_ctor_c_f64(idx_t m, idx_t n, double fill) {
    return new DMatrix<double>(m, n, fill);
}

DMatrix<double> *DMatrix_ctor_a_f64(idx_t m, idx_t n, double *fill) {
    return new DMatrix<double>(m, n, fill);
}

// DMatrix pointer returns
// these assume row-major storage
float *DMatrix_get_arr_ptr_f32(DMatrix<float> *A, idx_t m_start, idx_t n_start) {
    return A->A + (A->n * m_start + n_start);
}
double *DMatrix_get_arr_ptr_f64(DMatrix<double> *A, idx_t m_start, idx_t n_start) {
    return A->A + (A->n * m_start + n_start);
}

// DMatrix destructors
void DMatrix_dtor_f32(DMatrix<float> *A) { delete A; }
void DMatrix_dtor_f64(DMatrix<double> *A) { delete A; }

// SymCSRMatrix constructors
SymCSRMatrix<float> *SymCSRMatrix_ctor_f32(idx_t m, idx_t n, idx_t *arr_p, idx_t *arr_c,
                                           float *arr_v) {
    return new SymCSRMatrix<float>(m, n, arr_p, arr_c, arr_v);
}

SymCSRMatrix<double> *SymCSRMatrix_ctor_f64(idx_t m, idx_t n, idx_t *arr_p, idx_t *arr_c,
                                            double *arr_v) {
    return new SymCSRMatrix<double>(m, n, arr_p, arr_c, arr_v);
}

// SymCSRMatrix destructors
void SymCSRMatrix_dtor_f32(SymCSRMatrix<float> *A) { delete A; }
void SymCSRMatrix_dtor_f64(SymCSRMatrix<double> *A) { delete A; }

// SymCSRMatrix pointer returns
idx_t *SymCSRMatrix_get_ap_ptr_f32(SymCSRMatrix<float> *A) { return A->A_p; }
idx_t *SymCSRMatrix_get_ap_ptr_f64(SymCSRMatrix<double> *A) { return A->A_p; }

idx_t *SymCSRMatrix_get_ac_ptr_f32(SymCSRMatrix<float> *A) { return A->A_c; }
idx_t *SymCSRMatrix_get_ac_ptr_f64(SymCSRMatrix<double> *A) { return A->A_c; }

float *SymCSRMatrix_get_av_ptr_f32(SymCSRMatrix<float> *A) { return A->A_v; }
double *SymCSRMatrix_get_av_ptr_f64(SymCSRMatrix<double> *A) { return A->A_v; }
}

extern "C" {

// TODO: would love to not have our own version of this: at least MKL should have copy utilities
void DMatrix_set_submatrix_f32(const char op_A, const char op_B, const idx_t m, const idx_t n,
                               float *A, const idx_t lda, const float *B, const idx_t ldb) {

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

void DMatrix_set_submatrix_f64(const char op_A, const char op_B, const idx_t m, const idx_t n,
                               double *A, const idx_t lda, const double *B, const idx_t ldb) {

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
}