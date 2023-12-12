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

DMatrix<float> *sqr_mkl(const idx_t m, const idx_t n, float *A, const idx_t lda) {
    CBLAS_LAYOUT layout = CblasRowMajor;
    idx_t k = std::min(m, n);
    std::unique_ptr<float[]> tau(new float[k]);

    // Perform QR factorization
    LAPACKE_sgeqrf(layout, m, n, A, lda, tau.get());

    // Extract R from A
    DMatrix<float> *R = new DMatrix<float>(n, n, 0.0);
    for (auto i = 0; i < n; i++) {
        for (auto j = i; j < n; j++) {
            R->A[i * n + j] = A[i * lda + j];
        }
    }

    // Form explicit Q in columns of A
    LAPACKE_sorgqr(layout, m, n, k, A, lda, tau.get());

    return R;
}

DMatrix<double> *dqr_mkl(const idx_t m, const idx_t n, double *A, const idx_t lda) {
    CBLAS_LAYOUT layout = CblasRowMajor;
    idx_t k = std::min(m, n);
    std::unique_ptr<double[]> tau(new double[k]);

    // Perform QR factorization
    LAPACKE_dgeqrf(layout, m, n, A, lda, tau.get());

    // Extract R from A
    DMatrix<double> *R = new DMatrix<double>(n, n, 0.0);
    for (auto i = 0; i < n; i++) {
        for (auto j = i; j < n; j++) {
            R->A[i * n + j] = A[i * lda + j];
        }
    }

    // Form explicit Q in columns of A
    LAPACKE_dorgqr(layout, m, n, k, A, lda, tau.get());

    return R;
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

void DMatrix_sApB(const char op_A, const char op_B, const idx_t m, const idx_t n, float *A,
                  const idx_t lda, float *B, const idx_t ldb, float *C, const idx_t ldc) {
    ApB(op_A, op_B, m, n, A, lda, B, ldb, C, ldc);
}
void DMatrix_sAmB(const char op_A, const char op_B, const idx_t m, const idx_t n, float *A,
                  const idx_t lda, float *B, const idx_t ldb, float *C, const idx_t ldc) {
    AmB(op_A, op_B, m, n, A, lda, B, ldb, C, ldc);
}
void DMatrix_dApB(const char op_A, const char op_B, const idx_t m, const idx_t n, double *A,
                  const idx_t lda, double *B, const idx_t ldb, double *C, const idx_t ldc) {
    ApB(op_A, op_B, m, n, A, lda, B, ldb, C, ldc);
}
void DMatrix_dAmB(const char op_A, const char op_B, const idx_t m, const idx_t n, double *A,
                  const idx_t lda, double *B, const idx_t ldb, double *C, const idx_t ldc) {
    AmB(op_A, op_B, m, n, A, lda, B, ldb, C, ldc);
}
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

idx_t SymCSRMatrix_get_n_entries_f32(SymCSRMatrix<float> *A) { return A->A_p[A->m]; }
idx_t SymCSRMatrix_get_n_entries_f64(SymCSRMatrix<double> *A) { return A->A_p[A->m]; }
}

extern "C" {

// TODO: would love to not have our own version of this: at least MKL should have copy utilities
void DMatrix_set_submatrix_f32(const char op_A, const char op_B, const idx_t m, const idx_t n,
                               float *A, const idx_t lda, const float *B, const idx_t ldb) {
    set_submatrix(op_A, op_B, m, n, A, lda, B, ldb);
}

void DMatrix_set_submatrix_f64(const char op_A, const char op_B, const idx_t m, const idx_t n,
                               double *A, const idx_t lda, const double *B, const idx_t ldb) {

    set_submatrix(op_A, op_B, m, n, A, lda, B, ldb);
}
}