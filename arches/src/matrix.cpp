#include "matrix.h"
#include "mkl.h"
#include "mkl_spblas.h"

// ctypes matrix routine interfaces
extern "C" {

void sgemm_mkl(const char op_a, const char op_b, const idx_t m, const idx_t n, const idx_t k,
               const float alpha, const float *A, const idx_t lda, const float *B, const idx_t ldb,
               const float beta, float *C, const idx_t ldc) {

    const CBLAS_LAYOUT layout = CblasRowMajor;
    const CBLAS_TRANSPOSE trans_A = op_a == 't' ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE trans_B = op_b == 't' ? CblasTrans : CblasNoTrans;

    cblas_sgemm(layout, trans_A, trans_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void dgemm_mkl(const char op_a, const char op_b, const idx_t m, const idx_t n, const idx_t k,
               const double alpha, const double *A, const idx_t lda, const double *B,
               const idx_t ldb, const double beta, double *C, const idx_t ldc) {

    const CBLAS_LAYOUT layout = CblasRowMajor;
    const CBLAS_TRANSPOSE trans_A = op_a == 't' ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE trans_B = op_b == 't' ? CblasTrans : CblasNoTrans;

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

void sApB(float *A, float *B, float *C, idx_t m, idx_t n) { ApB(A, B, C, m, n); }
void dApB(double *A, double *B, double *C, idx_t m, idx_t n) { ApB(A, B, C, m, n); }
void sAmB(float *A, float *B, float *C, idx_t m, idx_t n) { AmB(A, B, C, m, n); }
void dAmB(double *A, double *B, double *C, idx_t m, idx_t n) { AmB(A, B, C, m, n); }
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
float *DMatrix_get_arr_ptr_f32(DMatrix<float> *A) { return A->A; }
double *DMatrix_get_arr_ptr_f64(DMatrix<double> *A) { return A->A; }

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