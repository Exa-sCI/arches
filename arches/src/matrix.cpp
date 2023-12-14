#include "matrix.h"
#include "mkl.h"
#include "mkl_spblas.h"

// ctypes matrix routine interfaces
extern "C" {

void dsyevd_mkl(const idx_t n, double *A, const idx_t lda, double *w) {
    CBLAS_LAYOUT layout = CblasRowMajor;

    long long status = LAPACKE_dsyevd(layout, 'v', 'U', n, A, lda, w);

    if (status < 0)
        std::cout << "in dysevd: parameter " << status << " had an illegal value" << std::endl;
    if (status > 0)
        std::cout << "in dysevd: failed to compute eigenvalue. status:  " << status << std::endl;
}

void ssyevd_mkl(const idx_t n, float *A, const idx_t lda, float *w) {
    std::unique_ptr<double[]> A_temp(new double[lda * n]);
    std::unique_ptr<double[]> w_temp(new double[n]);

    std::transform(A, A + lda * n, A_temp.get(),
                   [](const float &x) { return static_cast<double>(x); });

    dsyevd_mkl(n, A_temp.get(), lda, w_temp.get());

    std::transform(A_temp.get(), A_temp.get() + lda * n, A,
                   [](const double &x) { return static_cast<float>(x); });
    std::transform(w_temp.get(), w_temp.get() + n, w,
                   [](const double &x) { return static_cast<float>(x); });
}

void sgtsv_mkl(const idx_t n, float *d, float *e, float *b, const idx_t ldb) {
    CBLAS_LAYOUT layout = CblasRowMajor;

    std::unique_ptr<float[]> du(new float[n - 1]);
    std::copy(e, e + n - 1, du.get());

    long long status = LAPACKE_sgtsv(layout, n, (idx_t)1, du.get(), d, e, b, ldb);

    if (status < 0)
        std::cout << "in sgtsv: parameter " << status << " had an illegal value" << std::endl;
    if (status > 0)
        std::cout << "in sgtsv: diag element  " << status << " is exactly zero " << std::endl;
}

void dgtsv_mkl(const idx_t n, double *d, double *e, double *b, const idx_t ldb) {
    CBLAS_LAYOUT layout = CblasRowMajor;

    std::unique_ptr<double[]> du(new double[n - 1]);
    std::copy(e, e + n - 1, du.get());

    long long status = LAPACKE_dgtsv(layout, n, (idx_t)1, du.get(), d, e, b, ldb);

    if (status < 0)
        std::cout << "in dgtsv: parameter " << status << " had an illegal value" << std::endl;
    if (status > 0)
        std::cout << "in dgtsv: diag element  " << status << " is exactly zero " << std::endl;
}

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

void sym_csr_MKL_test() {
    const int M = 4;
    const int K = 4;
    const int N = 4;
    const double tol = 1e-8;
    const CBLAS_LAYOUT layout = CblasRowMajor;
    const CBLAS_TRANSPOSE trans = CblasNoTrans;

    std::unique_ptr<double[]> A(new double[M * K]);
    std::unique_ptr<double[]> B(new double[K * N]);
    std::unique_ptr<double[]> C_sref(new double[M * N]);
    std::unique_ptr<double[]> C_dref(new double[M * N]);
    std::unique_ptr<double[]> C_test(new double[M * N]);

    double *A_p = A.get();
    double *B_p = B.get();
    double *C_sref_p = C_sref.get();
    double *C_dref_p = C_dref.get();
    double *C_test_p = C_test.get();

    std::unique_ptr<idx_t[]> A_rows(new idx_t[M + 1]);
    std::unique_ptr<idx_t[]> A_cols(new idx_t[10]); // assume M = 4, all entries in upper triangle
    std::unique_ptr<double[]> A_vals(new double[10]);

    idx_t *A_rows_p = A_rows.get();
    A_rows_p[0] = 0;
    A_rows_p[1] = 4;
    A_rows_p[2] = 7;
    A_rows_p[3] = 9;
    A_rows_p[4] = 10;

    idx_t *A_cols_p = A_cols.get();
    A_cols_p[0] = 0;
    A_cols_p[1] = 1;
    A_cols_p[2] = 2;
    A_cols_p[3] = 3;
    A_cols_p[4] = 1;
    A_cols_p[5] = 2;
    A_cols_p[6] = 3;
    A_cols_p[7] = 2;
    A_cols_p[8] = 3;
    A_cols_p[9] = 3;

    double *A_vals_p = A_vals.get();
    // Sym CSR A
    std::fill(A_p, A_p + M * N, 0.0);
    for (auto i = 0; i < M; i++) {
        for (auto idx = A_rows_p[i]; idx < A_rows_p[i + 1]; idx++) {
            auto &j = A_cols_p[idx];
            double val = 2.3 * (i + 1) + 0.4 * (j + 1);
            A_vals_p[idx] = val;
            std::cout << val << ": "
                      << "(" << idx << ", (" << i << ", " << j << "))" << std::endl;
            A_p[i * M + j] = val; // Copy into dense A
            A_p[j * M + i] = val;
        }
    }

    for (auto i = 0; i < K; i++) {
        for (auto j = 0; j < N; j++) {
            B_p[i * K + j] = i == j ? 1.0 : 0.0;
        }
    }

    std::fill(C_sref_p, C_sref_p + M * N, 0.0);
    std::fill(C_dref_p, C_dref_p + M * N, 0.0);
    std::fill(C_test_p, C_test_p + M * N, 0.0);

    const sparse_operation_t op_A = SPARSE_OPERATION_NON_TRANSPOSE;
    struct matrix_descr descrA;
    descrA.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
    descrA.mode = SPARSE_FILL_MODE_UPPER;
    descrA.diag = SPARSE_DIAG_NON_UNIT;

    sparse_matrix_t csrA;
    mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO, M, K, A_rows_p, A_rows_p + 1, A_cols_p,
                            A_vals_p);
    sparse_status_t status = mkl_sparse_d_mm(op_A, 1.0, csrA, descrA, SPARSE_LAYOUT_ROW_MAJOR, B_p,
                                             N, K, 1.0, C_test_p, M);

    for (auto i = 0; i < M; i++) {
        for (auto j = 0; j < N; j++) {
            std::cout << "(" << i << ", " << j << ") : " << C_test_p[i * M + j] << std::endl;

            //    << std::abs(C_test_p[i * M + j] - C_dref_p[i * M + j]) << std::endl;
        }
    }
}

void sym_csr_s_MM_ref(const float alpha, idx_t *A_p, const idx_t M, idx_t *A_c, float *A_v,
                      const idx_t K, float *B, const idx_t ldb, const float beta, float *C,
                      const idx_t ldc) {

    // iterate over rows
    for (auto i = 0; i < M; i++) {
        // iterate over cols of output
        for (auto j = 0; j < K; j++) {

            for (auto idx = A_p[i]; idx < A_p[i + 1]; idx++) {
                auto &k = A_c[idx];
                C[i * ldc + j] += A_v[idx] * B[k * ldb + j]; // contribution from A_ik * B_kj

                if (k > i) {
                    C[k * ldc + j] += A_v[idx] * B[i * ldb + j]; // contribution from A_ki * B_ij;
                }
            }
        }
    }
}

void sym_csr_d_MM_ref(const double alpha, idx_t *A_p, const idx_t M, idx_t *A_c, double *A_v,
                      const idx_t K, double *B, const idx_t ldb, const double beta, double *C,
                      const idx_t ldc) {

    // iterate over rows
    for (auto i = 0; i < M; i++) {
        // iterate over cols of output
        for (auto j = 0; j < K; j++) {

            for (auto idx = A_p[i]; idx < A_p[i + 1]; idx++) {
                auto &k = A_c[idx];
                C[i * ldc + j] += A_v[idx] * B[k * ldb + j]; // contribution from A_ik * B_kj

                if (k > i) {
                    C[k * ldc + j] += A_v[idx] * B[i * ldb + j]; // contribution from A_ki * B_ij;
                }
            }
        }
    }
}

void sym_csr_s_MM_mkl(const float alpha, idx_t *A_rows, const idx_t M, idx_t *A_cols, float *A_vals,
                      const idx_t K, float *B, const idx_t ldb, const float beta, float *C,
                      const idx_t ldc) {
    std::cout << "Running internal test" << std::endl;
    sym_csr_MKL_test();
    // for (auto i = 0; i < 4; i++) {
    //     idx_t start_idx = A_rows[i];
    //     idx_t end_idx = A_rows[i + 1];
    //     for (auto j = start_idx; j < end_idx; j++) {
    //         idx_t col = A_cols[j];
    //         float val = A_vals[j];
    //         if (col < 4)
    //             std::cout << "(" << i << ", " << col << ") :" << val << std::endl;
    //     }
    // }

    const sparse_operation_t op_A = SPARSE_OPERATION_NON_TRANSPOSE;
    sparse_matrix_t csrA;
    sparse_status_t status;
    struct matrix_descr descr_A;
    descr_A.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
    descr_A.mode = SPARSE_FILL_MODE_UPPER;
    descr_A.diag = SPARSE_DIAG_NON_UNIT;
    status = mkl_sparse_s_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO, M, M, A_rows, A_rows + 1,
                                     A_cols, A_vals);

    std::cout << "csr create status: " << status << " " << SPARSE_STATUS_SUCCESS << std::endl;

    sparse_index_base_t base;
    idx_t N_rows;
    idx_t N_cols;
    std::unique_ptr<idx_t[]> A_p_starts_debug(new idx_t[M]);
    std::unique_ptr<idx_t[]> A_p_stops_debug(new idx_t[M]);
    std::unique_ptr<idx_t[]> A_cols_debug(new idx_t[A_rows[M]]);
    std::unique_ptr<float[]> A_vals_debug(new float[A_rows[M]]);
    idx_t *A_p_starts_test = A_p_starts_debug.get();
    idx_t *A_p_stops_test = A_p_stops_debug.get();
    idx_t *A_cols_test = A_cols_debug.get();
    float *A_v_test = A_vals_debug.get();
    status = mkl_sparse_s_export_csr(csrA, &base, &N_rows, &N_cols, &A_p_starts_test,
                                     &A_p_stops_test, &A_cols_test, &A_v_test);
    std::cout << "csr export status: " << status << " " << SPARSE_STATUS_SUCCESS << std::endl;

    for (auto i = 0; i < 4; i++) {
        for (auto j = 0; j < K; j++) {
            std::cout << "(" << i << ", " << j << ") : " << B[i * ldb + j] << std::endl;
        }
    }

    for (auto i = 0; i < M; i++) {
        for (auto j = 0; j < K; j++) {
            B[i * ldb + j] = i == j ? 1.0 : 0.0;
            C[i * ldc + j] = 0.0;
        }
    }

    // op_A, alpha, A, matrix_descr, layout, B, columns (of C), ldb, beta, C, ldc
    status =
        mkl_sparse_s_mm(op_A, 1.0, csrA, descr_A, SPARSE_LAYOUT_ROW_MAJOR, B, K, ldb, beta, C, ldc);
    std::cout << std::endl;
    for (auto i = 0; i < 4; i++) {
        for (auto j = 0; j < K; j++) {
            std::cout << "(" << i << ", " << j << ") : " << C[i * ldc + j] << std::endl;
        }
    }

    std::cout << "matmul status: " << status << " " << SPARSE_STATUS_SUCCESS << std::endl;
    mkl_sparse_destroy(csrA);
}

void sym_csr_d_MM_mkl(const double alpha, idx_t *A_rows, const idx_t M, idx_t *A_cols,
                      double *A_vals, const idx_t K, const double beta, const double *B,
                      const idx_t ldb, double *C, const idx_t ldc) {
    std::cout << "Hello I am instead here" << std::endl;
    const sparse_operation_t op_A = SPARSE_OPERATION_NON_TRANSPOSE;
    sparse_matrix_t csrA;
    sparse_status_t status;
    struct matrix_descr descr_A;
    descr_A.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
    descr_A.mode = SPARSE_FILL_MODE_UPPER;
    descr_A.diag = SPARSE_DIAG_NON_UNIT;

    mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO, M, K, A_rows, A_rows + 1, A_cols,
                            A_vals);
    status = mkl_sparse_d_mm(op_A, alpha, csrA, descr_A, SPARSE_LAYOUT_ROW_MAJOR, B, K, ldb, beta,
                             C, ldc);
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

void DMatrix_sAtB(const char op_A, const char op_B, const idx_t m, const idx_t n, float *A,
                  const idx_t lda, float *B, const idx_t ldb, float *C, const idx_t ldc) {
    AtB(op_A, op_B, m, n, A, lda, B, ldb, C, ldc);
}
void DMatrix_sAdB(const char op_A, const char op_B, const idx_t m, const idx_t n, float *A,
                  const idx_t lda, float *B, const idx_t ldb, float *C, const idx_t ldc) {
    AdB(op_A, op_B, m, n, A, lda, B, ldb, C, ldc);
}
void DMatrix_dAtB(const char op_A, const char op_B, const idx_t m, const idx_t n, double *A,
                  const idx_t lda, double *B, const idx_t ldb, double *C, const idx_t ldc) {
    AtB(op_A, op_B, m, n, A, lda, B, ldb, C, ldc);
}
void DMatrix_dAdB(const char op_A, const char op_B, const idx_t m, const idx_t n, double *A,
                  const idx_t lda, double *B, const idx_t ldb, double *C, const idx_t ldc) {
    AdB(op_A, op_B, m, n, A, lda, B, ldb, C, ldc);
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

void DMatrix_fill_diagonal_f32(const idx_t m, DMatrix<float> *A, const idx_t lda,
                               const float *fill) {
    fill_diagonal(m, A->A, lda, fill);
}

void DMatrix_fill_diagonal_f64(const idx_t m, DMatrix<double> *A, const idx_t lda,
                               const double *fill) {
    fill_diagonal(m, A->A, lda, fill);
}

void DMatrix_extract_diagonal_f32(const idx_t m, const DMatrix<float> *A, const idx_t lda,
                                  float *res) {
    extract_dense_diagonal(m, A->A, lda, res);
}

void DMatrix_extract_diagonal_f64(const idx_t m, const DMatrix<double> *A, const idx_t lda,
                                  double *res) {
    extract_dense_diagonal(m, A->A, lda, res);
}

void DMatrix_extract_superdiagonal_f32(const idx_t m, const DMatrix<float> *A, const idx_t lda,
                                       float *res) {
    extract_dense_superdiagonal(m, A->A, lda, res);
}

void DMatrix_extract_superdiagonal_f64(const idx_t m, const DMatrix<double> *A, const idx_t lda,
                                       double *res) {
    extract_dense_superdiagonal(m, A->A, lda, res);
}

void DMatrix_column_2norm_f32(const idx_t m, const idx_t n, const DMatrix<float> *A,
                              const idx_t lda, float *res) {
    column_2norm(m, n, A->A, lda, res);
}

void DMatrix_column_2norm_f64(const idx_t m, const idx_t n, const DMatrix<double> *A,
                              const idx_t lda, double *res) {
    column_2norm(m, n, A->A, lda, res);
}

//// SymCSR
void SymCSRMatrix_extract_diagonal_f32(const idx_t m, const SymCSRMatrix<float> *A, float *res) {
    extract_sparse_diagonal(m, A->A_p, A->A_v, res);
}

void SymCSRMatrix_extract_diagonal_f64(const idx_t m, const SymCSRMatrix<double> *A, double *res) {
    extract_sparse_diagonal(m, A->A_p, A->A_v, res);
}
void SymCSRMatrix_extract_superdiagonal_f32(const idx_t m, const SymCSRMatrix<float> *A,
                                            float *res) {
    extract_sparse_superdiagonal(m, A->A_p, A->A_c, A->A_v, res);
}

void SymCSRMatrix_extract_superdiagonal_f64(const idx_t m, const SymCSRMatrix<double> *A,
                                            double *res) {
    extract_sparse_superdiagonal(m, A->A_p, A->A_c, A->A_v, res);
}
}