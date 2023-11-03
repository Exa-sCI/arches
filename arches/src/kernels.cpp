#include "kernels.h"

// ctype interfaces
extern "C" {

//// Real integral pt2 kernels

/// Numerator contributions
// Single float
void Kernels_OE_pt2n_f32(float *J, idx_t *J_ind, idx_t N, det_t *psi_int, float *psi_coef,
                         idx_t N_int, idx_t N_states, det_t *psi_ext, idx_t N_ext, float *res) {
    e_pt2_ij_OE(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_C_pt2n_f32(float *J, idx_t *J_ind, idx_t N, det_t *psi_int, float *psi_coef,
                           idx_t N_int, idx_t N_states, det_t *psi_ext, idx_t N_ext, float *res) {
    C_pt2_kernel(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_D_pt2n_f32(float *J, idx_t *J_ind, idx_t N, det_t *psi_int, float *psi_coef,
                           idx_t N_int, idx_t N_states, det_t *psi_ext, idx_t N_ext, float *res) {
    D_pt2_kernel(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_E_pt2n_f32(float *J, idx_t *J_ind, idx_t N, det_t *psi_int, float *psi_coef,
                           idx_t N_int, idx_t N_states, det_t *psi_ext, idx_t N_ext, float *res) {
    E_pt2_kernel(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_F_pt2n_f32(float *J, idx_t *J_ind, idx_t N, det_t *psi_int, float *psi_coef,
                           idx_t N_int, idx_t N_states, det_t *psi_ext, idx_t N_ext, float *res) {
    F_pt2_kernel(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_G_pt2n_f32(float *J, idx_t *J_ind, idx_t N, det_t *psi_int, float *psi_coef,
                           idx_t N_int, idx_t N_states, det_t *psi_ext, idx_t N_ext, float *res) {
    G_pt2_kernel(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res);
}

// Double float
void Kernels_OE_pt2n_f64(double *J, idx_t *J_ind, idx_t N, det_t *psi_int, double *psi_coef,
                         idx_t N_int, idx_t N_states, det_t *psi_ext, idx_t N_ext, double *res) {
    e_pt2_ij_OE(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_C_pt2n_f64(double *J, idx_t *J_ind, idx_t N, det_t *psi_int, double *psi_coef,
                           idx_t N_int, idx_t N_states, det_t *psi_ext, idx_t N_ext, double *res) {
    C_pt2_kernel(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_D_pt2n_f64(double *J, idx_t *J_ind, idx_t N, det_t *psi_int, double *psi_coef,
                           idx_t N_int, idx_t N_states, det_t *psi_ext, idx_t N_ext, double *res) {
    D_pt2_kernel(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_E_pt2n_f64(double *J, idx_t *J_ind, idx_t N, det_t *psi_int, double *psi_coef,
                           idx_t N_int, idx_t N_states, det_t *psi_ext, idx_t N_ext, double *res) {
    E_pt2_kernel(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_F_pt2n_f64(double *J, idx_t *J_ind, idx_t N, det_t *psi_int, double *psi_coef,
                           idx_t N_int, idx_t N_states, det_t *psi_ext, idx_t N_ext, double *res) {
    F_pt2_kernel(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_G_pt2n_f64(double *J, idx_t *J_ind, idx_t N, det_t *psi_int, double *psi_coef,
                           idx_t N_int, idx_t N_states, det_t *psi_ext, idx_t N_ext, double *res) {
    G_pt2_kernel(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res);
}

/// Denominator contributions
// Single float
void Kernels_OE_pt2d_f32(float *J, idx_t *J_ind, idx_t N, idx_t N_states, det_t *psi_ext,
                         idx_t N_ext, float *res) {
    e_pt2_ii_OE(J, J_ind, N, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_A_pt2d_f32(float *J, idx_t *J_ind, idx_t N, idx_t N_states, det_t *psi_ext,
                           idx_t N_ext, float *res) {
    A_pt2_kernel(J, J_ind, N, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_B_pt2d_f32(float *J, idx_t *J_ind, idx_t N, idx_t N_states, det_t *psi_ext,
                           idx_t N_ext, float *res) {
    B_pt2_kernel(J, J_ind, N, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_F_pt2d_f32(float *J, idx_t *J_ind, idx_t N, idx_t N_states, det_t *psi_ext,
                           idx_t N_ext, float *res) {
    F_pt2_kernel_denom(J, J_ind, N, N_states, psi_ext, N_ext, res);
}

// Double float
void Kernels_OE_pt2d_f64(double *J, idx_t *J_ind, idx_t N, idx_t N_states, det_t *psi_ext,
                         idx_t N_ext, double *res) {
    e_pt2_ii_OE(J, J_ind, N, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_A_pt2d_f64(double *J, idx_t *J_ind, idx_t N, idx_t N_states, det_t *psi_ext,
                           idx_t N_ext, double *res) {
    A_pt2_kernel(J, J_ind, N, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_B_pt2d_f64(double *J, idx_t *J_ind, idx_t N, idx_t N_states, det_t *psi_ext,
                           idx_t N_ext, double *res) {
    B_pt2_kernel(J, J_ind, N, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_F_pt2d_f64(double *J, idx_t *J_ind, idx_t N, idx_t N_states, det_t *psi_ext,
                           idx_t N_ext, double *res) {
    F_pt2_kernel_denom(J, J_ind, N, N_states, psi_ext, N_ext, res);
}
}