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
    C_pt2(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_D_pt2n_f32(float *J, idx_t *J_ind, idx_t N, det_t *psi_int, float *psi_coef,
                           idx_t N_int, idx_t N_states, det_t *psi_ext, idx_t N_ext, float *res) {
    D_pt2(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_E_pt2n_f32(float *J, idx_t *J_ind, idx_t N, det_t *psi_int, float *psi_coef,
                           idx_t N_int, idx_t N_states, det_t *psi_ext, idx_t N_ext, float *res) {
    E_pt2(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_F_pt2n_f32(float *J, idx_t *J_ind, idx_t N, det_t *psi_int, float *psi_coef,
                           idx_t N_int, idx_t N_states, det_t *psi_ext, idx_t N_ext, float *res) {
    F_pt2(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_G_pt2n_f32(float *J, idx_t *J_ind, idx_t N, det_t *psi_int, float *psi_coef,
                           idx_t N_int, idx_t N_states, det_t *psi_ext, idx_t N_ext, float *res) {
    G_pt2(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res);
}

// Double float
void Kernels_OE_pt2n_f64(double *J, idx_t *J_ind, idx_t N, det_t *psi_int, double *psi_coef,
                         idx_t N_int, idx_t N_states, det_t *psi_ext, idx_t N_ext, double *res) {
    e_pt2_ij_OE(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_C_pt2n_f64(double *J, idx_t *J_ind, idx_t N, det_t *psi_int, double *psi_coef,
                           idx_t N_int, idx_t N_states, det_t *psi_ext, idx_t N_ext, double *res) {
    C_pt2(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_D_pt2n_f64(double *J, idx_t *J_ind, idx_t N, det_t *psi_int, double *psi_coef,
                           idx_t N_int, idx_t N_states, det_t *psi_ext, idx_t N_ext, double *res) {
    D_pt2(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_E_pt2n_f64(double *J, idx_t *J_ind, idx_t N, det_t *psi_int, double *psi_coef,
                           idx_t N_int, idx_t N_states, det_t *psi_ext, idx_t N_ext, double *res) {
    E_pt2(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_F_pt2n_f64(double *J, idx_t *J_ind, idx_t N, det_t *psi_int, double *psi_coef,
                           idx_t N_int, idx_t N_states, det_t *psi_ext, idx_t N_ext, double *res) {
    F_pt2(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_G_pt2n_f64(double *J, idx_t *J_ind, idx_t N, det_t *psi_int, double *psi_coef,
                           idx_t N_int, idx_t N_states, det_t *psi_ext, idx_t N_ext, double *res) {
    G_pt2(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res);
}

/// Denominator contributions
// Single float
void Kernels_OE_pt2d_f32(float *J, idx_t *J_ind, idx_t N, idx_t N_states, det_t *psi_ext,
                         idx_t N_ext, float *res) {
    e_pt2_ii_OE(J, J_ind, N, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_A_pt2d_f32(float *J, idx_t *J_ind, idx_t N, idx_t N_states, det_t *psi_ext,
                           idx_t N_ext, float *res) {
    A_pt2(J, J_ind, N, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_B_pt2d_f32(float *J, idx_t *J_ind, idx_t N, idx_t N_states, det_t *psi_ext,
                           idx_t N_ext, float *res) {
    B_pt2(J, J_ind, N, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_F_pt2d_f32(float *J, idx_t *J_ind, idx_t N, idx_t N_states, det_t *psi_ext,
                           idx_t N_ext, float *res) {
    F_pt2_denom(J, J_ind, N, N_states, psi_ext, N_ext, res);
}

// Double float
void Kernels_OE_pt2d_f64(double *J, idx_t *J_ind, idx_t N, idx_t N_states, det_t *psi_ext,
                         idx_t N_ext, double *res) {
    e_pt2_ii_OE(J, J_ind, N, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_A_pt2d_f64(double *J, idx_t *J_ind, idx_t N, idx_t N_states, det_t *psi_ext,
                           idx_t N_ext, double *res) {
    A_pt2(J, J_ind, N, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_B_pt2d_f64(double *J, idx_t *J_ind, idx_t N, idx_t N_states, det_t *psi_ext,
                           idx_t N_ext, double *res) {
    B_pt2(J, J_ind, N, N_states, psi_ext, N_ext, res);
}

void Kernels_TE_F_pt2d_f64(double *J, idx_t *J_ind, idx_t N, idx_t N_states, det_t *psi_ext,
                           idx_t N_ext, double *res) {
    F_pt2_denom(J, J_ind, N, N_states, psi_ext, N_ext, res);
}

//// Real integral explicit Hamiltonian kernels
// single float
void Kernels_H_OE_ii_f32(float *J, idx_t *J_ind, idx_t N, det_t *psi_det, idx_t N_det, idx_t *H_p,
                         float *H_v) {
    H_OE_ii(J, J_ind, N, psi_det, N_det, H_p, H_v);
}

void Kernels_H_OE_ij_f32(float *J, idx_t *J_ind, idx_t N, det_t *psi_det, idx_t N_det, idx_t *H_p,
                         idx_t *H_c, float *H_v) {
    H_OE_ij(J, J_ind, N, psi_det, N_det, H_p, H_c, H_v);
}

void Kernels_H_A_f32(float *J, idx_t *J_ind, idx_t N, det_t *psi_det, idx_t N_det, idx_t *H_p,
                     float *H_v) {
    H_A(J, J_ind, N, psi_det, N_det, H_p, H_v);
}

void Kernels_H_B_f32(float *J, idx_t *J_ind, idx_t N, det_t *psi_det, idx_t N_det, idx_t *H_p,
                     float *H_v) {
    H_B(J, J_ind, N, psi_det, N_det, H_p, H_v);
}

void Kernels_H_C_f32(float *J, idx_t *J_ind, idx_t N, det_t *psi_det, idx_t N_det, idx_t *H_p,
                     idx_t *H_c, float *H_v) {
    H_C(J, J_ind, N, psi_det, N_det, H_p, H_c, H_v);
}

void Kernels_H_D_f32(float *J, idx_t *J_ind, idx_t N, det_t *psi_det, idx_t N_det, idx_t *H_p,
                     idx_t *H_c, float *H_v) {
    H_D(J, J_ind, N, psi_det, N_det, H_p, H_c, H_v);
}

void Kernels_H_E_f32(float *J, idx_t *J_ind, idx_t N, det_t *psi_det, idx_t N_det, idx_t *H_p,
                     idx_t *H_c, float *H_v) {
    H_E(J, J_ind, N, psi_det, N_det, H_p, H_c, H_v);
}

void Kernels_H_F_ii_f32(float *J, idx_t *J_ind, idx_t N, det_t *psi_det, idx_t N_det, idx_t *H_p,
                        float *H_v) {
    H_F_ii(J, J_ind, N, psi_det, N_det, H_p, H_v);
}

void Kernels_H_F_ij_f32(float *J, idx_t *J_ind, idx_t N, det_t *psi_det, idx_t N_det, idx_t *H_p,
                        idx_t *H_c, float *H_v) {
    H_F_ij(J, J_ind, N, psi_det, N_det, H_p, H_c, H_v);
}

void Kernels_H_G_f32(float *J, idx_t *J_ind, idx_t N, det_t *psi_det, idx_t N_det, idx_t *H_p,
                     idx_t *H_c, float *H_v) {
    H_G(J, J_ind, N, psi_det, N_det, H_p, H_c, H_v);
}

// double float
void Kernels_H_OE_ii_f64(double *J, idx_t *J_ind, idx_t N, det_t *psi_det, idx_t N_det, idx_t *H_p,
                         double *H_v) {
    H_OE_ii(J, J_ind, N, psi_det, N_det, H_p, H_v);
}

void Kernels_H_OE_ij_f64(double *J, idx_t *J_ind, idx_t N, det_t *psi_det, idx_t N_det, idx_t *H_p,
                         idx_t *H_c, double *H_v) {
    H_OE_ij(J, J_ind, N, psi_det, N_det, H_p, H_c, H_v);
}

void Kernels_H_A_f64(double *J, idx_t *J_ind, idx_t N, det_t *psi_det, idx_t N_det, idx_t *H_p,
                     double *H_v) {
    H_A(J, J_ind, N, psi_det, N_det, H_p, H_v);
}

void Kernels_H_B_f64(double *J, idx_t *J_ind, idx_t N, det_t *psi_det, idx_t N_det, idx_t *H_p,
                     double *H_v) {
    H_B(J, J_ind, N, psi_det, N_det, H_p, H_v);
}

void Kernels_H_C_f64(double *J, idx_t *J_ind, idx_t N, det_t *psi_det, idx_t N_det, idx_t *H_p,
                     idx_t *H_c, double *H_v) {
    H_C(J, J_ind, N, psi_det, N_det, H_p, H_c, H_v);
}

void Kernels_H_D_f64(double *J, idx_t *J_ind, idx_t N, det_t *psi_det, idx_t N_det, idx_t *H_p,
                     idx_t *H_c, double *H_v) {
    H_D(J, J_ind, N, psi_det, N_det, H_p, H_c, H_v);
}

void Kernels_H_E_f64(double *J, idx_t *J_ind, idx_t N, det_t *psi_det, idx_t N_det, idx_t *H_p,
                     idx_t *H_c, double *H_v) {
    H_E(J, J_ind, N, psi_det, N_det, H_p, H_c, H_v);
}

void Kernels_H_F_ii_f64(double *J, idx_t *J_ind, idx_t N, det_t *psi_det, idx_t N_det, idx_t *H_p,
                        double *H_v) {
    H_F_ii(J, J_ind, N, psi_det, N_det, H_p, H_v);
}

void Kernels_H_F_ij_f64(double *J, idx_t *J_ind, idx_t N, det_t *psi_det, idx_t N_det, idx_t *H_p,
                        idx_t *H_c, double *H_v) {
    H_F_ij(J, J_ind, N, psi_det, N_det, H_p, H_c, H_v);
}

void Kernels_H_G_f64(double *J, idx_t *J_ind, idx_t N, det_t *psi_det, idx_t N_det, idx_t *H_p,
                     idx_t *H_c, double *H_v) {
    H_G(J, J_ind, N, psi_det, N_det, H_p, H_c, H_v);
}
}