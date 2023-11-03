#include "integrals.h"

// ctype handler interfacing
extern "C" {

// JChunk constructors
JChunk<float> *JChunk_ctor_f32(idx_t N, idx_t *J_ind, float *J_val) {
    return new JChunk<float>(N, J_ind, J_val);
}
JChunk<double> *JChunk_ctor_f64(idx_t N, idx_t *J_ind, double *J_val) {
    return new JChunk<double>(N, J_ind, J_val);
}

// JChunk destructors
void JChunk_dtor_f32(JChunk<float> *chunk) { delete chunk; }
void JChunk_dtor_f64(JChunk<double> *chunk) { delete chunk; }

// JChunk pointer returns
idx_t *JChunk_get_idx_ptr_f32(JChunk<float> *chunk){return chunk->ind};
idx_t *JChunk_get_idx_ptr_f64(JChunk<double> *chunk){return chunk->ind};

float *JChunk_get_idx_ptr_f32(JChunk<float> *chunk){return chunk->J};
double *JChunk_get_idx_ptr_f64(JChunk<double> *chunk){return chunk->J};
}