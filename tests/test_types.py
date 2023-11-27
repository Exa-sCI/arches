import numpy as np


class Test_f32:
    dtype = np.float32
    rtol = 1e-4
    atol = 1e-6


class Test_f64:
    dtype = np.float64
    rtol = 1e-8
    atol = 1e-10
