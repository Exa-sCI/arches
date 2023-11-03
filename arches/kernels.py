import pathlib
from ctypes import CDLL
from functools import singledispatch
from typing import Callable, Tuple, Union

from arches.linked_object import det_t_p, f32, f32_p, f64, f64_p, idx_t, idx_t_p, type_dict

run_folder = pathlib.Path(__file__).parent.resolve()
lib_kernels = CDLL(run_folder.joinpath("build/libkernels.so"))

### Register all of the offloaded pt2 kernels
for k in [f32, f64]:
    pfix = "Kernels_"
    sfix = "_" + type_dict[k][0]  # key : value is (c_type : (name, pointer, np ndpointer))
    k_p = type_dict[k][1]

    # Numerator kernels
    for cat in ["OE"] + ["TE_" + x for x in "CDEFG"]:
        f_kernel = getattr(lib_kernels, pfix + cat + "_pt2n" + sfix)
        f_kernel.argtypes = [k_p, idx_t_p, idx_t, det_t_p, k_p, idx_t, idx_t, det_t_p, idx_t, k_p]
        f_kernel.restype = None

    # Denominator kernels
    for cat in ["OE"] + ["TE_" + x for x in "ABF"]:
        f_kernel = getattr(lib_kernels, pfix + cat + "_pt2d" + sfix)
        f_kernel.argtypes = [k_p, idx_t_p, idx_t, idx_t, det_t_p, idx_t, k_p]
        f_kernel.restype = None


### Define python kernels
## Numerator kernels
@singledispatch
def k_OE_pt2n(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res):
    raise NotImplementedError


@singledispatch
def k_TE_C_pt2n(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res):
    raise NotImplementedError


@singledispatch
def k_TE_D_pt2n(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res):
    raise NotImplementedError


@singledispatch
def k_TE_E_pt2n(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res):
    raise NotImplementedError


@singledispatch
def k_TE_F_pt2n(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res):
    raise NotImplementedError


@singledispatch
def k_TE_G_pt2n(J, J_ind, N, psi_int, psi_coef, N_int, N_states, psi_ext, N_ext, res):
    raise NotImplementedError


## Denominator kernels
@singledispatch
def k_OE_pt2d(J, J_ind, N, N_states, psi_ext, N_ext, res):
    raise NotImplementedError


@singledispatch
def k_TE_A_pt2d(J, J_ind, N, N_states, psi_ext, N_ext, res):
    raise NotImplementedError


@singledispatch
def k_TE_B_pt2d(J, J_ind, N, N_states, psi_ext, N_ext, res):
    raise NotImplementedError


@singledispatch
def k_TE_F_pt2d(J, J_ind, N, N_states, psi_ext, N_ext, res):
    raise NotImplementedError


### Register offload kernels against python callers
n_kernels = (k_OE_pt2n, k_TE_C_pt2n, k_TE_D_pt2n, k_TE_E_pt2n, k_TE_F_pt2n, k_TE_G_pt2n)
d_kernels = (k_OE_pt2d, k_TE_A_pt2d, k_TE_B_pt2d, k_TE_F_pt2d)

for kern in n_kernels + d_kernels:
    name = kern.__name__
    for k in [f32, f64]:
        k_p = type_dict[k][1]
        pfix = "Kernels_"
        sfix = "_" + type_dict[k][0]
        target_name = pfix + name.split("_", 1)[1] + sfix
        f_dispatch = getattr(lib_kernels, target_name)
        kern.register(k_p, f_dispatch)


T_p = Union[f32_p, f64_p]


def dispatch_kernel(
    category: str
) -> Tuple[
    Union[None, Callable[[T_p, idx_t_p, det_t_p, T_p, idx_t, idx_t, det_t_p, idx_t, T_p], None]],
    Union[None, Callable[[T_p, idx_t_p, idx_t, idx_t, det_t_p, idx_t, T_p], None]],
]:
    match category:
        case "OE":
            return (k_OE_pt2n, k_OE_pt2d)
        case "A":
            return (None, k_TE_A_pt2d)
        case "B":
            return (None, k_TE_B_pt2d)
        case "C":
            return (k_TE_C_pt2n, None)
        case "D":
            return (k_TE_D_pt2n, None)
        case "E":
            return (k_TE_E_pt2n, None)
        case "F":
            return (k_TE_F_pt2n, k_TE_F_pt2d)
        case "G":
            return (k_TE_G_pt2n, None)
