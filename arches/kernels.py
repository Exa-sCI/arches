import pathlib
from ctypes import CDLL
from functools import singledispatch
from typing import Callable, Tuple, Union

from arches.linked_object import det_t_p, f32, f32_p, f64, f64_p, idx_t, idx_t_p, type_dict

run_folder = pathlib.Path(__file__).parent.resolve()
lib_kernels = CDLL(run_folder.joinpath("build/libkernels.so"))

### Register all of the offloaded pt2/H kernels
for k in [f32, f64]:
    pfix = "Kernels_"
    sfix = "_" + type_dict[k][0]  # key : value is (c_type : (name, pointer, np ndpointer))
    k_p = type_dict[k][1]

    # pt2 numerator kernels
    for cat in ["OE"] + ["TE_" + x for x in "CDEFG"]:
        f_kernel = getattr(lib_kernels, pfix + cat + "_pt2n" + sfix)
        f_kernel.argtypes = [k_p, idx_t_p, idx_t, det_t_p, k_p, idx_t, idx_t, det_t_p, idx_t, k_p]
        f_kernel.restype = None

    # pt2 denominator kernels
    for cat in ["OE"] + ["TE_" + x for x in "ABF"]:
        f_kernel = getattr(lib_kernels, pfix + cat + "_pt2d" + sfix)
        f_kernel.argtypes = [k_p, idx_t_p, idx_t, idx_t, det_t_p, idx_t, k_p]
        f_kernel.restype = None

    # H ii kernels
    for cat in ["OE_ii", "A", "B", "F_ii"]:
        f_kernel = getattr(lib_kernels, pfix + "H_" + cat + sfix)
        f_kernel.argtypes = [k_p, idx_t_p, idx_t, det_t_p, idx_t, idx_t_p, k_p]
        f_kernel.restype = None

    # H ij kernels
    for cat in ["OE_ij", "C", "D", "E", "F_ij", "G"]:
        f_kernel = getattr(lib_kernels, pfix + "H_" + cat + sfix)
        f_kernel.argtypes = [k_p, idx_t_p, idx_t, det_t_p, idx_t, idx_t_p, idx_t_p, k_p]
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


## H_ii kernels
@singledispatch
def k_H_OE_ii(J, J_ind, N, psi_det, N_det, H_p, H_v):
    raise NotImplementedError


@singledispatch
def k_H_A(J, J_ind, N, psi_det, N_det, H_p, H_v):
    raise NotImplementedError


@singledispatch
def k_H_B(J, J_ind, N, psi_det, N_det, H_p, H_v):
    raise NotImplementedError


@singledispatch
def k_H_F_ii(J, J_ind, N, psi_det, N_det, H_p, H_v):
    raise NotImplementedError


## H_ij kernels
@singledispatch
def k_H_OE_ij(J, J_ind, N, psi_det, N_det, H_p, H_c, H_v):
    raise NotImplementedError


@singledispatch
def k_H_C(J, J_ind, N, psi_det, N_det, H_p, H_c, H_v):
    raise NotImplementedError


@singledispatch
def k_H_D(J, J_ind, N, psi_det, N_det, H_p, H_c, H_v):
    raise NotImplementedError


@singledispatch
def k_H_E(J, J_ind, N, psi_det, N_det, H_p, H_c, H_v):
    raise NotImplementedError


@singledispatch
def k_H_F_ij(J, J_ind, N, psi_det, N_det, H_p, H_c, H_v):
    raise NotImplementedError


@singledispatch
def k_H_G(J, J_ind, N, psi_det, N_det, H_p, H_c, H_v):
    raise NotImplementedError


### Register offload kernels against python callers
pt2d_kernels = (k_OE_pt2d, k_TE_A_pt2d, k_TE_B_pt2d, k_TE_F_pt2d)
pt2n_kernels = (k_OE_pt2n, k_TE_C_pt2n, k_TE_D_pt2n, k_TE_E_pt2n, k_TE_F_pt2n, k_TE_G_pt2n)

h_ii_kernels = (k_H_OE_ii, k_H_A, k_H_B, k_H_F_ii)
h_ij_kernels = (k_H_OE_ij, k_H_C, k_H_D, k_H_E, k_H_F_ij, k_H_G)


for kern in pt2n_kernels + pt2d_kernels + h_ii_kernels + h_ij_kernels:
    name = kern.__name__
    for k in [f32, f64]:
        k_p = type_dict[k][1]
        pfix = "Kernels_"
        sfix = "_" + type_dict[k][0]
        target_name = pfix + name.split("_", 1)[1] + sfix
        f_dispatch = getattr(lib_kernels, target_name)
        kern.register(k_p, f_dispatch)


T_p = Union[f32_p, f64_p]


def dispatch_pt2_kernel(
    category: str,
) -> Tuple[
    Union[
        None, Callable[[T_p, idx_t_p, idx_t, det_t_p, T_p, idx_t, idx_t, det_t_p, idx_t, T_p], None]
    ],
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


def dispatch_H_kernel(
    category: str,
) -> Tuple[
    Union[None, Callable[[T_p, idx_t_p, idx_t, det_t_p, idx_t, idx_t_p, idx_t_p, T_p], None]],
    Union[None, Callable[[T_p, idx_t_p, idx_t, det_t_p, idx_t, idx_t_p, T_p], None]],
]:
    match category:
        case "OE":
            return (k_H_OE_ij, k_H_OE_ii)
        case "A":
            return (None, k_H_A)
        case "B":
            return (None, k_H_B)
        case "C":
            return (k_H_C, None)
        case "D":
            return (k_H_D, None)
        case "E":
            return (k_H_E, None)
        case "F":
            return (k_H_F_ij, k_H_F_ii)
        case "G":
            return (k_H_G, None)


def launch_H_ii_kernel(kernel, chunk, dets, H):
    kernel(
        chunk.J.p,
        chunk.idx.p,
        idx_t(chunk.chunk_size),
        dets.det_pointer,
        idx_t(dets.N_dets),
        H.A_p.p,
        H.A_v.p,
    )


def launch_H_ij_kernel(kernel, chunk, dets, H):
    kernel(
        chunk.J.p,
        chunk.idx.p,
        idx_t(chunk.chunk_size),
        dets.det_pointer,
        idx_t(dets.N_dets),
        H.A_p.p,
        H.A_c.p,
        H.A_v.p,
    )


def launch_denom_kernel(kernel, chunk, ext_dets, N_states, res):
    kernel(
        chunk.J.p,
        chunk.idx.p,
        idx_t(chunk.chunk_size),
        idx_t(N_states),
        ext_dets.det_pointer,
        ext_dets.N_dets,
        res.arr.p,
    )


def launch_num_kernel(kernel, chunk, int_dets, psi_coef, ext_dets, N_states, res):
    kernel(
        chunk.J.p,
        chunk.idx.p,
        idx_t(chunk.chunk_size),
        int_dets.det_pointer,
        psi_coef.arr.p,
        int_dets.N_dets,
        idx_t(N_states),
        ext_dets.det_pointer,
        ext_dets.N_dets,
        res.arr.p,
    )
