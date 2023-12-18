import pathlib

import numpy as np

from arches.algorithms import evaluate_H_entries, s_CI
from arches.determinant import det_t, spin_det_t
from arches.integrals import load_integrals_into_chunks
from arches.io import load_integrals
from arches.linked_object import LinkedArray_idx_t, get_LArray
from arches.log_timer import logtimer
from arches.matrix import DMatrix, SymCSRMatrix


class FakeComm:
    # to avoid initializing MPI for just tests
    def __init__(self, rank, size):
        self.rank = rank
        self.size = size

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size


def get_ground_state(N_orbs, N_elec):
    max_orb = N_elec // 2
    sdet = spin_det_t(N_elec, occ=True, max_orb=max_orb)
    return det_t(N_orbs, sdet, sdet)


def get_E0(ground_state, chunks, V_nn, dtype):
    A_p = LinkedArray_idx_t(2, np.array([0, 1]).astype(np.int64))
    A_c = LinkedArray_idx_t(1, np.array([0]).astype(np.int64))
    A_v = get_LArray(dtype=dtype)(1)
    H = SymCSRMatrix(1, 1, dtype, A_p=A_p.arr, A_c=A_c.arr, A_v=A_v.arr)

    evaluate_H_entries(H, ground_state, chunks, V_nn)
    return H.A_v[0]


def get_constraint(ground_det, N_holes, N_parts):
    occ_orbs = ground_det.alpha.as_orb_list
    all_orbs = set([x for x in range(ground_det.N_orbs)])
    h_constraint = occ_orbs[-N_holes:]
    p_constraint = tuple(all_orbs.difference(occ_orbs).difference(set(h_constraint)))[:N_parts]
    return (h_constraint, p_constraint)


config_map = (
    lambda fp, config, **kwargs: (str(fp).rsplit("/", 1)[1], config["constraint"]),
    lambda name, constraint: f"| Example: {name.split('.')[0]}, (N_h, N_p): {constraint}",
)


@logtimer(config_map, report_on=("call",))
def run_example(fp, config, dtype):
    N_orbs, N_elec = load_integrals(str(fp), return_size_only=True)
    ground_det = get_ground_state(N_orbs, N_elec)
    ground_psi = DMatrix(1, 1, 1.0, dtype=dtype)
    config["constraint"] = get_constraint(ground_det, *config["constraint"])

    N_orbs, N_elec, V_nn, chunks = load_integrals_into_chunks(
        str(fp),
        FakeComm(0, 1),
        dtype=dtype,
        constraint=config["constraint"],
        chunk_size=512,
        screening_threshold=config["screening_threshold"],
    )

    E0 = get_E0(ground_det, chunks, V_nn, dtype)

    (E_var, psi_var, dets) = s_CI(V_nn, E0, ground_det, ground_psi, chunks, **config)


def nh3(dtype):
    print("###### Running NH3 example ######\n\n")
    fp = pathlib.Path("../data/nh3.5det.fcidump")
    config = {
        "N_states": 1,
        "N_max_dets": 2000,
        "pt2_conv": 1e-6,
        "pt2_threshold": 1e-8,
        "constraint": (4, 4),
        "screening_threshold": 1e-8,
    }
    run_example(fp, config, dtype)
    print("###### NH3 example finished ######\n")


def f2(dtype):
    print("###### Running F2 example ######")
    fp = pathlib.Path("../data/f2_631g.18det.fcidump")
    config = {
        "N_states": 1,
        "N_max_dets": 2000,
        "pt2_conv": 1e-6,
        "pt2_threshold": 1e-8,
        "constraint": (4, 4),
        "screening_threshold": 1e-8,
    }
    run_example(fp, config, dtype)
    print("###### F2 example finished ######\n")


def c2(dtype):
    print("###### Running C2 example ######")
    fp = pathlib.Path("../data/c2_eq_hf_dz.fcidump")
    config = {
        "N_states": 1,
        "N_max_dets": 2000,
        "pt2_conv": 1e-6,
        "pt2_threshold": 1e-8,
        "constraint": (4, 4),
        "screening_threshold": 1e-8,
    }
    run_example(fp, config, dtype)
    print("###### C2 example finished ######\n")


def main(dtype=np.float64):
    nh3(dtype)
    print("------------------\n")
    f2(dtype)
    print("------------------\n")
    c2(dtype)


if __name__ == "__main__":
    main()
