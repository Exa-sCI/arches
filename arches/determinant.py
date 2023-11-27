import pathlib
from ctypes import CDLL

import numpy as np

from arches.linked_object import LinkedHandle, handle_t, idx_t, idx_t_p

run_folder = pathlib.Path(__file__).parent.resolve()
lib_dets = CDLL(run_folder.joinpath("build/libdeterminant.so"))


lib_dets.Dets_spin_det_t_empty_ctor.argtypes = [idx_t]
lib_dets.Dets_spin_det_t_empty_ctor.restype = handle_t

lib_dets.Dets_spin_det_t_fill_ctor.argtypes = [idx_t, idx_t]
lib_dets.Dets_spin_det_t_fill_ctor.restype = handle_t

lib_dets.Dets_spin_det_t_orb_list_ctor.argtypes = [idx_t, idx_t, idx_t_p]
lib_dets.Dets_spin_det_t_orb_list_ctor.restype = handle_t

lib_dets.Dets_spin_det_t_dtor.argtypes = [handle_t]
lib_dets.Dets_spin_det_t_dtor.restype = None


class spin_det_t(LinkedHandle):
    _empty_ctor = lib_dets.Dets_spin_det_t_empty_ctor
    _fill_ctor = lib_dets.Dets_spin_det_t_fill_ctor
    _orb_list_ctor = lib_dets.Dets_spin_det_t_orb_list_ctor
    _dtor = lib_dets.Dets_spin_det_t_dtor

    def __init__(self, N_orbs, occ=False, max_orb=None, handle=None, **kwargs):
        super().__init__(handle=handle, N_orbs=N_orbs, occ=occ, max_orb=max_orb, **kwargs)
        self.N_orbs = N_orbs

    def constructor(self, N_orbs, occ, max_orb, **kwargs):
        match occ:
            case True:
                return self._fill_ctor(idx_t(N_orbs), idx_t(max_orb))
            case False:
                return self._empty_ctor(idx_t(N_orbs))
            case tuple():
                occ_arr = np.array(occ).astype(np.int64).ctypes.data_as(idx_t_p)
                return self._orb_list_ctor(idx_t(N_orbs), idx_t(len(occ)), occ_arr)

    def destructor(self, handle):
        self._dtor(handle)
