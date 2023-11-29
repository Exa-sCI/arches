import pathlib
from abc import ABC, abstractmethod
from ctypes import CDLL, c_bool
from dataclasses import dataclass

import numpy as np

from arches.linked_object import LinkedArray_i32, LinkedHandle, handle_t, i32, i32_p, idx_t, idx_t_p

run_folder = pathlib.Path(__file__).parent.resolve()
lib_dets = CDLL(run_folder.joinpath("build/libdeterminant.so"))

#### spin det t
lib_dets.Dets_spin_det_t_empty_ctor.argtypes = [idx_t]
lib_dets.Dets_spin_det_t_empty_ctor.restype = handle_t

lib_dets.Dets_spin_det_t_fill_ctor.argtypes = [idx_t, idx_t]
lib_dets.Dets_spin_det_t_fill_ctor.restype = handle_t

lib_dets.Dets_spin_det_t_orb_list_ctor.argtypes = [idx_t, idx_t, idx_t_p]
lib_dets.Dets_spin_det_t_orb_list_ctor.restype = handle_t

lib_dets.Dets_spin_det_t_dtor.argtypes = [handle_t]
lib_dets.Dets_spin_det_t_dtor.restype = None

lib_dets.Dets_spin_det_t_print.argtypes = [handle_t]
lib_dets.Dets_spin_det_t_print.restype = None

lib_dets.Dets_spin_det_t_to_bit_tuple.argtypes = [handle_t, idx_t, idx_t, i32_p]
lib_dets.Dets_spin_det_t_to_bit_tuple.restype = None

lib_dets.Dets_spin_det_t_set_orb.argtypes = [handle_t, idx_t, c_bool]
lib_dets.Dets_spin_det_t_set_orb.restype = None

lib_dets.Dets_spin_det_t_get_orb.argtypes = [handle_t, idx_t]
lib_dets.Dets_spin_det_t_get_orb.restype = c_bool

lib_dets.Dets_spin_det_t_set_orb_range.argtypes = [handle_t, idx_t, idx_t, c_bool]
lib_dets.Dets_spin_det_t_set_orb_range.restype = None

lib_dets.Dets_spin_det_t_bit_flip.argtypes = [handle_t]
lib_dets.Dets_spin_det_t_bit_flip.restype = handle_t

lib_dets.Dets_spin_det_t_xor.argtypes = [handle_t, handle_t]
lib_dets.Dets_spin_det_t_xor.restype = handle_t

lib_dets.Dets_spin_det_t_and.argtypes = [handle_t, handle_t]
lib_dets.Dets_spin_det_t_and.restype = handle_t

lib_dets.Dets_spin_det_t_count.argtypes = [handle_t]
lib_dets.Dets_spin_det_t_count.restype = i32

lib_dets.Dets_spin_det_t_phase_single_exc.argtypes = [handle_t, idx_t, idx_t]
lib_dets.Dets_spin_det_t_phase_single_exc.restype = i32

lib_dets.Dets_spin_det_t_phase_double_exc.argtypes = [handle_t, idx_t, idx_t, idx_t, idx_t]
lib_dets.Dets_spin_det_t_phase_double_exc.restype = i32

lib_dets.Dets_spin_det_t_apply_single_exc.argtypes = [handle_t, idx_t, idx_t]
lib_dets.Dets_spin_det_t_apply_single_exc.restype = handle_t

lib_dets.Dets_spin_det_t_apply_double_exc.argtypes = [handle_t, idx_t, idx_t, idx_t, idx_t]
lib_dets.Dets_spin_det_t_apply_double_exc.restype = handle_t

#### det t
lib_dets.Dets_det_t_empty_ctor.argtypes = [idx_t]
lib_dets.Dets_det_t_empty_ctor.restype = handle_t

lib_dets.Dets_det_t_copy_ctor.argtypes = [handle_t, handle_t]
lib_dets.Dets_det_t_copy_ctor.restype = handle_t

lib_dets.Dets_det_t_dtor.argtypes = [handle_t]
lib_dets.Dets_det_t_dtor.restype = None

lib_dets.Dets_det_t_get_spin_det_handle.argtypes = [handle_t, c_bool]
lib_dets.Dets_det_t_get_spin_det_handle.restype = handle_t

lib_dets.Dets_det_t_phase_double_exc.argtypes = [handle_t, idx_t, idx_t, idx_t, idx_t]
lib_dets.Dets_det_t_phase_double_exc.restype = i32

lib_dets.Dets_det_t_apply_single_exc.argtypes = [handle_t, idx_t, idx_t, idx_t]
lib_dets.Dets_det_t_apply_single_exc.restype = handle_t

lib_dets.Dets_det_t_apply_double_exc.argtypes = [handle_t, idx_t, idx_t, idx_t, idx_t, idx_t, idx_t]
lib_dets.Dets_det_t_apply_double_exc.restype = handle_t


#### DetArray
lib_dets.Dets_DetArray_empty_ctor.argtypes = [idx_t, idx_t]
lib_dets.Dets_DetArray_empty_ctor.restype = handle_t

lib_dets.Dets_DetArray_dtor.argtypes = [handle_t]
lib_dets.Dets_DetArray_dtor.restype = None

lib_dets.Dets_DetArray_getitem.argtypes = [handle_t, idx_t]
lib_dets.Dets_DetArray_getitem.restype = handle_t

lib_dets.Dets_DetArray_setitem.argtypes = [handle_t, handle_t, idx_t]
lib_dets.Dets_DetArray_setitem.restype = None

lib_dets.Dets_DetArray_get_N_dets.argtypes = [handle_t]
lib_dets.Dets_DetArray_get_N_dets.restype = idx_t

lib_dets.Dets_DetArray_get_N_mos.argtypes = [handle_t]
lib_dets.Dets_DetArray_get_N_mos.restype = idx_t

#### Generation routines
lib_dets.Dets_get_all_connected_singles.argtypes = [handle_t]
lib_dets.Dets_get_all_connected_singles.restype = handle_t

lib_dets.Dets_get_connected_same_spin_doubles.argtypes = [handle_t]
lib_dets.Dets_get_connected_same_spin_doubles.restype = handle_t

lib_dets.Dets_get_connected_opp_spin_doubles.argtypes = [handle_t]
lib_dets.Dets_get_connected_opp_spin_doubles.restype = handle_t

lib_dets.Dets_get_connected_dets.argtypes = [handle_t]
lib_dets.Dets_get_connected_dets.restype = handle_t


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
            case tuple() | list():
                occ_arr = np.array(occ).astype(np.int64).ctypes.data_as(idx_t_p)
                return self._orb_list_ctor(idx_t(N_orbs), idx_t(len(occ)), occ_arr)

    def destructor(self, handle):
        self._dtor(handle)

    def debug_print(self):
        lib_dets.Dets_spin_det_t_print(self.handle)

    @property
    def as_bit_tuple(self):
        return self[0 : self.N_orbs]

    @property
    def as_orb_list(self):
        return tuple([i for i, b in enumerate(self.as_bit_tuple) if b])

    def __setitem__(self, k, v):
        if isinstance(k, slice):
            if k.step is None:
                lib_dets.Dets_spin_det_t_set_orb_range(
                    self.handle, idx_t(k.start), idx_t(k.stop), c_bool(v)
                )
            else:
                raise NotImplementedError
        else:
            lib_dets.Dets_spin_det_t_set_orb(self.handle, idx_t(k), c_bool(v))

    def __getitem__(self, k):
        if isinstance(k, slice):
            if k.step is None:
                res = LinkedArray_i32(k.stop - k.start)
                lib_dets.Dets_spin_det_t_to_bit_tuple(
                    self.handle, idx_t(k.start), idx_t(k.stop), res.arr.p
                )
                return tuple(res.arr.np_arr)
            else:
                raise NotImplementedError
        else:
            return lib_dets.Dets_spin_det_t_get_orb(self.handle, idx_t(k))

    def __invert__(self):
        handle = lib_dets.Dets_spin_det_t_bit_flip(self.handle)
        return spin_det_t(self.N_orbs, handle=handle, override_original=True)

    def __xor__(self, other):
        handle = lib_dets.Dets_spin_det_t_xor(self.handle, other.handle)
        return spin_det_t(self.N_orbs, handle=handle, override_original=True)

    def __and__(self, other):
        handle = lib_dets.Dets_spin_det_t_and(self.handle, other.handle)
        return spin_det_t(self.N_orbs, handle=handle, override_original=True)

    def popcount(self):
        return lib_dets.Dets_spin_det_t_count(self.handle)

    def compute_phase_single_exc(self, h, p):
        return lib_dets.Dets_spin_det_t_phase_single_exc(self.handle, idx_t(h), idx_t(p))

    def compute_phase_double_exc(self, h1, h2, p1, p2):
        return lib_dets.Dets_spin_det_t_phase_double_exc(
            self.handle, idx_t(h1), idx_t(h2), idx_t(p1), idx_t(p2)
        )


class det_t(LinkedHandle):
    _empty_ctor = lib_dets.Dets_det_t_empty_ctor
    _copy_ctor = lib_dets.Dets_det_t_copy_ctor
    _dtor = lib_dets.Dets_det_t_dtor

    def __init__(self, N_orbs=None, alpha=None, beta=None, handle=None, **kwargs):
        super().__init__(handle=handle, N_orbs=N_orbs, alpha=alpha, beta=beta, **kwargs)
        if N_orbs is None:
            self.N_orbs = alpha.N_orbs
        else:
            self.N_orbs = N_orbs

        self.alpha = spin_det_t(self.N_orbs, handle=self.get_spin_det_handle(False))
        self.beta = spin_det_t(self.N_orbs, handle=self.get_spin_det_handle(True))

    def constructor(self, N_orbs, alpha, beta, **kwargs):
        if (alpha is None) and (beta is None):
            return self._empty_ctor(N_orbs)
        else:
            return self._copy_ctor(alpha.handle, beta.handle)

    def destructor(self, handle):
        self._dtor(handle)

    def get_spin_det_handle(self, idx):
        return lib_dets.Dets_det_t_get_spin_det_handle(self.handle, c_bool(idx))

    def debug_print(self):
        self.alpha.debug_print()
        self.beta.debug_print()

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, sdet):
        if isinstance(sdet, spin_det_t):
            self._alpha = sdet
        else:
            raise TypeError

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, sdet):
        if isinstance(sdet, spin_det_t):
            self._beta = sdet
        else:
            raise TypeError

    def __getitem__(self, val):
        match val:
            case 0:
                return self.alpha
            case 1:
                return self.beta
            case _:
                raise ValueError

    def compute_phase_opp_spin_double_exc(self, h1, h2, p1, p2):
        return lib_dets.Dets_det_t_phase_double_exc(
            self.handle, idx_t(h1), idx_t(h2), idx_t(p1), idx_t(p2)
        )

    def get_exc_det(self, other):
        res_handle = lib_dets.Dets_det_t_exc_det(self.handle, other.handle)
        return det_t(handle=res_handle, override_original=True, N_orbs=self.N_orbs)

    def get_connected_singles(self):
        res_handle = lib_dets.Dets_get_all_connected_singles(self.handle)
        N_dets = lib_dets.Dets_DetArray_get_N_dets(res_handle)
        return DetArray(
            handle=res_handle, override_original=True, N_dets=N_dets, N_orbs=self.N_orbs
        )

    def get_connected_ss_doubles(self):
        res_handle = lib_dets.Dets_get_connected_same_spin_doubles(self.handle)
        N_dets = lib_dets.Dets_DetArray_get_N_dets(res_handle)
        return DetArray(
            handle=res_handle, override_original=True, N_dets=N_dets, N_orbs=self.N_orbs
        )

    def get_connected_os_doubles(self):
        res_handle = lib_dets.Dets_get_connected_opp_spin_doubles(self.handle)
        N_dets = lib_dets.Dets_DetArray_get_N_dets(res_handle)
        return DetArray(
            handle=res_handle, override_original=True, N_dets=N_dets, N_orbs=self.N_orbs
        )

    def generate_connected_dets(self):
        res_handle = lib_dets.Dets_get_connected_dets(self.handle)
        N_dets = lib_dets.Dets_DetArray_get_N_dets(res_handle)
        return DetArray(
            handle=res_handle, override_original=True, N_dets=N_dets, N_orbs=self.N_orbs
        )


class DetArray(LinkedHandle):
    _empty_ctor = lib_dets.Dets_DetArray_empty_ctor
    _dtor = lib_dets.Dets_DetArray_dtor

    def __init__(self, N_dets, N_orbs, handle=None, **kwargs):
        super().__init__(handle=handle, N_dets=N_dets, N_orbs=N_orbs, **kwargs)
        self.N_dets = N_dets
        self.N_orbs = N_orbs

    def constructor(self, N_dets, N_orbs, **kwargs):
        return self._empty_ctor(idx_t(N_dets), idx_t(N_orbs))

    def destructor(self, handle):
        self._dtor(handle)

    def __getitem__(self, k):
        if isinstance(k, slice):
            raise NotImplementedError
        if np.issubdtype(type(k), np.integer):
            res_handle = lib_dets.Dets_DetArray_getitem(self.handle, idx_t(k))
            return det_t(handle=res_handle, N_orbs=self.N_orbs)
        else:
            raise TypeError

    def __setitem__(self, k, v):
        if isinstance(k, slice):
            raise NotImplementedError
        if np.issubdtype(type(k), np.integer) and isinstance(v, det_t):
            lib_dets.Dets_DetArray_setitem(self.handle, v.handle, idx_t(k))
        else:
            raise TypeError


class exc(ABC):
    def __init__(self, h, p, spin):
        self.spin = spin
        self.h = h
        self.p = p

    @property
    def h(self):
        return self._h

    @h.setter
    @abstractmethod
    def h(self, v):
        pass

    @property
    def p(self):
        return self._p

    @p.setter
    @abstractmethod
    def p(self, v):
        pass

    @property
    def spin(self):
        return self._spin

    @spin.setter
    @abstractmethod
    def spin(self, v):
        pass

    @abstractmethod
    def __matmul__(self, other):
        pass


class single_exc(exc):
    def __init__(self, h, p, spin=None):
        super().__init__(h, p, spin)

    def h(self, v):
        if np.issubdtype(type(v), np.integer):
            self._h = v
        else:
            raise TypeError

    def p(self, v):
        if np.issubdtype(type(v), np.integer):
            self._p = v
        else:
            raise TypeError

    def spin(self, v):
        match v:
            case 1 | 0 | None:
                self._v = v
            case _:
                raise ValueError

    def __matmul__(self, other):
        match other:
            case spin_det_t():
                res_handle = lib_dets.Dets_spin_det_t_apply_single_exc(
                    other.handle, idx_t(self.h), idx_t(self.p)
                )
                return spin_det_t(handle=res_handle, override_handle=True, N_orbs=other.N_orbs)
            case det_t():
                res_handle = lib_dets.Dets_det_t_apply_single_exc(
                    other.handle, idx_t(self.spin), idx_t(self.h), idx_t(self.p)
                )
                return det_t(handle=res_handle, override_original=True, N_orbs=other.N_orbs)
            case single_exc():
                if self.spin is None and other.spin is None:
                    return double_exc((self.h, other.h), (self.p, other.p))
                elif self.spin is not None and other.spin is not None:
                    return double_exc((self.h, other.h), (self.p, other.p), (self.spin, other.spin))
                else:
                    raise ValueError
            case _:
                raise NotImplementedError


class double_exc(exc):
    def __init__(self, h, p, spin=None):
        if len(h) != 2:
            raise ValueError

        if len(h) != len(p):
            raise ValueError

        if (spin is not None) and (len(h) != len(spin)):
            raise ValueError

        super().__init__(h, p, spin)

    def h(self, v):
        if np.issubdtype(type(v[0]), np.integer) and np.issubdtype(type(v[1]), np.integer):
            if v[0] == v[1] and ((self.spin is None) or (self.spin[0] == self.spin[1])):
                raise ValueError
            self._h = v
        else:
            raise TypeError

    def p(self, v):
        if np.issubdtype(type(v[0]), np.integer) and np.issubdtype(type(v[1]), np.integer):
            if v[0] == v[1] and ((self.spin is None) or (self.spin[0] == self.spin[1])):
                raise ValueError
            self._p = v
        else:
            raise TypeError

    def spin(self, v):
        match v:
            case None:
                self._v = v
            case tuple() | list():
                if v[0] in (0, 1) and v[1] in (0, 1):
                    self._v = v
                else:
                    raise ValueError
            case _:
                raise ValueError

    def __matmul__(self, other):
        match other:
            case spin_det_t():
                res_handle = lib_dets.Dets_spin_det_t_apply_double_exc(
                    other.handle,
                    idx_t(self.h[0]),
                    idx_t(self.h[1]),
                    idx_t(self.p[0]),
                    idx_t(self.p[1]),
                )
                return spin_det_t(handle=res_handle, override_handle=True, N_orbs=other.N_orbs)
            case det_t():
                if self.spin is None:
                    raise ValueError
                res_handle = lib_dets.Dets_det_t_apply_double_exc(
                    other.handle,
                    idx_t(self.spin[0]),
                    idx_t(self.spin[1]),
                    idx_t(self.h[0]),
                    idx_t(self.h[1]),
                    idx_t(self.p[0]),
                    idx_t(self.p[1]),
                )
                return det_t(handle=res_handle, override_original=True, N_orbs=other.N_orbs)
            case _:
                raise NotImplementedError
