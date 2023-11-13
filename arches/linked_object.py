import pathlib
from abc import ABC, abstractmethod
from ctypes import CDLL, POINTER, c_double, c_float, c_int, c_long, c_uint, c_ulong, c_void_p
from functools import singledispatchmethod

import numpy as np
from numpy.ctypeslib import ndpointer

## Register all the types here for common use
# i64 is the indexing type for all
all_types = (c_int, c_long, c_uint, c_ulong, c_float, c_double)
i32, i64, ui32, ui64, f32, f64 = all_types
idx_t = i64  # alias for idx_t
handle_t = c_void_p  # void pointers

# TODO: Update when determinant array interface is implemented, though it's likely that
# the underlying determinant datatype will be a structured array anyway and it'll be easier
# to use a void pointer on the python side than creating a proper pointer via ctype structs
det_t_p = c_void_p

i32_p, i64_p, ui32_p, ui64_p, f32_p, f64_p = (POINTER(t) for t in all_types)
idx_t_p = i64_p
i32_a, i64_a, ui32_a, ui64_a, f32_a, f64_a = (ndpointer(t, flags="C_CONTIGUOUS") for t in all_types)
idx_t_a = i64_a
type_dict = {
    i32: ("i32", i32_p, i32_a),
    i64: ("i64", i64_p, i64_a),
    ui32: ("ui32", ui32_p, ui32_a),
    ui64: ("ui64", ui64_p, ui64_a),
    f32: ("f32", f32_p, f32_a),
    f64: ("f32", f64_p, f64_a),
}

np_type_map = {
    np.int32: i32,
    np.int64: i64,
    np.uint32: ui32,
    np.uint64: ui64,
    np.float32: f32,
    np.float64: f64,
}

run_folder = pathlib.Path(__file__).parent.resolve()
lib_interface = CDLL(run_folder.joinpath("build/libinterface.so"))


class LinkedHandle(ABC):
    """Base class for objects managed in memory via C++ and handled via Python."""

    def __init__(self, handle=None, **kwargs):
        if handle is None:
            self.handle = self.constructor(**kwargs)
            self._original = True
        else:
            self.handle = handle
            self._original = False

    def __del__(self):
        # only call destructor if the owning object is not a copied view
        # onto the array
        if self._original:
            self.destructor(self.handle)

    @abstractmethod
    def destructor(handle):
        pass

    @abstractmethod
    def constructor(self, **kwargs):
        pass


class ManagedArray(ABC):
    """Base class for arrays owned by LinkedHandle."""

    def __init__(self, pointer, size, ctype, dtype, ptype, data=None):
        self._p = pointer
        self._size = size
        self._dtype = dtype
        self._ctype = ctype
        self._ptype = ptype
        if data is not None:
            self[slice(0, size)] = data

    @property
    def p(self):
        return self._p

    @property
    def dtype(self):
        return self._dtype

    @property
    def ctype(self):
        return self._ctype

    @property
    def ptype(self):
        return self._ptype

    @property
    def size(self):
        return self._size

    @staticmethod
    @abstractmethod
    def at(p, k):
        pass

    @staticmethod
    @abstractmethod
    def set_val(p, k, v):
        pass

    @staticmethod
    @abstractmethod
    def set_range(p, k, v):
        pass

    @staticmethod
    @abstractmethod
    def set_strided_range(p, k, v):
        pass

    def __getitem__(self, k):
        if isinstance(k, slice):
            raise NotImplementedError
        else:
            return self.at(self.p, k)

    def __setitem__(self, k, v):
        if isinstance(k, slice):
            if isinstance(v, np.ndarray):
                d = v.ctypes.data_as(self.ptype)
            else:
                d = v
            if k.step:
                self.set_strided_range(self.p, k.start, k.stop, k.step, d)
            else:
                self.set_range(self.p, k.start, k.stop, d)
        else:
            self.set_val(self.p, k, self.ctype(v))


for k, v in type_dict.items():
    f_at = getattr(lib_interface, "at_" + v[0])
    f_set = getattr(lib_interface, "set_" + v[0])
    f_set_range = getattr(lib_interface, "set_range_" + v[0])
    f_set_strided_range = getattr(lib_interface, "set_strided_range_" + v[0])

    f_at.argtypes = [v[1], idx_t]
    f_set.argtypes = [v[1], idx_t, k]
    f_set_range.argtypes = [v[1], idx_t, idx_t, v[1]]
    f_set_strided_range.argtypes = [v[1], idx_t, idx_t, idx_t, v[1]]

    f_at.restype = k
    f_set.restype = None
    f_set_range.restype = None
    f_set_strided_range.restype = None


class ManagedArray_i32(ManagedArray):
    at = lib_interface.at_i32
    set_val = lib_interface.set_i32
    set_range = lib_interface.set_range_i32
    set_strided_range = lib_interface.set_strided_range_i32

    def __init__(self, pointer, size, data):
        super().__init__(pointer, size, ctype=i32, ptype=i32_p, dtype=np.int32, data=data)


class ManagedArray_i64(ManagedArray):
    at = lib_interface.at_i64
    set_val = lib_interface.set_i64
    set_range = lib_interface.set_range_i64
    set_strided_range = lib_interface.set_strided_range_i64

    def __init__(self, pointer, size, data):
        super().__init__(pointer, size, ctype=i64, ptype=i64_p, dtype=np.int64, data=data)


ManagedArray_idx_t = ManagedArray_i64  # alias for indexing arrays


class ManagedArray_ui32(ManagedArray):
    at = lib_interface.at_ui32
    set_val = lib_interface.set_ui32
    set_range = lib_interface.set_range_ui32
    set_strided_range = lib_interface.set_strided_range_ui32

    def __init__(self, pointer, size, data):
        super().__init__(pointer, size, ctype=ui32, ptype=ui32_p, dtype=np.uint32, data=data)


class ManagedArray_ui64(ManagedArray):
    at = lib_interface.at_ui64
    set_val = lib_interface.set_ui64
    set_range = lib_interface.set_range_ui64
    set_strided_range = lib_interface.set_strided_range_ui64

    def __init__(self, pointer, size, data):
        super().__init__(pointer, size, ctype=ui64, ptype=ui64_p, dtype=np.uint64, data=data)


class ManagedArray_f32(ManagedArray):
    at = lib_interface.at_f32
    set_val = lib_interface.set_f32
    set_range = lib_interface.set_range_f32
    set_strided_range = lib_interface.set_strided_range_f32

    def __init__(self, pointer, size, data):
        super().__init__(pointer, size, ctype=f32, ptype=f32_p, dtype=np.float32, data=data)


class ManagedArray_f64(ManagedArray):
    at = lib_interface.at_f64
    set_val = lib_interface.set_f64
    set_range = lib_interface.set_range_f64
    set_strided_range = lib_interface.set_strided_range_f64

    def __init__(self, pointer, size, data):
        super().__init__(pointer, size, ctype=f64, ptype=f64_p, dtype=np.float64, data=data)


class LinkedArray(LinkedHandle):
    """Base class for loose, singular arrays not part of a larger structured object."""

    def __init__(self, N, fill=None, handle=None, **kwargs):
        super().__init__(handle=handle, N=N, fill=None)
        self.arr = self.M_array_type(self.get_arr_ptr(self.handle), N)
        self.N = N

    def constructor(self, N, fill=None, **kwargs):
        if fill is None:
            return self._empty_ctor(idx_t(N))
        elif hasattr(fill, "__iter__"):
            if isinstance(fill, np.ndarray):
                d = fill.ctypes.data_as(self.ptype)
            else:
                d = fill  # Assume fill is an appropriate pointer type
            return self._copy_ctor(idx_t(N), d)
        else:
            return self._fill_ctor(idx_t(N), self.ctype(fill))

    def destructor(self):
        return self._dtor

    @property
    def M_array_type(self):
        return self._M_array_type

    @singledispatchmethod
    @abstractmethod
    @staticmethod
    def _add(b, a, c, N):  # b needs to be first arg to single dispatch over constant vs arr
        raise NotImplementedError

    @singledispatchmethod
    @abstractmethod
    @staticmethod
    def _iadd(b, a, N):
        raise NotImplementedError

    @singledispatchmethod
    @abstractmethod
    @staticmethod
    def _sub(b, a, c, N):
        raise NotImplementedError

    @singledispatchmethod
    @abstractmethod
    @staticmethod
    def _isub(b, a, N):
        raise NotImplementedError

    @singledispatchmethod
    @abstractmethod
    @staticmethod
    def _mul(b, a, c, N):
        raise NotImplementedError

    @singledispatchmethod
    @abstractmethod
    @staticmethod
    def _imul(b, a, N):
        raise NotImplementedError

    @singledispatchmethod
    @abstractmethod
    @staticmethod
    def _div(b, a, c, N):
        raise NotImplementedError

    @singledispatchmethod
    @abstractmethod
    @staticmethod
    def _idiv(b, a, N):
        raise NotImplementedError

    def __add__(self, b):
        res = self.allocate_result(self.N)
        match b:
            case LinkedArray():
                self._add(b.arr.p, self.arr.p, res.arr.p, self.N)
            case ManagedArray():
                self._add(b.p, self.arr.p, res.arr.p, self.N)
            case self.dtype:  # maybe need to track python scalar types
                self._add(self.ctype(b), self.arr.p, res.arr.p, self.N)

        return res

    def __iadd__(self, b):
        match b:
            case LinkedArray():
                self._iadd(b.arr.p, self.arr.p, self.arr.p, self.N)
            case ManagedArray():
                self._iadd(b.p, self.arr.p, self.arr.p, self.N)
            case self.dtype:  # maybe need to track python scalar types
                self._iadd(self.ctype(b), self.arr.p, self.arr.p, self.N)

        return self

    def __sub__(self, b):
        res = self.allocate_result(self.N)
        match b:
            case LinkedArray():
                self._sub(b.arr.p, self.arr.p, res.arr.p, self.N)
            case ManagedArray():
                self._sub(b.p, self.arr.p, res.arr.p, self.N)
            case self.dtype:  # maybe need to track python scalar types
                self._sub(self.ctype(b), self.arr.p, res.arr.p, self.N)

        return res

    def __isub__(self, b):
        match b:
            case LinkedArray():
                self._isub(b.arr.p, self.arr.p, self.arr.p, self.N)
            case ManagedArray():
                self._isub(b.p, self.arr.p, self.arr.p, self.N)
            case self.dtype:  # maybe need to track python scalar types
                self._isub(self.ctype(b), self.arr.p, self.arr.p, self.N)

        return self

    def __mul__(self, b):
        res = self.allocate_result(self.N)
        match b:
            case LinkedArray():
                self._mul(b.arr.p, self.arr.p, res.arr.p, self.N)
            case ManagedArray():
                self._mul(b.p, self.arr.p, res.arr.p, self.N)
            case self.dtype:  # maybe need to track python scalar types
                self._mul(self.ctype(b), self.arr.p, res.arr.p, self.N)

        return res

    def __imul__(self, b):
        match b:
            case LinkedArray():
                self._imul(b.arr.p, self.arr.p, self.arr.p, self.N)
            case ManagedArray():
                self._imul(b.p, self.arr.p, self.arr.p, self.N)
            case self.dtype:  # maybe need to track python scalar types
                self._imul(self.ctype(b), self.arr.p, self.arr.p, self.N)

        return self

    def __truediv__(self, b):
        res = self.allocate_result(self.N)
        match b:
            case LinkedArray():
                self._div(b.arr.p, self.arr.p, res.arr.p, self.N)
            case ManagedArray():
                self._div(b.p, self.arr.p, res.arr.p, self.N)
            case self.dtype:  # maybe need to track python scalar types
                self._div(self.ctype(b), self.arr.p, res.arr.p, self.N)

        return res

    def __itruediv__(self, b):
        match b:
            case LinkedArray():
                self._idiv(b.arr.p, self.arr.p, self.arr.p, self.N)
            case ManagedArray():
                self._idiv(b.p, self.arr.p, self.arr.p, self.N)
            case self.dtype:  # maybe need to track python scalar types
                self._idiv(self.ctype(b), self.arr.p, self.arr.p, self.N)

        return self

    def ipow2(self):
        self._ipow2(self.arr.p, self.N)

    @classmethod
    def allocate_result(cls, N):
        return cls(N)


class LinkedArray_f32(LinkedArray):
    _M_array_type = ManagedArray_f32
    _empty_ctor = lib_interface.LArray_ctor_e_f32
    _copy_ctor = lib_interface.LArray_ctor_a_f32
    _fill_ctor = lib_interface.LArray_ctor_c_f32
    _ipow2 = lib_interface.LArray_ipow2_f32

    # TODO: will all of these derived types conflict with eachother?
    for op in ["add", "iadd", "sub", "isub", "mul", "imul", "div", "idiv"]:
        _method = getattr(LinkedArray, "_" + op)
        _method.register(_M_array_type.pytpe, getattr(lib_interface, "LArray_" + op + "_f32"))
        _method.register(_M_array_type.ctype, getattr(lib_interface, "LArray_" + op + "_c_f32"))

    def __init__(self, N, fill=None, handle=None, **kwargs):
        super().__init__(N=N, fill=fill, handle=handle)


class LinkedArray_f64(LinkedArray):
    _M_array_type = ManagedArray_f64
    _empty_ctor = lib_interface.LArray_ctor_e_f64
    _copy_ctor = lib_interface.LArray_ctor_a_f64
    _fill_ctor = lib_interface.LArray_ctor_c_f64
    _ipow2 = lib_interface.LArray_ipow2_f64

    # TODO: will all of these derived types conflict with eachother?
    for op in ["add", "iadd", "sub", "isub", "mul", "imul", "div", "idiv"]:
        _method = getattr(LinkedArray, "_" + op)
        _method.register(_M_array_type.pytpe, getattr(lib_interface, "LArray_" + op + "_f64"))
        _method.register(_M_array_type.ctype, getattr(lib_interface, "LArray_" + op + "_c_f64"))

    def __init__(self, N, fill=None, handle=None, **kwargs):
        super().__init__(N=N, fill=fill, handle=handle)


class LinkedArray_i32(LinkedArray):
    _M_array_type = ManagedArray_i32
    _empty_ctor = lib_interface.LArray_ctor_e_i32
    _copy_ctor = lib_interface.LArray_ctor_a_i32
    _fill_ctor = lib_interface.LArray_ctor_c_i32
    _ipow2 = lib_interface.LArray_ipow2_i32

    # TODO: will all of these derived types conflict with eachother?
    for op in ["add", "iadd", "sub", "isub", "mul", "imul", "div", "idiv"]:
        _method = getattr(LinkedArray, "_" + op)
        _method.register(_M_array_type.pytpe, getattr(lib_interface, "LArray_" + op + "_i32"))
        _method.register(_M_array_type.ctype, getattr(lib_interface, "LArray_" + op + "_c_i32"))

    def __init__(self, N, fill=None, handle=None, **kwargs):
        super().__init__(N=N, fill=fill, handle=handle)


class LinkedArray_i64(LinkedArray):
    _M_array_type = ManagedArray_i64
    _empty_ctor = lib_interface.LArray_ctor_e_i64
    _copy_ctor = lib_interface.LArray_ctor_a_i64
    _fill_ctor = lib_interface.LArray_ctor_c_i64
    _ipow2 = lib_interface.LArray_ipow2_i64

    # TODO: will all of these derived types conflict with eachother?
    for op in ["add", "iadd", "sub", "isub", "mul", "imul", "div", "idiv"]:
        _method = getattr(LinkedArray, "_" + op)
        _method.register(_M_array_type.pytpe, getattr(lib_interface, "LArray_" + op + "_i64"))
        _method.register(_M_array_type.ctype, getattr(lib_interface, "LArray_" + op + "_c_i64"))

    def __init__(self, N, fill=None, handle=None, **kwargs):
        super().__init__(N=N, fill=fill, handle=handle)


class LinkedArray_ui32(LinkedArray):
    _M_array_type = ManagedArray_ui32
    _empty_ctor = lib_interface.LArray_ctor_e_ui32
    _copy_ctor = lib_interface.LArray_ctor_a_ui32
    _fill_ctor = lib_interface.LArray_ctor_c_ui32
    _ipow2 = lib_interface.LArray_ipow2_ui32

    # TODO: will all of these derived types conflict with eachother?
    for op in ["add", "iadd", "sub", "isub", "mul", "imul", "div", "idiv"]:
        _method = getattr(LinkedArray, "_" + op)
        _method.register(_M_array_type.pytpe, getattr(lib_interface, "LArray_" + op + "_ui32"))
        _method.register(_M_array_type.ctype, getattr(lib_interface, "LArray_" + op + "_c_ui32"))

    def __init__(self, N, fill=None, handle=None, **kwargs):
        super().__init__(N=N, fill=fill, handle=handle)


class LinkedArray_ui64(LinkedArray):
    _M_array_type = ManagedArray_ui64
    _empty_ctor = lib_interface.LArray_ctor_e_ui64
    _copy_ctor = lib_interface.LArray_ctor_a_ui64
    _fill_ctor = lib_interface.LArray_ctor_c_ui64
    _ipow2 = lib_interface.LArray_ipow2_ui64

    # TODO: will all of these derived types conflict with eachother?
    for op in ["add", "iadd", "sub", "isub", "mul", "imul", "div", "idiv"]:
        _method = getattr(LinkedArray, "_" + op)
        _method.register(_M_array_type.pytpe, getattr(lib_interface, "LArray_" + op + "_ui64"))
        _method.register(_M_array_type.ctype, getattr(lib_interface, "LArray_" + op + "_c_ui64"))

    def __init__(self, N, fill=None, handle=None, **kwargs):
        super().__init__(N=N, fill=fill, handle=handle)
