import pathlib
from abc import ABC, abstractmethod
from ctypes import (
    CDLL,
    POINTER,
    c_double,
    c_float,
    c_int32,
    c_int64,
    c_uint32,
    c_uint64,
    c_void_p,
)
from functools import singledispatchmethod

import numpy as np
from numpy.ctypeslib import ndpointer

## Register all the types here for common use
# i64 is the indexing type for all


all_types = (c_int32, c_int64, c_uint32, c_uint64, c_float, c_double)
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
    f64: ("f64", f64_p, f64_a),
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
lib_array = CDLL(run_folder.joinpath("build/libarrays.so"))


class LinkedHandle(ABC):
    """Base class for objects managed in memory via C++ and handled via Python."""

    def __init__(self, handle=None, **kwargs):
        if handle is None:
            self.handle = self.constructor(**kwargs)
            self._original = True
        else:
            self.handle = handle
            self._original = False
            if "override_original" in kwargs.keys():
                self._original = kwargs["override_original"]

    def __del__(self):
        # only call destructor if the owning object is not a copied view
        # onto the array
        if self._original:
            self.destructor(self.handle)

    @abstractmethod
    def destructor(self, handle):
        pass

    @abstractmethod
    def constructor(self, **kwargs):
        pass


class ManagedArray(ABC):
    """Base class for arrays owned by LinkedHandle."""

    def __init__(self, pointer, size, data=None):
        self._p = pointer
        self._size = size
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

    @property
    def np_arr(self):
        return np.fromiter(self.p, dtype=self.dtype, count=self.size)

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
    pfix = "LArray_"
    sfix = "_" + type_dict[k][0]  # key : value is (c_type : (name, pointer, np ndpointer))
    k_p = type_dict[k][1]

    f_at = getattr(lib_array, "at" + sfix)
    f_set = getattr(lib_array, "set" + sfix)
    f_set_range = getattr(lib_array, "set_range" + sfix)
    f_set_strided_range = getattr(lib_array, "set_strided_range" + sfix)

    f_at.argtypes = [k_p, idx_t]
    f_set.argtypes = [k_p, idx_t, k]
    f_set_range.argtypes = [k_p, idx_t, idx_t, k_p]
    f_set_strided_range.argtypes = [k_p, idx_t, idx_t, idx_t, k_p]

    f_at.restype = k
    f_set.restype = None
    f_set_range.restype = None
    f_set_strided_range.restype = None

    empty_ctor = getattr(lib_array, pfix + "ctor_e" + sfix)
    fill_ctor = getattr(lib_array, pfix + "ctor_c" + sfix)
    copy_ctor = getattr(lib_array, pfix + "ctor_c" + sfix)
    dtor = getattr(lib_array, pfix + "dtor" + sfix)
    ptr_return = getattr(lib_array, pfix + "get_arr_ptr" + sfix)

    empty_ctor.argtypes = [idx_t]
    fill_ctor.argtypes = [idx_t, k]
    copy_ctor.argtypes = [idx_t, k_p]
    dtor.argtypes = [handle_t]
    ptr_return.argtypes = [handle_t]

    empty_ctor.restype = handle_t
    fill_ctor.restype = handle_t
    copy_ctor.restype = handle_t
    dtor.restype = None
    ptr_return.restype = k_p

    for op in ["add", "iadd", "sub", "isub", "mul", "imul", "div", "idiv"]:
        f_a = getattr(lib_array, pfix + op + sfix)
        f_a.argtypes = [k_p, k_p, k_p, idx_t]
        f_a.restype = None

        f_c = getattr(lib_array, pfix + op + "_c" + sfix)
        f_c.argtypes = [k_p, k, k_p, idx_t]
        f_c.restype = None

    ipow2 = getattr(lib_array, pfix + "ipow2" + sfix)
    ipow2.argtypes = [k_p, idx_t]
    ipow2.restype = None


class ManagedArray_i32(ManagedArray):
    at = lib_array.at_i32
    set_val = lib_array.set_i32
    set_range = lib_array.set_range_i32
    set_strided_range = lib_array.set_strided_range_i32
    _ctype = i32
    _ptype = i32_p
    _dtype = np.dtype(np.int32)

    def __init__(self, pointer, size, data):
        super().__init__(pointer, size, data=data)


class ManagedArray_i64(ManagedArray):
    at = lib_array.at_i64
    set_val = lib_array.set_i64
    set_range = lib_array.set_range_i64
    set_strided_range = lib_array.set_strided_range_i64
    _ctype = i64
    _ptype = i64_p
    _dtype = np.dtype(np.int64)

    def __init__(self, pointer, size, data):
        super().__init__(pointer, size, data=data)


ManagedArray_idx_t = ManagedArray_i64  # alias for indexing arrays


class ManagedArray_ui32(ManagedArray):
    at = lib_array.at_ui32
    set_val = lib_array.set_ui32
    set_range = lib_array.set_range_ui32
    set_strided_range = lib_array.set_strided_range_ui32
    _ctype = ui32
    _ptype = ui32_p
    _dtype = np.dtype(np.uint32)

    def __init__(self, pointer, size, data):
        super().__init__(pointer, size, data=data)


class ManagedArray_ui64(ManagedArray):
    at = lib_array.at_ui64
    set_val = lib_array.set_ui64
    set_range = lib_array.set_range_ui64
    set_strided_range = lib_array.set_strided_range_ui64
    _ctype = ui64
    _ptype = ui64_p
    _dtype = np.dtype(np.uint64)

    def __init__(self, pointer, size, data):
        super().__init__(pointer, size, data=data)


class ManagedArray_f32(ManagedArray):
    at = lib_array.at_f32
    set_val = lib_array.set_f32
    set_range = lib_array.set_range_f32
    set_strided_range = lib_array.set_strided_range_f32
    _ctype = f32
    _ptype = f32_p
    _dtype = np.dtype(np.float32)

    def __init__(self, pointer, size, data):
        super().__init__(pointer, size, data=data)


class ManagedArray_f64(ManagedArray):
    at = lib_array.at_f64
    set_val = lib_array.set_f64
    set_range = lib_array.set_range_f64
    set_strided_range = lib_array.set_strided_range_f64
    _ctype = f64
    _ptype = f64_p
    _dtype = np.dtype(np.float64)

    def __init__(self, pointer, size, data):
        super().__init__(pointer, size, data=data)


class LinkedArray(LinkedHandle):
    """Base class for loose, singular arrays not part of a larger structured object."""

    def __init__(self, N, fill=None, handle=None, **kwargs):
        super().__init__(handle=handle, N=N, fill=fill)
        self.arr = self.M_array_type(self.get_arr_ptr(self.handle), N, None)
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

    def destructor(self, handle):
        self._dtor(handle)

    @property
    def get_arr_ptr(self):
        return self._get_arr_ptr

    @property
    def M_array_type(self):
        return self._M_array_type

    @singledispatchmethod
    @staticmethod
    def _add(b, a, c, N):  # b needs to be first arg to single dispatch over constant vs arr
        raise NotImplementedError

    @singledispatchmethod
    @staticmethod
    def _iadd(b, a, N):
        raise NotImplementedError

    @singledispatchmethod
    @staticmethod
    def _sub(b, a, c, N):
        raise NotImplementedError

    @singledispatchmethod
    @staticmethod
    def _isub(b, a, N):
        raise NotImplementedError

    @singledispatchmethod
    @staticmethod
    def _mul(b, a, c, N):
        raise NotImplementedError

    @singledispatchmethod
    @staticmethod
    def _imul(b, a, N):
        raise NotImplementedError

    @singledispatchmethod
    @staticmethod
    def _div(b, a, c, N):
        raise NotImplementedError

    @singledispatchmethod
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


class LinkedArray_i32(LinkedArray):
    _M_array_type = ManagedArray_i32
    _empty_ctor = lib_array.LArray_ctor_e_i32
    _copy_ctor = lib_array.LArray_ctor_a_i32
    _fill_ctor = lib_array.LArray_ctor_c_i32
    _dtor = lib_array.LArray_dtor_i32
    _get_arr_ptr = lib_array.LArray_get_arr_ptr_i32
    _ipow2 = lib_array.LArray_ipow2_i32

    # TODO: will all of these derived types conflict with eachother?
    for op in ["add", "iadd", "sub", "isub", "mul", "imul", "div", "idiv"]:
        _method = getattr(LinkedArray, "_" + op)
        _method.register(i32_p, getattr(lib_array, "LArray_" + op + "_i32"))
        _method.register(i32, getattr(lib_array, "LArray_" + op + "_c_i32"))

    def __init__(self, N, fill=None, handle=None, **kwargs):
        super().__init__(N=N, fill=fill, handle=handle)


class LinkedArray_i64(LinkedArray):
    _M_array_type = ManagedArray_i64
    _empty_ctor = lib_array.LArray_ctor_e_i64
    _copy_ctor = lib_array.LArray_ctor_a_i64
    _fill_ctor = lib_array.LArray_ctor_c_i64
    _dtor = lib_array.LArray_dtor_i64
    _get_arr_ptr = lib_array.LArray_get_arr_ptr_i64
    _ipow2 = lib_array.LArray_ipow2_i64

    # TODO: will all of these derived types conflict with eachother?
    for op in ["add", "iadd", "sub", "isub", "mul", "imul", "div", "idiv"]:
        _method = getattr(LinkedArray, "_" + op)
        _method.register(i64_p, getattr(lib_array, "LArray_" + op + "_i64"))
        _method.register(i64, getattr(lib_array, "LArray_" + op + "_c_i64"))

    def __init__(self, N, fill=None, handle=None, **kwargs):
        super().__init__(N=N, fill=fill, handle=handle)


class LinkedArray_ui32(LinkedArray):
    _M_array_type = ManagedArray_ui32
    _empty_ctor = lib_array.LArray_ctor_e_ui32
    _copy_ctor = lib_array.LArray_ctor_a_ui32
    _fill_ctor = lib_array.LArray_ctor_c_ui32
    _dtor = lib_array.LArray_dtor_ui32
    _get_arr_ptr = lib_array.LArray_get_arr_ptr_ui32
    _ipow2 = lib_array.LArray_ipow2_ui32

    # TODO: will all of these derived types conflict with eachother?
    for op in ["add", "iadd", "sub", "isub", "mul", "imul", "div", "idiv"]:
        _method = getattr(LinkedArray, "_" + op)
        _method.register(ui32_p, getattr(lib_array, "LArray_" + op + "_ui32"))
        _method.register(ui32, getattr(lib_array, "LArray_" + op + "_c_ui32"))

    def __init__(self, N, fill=None, handle=None, **kwargs):
        super().__init__(N=N, fill=fill, handle=handle)


class LinkedArray_ui64(LinkedArray):
    _M_array_type = ManagedArray_ui64
    _empty_ctor = lib_array.LArray_ctor_e_ui64
    _copy_ctor = lib_array.LArray_ctor_a_ui64
    _fill_ctor = lib_array.LArray_ctor_c_ui64
    _dtor = lib_array.LArray_dtor_ui64
    _get_arr_ptr = lib_array.LArray_get_arr_ptr_ui64
    _ipow2 = lib_array.LArray_ipow2_ui64

    # TODO: will all of these derived types conflict with eachother?
    for op in ["add", "iadd", "sub", "isub", "mul", "imul", "div", "idiv"]:
        _method = getattr(LinkedArray, "_" + op)
        _method.register(ui64_p, getattr(lib_array, "LArray_" + op + "_ui64"))
        _method.register(ui64, getattr(lib_array, "LArray_" + op + "_c_ui64"))

    def __init__(self, N, fill=None, handle=None, **kwargs):
        super().__init__(N=N, fill=fill, handle=handle)


class LinkedArray_f32(LinkedArray):
    _M_array_type = ManagedArray_f32
    _empty_ctor = lib_array.LArray_ctor_e_f32
    _copy_ctor = lib_array.LArray_ctor_a_f32
    _fill_ctor = lib_array.LArray_ctor_c_f32
    _dtor = lib_array.LArray_dtor_f32
    _get_arr_ptr = lib_array.LArray_get_arr_ptr_f32
    _ipow2 = lib_array.LArray_ipow2_f32

    # TODO: will all of these derived types conflict with eachother?
    for op in ["add", "iadd", "sub", "isub", "mul", "imul", "div", "idiv"]:
        _method = getattr(LinkedArray, "_" + op)
        _method.register(f32_p, getattr(lib_array, "LArray_" + op + "_f32"))
        _method.register(f32, getattr(lib_array, "LArray_" + op + "_c_f32"))

    def __init__(self, N, fill=None, handle=None, **kwargs):
        super().__init__(N=N, fill=fill, handle=handle)


class LinkedArray_f64(LinkedArray):
    _M_array_type = ManagedArray_f64
    _empty_ctor = lib_array.LArray_ctor_e_f64
    _copy_ctor = lib_array.LArray_ctor_a_f64
    _fill_ctor = lib_array.LArray_ctor_c_f64
    _dtor = lib_array.LArray_dtor_f64
    _get_arr_ptr = lib_array.LArray_get_arr_ptr_f64
    _ipow2 = lib_array.LArray_ipow2_f64

    # TODO: will all of these derived types conflict with eachother?
    for op in ["add", "iadd", "sub", "isub", "mul", "imul", "div", "idiv"]:
        _method = getattr(LinkedArray, "_" + op)
        _method.register(f64_p, getattr(lib_array, "LArray_" + op + "_f64"))
        _method.register(f64, getattr(lib_array, "LArray_" + op + "_c_f64"))

    def __init__(self, N, fill=None, handle=None, **kwargs):
        super().__init__(N=N, fill=fill, handle=handle)
