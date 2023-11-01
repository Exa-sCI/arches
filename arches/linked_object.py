import pathlib
from abc import ABC, abstractmethod
from ctypes import CDLL, POINTER, c_double, c_float, c_int, c_long, c_uint, c_ulong

import numpy as np
from numpy.ctypeslib import ndpointer

run_folder = pathlib.Path(__file__).parent.resolve()
chunk_lib = CDLL(run_folder.joinpath("build/libchunk.so"))


class LinkedHandle(ABC):
    """Base class for objects managed in memory via C++ and handled via Python."""

    def __init__(self, *args, **kwargs):
        self.handle = self.constructor(*args, **kwargs)

    def __del__(self):
        self.destructor(self.handle)

    @abstractmethod
    def destructor(handle):
        pass

    @abstractmethod
    def constructor(self, *args, **kwargs):
        pass


class LinkedArray(ABC):
    """Base class for arrays owned by LinkedHandle."""

    def __init__(self, pointer, size, ctype, dtype):
        self._p = pointer
        self._dtype = dtype
        self._size = size
        self._ctype = ctype

    @property
    def p(self):
        return self._p

    # TODO: figure out type checking to make sure pointer aligns with dtype and ctype
    @property
    def dtype(self):
        return self._dtype

    @property
    def ctype(self):
        return self._ctype

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
            if k.step:
                self.set_strided_range(self.p, k.start, k.stop, k.step, v)
            else:
                self.set_range(self.p, k.start, k.stop, v)
        else:
            self.set_val(self.p, k, self.ctype(v))


## Register all the types
# i64 is the indexing type for all
all_types = (c_int, c_long, c_uint, c_ulong, c_float, c_double)
i32, i64, ui32, ui64, f32, f64 = all_types
i32_p, i64_p, ui32_p, ui64_p, f32_p, f64_p = (POINTER(t) for t in all_types)
i32_a, i64_a, ui32_a, ui64_a, f32_a, f64_a = (ndpointer(t, flags="C_CONTIGUOUS") for t in all_types)
type_dict = {
    i32: ("i32", i32_p, i32_a),
    i64: ("i64", i64_p, i64_a),
    ui32: ("ui32", ui32_p, ui32_a),
    ui64: ("ui64", ui64_p, ui64_a),
    f32: ("f32", f32_p, f32_a),
    f64: ("f32", f64_p, f64_a),
}

for k, v in type_dict.items():
    f_at = getattr(chunk_lib, "at_" + v[0])
    f_set = getattr(chunk_lib, "set_" + v[0])
    f_set_range = getattr(chunk_lib, "set_range_" + v[0])
    f_set_strided_range = getattr(chunk_lib, "set_strided_range_" + v[0])

    f_at.argtypes = [v[1], i64]
    f_set.argtypes = [v[1], i64, k]
    f_set_range.argtypes = [v[1], i64, i64, v[2]]
    f_set_strided_range.argtypes = [v[1], i64, i64, i64, v[2]]

    f_at.restype = k
    f_set.restype = None
    f_set_range.restype = None
    f_set_strided_range.restype = None


class LinkedArray_i32(LinkedArray):
    at = chunk_lib.at_i32
    set_val = chunk_lib.set_i32
    set_range = chunk_lib.set_range_i32
    set_strided_range = chunk_lib.set_strided_range_i32

    def __init__(self, pointer, size, data):
        super().__init__(pointer, size, ctype=i32, dtype=np.int32)
        self[slice(0, size)] = data


class LinkedArray_i64(LinkedArray):
    at = chunk_lib.at_i64
    set_val = chunk_lib.set_i64
    set_range = chunk_lib.set_range_i64
    set_strided_range = chunk_lib.set_strided_range_i64

    def __init__(self, pointer, size, data):
        super().__init__(pointer, size, ctype=i64, dtype=np.int64)
        self[slice(0, size)] = data


class LinkedArray_ui32(LinkedArray):
    at = chunk_lib.at_ui32
    set_val = chunk_lib.set_ui32
    set_range = chunk_lib.set_range_ui32
    set_strided_range = chunk_lib.set_strided_range_ui32

    def __init__(self, pointer, size, data):
        super().__init__(pointer, size, ctype=ui32, dtype=np.uint32)
        self[slice(0, size)] = data


class LinkedArray_f32(LinkedArray):
    at = chunk_lib.at_f32
    set_val = chunk_lib.set_f32
    set_range = chunk_lib.set_range_f32
    set_strided_range = chunk_lib.set_strided_range_f32

    def __init__(self, pointer, size, data):
        super().__init__(pointer, size, ctype=f32, dtype=np.float32)
        self[slice(0, size)] = data


class LinkedArray_f64(LinkedArray):
    at = chunk_lib.at_f64
    set_val = chunk_lib.set_f64
    set_range = chunk_lib.set_range_f64
    set_strided_range = chunk_lib.set_strided_range_f64

    def __init__(self, pointer, size, data):
        super().__init__(pointer, size, ctype=f64, dtype=np.float64)
        self[slice(0, size)] = data
