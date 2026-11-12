"""面板代次写入：支持 APFS 写时复制，保持旧读者映射有效。"""

import ctypes
import io
import os
import sys

import numpy as np


def clone_append(previous, target, shape):
    """仅列不变且 NPY 头等长时克隆；其他平台返回 None，由调用方复制。"""
    if sys.platform != "darwin":
        return None
    original = np.load(previous, mmap_mode="r")
    if original.shape[1] != shape[1] or original.shape[0] > shape[0]:
        return None
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header,
        dict(
            descr=np.lib.format.dtype_to_descr(original.dtype),
            fortran_order=False,
            shape=shape,
        ),
    )
    encoded = header.getvalue()
    if len(encoded) != original.offset:
        return None
    library = ctypes.CDLL(None, use_errno=True)
    clone = getattr(library, "clonefile", None)
    if clone is None:
        return None
    clone.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    clone.restype = ctypes.c_int
    if clone(os.fsencode(previous), os.fsencode(target), 0) != 0:
        return None
    with target.open("r+b") as stream:
        stream.write(encoded)
        stream.truncate(len(encoded) + int(np.prod(shape)) * original.dtype.itemsize)
        stream.flush()
        os.fsync(stream.fileno())
    return np.load(target, mmap_mode="r+")
