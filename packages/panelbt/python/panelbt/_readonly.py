"""Temporarily protect borrowed views, including overlapping weight calls.

NumPy writeability is per array object: pre-existing aliases are not locked.
Callers must still refrain from mutating any input storage during a run.
"""

from contextlib import contextmanager
from threading import RLock

import numpy as np

_lock = RLock()
_borrowed = {}


def contiguous(value, dtype):
    """Make a borrowable array without copying ordinary contiguous inputs."""
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.flags.writeable:
        base = array.base
        while isinstance(base, np.ndarray):
            if not base.flags.writeable:
                # A view created before its owner became read-only can still
                # be writable. NumPy would refuse to restore that flag later;
                # isolate it instead of ever changing the owner's original flag.
                return array.copy()
            base = base.base
    return array


def _restore_ready():
    # A writable view cannot be restored before its borrowed owner. Retain a
    # finished view until that owner is released, even across separate calls.
    pending = True
    while pending:
        pending = False
        for key, (array, writable, count) in list(_borrowed.items()):
            if count:
                continue
            base = array.base
            blocked = False
            while isinstance(base, np.ndarray):
                if id(base) in _borrowed and not base.flags.writeable:
                    blocked = True
                    break
                base = base.base
            if writable and blocked:
                continue
            if writable:
                array.flags.writeable = True
            del _borrowed[key]
            pending = True


@contextmanager
def readonly(arrays):
    """Restore each view's original flag after its last concurrent borrower."""
    unique = {id(a): a for a in arrays}
    acquired = []
    try:
        with _lock:
            for key, array in unique.items():
                if key not in _borrowed:
                    writable = array.flags.writeable
                    array.flags.writeable = False
                    _borrowed[key] = [array, writable, 0]
                _borrowed[key][2] += 1
                acquired.append(key)
        yield
    finally:
        with _lock:
            for key in reversed(acquired):
                _borrowed[key][2] -= 1
            _restore_ready()
