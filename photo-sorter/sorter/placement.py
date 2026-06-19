import ctypes
import ctypes.util
import os
import shutil
import sys

_libc = ctypes.CDLL(ctypes.util.find_library("System"), use_errno=True) if sys.platform == "darwin" else None
if _libc is not None:
    _libc.clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32]
    _libc.clonefile.restype = ctypes.c_int


def place_file(src: str, dst: str, mode: str = "clone") -> str:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if mode == "move":
        shutil.move(src, dst)
        return "moved"
    if mode == "clone" and _libc is not None and not os.path.exists(dst):
        if _libc.clonefile(os.fsencode(src), os.fsencode(dst), 0) == 0:
            return "cloned"
    shutil.copy2(src, dst)
    return "copied"
