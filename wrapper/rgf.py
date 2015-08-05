import os
import ctypes
from shutil import copyfile
import tempfile

def load_rgflib():
    dll_path = '../rgf1.2/bin/rgf.so'
    with tempfile.NamedTemporaryFile() as fd:
        fd.file.close()
        os.unlink(fd.name)
        copyfile(dll_path, fd.name)
        lib = ctypes.cdll.LoadLibrary(fd.name)

    return lib
