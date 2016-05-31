from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
import numpy as np
cimport numpy as cnp


cnp.import_array()


cdef class _finalizer:
    """Object whose sole purpose is to de-allocate the memory
    of the array of data it is pointing to. Used by `set_base`
    to properly clean up numpy arrays created by C arrays.
    """
    def __dealloc__(self):
        if self._data is not NULL:
            free(self._data)


cdef void set_base(cnp.ndarray np_array, void *c_array):
    """set the base of a newly allocated numpy array to
    the underlying C array, so that it is properly cleaned
    up by python (handled by the finalizer class).
    """
    cdef _finalizer f = _finalizer()
    f._data = <void*>c_array
    cnp.set_array_base(np_array, f)


cdef double *copy_array(double *array, int n_elements) nogil:
    """creates a new array (`array_copy`) and copies elements of
    `array` into it.
    """
    cdef double *array_copy = <double*>malloc(sizeof(double) * n_elements)
    memcpy(array_copy, array, sizeof(double) * n_elements)
    return array_copy

cdef cnp.ndarray[double, ndim=1] make_1d_ndarray(double *c_array, int n_elements):
    """Creates a numpy ndarray from a C array that will clean up after itself
    during destruction.
    """
    cdef double[:] mv = <double[:n_elements]>c_array
    cdef cnp.ndarray[double, ndim=1] np_array = np.asarray(mv)
    set_base(np_array, c_array)
    return np_array
