import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef class _finalizer:
    cdef void *_data

cdef void set_base(cnp.ndarray np_array, void *c_array)
cdef double *copy_array(double *array, int n_elements) nogil
cdef cnp.ndarray[double, ndim=1] make_1d_ndarray(double *c_array, int n_elements)
