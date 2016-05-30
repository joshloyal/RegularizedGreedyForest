from libcpp.memory cimport shared_ptr, make_shared
import numpy as np
cimport numpy as cnp
cnp.import_array()

cdef class RGFMatrix:
    def __cinit__(self, cnp.ndarray[double_t, ndim=2, mode='c'] X, cnp.ndarray[double_t, ndim=1] y):
        cdef int n_rows = X.shape[0]
        cdef int n_cols = X.shape[1]
        self._data_ptr = new AzSvDataS(<double_t*>X.data, <double_t*>y.data, n_rows, n_cols)

        if self._data_ptr is NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self._data_ptr is not NULL:
            del self._data_ptr

    cpdef double_t sum_features(self):
        return self._data_ptr.feat().sum()

    cpdef double_t feature_get(self, row_idx, col_idx):
        return self._data_ptr.feat().get(col_idx, row_idx)

    cpdef double_t target_get(self, row_idx):
        return self._data_ptr.targets().get(row_idx)

    cdef const AzSmat *feat(self):
        return self._data_ptr.feat()

    cdef const AzDvect *targets(self):
        return self._data_ptr.targets()

    cdef const AzSvFeatInfo *featInfo(self):
        return self._data_ptr.featInfo()

    property n_features:
        def __get__(self):
            return self._data_ptr.featNum()

    property n_samples:
        def __get__(self):
            return self._data_ptr.size()

cdef AzDvect* _init_dvect_from_numpy(cnp.ndarray[double, ndim=1] arr):
    cdef AzDvect *vect
    cdef int inp_num = arr.shape[0]

    # this does a memory copy
    vect = new AzDvect(<double*>arr.data, inp_num)

    return vect

def dataset(cnp.ndarray[double, ndim=2, mode='c'] X, cnp.ndarray[double, ndim=1] y):
    cdef AzSvDataS* data
    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]
    data = new AzSvDataS(<double*>X.data, <double*>y.data, n_rows, n_cols)
    try:
        print data.featNum()
        print data.size()
    finally:
        del data

def _init_mat_from_numpy(cnp.ndarray[double, ndim=2, mode='c'] arr):
    cdef AzDmat *mat
    cdef int row_num = arr.shape[0], col_num = arr.shape[1]
    cdef int i, j
    mat = new AzDmat(<double*>arr.data, row_num, col_num)
    try:
        for i in range(row_num):
            for j in range(col_num):
                print mat.get(i, j)
    finally:
        del mat

def _init_svec_from_numpy(cnp.ndarray[double, ndim=1] arr):
    cdef:
        AzSvect *vect
        int row_num = arr.shape[0]
        int i
    vect = new AzSvect(<double*>arr.data, row_num)
    try:
        for i in range(row_num):
            print vect.get(i)
        print vect.sum()
    finally:
        del vect

cdef AzSmat* _init_smat_from_numpy(cnp.ndarray[double, ndim=2, mode='c'] arr):
    # use a shared_ptr in practice
    cdef AzSmat* mat
    cdef int row_num = arr.shape[0], col_num = arr.shape[1]
    mat = new AzSmat(<double*>arr.data, row_num, col_num)
    return mat

