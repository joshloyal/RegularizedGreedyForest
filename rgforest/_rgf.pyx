from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
import numpy as np
cimport numpy as cnp
from scipy.special import expit

from rgforest.dataset cimport RGFMatrix

cdef extern from "<math.h>" nogil:
    double exp(double x)

cnp.import_array()

cdef double sigmoid(double x):
    return 1 / (1 + exp(-x))


cdef class _finalizer:
    cdef void *_data
    def __dealloc__(self):
        if self._data is not NULL:
            free(self._data)


cdef void set_base(cnp.ndarray arr, void *carr):
    cdef _finalizer f = _finalizer()
    f._data = <void*>carr
    cnp.set_array_base(arr, f)


cdef double *copy_array(double *buf, int n_elems):
    cdef double *arr = <double*>malloc(sizeof(double) * n_elems)
    memcpy(arr, buf, sizeof(double) * n_elems)
    return arr

cdef cnp.ndarray make_ndarray(double *buf, int n_samples):
    cdef double[:] mv = <double[:n_samples]>buf
    cdef cnp.ndarray arr = np.asarray(mv)
    set_base(arr, buf)
    return arr


cdef class RegularizedGreedyForest:
    cdef AzRgforest* forest
    cdef AzOut* out
    cdef int verbose
    cdef bytes parameters
    def __cinit__(self, parameters, verbose=0):
        self.forest = new AzRgforest()
        if self.forest is NULL:
            raise MemoryError()

        self.parameters = parameters

        self.verbose = verbose
        if verbose:
            self.out = new AzOut(&cout)
        else:
            self.out = new AzOut()
        if self.out is NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self.forest is not NULL:
            del self.forest

        if self.out is not NULL:
            del self.out

    def fit(self, X, y):
        y = y * 2 - 1
        self.train(X, y)

    cpdef train(self, cnp.ndarray[double, ndim=2, mode='c'] X, cnp.ndarray[double, ndim=1] y):
        cdef RGFMatrix rgf_matrix = RGFMatrix(X, y)
        cdef char* param = self.parameters
        cdef int inp_num = X.shape[0]
        cdef AzDvect* v_y = new AzDvect()
        cdef AzSmat* m_x = new AzSmat()
        cdef AzTETrainer_Ret ret
        cdef AzSvFeatInfoClone* featInfo = new AzSvFeatInfoClone()

        # set-up input data
        v_y.set(<double*>y.data, inp_num)
        m_x.set(rgf_matrix.feat())
        featInfo.reset(rgf_matrix.featInfo())

        # training loop
        try:
            self.forest.startup(self.out[0], param, m_x, v_y, <AzSvFeatInfo*>featInfo, NULL, NULL)
            while True:
                ret = self.forest.proceed_until()
                if ret == AzTETrainer_Ret_Exit:
                    break
        finally:
            del m_x
            del v_y
            del rgf_matrix
            del featInfo

    cpdef predict_c(self, cnp.ndarray[double, ndim=2, mode='c'] X):
        cdef int row_num = X.shape[0], col_num = X.shape[1]
        cdef AzSmat* m_test_x = new AzSmat(<double*>X.data, row_num, col_num)
        cdef AzTreeEnsemble ens
        cdef AzDvect v_p
        cdef AzTE_ModelInfo info
        cdef AzTETrainer_TestData *td = new AzTETrainer_TestData(self.out[0], m_test_x)
        cdef int i
        cdef double *out
        try:

            self.forest.apply(td, &v_p, &info, &ens)
            out = copy_array(v_p.point_u(), row_num)
            return make_ndarray(out, row_num)
        finally:
            del m_test_x
            del td

    def predict(self, cnp.ndarray[double, ndim=2, mode='c'] X):
        cdef cnp.ndarray[double, ndim=1] y_pred
        y_pred = self.predict_c(X)
        return y_pred

