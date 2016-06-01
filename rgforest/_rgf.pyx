from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libcpp cimport bool
import numpy as np
cimport numpy as cnp

from rgforest.dataset cimport RGFMatrix
cimport rgforest._memory as mem

cnp.import_array()

cdef class RGFTreeEnsemble:
    """wrap AzTreeEnsemble in an extension type for use in python (may need to also wrap AzTree)"""
    cdef AzTreeEnsemble *ensemble

    def __cinit__(self):
        self.ensemble = new AzTreeEnsemble()
        if self.ensemble is NULL:
            raise MemoryError('Unable to initialize tree ensemble.')

    def __dealloc__(self):
        if self.ensemble is not NULL:
            del self.ensemble

    cdef AzTreeEnsemble *get_ensemble_ptr(self):
        return self.ensemble


cdef class RGFTree:
    """wrap AzTree"""
    pass


# also need a warm start version
cdef RGFTreeEnsemble train_rgf(cnp.ndarray[double, ndim=2, mode='c'] X,
                               cnp.ndarray[double, ndim=1] y,
                               cnp.ndarray[double, ndim=1] sample_weight,
                               bytes parameters,
                               bool verbose=False):
    cdef RGFMatrix rgf_matrix = RGFMatrix(X, y)
    cdef AzRgforest *forest  = new AzRgforest()
    cdef char* param = parameters
    cdef int inp_num = X.shape[0]
    cdef AzDvect* v_y = new AzDvect()
    cdef AzDvect* v_fixed_dw = new AzDvect()
    cdef AzSmat* m_x = new AzSmat()
    cdef AzTETrainer_Ret ret
    cdef AzSvFeatInfoClone* featInfo = new AzSvFeatInfoClone()
    #cdef AzTreeEnsemble *ensemble = new AzTreeEnsemble()
    cdef RGFTreeEnsemble ensemble = RGFTreeEnsemble()
    cdef AzOut *out

    if verbose:
        out = new AzOut(&cout)
    else:
        out = new AzOut()

    # set-up input data
    v_y.set(<double*>y.data, inp_num)
    v_fixed_dw.set(<double*>sample_weight.data, inp_num)
    m_x.set(rgf_matrix.feat())
    featInfo.reset(rgf_matrix.featInfo())

    # training loop
    try:
        # for warm start need to pass prev_ensemble to NULL
        with nogil:
            forest.startup(
                out[0],
                param,
                m_x,
                v_y,
                <AzSvFeatInfo*>featInfo,
                v_fixed_dw,
                NULL)
            while True:
                ret = forest.proceed_until()  # proceed until test_interval
                if ret == AzTETrainer_Ret_Exit:
                    break
        forest.copy_to(ensemble.get_ensemble_ptr())
    finally:
        del rgf_matrix
        del forest
        del out
        del v_y
        del v_fixed_dw
        del m_x
        del featInfo

    return ensemble


cdef class RegularizedGreedyForest:
    cdef AzRgforest* forest
    cdef AzOut* out
    cdef bool verbose
    cdef bytes parameters
    def __cinit__(self, parameters, verbose=False):
        self.forest = new AzRgforest()
        if self.forest is NULL:
            raise MemoryError()

        self.parameters = parameters

        self.verbose = verbose
        if self.verbose:
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

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones_like(y)
        self.train(X, y, sample_weight)

    cpdef save_model(self, bytes file_name):
        cdef const char* output_fn = 'test_model'
        cdef AzTreeEnsemble ens
        self.forest.copy_to(&ens)
        ens.write(file_name)

    cdef void train(self, cnp.ndarray[double, ndim=2, mode='c'] X, cnp.ndarray[double, ndim=1] y, cnp.ndarray[double, ndim=1] sample_weight):
        cdef RGFMatrix rgf_matrix = RGFMatrix(X, y)
        cdef char* param = self.parameters
        cdef int inp_num = X.shape[0]
        cdef AzDvect* v_y = new AzDvect()
        cdef AzDvect* v_fixed_dw = new AzDvect()
        cdef AzSmat* m_x = new AzSmat()
        cdef AzTETrainer_Ret ret
        cdef AzSvFeatInfoClone* featInfo = new AzSvFeatInfoClone()

        # set-up input data
        v_y.set(<double*>y.data, inp_num)
        v_fixed_dw.set(<double*>sample_weight.data, inp_num)
        m_x.set(rgf_matrix.feat())
        featInfo.reset(rgf_matrix.featInfo())

        # training loop
        try:
            # for warm start need to pass prev_ensemble to NULL
            with nogil:
                self.forest.startup(self.out[0], param, m_x, v_y, <AzSvFeatInfo*>featInfo, v_fixed_dw, NULL)
                while True:
                    ret = self.forest.proceed_until()  # proceed until test_interval
                    if ret == AzTETrainer_Ret_Exit:
                        break
        finally:
            del m_x
            del v_y
            del v_fixed_dw
            del rgf_matrix
            del featInfo

    cdef cnp.ndarray[double, ndim=1] predict_c(self, cnp.ndarray[double, ndim=2, mode='c'] X):
        cdef int row_num = X.shape[0], col_num = X.shape[1]
        cdef AzSmat* m_test_x = new AzSmat(<double*>X.data, row_num, col_num)
        cdef AzTreeEnsemble ens
        cdef AzDvect v_p
        cdef AzTE_ModelInfo info
        cdef AzTETrainer_TestData *td = new AzTETrainer_TestData(self.out[0], m_test_x)
        cdef double *out
        try:
            self.forest.apply(td, &v_p, &info, &ens)
            out = mem.copy_array(v_p.point_u(), row_num)
            return mem.make_1d_ndarray(out, row_num)
        finally:
            del m_test_x
            del td

    def predict(self, cnp.ndarray[double, ndim=2, mode='c'] X):
        cdef cnp.ndarray[double, ndim=1] y_pred
        y_pred = self.predict_c(X)
        return y_pred

