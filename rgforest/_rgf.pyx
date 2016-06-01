from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libcpp cimport bool
import numpy as np
cimport numpy as cnp

from rgforest.dataset cimport RGFMatrix
cimport rgforest._memory as mem

cnp.import_array()


PARAM_FMT = ('reg_L2={l2},loss={loss},test_interval={test_interval},'
             'max_leaf_forest={max_leaf_forest}')


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

    cpdef cnp.ndarray[double, ndim=1] predict(self, cnp.ndarray[double, ndim=2, mode='c'] X):
        cdef int row_num = X.shape[0], col_num = X.shape[1]
        cdef AzDvect v_p
        cdef AzSmat *m_test_x = new AzSmat(<double*>X.data, row_num, col_num)

        try:
            with nogil:
                self.ensemble.apply(m_test_x, &v_p)
                out = mem.copy_array(v_p.point_u(), row_num)
            return mem.make_1d_ndarray(out, row_num)
        finally:
            del m_test_x

    cpdef save(self, bytes file_name):
        self.ensemble.write(file_name)


cdef class RGFTree:
    """wrap AzTree"""
    pass


cdef class RGFBuilder:
    cdef AzRgforest* forest
    cdef int max_leaf_nodes
    cdef double l2
    cdef bytes loss
    cdef int test_interval
    cdef bool verbose
    cdef bytes parameter_string
    def __cinit__(self, max_leaf_nodes=500, l2=1, loss='LS',
                  test_interval=100, verbose=False):
        self.max_leaf_nodes = max_leaf_nodes
        self.l2 = l2
        self.loss = loss
        self.test_interval = test_interval
        self.verbose = verbose

        self.forest = new AzRgforest()
        if self.forest is NULL:
            raise MemoryError()

    def __init__(self, max_leaf_nodes=500, l2=1, loss='LS',
                 test_interval=100, verbose=False):
        self.parameter_string = self.get_param_str()

    def get_param_str(self):
        parameters = PARAM_FMT.format(
            l2=self.l2,
            loss=self.loss,
            test_interval=self.test_interval,
            max_leaf_forest=self.max_leaf_nodes)
        if self.verbose:
            parameters += ',Verbose'
        return parameters

    def __dealloc__(self):
        if self.forest is not NULL:
            del self.forest

    ### figure this one out for pickling
    #def __reduce__(self):
    #    pass

    #def __getstate__(self):
    #    pass

    #def __setstate__(self, d):
    #    pass


    cpdef build(self,
                RGFTreeEnsemble ensemble,
                cnp.ndarray[double, ndim=2, mode='c'] X,
                cnp.ndarray[double, ndim=1] y,
                cnp.ndarray[double, ndim=1] sample_weight):
        cdef RGFMatrix rgf_matrix = RGFMatrix(X, y)
        cdef AzRgforest *forest  = new AzRgforest()
        cdef char* param = self.parameter_string
        cdef int inp_num = X.shape[0]
        cdef AzDvect* v_y = new AzDvect()
        cdef AzDvect* v_fixed_dw = new AzDvect()
        cdef AzSmat* m_x = new AzSmat()
        cdef AzTETrainer_Ret ret
        cdef AzSvFeatInfoClone* featInfo = new AzSvFeatInfoClone()
        cdef AzOut *logger
        cdef AzTreeEnsemble *ens = ensemble.get_ensemble_ptr()

        if self.verbose:
            logger = new AzOut(&cout)
        else:
            logger = new AzOut()

        # set-up input data
        v_y.set(<double*>y.data, inp_num)

        if sample_weight is not None:
            v_fixed_dw.set(<double*>sample_weight.data, inp_num)
        else:
            v_fixed_dw = NULL

        m_x.set(rgf_matrix.feat())
        featInfo.reset(rgf_matrix.featInfo())

        # training loop
        try:
            with nogil:
                forest.startup(
                    logger[0],
                    param,
                    m_x,
                    v_y,
                    <AzSvFeatInfo*>featInfo,
                    v_fixed_dw,
                    ens)  # this destroys the previous ensemble
                while True:
                    ret = forest.proceed_until()  # proceed until test_interval
                    if ret == AzTETrainer_Ret_Exit:
                        break
            # (maybe the want to preserve the old one?)
            forest.copy_to(ens)
        finally:
            del rgf_matrix
            del forest
            del logger
            del v_y
            del v_fixed_dw
            del m_x
            del featInfo
