import numpy as np
cimport numpy as np

from rgforest.dataset cimport AzSmat, AzDvect, AzSvFeatInfoClone
from rgforest._tree cimport AzTree
from rgforest.typedefs cimport SIZE_t, DOUBLE_t

cdef extern from "AzTreeEnsemble.hpp":
    cdef cppclass AzTreeEnsemble:
       AzTreeEnsemble()
       void apply(AzSmat *m_data, AzDvect *v_pred) nogil except +
       void write(char *fn) nogil
       const AzTree *tree(int tx) nogil
       int size() nogil
       double constant()
       int orgdim()
       char *signature()
       char *configuration()
       void transfer_from(AzTree *inp_tree[],
                          int inp_tree_num,
                          double const_val,
                          int orgdim,
                          char *config,
                          char *sign)


cdef class RGFTreeEnsemble:
    cdef AzTreeEnsemble *ensemble

    cdef public SIZE_t n_trees
    cdef DOUBLE_t constant
    cdef SIZE_t org_dim
    cdef bytes signature
    cdef bytes configuration

    # methods
    cdef inline AzTreeEnsemble *get_ensemble_ptr(self):
        return self.ensemble
    cdef end_training(self)
    cdef _rebuild_ensemble(self, list trees)
    cpdef np.ndarray[double, ndim=1] predict(self, np.ndarray[double, ndim=2, mode='c'] X)
