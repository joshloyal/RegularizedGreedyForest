import numpy as np
cimport numpy as np

from rgforest.dataset cimport AzSmat, AzDvect, AzSvFeatInfoClone
from rgforest._tree cimport AzTree


cdef extern from "AzTreeEnsemble.hpp":
    cdef cppclass AzTreeEnsemble:
       AzTreeEnsemble()
       void apply(AzSmat *m_data, AzDvect *v_pred) nogil except +
       void write(char *fn) nogil
       const AzTree *tree(int tx) nogil
       int size() nogil


cdef class RGFTreeEnsemble:
    cdef AzTreeEnsemble *ensemble
    cdef inline AzTreeEnsemble *get_ensemble_ptr(self):
        return self.ensemble
    cpdef np.ndarray[double, ndim=1] predict(self, np.ndarray[double, ndim=2, mode='c'] X)
    cpdef save(self, bytes file_name)
