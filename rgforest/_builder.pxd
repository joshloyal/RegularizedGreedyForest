from libcpp cimport bool

import numpy as np
cimport numpy as np

from rgforest.dataset cimport AzSmat, AzDvect
from rgforest.dataset cimport AzSvFeatInfo, AzSvFeatInfoClone
from rgforest._ensemble cimport AzTreeEnsemble, RGFTreeEnsemble


cdef extern from "<iostream>" namespace "std":
    cdef cppclass ostream:
        pass
    ostream cout


cdef extern from  "AzOut.hpp":
    cdef cppclass AzOut:
        AzOut()
        AzOut(ostream *o_ptr)


cdef extern from "AzTE_ModelInfo.hpp":
    cdef cppclass AzTE_ModelInfo:
        AzTE_ModelInfo()


cdef extern from "AzTETrainer.hpp":
    cdef enum AzTETrainer_Ret:
        AzTETrainer_Ret_TestNow = 1
        AzTETrainer_Ret_Exit = 2

    cdef cppclass AzTETrainer_TestData:
        AzTETrainer_TestData(AzOut &out, AzSmat *m_test_x)


cdef extern from "AzRgforest.hpp":
    cdef cppclass AzRgforest:
        AzRgforest()
        void startup(AzOut &out,
                     char *param,
                     AzSmat *m_x,
                     AzDvect *v_y,
                     AzSvFeatInfo *featInfo,
                     AzDvect *v_data_weights,
                     AzTreeEnsemble *inps_ens) nogil except +

        AzTETrainer_Ret proceed_until() nogil except +
        void copy_to(AzTreeEnsemble *out_ens) except +
        void apply(AzTETrainer_TestData *td,
                   AzDvect *v_test_p,
                   AzTE_ModelInfo *info,
                   AzTreeEnsemble *out_ens) except +


cdef class RGFBuilder:
    cdef AzRgforest* forest
    cdef int max_leaf_nodes
    cdef double l2
    cdef bytes loss
    cdef int test_interval
    cdef bool verbose
    cdef bytes parameter_string
    cpdef build(self,
                RGFTreeEnsemble ensemble,
                np.ndarray[double, ndim=2, mode='c'] X,
                np.ndarray[double, ndim=1] y,
                np.ndarray[double, ndim=1] sample_weight)
