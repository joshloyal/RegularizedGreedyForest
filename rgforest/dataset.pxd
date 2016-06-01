cimport numpy as cnp

cdef extern from "AzDmat.hpp":
    cdef cppclass AzDvect:
        AzDvect()
        AzDvect(double *inp, int inp_num)
        int rowNum()
        double get(int row)
        double *point_u() nogil
        void set(AzDvect* inp,  double coeff=1) except +
        void set(const double* inp, int inp_num) except +

cdef extern from "AzDmat.hpp":
    cdef cppclass AzDmat:
        AzDmat(double* inp, int row_num, int col_num)
        double get(int row, int col)

cdef extern from "AzSmat.hpp":
    cdef cppclass AzSvect:
        AzSvect(double* data, int inp_row_num)
        int rowNum()
        double get(int row_no)
        double sum()

cdef extern from "AzDmat.hpp":
    cdef cppclass AzSmat:
        AzSmat()
        AzSmat(double* data, int inp_row_num, int inp_col_num)
        double get(int row_no, int col_no)
        double sum()
        void set(AzSmat *mv) except +

cdef extern from "AzSvFeatInfo.hpp":
    cdef cppclass AzSvFeatInfo:
        pass

cdef extern from "AzSvFeatInfoClone.hpp":
    cdef cppclass AzSvFeatInfoClone(AzSvFeatInfo):
        AzSvFeatInfoClone()
        void reset(AzSvFeatInfo *inp)

cdef extern from "AzSvDataS.hpp":
    cdef cppclass AzSvDataS:
        AzSvDataS(double* features, double* target, int n_rows, int n_cols) except +
        int featNum() except +
        int size() except +
        inline const AzSmat *feat() except +
        inline const AzDvect *targets() except +
        inline const AzSvFeatInfo *featInfo()


cdef AzDvect* _init_dvect_from_numpy(cnp.ndarray[double, ndim=1] arr)
cdef AzSmat* _init_smat_from_numpy(cnp.ndarray[double, ndim=2, mode='c'] arr)

ctypedef double double_t

cdef class RGFMatrix:
    cdef AzSvDataS* _data_ptr
    cpdef double_t sum_features(self)
    cpdef double_t feature_get(self, row_idx, col_idx)
    cpdef double_t target_get(self, row_idx)
    cdef const AzSmat *feat(self)
    cdef const AzDvect *targets(self)
    cdef const AzSvFeatInfo *featInfo(self)
