from dataset cimport AzSmat, AzDvect, AzSvFeatInfoClone

cdef extern from "AzTETselector.hpp":
    cdef cppclass AzTETselector:
        AzTETselector()

cdef extern from "AzTE_ModelInfo.hpp":
    cdef cppclass AzTE_ModelInfo:
        AzTE_ModelInfo()

cdef extern from "AzTreeEnsemble.hpp":
    cdef cppclass AzTreeEnsemble:
       AzTreeEnsemble()
       void write(char *fn)

cdef extern from "AzTETrainer.hpp":
    cdef enum AzTETrainer_Ret:
        AzTETrainer_Ret_TestNow = 1
        AzTETrainer_Ret_Exit = 2

    cdef cppclass AzTETrainer_TestData:
        AzTETrainer_TestData(AzOut &out, AzSmat *m_test_x)

cdef extern from "AzRgfTrainerSel.hpp":
    cdef cppclass AzRgfTrainerSel(AzTETselector):
        AzRgfTrainerSel()

cdef extern from "AzTET_Eval.hpp":
    cdef cppclass AzTET_Eval:
        AzTET_Eval()

cdef extern from "AzTET_Eval_Dflt.hpp":
    cdef cppclass AzTET_Eval_Dflt(AzTET_Eval):
        AzTET_Eval_Dflt()

cdef extern from "AzTETmain.hpp":
    cdef cppclass AzTETmain:
        AzTETmain(AzTETselector *inp_alg_sel, AzTET_Eval *inp_eval)
        void train(char *argv[], int argc) except +

cdef extern from "<iostream>" namespace "std":
    cdef cppclass ostream:
        pass
    ostream cout

cdef extern from  "AzOut.hpp":
    cdef cppclass AzOut:
        AzOut()
        AzOut(ostream *o_ptr)

cdef extern from "AzSvFeatInfo.hpp":
    cdef cppclass AzSvFeatInfo:
        pass

cdef extern from "AzTreeEnsemble.hpp":
    cdef cppclass AzTreeEnsemble:
        void write(char *fn)

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
