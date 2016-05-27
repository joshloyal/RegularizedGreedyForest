cdef extern from "AzTETselector.hpp":
    cdef cppclass AzTETselector:
        AzTETselector()

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
