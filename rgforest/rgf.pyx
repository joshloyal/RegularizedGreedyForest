# cython: c_string_type=unicode, c_string_encoding=utf8

# argv needs to a vector of
# ['train', 'train_x_fn', 'train_y_fn', 'model_fn_prefix', 'reg_L2', 'algorithm', 'loss', 'test_interval', 'max_leaf_forest', 'verbose']

# eventually wrap AZTETmain in a cdef class.

cdef class RegularizedGreedyForest:
    cdef:
        AzTETmain *_rgf_ptr
        AzTET_Eval_Dflt _eval
        AzRgfTrainerSel _alg_sel

    def __cinit__(self):
        self._rgf_ptr = new AzTETmain(&self._alg_sel, &self._eval)

    def __dealloc__(self):
        if self._rgf_ptr != NULL:
            del self._rgf_ptr

    def train(self):
        cdef const char *test[3]
        test[0] = '../bin/rgf'
        test[1] = 'train'
        test[2] = 'train_x_fn=./test/sample/train.data.x,train_y_fn=./test/sample/train.data.y,model_fn_prefix=./test/output/sample.model,reg_L2=1,algorithm=RGF,loss=LS,test_interval=100,max_leaf_forest=500,Verbose'
        self._rgf_ptr.train(test, 3)

#def train():
#    cdef:
#        AzRgfTrainerSel alg_sel
#        AzTET_Eval_Dflt eval_
#        AzTETmain *driver = new AzTETmain(&alg_sel, &eval_)
#
#    try:
#
#    finally:
#        del driver
