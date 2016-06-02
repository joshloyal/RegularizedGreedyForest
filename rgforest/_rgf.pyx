from cpython cimport Py_INCREF, PyObject

from libc.stdlib cimport malloc, free, realloc
from libc.string cimport memcpy
from libcpp cimport bool

import numpy as np
cimport numpy as np
np.import_array()

from rgforest.dataset cimport RGFMatrix
cimport rgforest._memory as mem


ctypedef np.npy_intp SIZE_t
ctypedef np.npy_float64 DOUBLE_t


cdef struct Node:
    SIZE_t left_child
    SIZE_t right_child
    SIZE_t feature
    DOUBLE_t threshold


# numpy struct dtype
NODE_DTYPE = np.dtype({
    'names': ['left_child', 'right_child', 'feature', 'threshold'],
    'formats': [np.intp, np.intp, np.intp, np.float64],
    'offsets': [
        <Py_ssize_t> &(<Node*> NULL).left_child,
        <Py_ssize_t> &(<Node*> NULL).right_child,
        <Py_ssize_t> &(<Node*> NULL).feature,
        <Py_ssize_t> &(<Node*> NULL).threshold
        ]
})


cdef Node _to_node(const AzTreeNode *tree_node) nogil:
    cdef Node node
    node.left_child = tree_node.le_nx
    node.right_child = tree_node.gt_nx
    node.feature = tree_node.fx
    node.threshold = tree_node.border_val
    return node


cdef class RGFTree:
    cdef public int node_count
    cdef Node* nodes
    def __cinit__(self):
        self.node_count = 0
        self.nodes = NULL

    def __dealloc__(self):
        if self.nodes is not NULL:
            free(self.nodes)

    def __reduce__(self):
        """Reduce re-implementaiton for pickling."""
        return (RGFTree, (), self.__getstate__())

    def __getstate__(self):
        d = {}
        d["node_count"] = self.node_count
        d["nodes"] = self._get_node_ndarray()
        return d

    def __setstate__(self, d):
        self.node_count = d["node_count"]

        if 'nodes' not in d:
            raise ValueError("You have loaded a RGFTree version which ",
                             "cannot be imported.")

        node_ndarray = d['nodes']
        self.nodes = <Node*> realloc(self.nodes, <SIZE_t> node_ndarray.shape[0] * sizeof(Node))
        nodes = memcpy(self.nodes, (<np.ndarray> node_ndarray).data,
                       <SIZE_t> node_ndarray.shape[0] * sizeof(Node))

    property children_left:
        def __get__(self):
            return self._get_node_ndarray()['left_child'][:self.node_count]

    property children_right:
        def __get__(self):
            return self._get_node_ndarray()['right_child'][:self.node_count]

    property feature:
        def __get__(self):
            return self._get_node_ndarray()['feature'][:self.node_count]

    property threshold:
        def __get__(self):
            return self._get_node_ndarray()['threshold'][:self.node_count]

    cdef copy_from(self, const AzTree *tree):
        cdef int node_idx
        self.node_count = tree.nodeNum()

        self.nodes = <Node*>realloc(self.nodes, sizeof(Node) * self.node_count)
        if self.nodes is NULL:
            raise MemoryError('Could not allocate Nodes')

        for node_idx in range(self.node_count):
            self.nodes[node_idx] = _to_node(tree.node(node_idx))

    cdef np.ndarray _get_node_ndarray(self):
        # create numpy array
        Py_INCREF(NODE_DTYPE)
        cdef Node[:] mv = <Node[:self.node_count]>self.nodes
        cdef np.ndarray arr = np.asarray(mv)
        Py_INCREF(self)

        arr.base = <PyObject*> self
        return arr


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

    def __iter__(self):
        cdef int n_trees = self.ensemble.size()
        cdef int tree_idx
        for tree_idx in range(n_trees):
            yield self._tree(tree_idx)

    def __len__(self):
        return self.ensemble.size()

    def __getitem__(self, tree_idx):
        if tree_idx >= self.ensemble.size():
            raise IndexError('tree index out of range')

        if tree_idx < 0:
            raise IndexError('negative indices not supported')

        return self._tree(tree_idx)

    def _tree(self, tree_idx):
        cdef RGFTree tree = RGFTree()
        tree.copy_from(self.ensemble.tree(tree_idx))
        return tree

    cdef AzTreeEnsemble *get_ensemble_ptr(self):
        return self.ensemble

    cpdef np.ndarray[double, ndim=1] predict(self, np.ndarray[double, ndim=2, mode='c'] X):
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
                np.ndarray[double, ndim=2, mode='c'] X,
                np.ndarray[double, ndim=1] y,
                np.ndarray[double, ndim=1] sample_weight):
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
