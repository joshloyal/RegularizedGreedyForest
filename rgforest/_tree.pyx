from cpython cimport Py_INCREF, PyObject
from libc.stdlib cimport malloc, free, realloc
from libc.string cimport memcpy

import numpy as np
cimport numpy as np

from .typedefs cimport SIZE_t, DOUBLE_t

# numpy struct dtype
NODE_DTYPE = np.dtype({
    'names': ['parent', 'left_child', 'right_child', 'feature', 'threshold', 'weight'],
    'formats': [np.intp, np.intp, np.intp, np.intp, np.float64, np.float64],
    'offsets': [
        <Py_ssize_t> &(<Node*> NULL).parent,
        <Py_ssize_t> &(<Node*> NULL).left_child,
        <Py_ssize_t> &(<Node*> NULL).right_child,
        <Py_ssize_t> &(<Node*> NULL).feature,
        <Py_ssize_t> &(<Node*> NULL).threshold,
        <Py_ssize_t> &(<Node*> NULL).weight,
        ]
})


cdef Node _to_node(const AzTreeNode *tree_node) nogil:
    cdef Node node
    node.parent = tree_node.parent_nx
    node.left_child = tree_node.le_nx
    node.right_child = tree_node.gt_nx
    node.feature = tree_node.fx
    node.threshold = tree_node.border_val
    node.weight = tree_node.weight
    return node


cdef class RGFTree:
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
        d["root"] = self.root
        d["n_outputs"] = self.n_outputs
        d["nodes"] = self._get_node_ndarray()
        return d

    def __setstate__(self, d):
        self.node_count = d["node_count"]
        self.root = d["root"]
        self.n_outputs = d["n_outputs"]

        if 'nodes' not in d:
            raise ValueError("You have loaded a RGFTree version which ",
                             "cannot be imported.")

        node_ndarray = d['nodes']
        self.nodes = <Node*> realloc(self.nodes, <SIZE_t> node_ndarray.shape[0] * sizeof(Node))
        memcpy(self.nodes, (<np.ndarray> node_ndarray).data,
               <SIZE_t> node_ndarray.shape[0] * sizeof(Node))

    property parent:
        def __get__(self):
            return self._get_node_ndarray()['parent'][:self.node_count]

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

    property weight:
        def __get__(self):
            return self._get_node_ndarray()['weight'][:self.node_count]

    property nodes:
        def __get__(self):
            return self._get_node_ndarray()

    cdef copy_from(self, const AzTree *tree):
        cdef int node_idx
        self.node_count = tree.nodeNum()
        self.n_outputs = tree.leafNum()
        self.root = tree.root()

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
