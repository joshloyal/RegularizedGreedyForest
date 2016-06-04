import numpy as np
cimport numpy as np

from .typedefs cimport SIZE_t, DOUBLE_t

cdef extern from "AzTreeNodes.hpp":
    cdef cppclass AzTreeNode:
        int fx
        int le_nx
        int gt_nx
        double border_val

    cdef cppclass AzTreeNodes:
        pass


cdef extern from "AzTree.hpp":
    cdef cppclass AzTree:
        AzTree()
        void copy_from(const AzTreeNodes *tree_nodes)
        const AzTreeNode *node(int nx)
        int nodeNum()
        int leafNum()


cdef struct Node:
    SIZE_t left_child
    SIZE_t right_child
    SIZE_t feature
    DOUBLE_t threshold


cdef class RGFTree:
    cdef public SIZE_t node_count
    cdef public SIZE_t n_outputs
    cdef Node* nodes

    # methods
    cdef copy_from(self, const AzTree *tree)
    cdef np.ndarray _get_node_ndarray(self)


cdef Node _to_node(const AzTreeNode *tree_node) nogil
