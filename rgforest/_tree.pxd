import numpy as np
cimport numpy as np

from .typedefs cimport SIZE_t, DOUBLE_t

cdef extern from "AzTreeNodes.hpp":
    cdef cppclass AzTreeNode:
        int parent_nx
        int le_nx
        int gt_nx
        int fx
        double border_val
        double weight

    cdef cppclass AzTreeNodes:
        pass


cdef extern from "AzTree.hpp":
    cdef struct Node:
        int parent
        int left_child
        int right_child
        int feature
        double threshold
        double weight

    cdef cppclass AzTree:
        AzTree()
        AzTree(int root_nx, int nodes_used, Node *inp_nodes)
        void copy_from(const AzTreeNodes *tree_nodes)
        const AzTreeNode *node(int nx)
        int nodeNum()
        int leafNum()
        int root()


#cdef struct Node:
#    SIZE_t parent
#    SIZE_t left_child
#    SIZE_t right_child
#    SIZE_t feature
#    DOUBLE_t threshold
#    DOUBLE_t weight


cdef class RGFTree:
    cdef public SIZE_t node_count  # nodes_used
    cdef public SIZE_t n_outputs
    cdef public SIZE_t root
    cdef Node* nodes

    # methods
    cdef copy_from(self, const AzTree *tree)
    #cdef AzTree *to_aztree(self)
    cdef np.ndarray _get_node_ndarray(self)
    cdef inline Node *get_nodes(self):
        return self.nodes


cdef Node _to_node(const AzTreeNode *tree_node) nogil
