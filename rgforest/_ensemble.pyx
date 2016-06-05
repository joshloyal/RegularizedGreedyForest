from cython.operator cimport dereference as deref
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

import numpy as np
cimport numpy as np
np.import_array()


from rgforest._tree cimport RGFTree, Node
cimport rgforest._memory as mem


cdef class RGFTreeEnsemble:
    """wrap AzTreeEnsemble in an extension type for use in python (may need to also wrap AzTree)"""
    def __cinit__(self):
        self.ensemble = new AzTreeEnsemble()
        if self.ensemble is NULL:
            raise MemoryError('Unable to initialize tree ensemble.')

    def __dealloc__(self):
        if self.ensemble is not NULL:
            del self.ensemble

    def __reduce__(self):
        return (RGFTreeEnsemble, (), self.__getstate__())

    def __getstate__(self):
        d = {}
        d["n_trees"] = self.n_trees
        d["constant"] = self.constant
        d["org_dim"] = self.org_dim
        d["signature"] = self.signature
        d["configuration"] = self.configuration
        d["trees"] = [tree for tree in self]
        return d

    def __setstate__(self, d):
        self.n_trees = d["n_trees"]
        self.constant = d["constant"]
        self.org_dim = d["org_dim"]
        self.signature = d["signature"]
        self.configuration = d["configuration"]

        self._rebuild_ensemble(d["trees"])

    cdef _rebuild_ensemble(self, list trees):
        # create an array of AzTrees
        cdef AzTree **inp_trees = <AzTree**>malloc(sizeof(AzTree*) * self.n_trees)
        cdef AzTree *inp_tree
        cdef Node *nodes
        for tree_idx, tree in enumerate(trees):
            # we sadly need to allocate another node array...
            node_ndarray = tree.nodes
            nodes = <Node*>malloc(<SIZE_t> node_ndarray.shape[0] * sizeof(Node))
            if nodes is NULL:
                raise MemoryError('Could not allocate Nodes')
            memcpy(nodes, (<np.ndarray> node_ndarray).data,
                   <SIZE_t> node_ndarray.shape[0] * sizeof(Node))

            inp_tree = new AzTree(tree.root, tree.node_count, nodes)
            inp_trees[tree_idx] = inp_tree

        try:
            self.ensemble = new AzTreeEnsemble()
            self.ensemble.transfer_from(inp_trees,
                                        <int>self.n_trees,
                                        <double>self.constant,
                                        <int>self.org_dim,
                                        self.configuration,
                                        self.signature)
        finally:
            free(inp_trees)
            free(nodes)

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

    cdef end_training(self):
        self.n_trees = self.ensemble.size()
        self.org_dim = self.ensemble.orgdim()
        self.constant = self.ensemble.constant()
        self.signature = self.ensemble.signature()
        self.configuration = self.ensemble.configuration()
