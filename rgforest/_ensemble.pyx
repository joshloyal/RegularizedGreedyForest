import numpy as np
cimport numpy as np
np.import_array()


from rgforest._tree cimport RGFTree
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

    cpdef save(self, bytes file_name):
        self.ensemble.write(file_name)
