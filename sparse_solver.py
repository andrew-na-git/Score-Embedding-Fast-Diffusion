import ctypes
import numpy as np

import sys

sys.path.insert(0, "/home/w56gao/src/sparse/")


def sparse_solve(A, b):
    n = len(A)

    c_lib = ctypes.CDLL("./sparse_gaussian_elimination/a.so")
    c_lib.solve_sparse.argtypes = (
        np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
    )

    c_lib.solve_sparse.restype = ctypes.POINTER(ctypes.c_double)

    ans = c_lib.solve_sparse(A, b, n)
    sparse_ans = np.zeros(n)
    for i in range(n):
        sparse_ans[i] = ans[i]

    return sparse_ans
