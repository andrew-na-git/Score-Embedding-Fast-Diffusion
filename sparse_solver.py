import ctypes
import numpy as np
import os
dir = os.path.dirname(os.path.abspath(__file__))

def sparse_solve(A, b, order="rcm"):
    assert (
        order == "rcm" or order == "min_degree" or order == "original"
    ), "Order must be either 'rcm', 'min_degree', or 'original'"
    n = len(A)

    if order == "rcm":
        num_order = 1
    elif order == "min_degree":
        num_order = 2
    elif order == "original":
        num_order = 0

    c_lib = ctypes.CDLL(os.path.join(dir, "sparse_gaussian_elimination/a.so"))
    c_lib.solve_sparse.argtypes = (
        np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int,
    )

    result = np.zeros(n, dtype=np.double)
    c_lib.solve_sparse(A, b, result, n, num_order)

    return result
