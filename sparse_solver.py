import ctypes
import numpy as np


def sparse_solve(A, b, order="rcm"):
    assert (
        order == "rcm" or order == "min_degree" or order == "original"
    ), "Order musst be either 'rcm', 'min_degree', or 'original'"
    n = len(A)

    if order == "rcm":
        num_order = 1
    elif order == "min_degree":
        num_order = 2
    elif order == "original":
        num_order = 0

    c_lib = ctypes.CDLL("./sparse_gaussian_elimination/a.so")
    c_lib.solve_sparse.argtypes = (
        np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int,
    )

    c_lib.solve_sparse.restype = ctypes.POINTER(ctypes.c_double)

    ans = c_lib.solve_sparse(A, b, n, num_order)
    sparse_ans = np.zeros(n)
    for i in range(n):
        sparse_ans[i] = ans[i]

    return sparse_ans
