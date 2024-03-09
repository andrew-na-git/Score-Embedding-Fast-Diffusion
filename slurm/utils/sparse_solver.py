import ctypes
import numpy as np
import os
dir = os.path.dirname(os.path.abspath(__file__))
sparse_gaussian_elimination_folder = os.path.join(dir, "../../sparse_gaussian_elimination")
a_so_location = os.path.join(sparse_gaussian_elimination_folder, "a.so")

# create a.so if doesnt exists
if not os.path.isfile(a_so_location):
    os.system(f"make -C {sparse_gaussian_elimination_folder} a.so")

c_lib = ctypes.CDLL(a_so_location)
c_lib.solve_sparse.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int,
)

c_lib.solve_system.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.intc, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.intc, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int,
)

c_lib.init()

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

    result = np.zeros(n, dtype=np.double)
    c_lib.solve_sparse(A, b, result, n, num_order)

    return result
  
def solve_csr(A, b, order = "rcm"):
    assert (
        order == "rcm" or order == "min_degree" or order == "original"
    ), "Order must be either 'rcm', 'min_degree', or 'original'"
    
    n = A.shape[0]

    if order == "rcm":
        num_order = 1
    elif order == "min_degree":
        num_order = 2
    elif order == "original":
        num_order = 0

    result = np.zeros(n, dtype=np.double)

    c_lib.solve_system(A.data.astype(np.double), b.astype(np.double), A.indptr, A.indices, result, n, num_order)

    return result
