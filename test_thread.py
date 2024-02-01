import threading
import numpy as np
from sparse_solver import sparse_solve

def generate_rand():
  n = 5 # size of invertible matrix I wish to generate
  m = np.random.rand(n, n)
  mx = np.sum(np.abs(m), axis=1)
  np.fill_diagonal(m, mx)
  
  return m, np.random.rand(n)


def verify(A, b, x):
  x_real = np.linalg.solve(A, b)
  
  return np.all(np.around(x, 3) == np.around(x_real, 3))


def target():
  for _ in range(10000):
    A, b = generate_rand()
    x_sparse = sparse_solve(A, b)
    v = verify(A, b, x_sparse)
    if v == False:
      print(v)
      
      
num_threads = 10

threads = []

for _ in range(num_threads):
  threads.append(threading.Thread(target=target))
  threads[-1].start()
  
for thread in threads:
  thread.join()