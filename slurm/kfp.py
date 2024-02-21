import sys
sys.path.append("..") # Adds higher directory to python modules path so we can access sparse_solve.py

import numpy as np
import scipy as sp
from scipy import sparse

from sparse_solver import sparse_solve

## we construct coefficient matrix and constant matrix
# def construct_A(dx,dy,dt,g,s,H,W):
#   A = np.eye(H*W) + (np.eye(H*W)*(g**2)*dt)/(dx**2) + (np.eye(H*W)*(g**2)*dt)/(dy**2)\
#                   - np.diag((0.5*np.ones(H*W)*(g**2)*dt)[1:],-1)/(dx**2) - np.diag((0.5*np.ones(H*W)*(g**2)*dt)[1:],1)/(dy**2) \
#                   - np.diag((0.5*np.ones(H*W)*(g**2)*dt)[:-1],1)/(dx**2) - np.diag((0.5*np.ones(H*W)*(g**2)*dt)[:-1],-1)/(dy**2)
#   return A

def get_sparse_A_block(dx,dy,dt,g,s,H,W, N):
  diag = (np.ones((N - 1, H*W)) * (1 + (g**2)[:, None]*dt[:, None] * (1/dx**2 + 1/(dy**2)))).ravel()
  
  diag_off = (-np.ones((N - 1, H*W)) * (g**2)[:, None]*dt[:, None])/(dy**2)
  diag_off[:-1, -1] = 0
  diag_off = diag_off.ravel()[:-1]
  
  t_diag = -(np.ones((N - 2, H*W))/(dt[1:, None])).ravel()

  return sparse.diags([diag, diag_off, diag_off, t_diag], [0, 1, -1, -H*W], format="csr")

def construct_B(dx, dy, dt, m_prev, f, g, s, i):
  df = np.diff(f, axis=0, prepend=f[0,0]).ravel()
  f = f.ravel()
  if i == 0:
    B = m_prev - (df*dt/dx + df*dt/dy) - (f*s*dt/(2*dx) + f*s*dt/(2*dy)) + (0.5*(g**2)*(s**2)*dt)
  else:
    B = - (df*dt/dx + df*dt/dy) - (f*s*dt/dx + f*s*dt/dy) + (0.5*(g**2)*(s**2)*dt)
  return B

def construct_P(M,N):
  P = np.zeros((M,N))
  for j in range(N):
    if j == 0:
      P[0,j] = 1
      P[2*j+1,j] = 1/2
    elif j > 0 and j < N-1:
      P[2*j,j] = 1/2
      P[2*j+1,j] = 1
      P[2*j+2,j] = 1/2
    elif j == N-1:
      P[-2,j] = 1/2
      P[-1,j] = 1
  P = np.kron(P,P)
  return P

def construct_R(P):
  return 0.25*P.T

def solve_pde(A,b,mode='dense'):
  if mode == 'dense':
    return sp.linalg.solve(A, b)
  if mode == 'sparse':
    return sparse_solve(A, b)
  if mode == "sp_sparse":
    return sparse.linalg.spsolve(A, b)

def gauss_seidel(A, b, x, N):
  iteration = 10
  lu_A = sparse.linalg.splu(A.tocsc())
  I = sparse.identity(N, format='csc')
  L = lu_A.L - I
  D = sparse.diags(A.diagonal(),format='csc')
  U = -(lu_A.U - D)
  
  lu = sparse.linalg.splu(D + L)
  for _ in range(iteration):
    x = lu.solve(U.dot(x) + b)
  return x

def jacobi(A, b, x, N):
  iteration = 10
  for _ in range(iteration):
    lu = sparse.linalg.splu(sparse.csc_matrix(A))
    diag_A = np.diag(A)
    I = sparse.identity(N, format='csr')
    L = lu.L.tocsr() - I
    D = sparse.diags(diag_A,format='csr')
    U = lu.U.tocsr() - D
    T = -(L + U)
    Dinv = sparse.diags(1/diag_A,format='csr')
    x = Dinv.dot(T.dot(x) + b)
  return x

def logsumexp(x):
  c = x.max()
  return c + np.log(np.sum(np.exp(x - c)))

def marginal_prob_std(t, sigma):
  """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

  Returns:
    The standard deviation.
  """
  return np.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

  Returns:
    The vector of diffusion coefficients.
  """
  return sigma**t