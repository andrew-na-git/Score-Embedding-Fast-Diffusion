import sys
sys.path.append("..") # Adds higher directory to python modules path so we can access sparse_solve.py

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.special import logsumexp, softmax, log_softmax

from sparse_solver import sparse_solve

## we construct coefficient matrix and constant matrix
def construct_A(dx,dy,dt,g,s,H,W):
  A = np.eye(H*W) + (np.eye(H*W)*(g**2)*dt)/(dx**2) + (np.eye(H*W)*(g**2)*dt)/(dy**2)\
                  - np.diag((0.5*np.ones(H*W)*(g**2)*dt)[1:],-1)/(dx**2) - np.diag((0.5*np.ones(H*W)*(g**2)*dt)[1:],1)/(dy**2) \
                  - np.diag((0.5*np.ones(H*W)*(g**2)*dt)[:-1],1)/(dx**2) - np.diag((0.5*np.ones(H*W)*(g**2)*dt)[:-1],-1)/(dy**2)
  return A

def construct_B(dx,dy,dt,m_prev,f,g,df,s,i):
  if i == 1:
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

def construct_R_block(R, R_block, i):
  if i == 1:
    R_block = sp.linalg.block_diag(R)
  else:
    R_block = sp.linalg.block_diag(R_block, R)
  return R_block

def construct_P_block(P, P_block, i):
  if i == 1:
    P_block = sp.linalg.block_diag(P)
  else:
    P_block = sp.linalg.block_diag(P_block, P)
  return P_block

def gauss_seidel(A, b, x, N):
  iteration = 10
  for _ in range(iteration):
    lu = sparse.linalg.splu(sparse.csc_matrix(A))
    I = sparse.identity(N, format='csr')
    L = lu.L.tocsr() - I
    D = sparse.diags(np.diag(A),format='csr')
    U = -(lu.U.tocsr() - D)
    x = sparse.linalg.spsolve((D + L), U.dot(x) + b)
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