import sys
sys.path.append("..") # Adds higher directory to python modules path so we can access sparse_solve.py

import numpy as np
import scipy as sp

from sparse_solver import sparse_solve

## we construct coefficient matrix and constant matrix
def construct_A(dx,dy,dt,f,g,s,H,W):
  A = np.eye(H*W)/dt + np.diag((f - 0.5*((g**2)*s)).ravel())/dx + np.diag((f - 0.5*((g**2)*s)).ravel())/dy \
                  + np.diag((-f + 0.5*((g**2)*s)).ravel()[1:],1)/dx + np.diag((-f + 0.5*((g**2)*s)).ravel()[1:],-1)/dy
  return A

def construct_B(dx,dy,dt,m_prev,f,g,s):
  B = m_prev - (np.diff(f, axis=0, prepend=f[0,0]).ravel()*dt/dx + np.diff(f, axis=-1, prepend=f[0,0]).ravel()*dt/dy \
               - 0.5*(g**2)*(np.diff(s, axis=0, prepend = s[0,0]).ravel()*dt/dx + np.diff(s, axis=-1, prepend = s[0,0]).ravel()*dt/dy))
  return B/dt

def construct_R(N,M):
  R = np.zeros((N, M))
  for i in range(N):
    for j in range(M-3):
      if i  == 0 and j == 0:
        R[i,j] = 1/2
        R[i,j+1] = 1
        R[i,j+2] = 1/2
      elif i > 0 and j == 2 * i:
        R[i,j] = 1/2
        R[i,j+1] = 1
        R[i,j+2] = 1/2
      elif i == N-1 and j == M-4:
        R[i,-2] = 1/2
        R[i,-1] = 1/2
  R = 0.5*R
  R = np.kron(R,R)
  return R

def construct_P(R):
  return R.transpose()

def solve_pde(A,b,mode='dense'):
  if mode == 'dense':
    return sp.linalg.solve(A, b)
  if mode == "sp_sparse":
    return sp.sparse.linalg.spsolve(A, b)
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

def logsumexp(x):
  c = x.max()
  return c + np.log(np.sum(np.exp(x - c)))

def gauss_seidel(A, b, x):
    #x is the initial condition
    x_old  = x.copy()

    #Loop over rows
    for i in range(A.shape[0]):
        x[i] = (b[i] - np.dot(A[i,:i], x[:i]) - np.dot(A[i,(i+1):], x_old[(i+1):])) / A[i ,i]

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