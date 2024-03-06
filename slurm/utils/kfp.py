import numpy as np
import scipy as sp
from scipy import sparse
import torch


def get_sparse_A_block(dx,dt,g,s,H,W, N):
  h = dt/(2*dx)
  h2 = dt/(dx**2)
  f = np.zeros((H*W))

  base_diagonal = np.ones((N - 1, H*W)) * (g**2)[:, None] * h2
  diag = 1 + base_diagonal.ravel()
  up_diagonal = -0.5 * (base_diagonal + (g**2)[:, None] * h * s)
  up_diagonal[:-1, -1] = 0
  up_diagonal = up_diagonal.ravel()[:-1]

  down_diagonal = -0.5 * (base_diagonal - (g**2)[:, None] * h * s)
  down_diagonal[:-1, -1] = 0
  down_diagonal = down_diagonal.ravel()[:-1]

  t_diag = -np.ones(((N - 2) * H*W))
  return sparse.diags([diag, up_diagonal, down_diagonal, t_diag], [0, 1, -1, -H*W], format="csr")

def construct_B(dx,dt,m_prev,df,i):
  h = dt/(2*dx)
  if i == 1:
    B = m_prev - (df*h)
  else:
    B = - (df*h)
  return B

def get_B_block(dx, dt, m, channel, H, W, N):
  B_block = []
  for i in range(1, N):
    df = np.zeros((H*W))
    B = construct_B(dx, dt, m[i-1, channel], df, i)
    B_block.append(B)
  B_block = np.concatenate(B_block)
  return B_block

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
  return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

  Returns:
    The vector of diffusion coefficients.
  """
  return sigma**t