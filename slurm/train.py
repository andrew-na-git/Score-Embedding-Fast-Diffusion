import os

import numpy as np
import scipy as sp
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity

import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils.data import DataLoader

from kfp import construct_A, construct_B, diffusion_coeff, construct_R, construct_P, construct_P_block, construct_R_block, gauss_seidel, solve_pde, logsumexp
from network import ScoreNet

torch.set_default_device('cuda')


# create a.so if doesnt exists
if not os.path.isfile("../sparse_gaussian_elimination/a.so"):
    os.system("make -C ../sparse_gaussian_elimination a.so")

# download mnist dataset
mnist = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
#data_loader = DataLoader(list(filter(lambda i: i[1] == 5, mnist))[:1], shuffle=True, generator=torch.Generator(device='cuda'))
mnist_data = mnist.data[mnist.targets == 5][:10]
# mnist_data = np.moveaxis(mnist.data.numpy(), 0, -1)
# mnist_data.shape
#mnist_data = np.array(mnist)

## construct the grid and Initial values
batch_size = 32
N = 10
H = 28
W = 28
epoch = 3
eps = 1e-6

#data_loader = DataLoader(mnist_subset, batch_size=1, shuffle=True, num_workers=0, generator=torch.Generator(device='cuda'))

t = np.linspace(eps, 1, N)
dt = 1/N

sigma = 25

# create model
model_score = ScoreNet(H=H, W=W)
loss_fn = torch.nn.MSELoss()
optimizer = Adam(model_score.parameters(), lr=1e-3)
mm_scaler = MinMaxScaler()
model_score.train()

# we want to sample from random time steps to construct training samples
random_t = np.random.rand(N-3)
random_t = np.insert(random_t, 0, dt) # first step we want something small
random_t = np.insert(random_t, 0, 1)
random_t = np.sort(random_t) # we sort the time in increasing order for denoising
time_ = np.insert(random_t, 0, eps).astype(np.float32) # for denoising we want time 0 to always be in sample to train
sigma_ = diffusion_coeff(torch.tensor(time_), sigma).detach().cpu().numpy()

for x_ in mnist_data:
  data = x_

  x = torch.zeros((N, 1, H, W))
  m = np.zeros((N, H*W), dtype=np.float32)
  del_m = np.zeros_like(m, dtype=np.float32)
  
  dx = data.detach().numpy().max()/H
  dy = data.detach().numpy().max()/W
  x[0] = torch.tensor(mm_scaler.fit_transform(data.ravel()[:, None]).astype(np.float32)).reshape((1, 1, H, W))
  kde = KernelDensity(kernel='gaussian').fit(data.ravel()[:, None])
  m[0] = kde.score_samples(data.ravel()[:, None])
  del_m[0] = np.diff(m[0].ravel(), axis=0, prepend=m[0,0])
  m_c = np.zeros((N, int((H*W/4))), dtype=np.float32)
  del_m_c = np.zeros_like(m_c, dtype=np.float32)

  perturbed_x = torch.zeros_like(x)
  scores = np.zeros((N, H, W), dtype=np.float32)

  for e in tqdm(range(epoch)):
    # we normalize for sigma to ensure the dynamics doesn't blow up
    A_block = []
    for i, t_ in enumerate(random_t, 1):
      A = construct_A(dx, dy, t_ - time_[i-1], np.zeros((H, W)), sigma_[i], scores[i], H, W)
      if i == 1:
        A_block = sp.linalg.block_diag(A)
      else:
        A_block = sp.linalg.block_diag(A_block, A)
        A_block[(i-1)*H*W:i*H*W, (i-2)*H*W:(i-1)*H*W] = -np.eye((H*W))/(t_ - time_[i-1])

    B = construct_B(dx, dy, time_[1] - time_[0], m[0], np.zeros((H, W)), sigma_[1], scores[1])
    B_block = np.zeros(A_block.shape[0])
    B_block[:H*W] = B

    # update m (pre-smoothing)
    m[1:] = gauss_seidel(A_block, B_block, scores[1:].flatten()).reshape(((N-1), H*W))
    R = construct_R(int(H/2), int(H))
    P = construct_P(R)
    R_block = []
    P_block = []
    for i, t_ in enumerate(random_t, 1):
      ####### kernal preserving restriction ####
      R_block = construct_R_block(R, R_block, i)
      ####### bilinear interpolation ###########
      P_block = construct_P_block(P, P_block, i)

    # we want to perform the coarse grid
    # compute residual r = b - Am[1:]
    r = B_block - A_block@m[1:].flatten()
    # coursening step 1: r_c = R_c@r
    r_c = R_block@r
    # coursening A_c = R_c@A@P_c (Petrov-Galerkin Coursening)
    A_c = R_block@A_block@P_block
    # compute course err: err_c = solve_pde(A_c,r_c)
    err_c = solve_pde(A_c, r_c, mode='sparse')
    # interpolate to fine grid: err = P_c@err_c
    err = P_block@err_c
    # we apply fine grid-correction
    m[1:] = (m[1:].flatten() + err).reshape((N-1, H*W))
    # post smoothing
    m[1:] = gauss_seidel(A_block, B_block, m[1:].flatten()).reshape(((N-1), H*W))
    # we want to coarsen the score function to train on coarse data
    m_c[1:] = (R_block@m[1:].flatten()).reshape((-1, int(H*W/4)))

    # constructing the training data and labels
    for i, t_ in enumerate(random_t, 1):
      del_m[i] = np.diff(m[i].ravel(), axis=0, prepend=m[i, 0])

    x = torch.tensor(mm_scaler.fit_transform(np.exp((-m.ravel() - logsumexp(-m.ravel())))[:, None])).reshape((N, 1, H, W))
    perturbed_x = x + torch.randn_like(x) * torch.sqrt(2 * torch.tensor(sigma_)**2)[:, None, None, None]
    print(perturbed_x.max(), perturbed_x.min())

    train_x_data = perturbed_x
    train_y_data = torch.tensor(del_m.astype(np.float32)).reshape((N, 1, H, W))

    # generate coarse dataset
    x_c = torch.tensor(mm_scaler.fit_transform(np.exp((-m_c.ravel() - logsumexp(-m_c.ravel())))[:, None])).reshape((N, 1, int(H/2), int(W/2)))
    perturbed_xc = x_c + torch.randn_like(x_c) * torch.sqrt(2 * torch.tensor(sigma_)**2)[:, None, None, None]
    train_xc_data = perturbed_xc

    yc_pred = model_score(train_xc_data, torch.tensor(time_), coarse=True)
    lm = (2*torch.tensor(sigma_)**2)[:, None, None, None]
    loss = loss_fn(yc_pred/lm, train_y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses = loss.item()

    y_pred = model_score(train_x_data, torch.tensor(time_))
    lm = (2*torch.tensor(sigma_)**2)[:, None, None, None]
    loss = loss_fn(y_pred/lm, train_y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses = loss.item()

    scores = (y_pred/lm).clone().detach().cpu().numpy().reshape((N, H, W)) # we normalize before fedding back into PDE

torch.save(model_score.state_dict(), 'model_test.pth')
print(f"\nmodel has been saved")