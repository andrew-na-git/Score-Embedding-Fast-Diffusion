import argparse

from FDM.train_FDM_fast import train as train_fdm
from MG.train_MG_fast import train as train_mg
from create_report import create_report
from data.Dataset import CIFARDataset, FlowersDataset
from network.ddim_network import Model as DDIM
from network.ddpm.ddpm import DDPM
from network.openai.unet import UNetModel
from network.network import ScoreNet

import torch

if __name__ == "__main__":
  torch.multiprocessing.set_start_method('spawn')
  parser = argparse.ArgumentParser(prog="Faster Diffusion with KFP")
  parser.add_argument("--model", default="FDM", choices=["FDM", "MG"])
  parser.add_argument("--sigma", default=2, type=float)
  parser.add_argument("--no-train", action="store_true")
  parser.add_argument("--n-timestep", default=50, type=int) 
  parser.add_argument("--epochs", default=300, type=int)

  args = parser.parse_args()

  H = 32
  W = 32

  model = DDIM(H=H)
  dataset = CIFARDataset(H, W, n=3)
  n_data = len(dataset)
  n_channels = dataset.channels

  train = train_fdm if args.model == "FDM" else train_mg

  print("Begin training...")
  if not args.no_train:
    train(model, dataset, args.n_timestep, H, W, n_channels, args.epochs, args.sigma)

  print("Training finished. Generating report...")
  create_report(model, dataset, args.model, f"model_{args.model}.pth", args.sigma, args.n_timestep, n_data, H, W, torch.cuda.is_available())
  print("Done")