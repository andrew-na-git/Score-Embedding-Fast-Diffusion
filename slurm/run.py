import argparse

from FDM.train_FDM_fast import train as train_fdm
from create_report import create_report
from data.Dataset import CIFARDataset, FlowersDataset, CelebDataset, ImageNetDataset
from network.ddim_network import Model as DDIM
from network.ddpm.ddpm import DDPM
from network.openai.unet import UNetModel
from network.network import ScoreNet
from pathlib import Path
from datetime import datetime
from pytz import timezone

import os

import torch
import numpy as np
np.random.seed(2)
torch.manual_seed(2);

if __name__ == "__main__":
  torch.multiprocessing.set_start_method('spawn')
  parser = argparse.ArgumentParser(prog="Faster Diffusion with KFP")
  parser.add_argument("--model", default="DDPM", choices=["DDPM", "DDIM", "OPENAI"])
  parser.add_argument("--sigma", default=2, type=float)
  parser.add_argument("--no-train", action="store_true")
  parser.add_argument("--n-timestep", default=50, type=int) 
  parser.add_argument("--epochs", default=500, type=int)
  parser.add_argument("--comparison", action="store_true")
  parser.add_argument("--use-folder")
  parser.add_argument("--dataset", default="cifar", choices=["celeb", "cifar", "imagenet"])
  parser.add_argument("--res", default=32, type=int)
  parser.add_argument("--n", default=3, type=int)
  parser.add_argument("--seed", default=0, type=int)
  parser.add_argument("--time-btw-samples", default=30, type=int)
  parser.add_argument("--folder-name")
  parser.add_argument("--step-size", default=1, type=float)
  args = parser.parse_args()

  n = args.n
  seed = args.seed
  H = W = args.res
  tol = 2e-8
  dx_num = args.step_size
  
  if args.dataset =="celeb":
    dataset = CelebDataset(H, W, n=n)
  elif args.dataset == "cifar":
    dataset = CIFARDataset(H, W, n=n, seed=seed)
  elif args.dataset == "imagenet":
    dataset = ImageNetDataset(H, W, n=n, seed=seed)

  temb_method = "fourier" if n == 1 else "linear"

  if args.model == "DDIM":
    model = DDIM(H=H, time_embedding_method=temb_method)
  elif args.model == "DDPM":
    model = DDPM(H=H, time_embedding_method=temb_method)
  elif args.model == "OPENAI":
    model = UNetModel(time_embedding_method=temb_method)
    
  pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"Num Parameters: {pytorch_total_params}")
  n_data = len(dataset)
  n_channels = dataset.channels

  
  if not args.no_train:
    today = datetime.now(timezone('EST'))

    if not args.folder_name:
      folder = os.path.join(os.path.dirname(__file__), f"reports/{args.model}_{today.strftime('%B-%d-%H:%M')}")
    else:
      folder = os.path.join(os.path.dirname(__file__), f"reports/{args.model}_{today.strftime('%B-%d-%H:%M')}_{args.folder_name}")
    Path(folder).mkdir(parents=True, exist_ok=True)

    model_save_dir = os.path.join(folder, "models")
    
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(model_save_dir, "data.npy"), dataset.data)

    print("Begin training...")
    diffusion_time, training_time = train_fdm(model, dataset, args.n_timestep, H, W, n_channels, args.epochs, args.sigma, comparison=args.comparison, model_save_folder=model_save_dir, tol=tol, dx_num=dx_num, temb_method=temb_method, time_between_samples=args.time_btw_samples)
    print("Training finished. ")
  print("Generating report...")
  create_report(model, dataset, args.use_folder if args.use_folder else folder, args.model, os.path.join(args.use_folder if args.use_folder else  folder, f"models/model_final.pth"), args.sigma, args.n_timestep, n_data, H, W, torch.cuda.is_available(), temb_method)
  print("Done")