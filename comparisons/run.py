import sys
sys.path.insert(0, "..")

import argparse
from argparse import RawTextHelpFormatter
import os
from pathlib import Path

from model import train
from create_report import create_report
import yaml

import torch
import numpy as np
np.random.seed(100)
torch.manual_seed(100);

prog="DDIM and DDPM implementation for comparison to Fast Score-Based Diffusion"

description="""Authors: Andrew S. Na, William Gao, Mykhailo Briazkalo, Justin W.L. Wan
Contact: andrew.na@uwaterloo.ca"""

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    prog=prog,
    description=description,
    formatter_class=RawTextHelpFormatter
  )
  parser.add_argument("--config",
                      help="Config yaml file name in /configs folder (e.g. cifar1.yml)", 
                      required=True)
  
  parser.add_argument("--no-train", 
                      help="Skip training and just generate report from pretrained model", 
                      required=False, 
                      action="store_true")
  
  parser.add_argument("--folder-ext",
                      help="Name to append to default save folder name",
                      required=False)
  
  parser.add_argument("--profile",
                      help="Profile the SSIM and MSE over time (considerably slower)",
                      required=False,
                      action="store_true")
  
  args = parser.parse_args()
  
  ### Read config file
  config_file_loc = os.path.join("configs", args.config)
  assert os.path.isfile(config_file_loc), f"Could not find config file at {config_file_loc}"
  
  with open(config_file_loc, 'r') as f:
    config = yaml.safe_load(f)

  ### Create save folder
  folder_name = f"{config['name']}_{args.folder_ext}" if args.folder_ext else config['name']
  save_folder_loc = os.path.join("saves", folder_name)
  
  if args.no_train:
    assert os.path.isdir(save_folder_loc), f"Save folder cannot be found at {save_folder_loc}"
    create_report(save_folder_loc)
    exit(0)
    
  # create folder if not exists
  Path(save_folder_loc).mkdir(parents=True, exist_ok=True)
  
  ### train
  train.train(config, save_folder_loc, args.profile)

  ### create report pdf
  create_report(save_folder_loc)
