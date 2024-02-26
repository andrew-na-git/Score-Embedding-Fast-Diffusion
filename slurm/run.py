import argparse

from FDM.train_FDM_fast import train as train_fdm
from MG.train_MG_fast import train as train_mg
from create_report import create_report
from data.Dataset import CIFARDataset

parser = argparse.ArgumentParser(prog="Faster Diffusion with KFP")
parser.add_argument("--model", default="FDM", choices=["FDM", "MG"])
parser.add_argument("--sigma", default=25, type=int)
parser.add_argument("--n-timestep", default=10, type=int)
parser.add_argument("--epochs", default=60, type=int)

args = parser.parse_args()

H = 28
W = 28
dataset = CIFARDataset(H, W)
n_data = len(dataset)
n_channels = dataset.channels

train = train_fdm if args.model == "FDM" else train_mg

print("Begin training...")
train(dataset, args.n_timestep, H, W, n_channels, args.epochs, args.sigma)

print("Training finished. Generating report...")
create_report(args.model, f"model_{args.model}.pth", args.sigma, args.n_timestep, n_data, H, W)
print("Done")