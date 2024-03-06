from fpdf import FPDF
from datetime import datetime
from pytz import timezone
from pathlib import Path
from sample.sample import sample
import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from utils.metrics import mse_metric, ssim_metric

def plot_losses(model_folder):
    files = [os.path.join(model_folder, x) for x in os.listdir(model_folder) if "sample" in x]
    dataset = np.load(os.path.join(model_folder, "data.npy"))

    mses = []
    ssims = []

    for file in files: 
        file_name = file.split(os.sep)[-1]

        time = float(file_name.split("_", maxsplit=1)[1].replace(".npy", ""))

        samples = np.load(file)

        avg_mse = np.mean(mse_metric(dataset, samples[-1]))
        avg_ssim = np.mean(ssim_metric(dataset, samples[-1]))

        mses.append((avg_mse, time))
        ssims.append((avg_ssim, time))
    
    mse_plot_data_y = [x[0] for x in sorted(mses, key=lambda x: x[1])]
    mse_plot_data_x = [x[1] for x in sorted(mses, key=lambda x: x[1])]

    ssim_plot_data_y = [x[0] for x in sorted(ssims, key=lambda x: x[1])]
    ssim_plot_data_x = [x[1] for x in sorted(ssims, key=lambda x: x[1])]

    fig, ax = plt.subplots( nrows=1, ncols=1 );  # create figure & 1 axis
    ax.plot(mse_plot_data_x, mse_plot_data_y);
    ax.set_xlabel("Seconds")
    ax.set_ylabel("MSE")
    fig.suptitle("MSE Vs. Time")

    fig.savefig(os.path.join(os.path.dirname(model_folder), "mse_losses.png"))

    fig, ax = plt.subplots( nrows=1, ncols=1 );  # create figure & 1 axis
    ax.plot(ssim_plot_data_x, ssim_plot_data_y);
    ax.set_xlabel("Seconds")
    ax.set_ylabel("SSIM")
    fig.suptitle("SSIM Vs. Time")
    fig.savefig(os.path.join(os.path.dirname(model_folder), "ssim_losses.png"))


def create_report(model, dataset, folder, model_name="FDM", model_path="model_fdm.pth", sigma=2, N=20, n_data=1, H=28, W=28, with_cuda = False, temb_method="linear"):
    today = datetime.now(timezone('EST'))

    model.load_state_dict(torch.load(model_path))
    model_title = model_name

    log_df = pd.read_csv(folder + "/loss.log")

    epochs = len(log_df)
    total_run_time = log_df['running_time'][0] + log_df['diffusion_time'][0]

    total_run_time = f"{int(total_run_time//60)} Minutes {int(total_run_time % 60)} Seconds"
    running_time = f"{int(log_df['running_time'][0]//60)} Minutes {int(log_df['running_time'][0] % 60)} Seconds"
    diffusion_time = f"{int(log_df['diffusion_time'][0]//60)} Minutes {int(log_df['diffusion_time'][0] % 60)} Seconds"

    # ax = log_df["running_time"].plot(use_index = True)
    # ax.set_ylabel("Time (s)")
    # ax.set_xlabel("Epoch")
    # fig = ax.get_figure()
    # fig.suptitle("Time per Epoch")

    #fig.savefig(folder + "/time.jpeg")

    ax = log_df[[x for x in log_df.columns if "loss" in x]][100:].plot(use_index = True)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    fig = ax.get_figure()
    fig.suptitle("Loss per Epoch")

    fig.savefig(folder + "/loss.jpeg")

    fig, n_eval, samples = sample(model, H=H, W=W, N=N, sigma=sigma, n_data=n_data, temb_method=temb_method)
    fig.savefig(folder + "/sample.png", bbox_inches='tight')
    mse_loss = [round(m, 3) for m in mse_metric([x.numpy() for x in dataset], samples[-1])]
    ssim_loss = [round(m, 3) for m in ssim_metric([x.numpy() for x in dataset], samples[-1])]
    print(ssim_loss)
    plot_losses(folder + "/models")
    class PDF(FPDF):
        def header(self):
            # Arial bold 15
            self.set_font('Times', 'B', 20)
            # Move to the right
            self.cell(80)
            # Title
            self.cell(30, 10, 'Fast Diffusion Model Run Report', align='C')
            # Line break
            self.ln(13)
            self.set_font('Times', size=12)
            self.cell(80)
            self.cell(30, 0, f'Time Generated: {today.strftime("%b-%d-%Y %I:%M %p EST")}', align='C')

            # Line break
            self.ln(15)

    # Instantiation of inherited class
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font('Times', '', 12)

    pdf.cell(20)
    pdf.cell(10, 0, f"Model Type: {model_title}")
    pdf.cell(80)
    pdf.cell(10, 0, f"Number of training images: {n_data}")
    pdf.ln(7)

    pdf.cell(20)
    pdf.cell(10, 0, f"Epochs: {epochs}")
    pdf.cell(80)
    pdf.cell(10, 0, f"Image dimensions: {W} x {H}")
    pdf.ln(7)

    pdf.cell(20)
    pdf.cell(10, 0, f"Total Run Time: {total_run_time}")
    pdf.cell(80)
    pdf.cell(10, 0, f"Number of time steps sampled: {N}")
    pdf.ln(7)

    pdf.cell(20)
    pdf.cell(10, 0, f"Diffusion Time: {diffusion_time}")
    pdf.cell(80)
    pdf.cell(10, 0, f"Training Time: {running_time}")
    pdf.ln(7)
    
    pdf.cell(20)
    pdf.cell(10, 0, f"With CUDA: {with_cuda}")
    pdf.cell(80)
    pdf.cell(10, 0, f"Sigma: {sigma}")
    pdf.ln(7)

    pdf.cell(43)
    pdf.set_font('Times', 'B', 14)
    pdf.ln()
    pdf.image(folder + "/loss.jpeg", w=50, h=45)
    
    pdf.cell(50)
    pdf.cell(0, -55, f"MSE Metric: {mse_loss}")
    pdf.ln(1)
    pdf.cell(50)
    pdf.cell(0, -45, f"SSIM Metric: {ssim_loss}")
    pdf.ln(10)
    pdf.cell(0, 0, f"Sample (Num Function Eval: {n_eval})", align="C")
    pdf.ln(7)
    pdf.image(folder + "/sample.png", y=140, w=190, h=18 * n_data)
    pdf.output(folder + "/report.pdf", 'F')
    print(f"Report saved to {folder + '/report.pdf'}")

if __name__ == "__main__":
  create_report()