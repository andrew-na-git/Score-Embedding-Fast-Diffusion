from fpdf import FPDF
from datetime import datetime
from pytz import timezone
from datetime import datetime
from pytz import timezone

from network.network import Net
from model.sample import unconditional_sample, conditional_sample
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from data.metrics import ssim_metric, mse_metric

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_interpolated_values(values, x, y):
  # Interpolate the y-values corresponding to the given x-values
  interpolated_y_values = []
  for x_val in values:
      for i in range(len(x) - 1):
          if x[i] <= x_val <= x[i+1]:
              # Linear interpolation
              interpolated_y = y[i] + (y[i+1] - y[i]) * (x_val - x[i]) / (x[i+1] - x[i])
              interpolated_y_values.append(interpolated_y)
              break  # Break the loop as soon as the first interpolation point is found
      else:
          interpolated_y_values.append(np.nan)  # If x_val is outside the range, mark it as NaN

  return interpolated_y_values

def plot_metrics(metrics):
    losses = metrics["losses"]
    ssims = [np.mean(x) for x in metrics["ssim"]]
    mses = [np.mean(x) for x in metrics["mse"]]
    times = metrics["times"]
    epochs = metrics["epochs"]

    if len(ssims) > 0 and len(ssims) == len(mses):
        ## run was profiled, plot both
        # Values to annotate
        highlight_y_values = [0.95, 0.90]
        interpolated_x_values_ssim = get_interpolated_values(highlight_y_values, ssims, times)
        # interpolated corresponding mse values
        interpolated_y_values_mse = get_interpolated_values(interpolated_x_values_ssim, times, mses)

        fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))
        for i in range(len(losses)):
            axes[0].plot(losses[i], label=f"loss_img_{i}")
        axes[1].plot(times, ssims)
        axes[1].set_title("SSIM vs Time/Epoch")
        axes[1].set_xlabel("Time (s)")
        ax2_1 = axes[1].twiny()
        ax2_1.plot(epochs, ssims, linestyle='None')
        ax2_1.set_xlabel("Epoch")

        axes[2].plot(times, mses)
        axes[2].set_title("MSE vs Time/Epoch")
        axes[2].set_xlabel("Time (s)")
        ax2_2 = axes[2].twiny()
        ax2_2.plot(epochs, mses, linestyle='None')
        ax2_2.set_xlabel("Epoch")

        # Annotate the important points on ssim
        for x_val, y_val in zip(interpolated_x_values_ssim, highlight_y_values):
          axes[1].annotate(f'({x_val:.2f}, {y_val})', (x_val, y_val), textcoords="offset points", xytext=(0,10), ha='center')

        # annotate the important points on mse
        for x_val, y_val in zip(interpolated_x_values_ssim, interpolated_y_values_mse):
          axes[2].annotate(f'({x_val:.2f}, {y_val:.4f})', (x_val, y_val), textcoords="offset points", xytext=(0,10), ha='center')

        axes[0].set_title("Loss vs Epochs")
        axes[0].legend(loc="upper right")
    else:
        ## run was not profiled
        fig, axes = plt.subplots(1, 1,  figsize=(7, 5))
        for i in range(len(losses)):
            axes.plot(losses[i], label=f"loss_img_{i}")
        axes.set_title("Loss vs Epochs")
        axes.legend(loc="upper right")
    return fig

def plot_sample(samples, n):
  sample_batch_size = samples.shape[1]
  def get_frame(i, sample_idx):
    return ((samples[i, sample_idx] - samples[i, sample_idx].min())/(samples[i, sample_idx].max() - samples[i, sample_idx].min())).transpose(1, 2, 0)

  fig, ax = plt.subplots(sample_batch_size, n, figsize=(32, 30 * sample_batch_size/n), layout="tight")
  s = np.round(np.linspace(0, samples.shape[0] - 1, num=n)).astype(int)

  if sample_batch_size == 1:
    for idx, col in zip(s, ax):
        col.imshow(get_frame(idx, 0), aspect="auto")
  else:
    for sample_number, row in enumerate(ax):
      for idx, col in zip(s, row):
        col.imshow(get_frame(idx, sample_number), aspect="auto")
  
  return fig


def create_report(folder_path):
    today = datetime.now(timezone('US/Eastern'))

    state = torch.load(os.path.join(folder_path, "model.pth"), map_location=device)
    config = state["config"]
    sample_method = config["sample"]["type"]

    model = Net(state["config"])

    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    epochs = config["training"]["epochs"]
    total_run_time = state["train_time"] + state['diffusion_time']
    profiled = len(state["metrics"]["times"]) > 0

    total_run_time = f"{int(total_run_time//60)} Minutes {int(total_run_time % 60)} Seconds"
    running_time = f"{int(state['train_time']//60)} Minutes {int(state['train_time'] % 60)} Seconds"
    diffusion_time = f"{int(state['diffusion_time']//60)} Minutes {int(state['diffusion_time'] % 60)} Seconds"
    H = W = config['data_loader']['image_size']
    n_data = config['data_loader']['num_images']
    N = config["diffusion"]["num_timesteps"]
    sigma = config["diffusion"]["sigma"]

    fig = plot_metrics(state["metrics"])

    fig.savefig(os.path.join(folder_path, "metrics.png"))
    ground_truths = torch.from_numpy(np.load(os.path.join(folder_path, "dataset.npy")))

    if sample_method == "unconditional":
      samples, n_iter = unconditional_sample(model, config)
    else:
      conditional_weight = config["sample"]["conditional_weight"]
      samples, n_iter = conditional_sample(model, config, ground_truths, conditional_weight)
    
    np.save(os.path.join(folder_path, "samples.npy"), samples)

    mse = mse_metric(ground_truths.numpy(), samples[-1])
    mse = [np.round(x, 4) for x in mse]
    ssim = ssim_metric(ground_truths.numpy(), samples[-1])
    ssim = [np.round(x, 4) for x in ssim]
    
    print("SSIM: ", ssim)
    print("MSE: ", mse)
    
    sample_plot = plot_sample(samples, 5)
    sample_plot.savefig(os.path.join(folder_path, "sample.png"))

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
    pdf.cell(10, 0, f"Model Type: Fast Diffusion")
    pdf.cell(80)
    pdf.cell(10, 0, f"Number of training images: {n_data}")
    pdf.ln(7)

    pdf.cell(20)
    pdf.cell(10, 0, f"Epochs: {epochs}")
    pdf.cell(80)
    pdf.cell(10, 0, f"Image dimensions: {H} x {W}")
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
    pdf.cell(10, 0, f"Sample method: {sample_method}")
    pdf.cell(80)
    pdf.cell(10, 0, f"Sigma: {sigma}")
    pdf.ln(7)

    pdf.set_font('Times', 'B', 14)
    pdf.ln()
    pdf.image(os.path.join(folder_path, "metrics.png"), w=(200 if profiled else 70), h=45)
    
    pdf.cell(0, 10, f"MSE Metric: {mse}", align='C')
    pdf.ln()
    pdf.cell(0, 0, f"SSIM Metric: {ssim}", align='C')
    pdf.ln(10)
    pdf.cell(0, 0, f"Sample (Num Function Eval: {n_iter})", align="C")
    pdf.ln(7)
    pdf.image(os.path.join(folder_path, "sample.png"), w=190, h=35 * n_data)
    pdf.output(os.path.join(folder_path, "report.pdf"), 'F')
    print(f"Report saved to {os.path.join(folder_path, 'report.pdf')}")
