from fpdf import FPDF
from datetime import datetime
from pytz import timezone
from pathlib import Path
from sample.sample import sample
import pandas as pd

def create_report(model, dataset, model_name="FDM", model_path="model_fdm.pth", sigma=2, N=20, n_data=1, H=28, W=28, with_cuda = False):
    today = datetime.now(timezone('EST'))

    model_title = model_name
    folder = f"reports/{model_name}_{today.strftime('%B-%d-%H:%M')}"

    Path(folder).mkdir(parents=True, exist_ok=True)

    log_df = pd.read_csv("loss.log")

    epochs = len(log_df)
    total_time = f"{round(log_df['time'][0]/60, 2)} Minutes"
    median_time_per_epoch = f"N/A seconds"

    ax = log_df["time"].plot(use_index = True)
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Epoch")
    fig = ax.get_figure()
    fig.suptitle("Time per Epoch")

    fig.savefig(folder + "/time.jpeg")

    ax = log_df[[x for x in log_df.columns if x.endswith("loss")]][100:].plot(use_index = True)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    fig = ax.get_figure()
    fig.suptitle("Loss per Epoch")

    fig.savefig(folder + "/loss.jpeg")

    fig, n_eval = sample(model, H=H, W=W, N=N, sigma=sigma, n_data=n_data)
    fig.savefig(folder + "/sample.jpeg", bbox_inches='tight')

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
    pdf.cell(10, 0, f"Total Run Time: {total_time}")
    pdf.cell(80)
    pdf.cell(10, 0, f"Number of time steps sampled: {N}")
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

    pdf.cell(80)
    pdf.cell(30, 0, f"Sample (Num Function Eval: {n_eval})", align="C")
    pdf.ln(7)
    pdf.image(folder + "/sample.jpeg", w=190, h=35 * min(n_data, 4))
    pdf.output(folder + "/report.pdf", 'F')
    print(f"Report saved to {folder + '/report.pdf'}")

if __name__ == "__main__":
  create_report()