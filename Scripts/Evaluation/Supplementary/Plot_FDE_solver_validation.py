from pathlib import Path
import sys
import time

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import Constants as CONST
from src.models.FDEint.FixedStepFDEint_with_forcing import FDEint


DATA_PATH = CONST.DATA_PATH / "Fractional_Data" / "fractional_forced_smooth_testsystem.csv"
OUTPUT_DIR = CONST.OUTPUT_PATH / "pdf/Evaluation/new_runs/Supplementary/FDE_Solver_Validation"
RESULTS_DIR = CONST.NEW_RESULTS_PATH / "solver/forced"
TABLE_DIR = OUTPUT_DIR / "Tables"
FIG_BASENAME = "FDE_Solver_validation"
TABLE_PATH = TABLE_DIR / f"{FIG_BASENAME}_metrics.tex"
METRICS_PATH = RESULTS_DIR / f"{FIG_BASENAME}_metrics.csv"


class LinearForcedFDE(nn.Module):
    def __init__(self, a: float, b: float):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, t, y, u_t):
        if u_t.dim() == 1:
            u_t = u_t.unsqueeze(-1)
        return -self.a * y + self.b * u_t


def solve_nfde(t_np, u_np, y0, alpha, dtype=torch.float32):
    device = torch.device("cpu")
    t = torch.tensor(t_np, dtype=dtype, device=device)
    u = torch.tensor(u_np, dtype=dtype, device=device)
    y0_t = torch.tensor([y0], dtype=dtype, device=device)
    alpha_t = torch.tensor(alpha, dtype=dtype, device=device)
    model = LinearForcedFDE(a=1.0, b=1.0).to(device)

    start = time.perf_counter()
    with torch.no_grad():
        y_pred = FDEint(
            f=model,
            t=t,
            y0=y0_t,
            alpha=alpha_t,
            h=None,
            dtype=dtype,
            DEBUG=False,
            return_internals=False,
            u=u,
            u_fn=None,
        )
    runtime_s = time.perf_counter() - start
    return y_pred.squeeze(0).squeeze(-1).cpu().numpy(), runtime_s


def compute_metrics(t_np, y_ref, y_pred, runtime_s):
    error = y_ref - y_pred
    abs_error = np.abs(error)
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(abs_error))
    max_abs = float(np.max(abs_error))
    nrmse_range = float(rmse / (np.max(y_ref) - np.min(y_ref)))
    nrmse_rms = float(rmse / np.sqrt(np.mean(y_ref**2)))
    final_abs = float(abs_error[-1])
    return {
        "n_samples": len(t_np),
        "t_start": float(t_np[0]),
        "t_end": float(t_np[-1]),
        "dt": float(np.median(np.diff(t_np))),
        "sampling_rate": float(1.0 / np.median(np.diff(t_np))),
        "mae": mae,
        "rmse": rmse,
        "max_abs_error": max_abs,
        "nrmse_range": nrmse_range,
        "nrmse_rms": nrmse_rms,
        "final_abs_error": final_abs,
        "runtime_s": runtime_s,
    }


def configure_matplotlib():
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "text.latex.preamble": r"\usepackage{amsmath}",
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 10,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def latex_sci_body(value, digits=3):
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"{float(mantissa):.{digits}f}\times 10^{{{int(exponent)}}}"


def latex_sci(value, digits=3):
    return rf"${latex_sci_body(value, digits)}$"


def write_metrics_table(metrics):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        ("RMSE", latex_sci(metrics["rmse"])),
        ("Maximum absolute error", latex_sci(metrics["max_abs_error"])),
    ]

    table = [
        "\\begin{table}[H]",
        "    \\centering",
        "    \\caption[Forced FDE solver verification]{Numerical deviation between the FOMCON reference and the \\protect\\texttt{FDEint} implementation introduced in this work for the forced test problem.}",
        "    \\label{tab:FDE_solver_validation_metrics}",
        "    \\begin{tabular}{lc}",
        "        \\toprule",
        "        Quantity & Value \\\\",
        "        \\midrule",
    ]
    for name, value in rows:
        table.append(f"        {name} & {value} \\\\")
    table.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    TABLE_PATH.write_text("\n".join(table), encoding="utf-8")


def plot_validation(t_np, u_np, y_ref, y_pred, metrics):
    configure_matplotlib()
    error = y_ref - y_pred

    # 0.95 column widths in a 12pt book layout: readable labels without crowding.
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(5.9, 5.9),
        sharex=True,
        gridspec_kw={"height_ratios": [0.9, 1.25, 0.9], "hspace": 0.12},
    )

    fomcon_color = "#1f5aa6"
    nfde_color = "#d95f02"
    input_color = "#2b7a78"
    error_color = "#3f3f46"

    axes[0].plot(t_np, u_np, color=input_color, linewidth=1.25)
    axes[0].set_ylabel(r"$u(t)$")

    axes[1].plot(t_np, y_ref, color=fomcon_color, linewidth=1.35, label=r"FOMCON")
    axes[1].plot(
        t_np,
        y_pred,
        color=nfde_color,
        linewidth=1.35,
        linestyle=(0, (4, 2)),
        label=r"\texttt{FDEint}",
    )
    axes[1].set_ylabel(r"$y(t)$")
    axes[1].legend(loc="upper left", frameon=False, ncol=2, handlelength=2.2, columnspacing=1.2)

    axes[2].axhline(0.0, color="#9ca3af", linewidth=0.7)
    axes[2].plot(t_np, error, color=error_color, linewidth=1.15)
    axes[2].set_ylabel(r"$e(t)$")
    axes[2].set_xlabel(r"$t$")

    annotation = (
        rf"$\mathrm{{RMSE}}={latex_sci_body(metrics['rmse'], 2)}$"
        "\n"
        rf"$\max |e|={latex_sci_body(metrics['max_abs_error'], 2)}$"
    )
    axes[2].text(
        0.015,
        0.92,
        annotation,
        transform=axes[2].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.5},
    )

    for ax in axes:
        ax.grid(True, color="#d4d4d8", linewidth=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3, width=0.7)
        ax.margins(x=0)

    axes[2].set_xlim(float(t_np[0]), float(t_np[-1]))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"{FIG_BASENAME}.{suffix}", dpi=300)
    plt.close(fig)


def main():
    data = pd.read_csv(DATA_PATH)
    t_np = data["t"].to_numpy(dtype=float)
    u_np = data["u"].to_numpy(dtype=float)
    y_ref = data["y"].to_numpy(dtype=float)

    alpha = 0.7
    y_pred, runtime_s = solve_nfde(t_np, u_np, y_ref[0], alpha=alpha)
    metrics = compute_metrics(t_np, y_ref, y_pred, runtime_s)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)
    write_metrics_table(metrics)
    plot_validation(t_np, u_np, y_ref, y_pred, metrics)

    print(f"Wrote {OUTPUT_DIR / (FIG_BASENAME + '.pdf')}")
    print(f"Wrote {OUTPUT_DIR / (FIG_BASENAME + '.png')}")
    print(f"Wrote {TABLE_PATH}")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
