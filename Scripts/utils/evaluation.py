from pathlib import Path

import matplotlib.pyplot as plt


def configure_plot():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8.2,
        "axes.titlesize": 8.7,
        "axes.labelsize": 8.2,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.3,
        "ytick.labelsize": 7.3,
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.axisbelow": True,
        "savefig.facecolor": "white",
    })


def save_figure(fig, path, tight=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {"bbox_inches": "tight", "pad_inches": 0.02} if tight else {}
    fig.savefig(path.with_suffix(".pdf"), **options)
    fig.savefig(path.with_suffix(".png"), dpi=300, **options)
    plt.close(fig)
