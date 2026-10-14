"""RQ1.3: read saved fractional runs, compare full data, and plot data efficiency.

Edit the paths and lists below. No models or datasets are constructed here.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Constants import EVALUATION_RESULTS_PATH, EVALUATION_SOURCE, OUTPUT_PATH
from Scripts.utils.naming import model_table_label, variant_key, variant_directory, variant_result_name

reference_root = EVALUATION_RESULTS_PATH / "fractional/references"
reference_curve_root = EVALUATION_RESULTS_PATH / "fractional/reference_curves"
nfde_root = EVALUATION_RESULTS_PATH / "fractional/nfde"
output_dir = OUTPUT_PATH / "pdf/Evaluation" / EVALUATION_SOURCE / "RQ1_3"
make_figures = True

datasets = ["FractionalBenchmarkExtrapolation", "FractionalBenchmarkForcing", "FractionalBenchmarkReconstruction"]
dataset_labels = dict(zip(datasets, ["Extrapolation", "Modified forcing", "Reconstruction"]))
nfde_variants = ["NFDE", "NFDE-t"]
node_variants = ["NO-Aug0", "NO-Aug0-t", "NO-Obs", "NO-Obs-t"]
baseline_variants = ["I-LSTM", "TCN", "FC-dt"]
variants = nfde_variants + node_variants + baseline_variants
table_variants = nfde_variants + node_variants + ["I-LSTM", "TCN"]
full_seeds = [42, 1, 2, 3]
reference_seeds = [42, 1, 2, 3]
fractions = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
dense_references = {
    "FractionalBenchmarkExtrapolation": ["NO-Aug0-t", "TCN"],
    "FractionalBenchmarkForcing": ["NO-Aug0", "I-LSTM"],
    "FractionalBenchmarkReconstruction": ["NO-Aug0", "I-LSTM", "FC-dt"],
}


def read_seed(path, dataset, model, seed, percent, expected_n, source_kind):
    data = json.loads(path.read_text()) if path.is_file() else {}
    metrics = data.get("metrics", {})
    values = {}
    for split in ("val", "test"):
        value = metrics.get(split, {}).get("global", {}).get("rmse")
        values[f"{split}_rmse"] = float(value) if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 else np.nan
    success = (bool(data) and data.get("success", True) is True
               and (data.get("evaluation_status") or {}).get("success", True) is True
               and math.isfinite(values["test_rmse"]))
    failure = json.dumps(data.get("evaluation_status") or {}).lower()
    marker = "N" if "solver_failure" in failure or "underflow in dt" in failure else ""
    if not marker and any(word in failure for word in ["diverg", "nan", "nonfinite", "non-finite"]):
        marker = "D"
    return {"dataset": dataset, "model": variant_key(model), "seed": seed, "percent": percent,
            "success": success, "expected_n": expected_n, "present": path.is_file(),
            "failure_markers": marker if not success else "", "failure_type": (data.get("evaluation_status") or {}).get("failure_type", ""),
            "source_path": str(path), "source_kind": source_kind, **values}


def read_results():
    """Return full-table and curve rows separately: their NODE cohorts differ."""
    full_rows, curve_rows = [], []
    for dataset in datasets:
        for model in variants:
            model = variant_key(model)
            if model == "FC-dt" and dataset == "FractionalBenchmarkExtrapolation":
                continue
            root = nfde_root if model in nfde_variants else reference_curve_root
            source_label = variant_result_name(root, dataset, model)
            grid = fractions if model in nfde_variants or model in dense_references[dataset] else [10, 25, 50, 100]
            for percent in grid:
                seeds = full_seeds if model in nfde_variants and percent == 100 else reference_seeds
                for seed in seeds:
                    path = root / dataset / f"{source_label}_seed{seed}_{percent:03d}pct.json"
                    row = read_seed(path, dataset, model, seed, percent, len(seeds), "nfde" if model in nfde_variants else "reference_curve")
                    curve_rows.append(row)
                    if percent == 100 and model in nfde_variants + ["FC-dt", "TCN"]:
                        full_rows.append(row.copy())
            if model in node_variants + ["I-LSTM"]:
                for seed in reference_seeds:
                    run = variant_directory(reference_root, dataset, model) / f"Seed_{seed}"
                    path = run / "evaluation_results_copy.json"
                    if not path.is_file():
                        path = run / "evaluation_results.json"
                    full_rows.append(read_seed(path, dataset, model, seed, 100, len(reference_seeds), "reference_training"))
    return pd.DataFrame(full_rows), pd.DataFrame(curve_rows)


def aggregate_results(records):
    if records.duplicated(["dataset", "model", "percent", "seed"]).any():
        raise ValueError("Duplicate fractional dataset/model/percent/seed results")
    rows = []
    for (dataset, model, percent), group in records.groupby(["dataset", "model", "percent"], sort=False):
        successful = group[group["success"] & np.isfinite(group["test_rmse"])]
        row = {"dataset": dataset, "model": model, "percent": percent, "n": len(successful),
               "expected_n": int(group["expected_n"].iloc[0]),
               "n_missing": int((~group["present"]).sum()),
               "n_failed": int((group["present"] & ~group["success"]).sum()),
               "failure_markers": ",".join(sorted({value for value in group.failure_markers if value}))}
        for split in ("val", "test"):
            values = successful[f"{split}_rmse"].dropna()
            row[f"{split}_rmse_mean"] = values.mean()
            row[f"{split}_rmse_std"] = values.std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def select_families(curves):
    curves = curves.assign(model=curves.model.map(variant_key))
    rows = []
    for dataset in datasets:
        full = curves[(curves.dataset == dataset) & (curves.percent == 100)]
        for family, candidates in (("baseline", baseline_variants), ("node", node_variants), ("nfde", nfde_variants)):
            options = full[full.model.isin(candidates)].dropna(subset=["test_rmse_mean"])
            if options.empty:
                continue
            chosen = options.sort_values(["test_rmse_mean", "model"]).iloc[0]
            rows.append({"dataset": dataset, "family": family, "model": chosen.model,
                         "selection_percent": 100, "test_rmse_mean": chosen.test_rmse_mean})
    return pd.DataFrame(rows, columns=["dataset", "family", "model", "selection_percent", "test_rmse_mean"])


def write_full_table(summary):
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\scriptsize",
             r"\caption{Fractional benchmark test errors: mean and sample standard deviation over four seeds.}",
             r"\label{tab:rq13_p1_full_test_rmse}", r"\setlength{\tabcolsep}{3pt}",
             r"\renewcommand{\arraystretch}{1.08}", r"\begin{tabular}{lrrr}", r"\toprule",
             "Model & " + " & ".join(dataset_labels[d] for d in datasets) + r" \\", r"\midrule"]
    for model in table_variants:
        if model in (node_variants[0], baseline_variants[0]):
            lines.append(r"\addlinespace[2pt]")
        cells = [model_table_label(model)]
        for dataset in datasets:
            selected = summary[(summary.dataset == dataset) & summary.model.isin(table_variants)]
            row = selected[selected.model == model]
            if row.empty:
                cells.append("--")
                continue
            row = row.iloc[0]
            if pd.isna(row.test_rmse_mean):
                text = f"-- [{int(row.n)}/{int(row.expected_n)}]"
                if row.failure_markers:
                    text += r"\textsuperscript{" + row.failure_markers + "}"
                cells.append(text)
                continue
            values = []
            for key in ("test_rmse_mean", "test_rmse_std"):
                value = row[key]
                text = f"{value:.3g}" if pd.notna(value) else "--"
                if pd.notna(value) and math.isclose(value, selected[key].min(), rel_tol=0, abs_tol=1e-15):
                    text = r"\textbf{" + text + "}"
                values.append(text)
            text = values[0] + r" $\pm$ " + values[1]
            if row.n != row.expected_n:
                text += f" [{int(row.n)}/{int(row.expected_n)}]"
            if row.failure_markers:
                text += r"\textsuperscript{" + row.failure_markers + "}"
            cells.append(text)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (output_dir / "rq1_3_full_test_rmse_table.tex").write_text("\n".join(lines) + "\n")


def plot_results(curves, selection):
    import matplotlib.pyplot as plt
    from Scripts.utils.evaluation import configure_plot, save_figure
    configure_plot()
    for time_comparison in (True, False):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55 if time_comparison else 2.75), sharey=True, constrained_layout=True)
        handles = {}
        for ax, dataset in zip(axes, datasets):
            if time_comparison:
                selected = [("NFDE", "NFDE", "#56B4E9", "s"),
                            ("NFDE-t", "NFDE-t", "#0072B2", "s")]
            else:
                selected = []
                for family, label, color, marker in (("baseline", "Best baseline", "#4d4d4d", "s"),
                                                       ("node", "Best NODE variant", "#D55E00", "^"),
                                                       ("nfde", "Best NFDE variant", "#0072B2", "o")):
                    choice = selection[(selection.dataset == dataset) & (selection.family == family)]
                    if not choice.empty:
                        selected.append((choice.iloc[0].model, label, color, marker))
            for model, label, color, marker in selected:
                values = curves[(curves.dataset == dataset) & (curves.model == model)].sort_values("percent").dropna(subset=["test_rmse_mean"])
                y = values.test_rmse_mean.to_numpy()
                std = values.test_rmse_std.fillna(0).to_numpy()
                lower = np.minimum(std, np.maximum(y * .95, 1e-12))
                line = ax.errorbar(values.percent, y, yerr=np.array([lower, std]), label=label,
                                   color=color, marker=marker, markersize=4, linewidth=1.5, capsize=2.2, elinewidth=.8)
                handles[label] = line
            if not time_comparison:
                names = "\n".join(f"{label.split()[1]}: {model_table_label(model)}" for model, label, _, _ in selected)
                ax.text(.97, .97, names, transform=ax.transAxes, ha="right", va="top", fontsize=8,
                        bbox={"facecolor": "white", "edgecolor": "#d9d9d9", "alpha": .88})
            ax.set(title=dataset_labels[dataset], xlabel="Training data (%)", xlim=(0, 105), yscale="log")
            ax.set_ylim(8e-4, .12 if time_comparison else 1)
            ax.set_xticks([5, 25, 50, 75, 100])
            ax.grid(True, which="major", color="#e7e7e7", linewidth=.7)
            ax.grid(True, which="minor", color="#f0f0f0", linewidth=.45)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Test RMSE")
        if handles:
            fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=len(handles), frameon=False)
        stem = "rq1_3_nfde_time_input_data_fraction_rmse" if time_comparison else "rq1_3_fractional_best_family_data_fraction_rmse"
        save_figure(fig, output_dir / stem)


def main(dry_run=False):
    if dry_run:
        print(f"Source: {EVALUATION_SOURCE}\nReference training: {reference_root}\nReference curves: {reference_curve_root}\nFixed NFDE: {nfde_root}\nOutput: {output_dir}")
        print("Table 8.11; figures 8.8 and 8.9. Four seeds: 42, 1, 2, 3.")
        return 0
    full, curves = read_results()
    full_summary, curve_summary = aggregate_results(full), aggregate_results(curves)
    selection = select_families(curve_summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("full_budget_seeds", full), ("full_budget_summary", full_summary),
                        ("data_fraction_seeds", curves), ("data_fraction_summary", curve_summary), ("family_selection", selection)):
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    write_full_table(full_summary)
    if make_figures:
        plot_results(curve_summary, selection)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    raise SystemExit(main(parser.parse_args().dry_run))
