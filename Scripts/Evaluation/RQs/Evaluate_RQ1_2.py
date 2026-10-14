"""RQ1.2: partial physical prior knowledge on the RLC dataset."""

import argparse
import math
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Constants import EVALUATION_RESULTS_PATH, EVALUATION_SOURCE, OUTPUT_PATH
from Scripts.Evaluation.RQs.Evaluate_RQ1_1_RQ2_2 import (
    aggregate_results, format_number, model_label, read_json, read_training_result,
    read_result, read_rlc_full_result, write_table,
)
from Scripts.utils.naming import MODEL_COLORS, MODEL_MARKERS, model_key, model_directory

training_path = EVALUATION_RESULTS_PATH / "full_data_hpo"
data_fraction_path = EVALUATION_RESULTS_PATH / "data_fraction"
ablation_path = model_directory(EVALUATION_RESULTS_PATH / "physics_components", "RLCDataset", "NO-Prior") / "Seed_42/rlc_no_prior_physics_only_channel_ablation.json"
output_path = OUTPUT_PATH / "pdf/Evaluation" / EVALUATION_SOURCE / "RQ1_2"
dataset = "RLCDataset"
algorithms = ["GRU", "NO-Obs", "NO-Prior", "NO-Aug0", "NO-AugH"]
plot_algorithms = ["GRU", "NO-Obs", "NO-Prior", "NO-Aug0"]
seeds = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
fraction_seeds = seeds if EVALUATION_SOURCE == "thesis_results" else [42, 1, 2, 3]
fractions = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
plot_fractions = [5, 10, 20, 40, 60, 80, 100]
rlc_full_source = "data_fraction" if EVALUATION_SOURCE == "thesis_results" else "training"
ablation_variants = {
    "physics_only_initial_greybox_params": "Physics only, initial parameters",
    "physics_only_with_learned_physics_params": "Physics only, learned parameters",
    "nn_only_with_learned_weights": "Learned neural term only",
    "full_physics_plus_nn": "NO-Prior: physics + neural term",
}


def read_results():
    if rlc_full_source not in ["data_fraction", "training"]:
        raise ValueError("rlc_full_source must be 'data_fraction' or 'training'")
    full, curves = [], []
    for model in algorithms:
        model = model_key(model)
        for seed in seeds:
            if rlc_full_source == "data_fraction":
                row = read_rlc_full_result(model, seed, len(seeds), training_path, data_fraction_path)
            else:
                row = read_training_result(dataset, model, seed, len(seeds), training_path)
            full.append(row)
        for percent in fractions:
            for seed in fraction_seeds:
                source = data_fraction_path / dataset / "raw" / f"{model}_seed{seed}_{percent:03d}pct.json"
                curves.append(read_result(source, dataset, model, seed, percent, len(fraction_seeds), "data_fraction", training_path))
    return pd.DataFrame(full), pd.DataFrame(curves)


def read_ablation():
    source = ablation_path
    if not source.exists() and source.name == "rlc_no_prior_physics_only_channel_ablation.json":
        source = source.with_name("rlc_m2_physics_only_channel_ablation.json")
    if not source.exists():
        raise FileNotFoundError(f"Missing component evaluation: {ablation_path}. Generate it with Evaluate_Physics.py.")
    data = read_json(source)
    if data.get("dataset") != dataset or data.get("split") != "test":
        raise ValueError(f"Expected an RLC test-split component evaluation: {ablation_path}")
    rows = []
    legacy_keys = {
        "physics_only_with_learned_physics_params": "physics_only_with_m2_learned_physics_params",
        "nn_only_with_learned_weights": "nn_only_with_m2_learned_weights",
        "full_physics_plus_nn": "m2_full_physics_plus_nn",
    }
    for variant in ablation_variants:
        source_variant = variant if variant in data["metrics"] else legacy_keys.get(variant, variant)
        for group in ["symbolic_channels", "internal_channels", "all_channels"]:
            values = data["metrics"][source_variant]["groups"][group]
            if not math.isfinite(values["rmse"]) or values["rmse"] < 0:
                raise ValueError(f"Invalid component RMSE: {variant}/{group}")
            rows.append({"variant": variant, "group": group, **values, "source_path": str(source)})
    return pd.DataFrame(rows)


def write_prior_tables(full, ablation, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    full = full.sort_values("test_rmse_mean", na_position="last")
    full.to_csv(destination / "rq1_2_rlc_prior_summary_table.csv", index=False)
    minimum = full.test_rmse_mean.min()
    rows = []
    for _, row in full.iterrows():
        mean = format_number(row.test_rmse_mean, 3)
        if row.test_rmse_mean == minimum:
            mean = rf"\mathbf{{{mean}}}"
        metric = rf"${mean} \pm {format_number(row.test_rmse_std, 2)}$"
        rows.append([model_label(row.model), metric, f"{int(row.n)}/{int(row.expected_n)}"])
    write_table(destination / "rq1_2_rlc_prior_summary_table.tex", ["Model", "Test RMSE", "Seeds"], rows,
                "Test RMSE for the RLC dataset (E2a): NO-Prior and models without a physical component. Mean and sample standard deviation over successful seeds.",
                "tab:rq12_rlc_prior_summary")
    ablation.to_csv(destination / "rlc_prior_physics_only_channel_ablation_table.csv", index=False)
    rows = []
    for variant, label in ablation_variants.items():
        group = ablation[ablation.variant == variant].set_index("group")
        values = [format_number(group.loc[key, "rmse"], 4) for key in ["all_channels", "symbolic_channels"]]
        if variant == "full_physics_plus_nn":
            values = [rf"\mathbf{{{value}}}" for value in values]
        rows.append([label] + [f"${value}$" for value in values])
    write_table(destination / "rlc_prior_physics_only_channel_ablation_table.tex",
                ["Component", "All channels", r"Symbolic channels ($V_{out}, I_{in}$)"], rows,
                "RLC component ablation for the saved Seed 42 model without retraining. RMSE is computed jointly over the channels in each group.",
                "tab:rlc_prior_physics_only_ablation")


def plot_data_efficiency(aggregates, destination):
    import matplotlib.pyplot as plt
    from Scripts.utils.evaluation import configure_plot, save_figure
    configure_plot()
    fig, axis = plt.subplots(figsize=(5.9, 2.65))
    for model in plot_algorithms:
        frame = aggregates[(aggregates.model == model) & aggregates.percent.isin(plot_fractions)].sort_values("percent")
        axis.errorbar(frame.percent, frame.test_rmse_mean, yerr=frame.test_rmse_std,
                      color=MODEL_COLORS[model], marker=MODEL_MARKERS[model], markersize=4.4,
                      linewidth=1.35, capsize=2.2, label=model_label(model))
    axis.set(xlabel="Training data (%)", ylabel="Test RMSE", xticks=[5, 20, 40, 60, 80, 100])
    axis.grid(alpha=.25)
    fig.legend(*axis.get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(left=.1, right=.99, top=.96, bottom=.30)
    save_figure(fig, Path(destination) / "rq1_2_rlc_prior_data_budget_rmse")


def main(dry_run=False):
    full, curves = read_results()
    full_stats, curve_stats = aggregate_results(full), aggregate_results(curves)
    ablation = read_ablation()
    print(f"RLC full data ({rlc_full_source}): {int(full.success.sum())}/{len(full)} successful cells")
    print(f"RLC fractions: {int(curves.success.sum())}/{len(curves)}; ablation: {len(ablation)} groups. Output: {output_path}")
    if dry_run:
        return
    tables = output_path / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    full.to_csv(tables / "full_seed_results.csv", index=False)
    curves.to_csv(tables / "data_fraction_seed_results.csv", index=False)
    curve_stats.to_csv(tables / "data_fraction_aggregate.csv", index=False)
    write_prior_tables(full_stats, ablation, tables)
    plot_data_efficiency(curve_stats, output_path / "plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Read and validate results, without writing tables or plots.")
    main(dry_run=parser.parse_args().dry_run)
