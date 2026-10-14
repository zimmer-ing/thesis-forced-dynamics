"""RQ1.1 / RQ2.2: prediction accuracy and training-data efficiency."""

import argparse
import json
import math
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Constants import EVALUATION_RESULTS_PATH, EVALUATION_SOURCE, OUTPUT_PATH
from Scripts.utils.naming import DATASET_LABELS, MODEL_TABLE_LABELS, model_key, model_directory

training_path = EVALUATION_RESULTS_PATH / "full_data_hpo"
data_fraction_path = EVALUATION_RESULTS_PATH / "data_fraction"
output_path = OUTPUT_PATH / "pdf/Evaluation" / EVALUATION_SOURCE / "RQ1_1_RQ2_2"
synthetic_datasets = ["SMDSystem", "RLCDataset", "IndPenSim"]
real_datasets = ["CascadedTanks", "IndustrialRobotForward", "NanoDroneOOD", "NanoDroneRunSplit"]
datasets = synthetic_datasets + real_datasets
algorithms = [
    "PersistenceBaseline", "FC", "LSTM", "GRU", "TCN",
    "NO-Obs", "NO-Aug0", "NO-AugH", "NO-Lat", "LaplaceNet",
]
seeds = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
persistence_seeds = {dataset: (0 if EVALUATION_SOURCE == "new_runs" or dataset in ["CascadedTanks", "NanoDroneOOD", "NanoDroneRunSplit"] else 42)
                     for dataset in datasets}
fractions = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
fraction_seeds = [42, 1, 2, 3]
# The submitted RLC curves contain ten seeds; new curves use four.
fraction_seed_overrides = ({("RLCDataset", model): seeds for model in ["GRU", "NO-Aug0"]}
                          if EVALUATION_SOURCE == "thesis_results" else {})
selected_models = {
    "SMDSystem": {"baseline": "LSTM", "node": "NO-AugH", "laplace": "LaplaceNet"},
    "RLCDataset": {"baseline": "GRU", "node": "NO-Aug0", "laplace": "LaplaceNet"},
    "IndPenSim": {"baseline": "GRU", "node": "NO-Lat", "laplace": "LaplaceNet"},
    "CascadedTanks": {"baseline": "GRU", "node": "NO-AugH", "laplace": "LaplaceNet"},
    "IndustrialRobotForward": {"baseline": "FC", "node": "NO-Aug0", "laplace": "LaplaceNet"},
    "NanoDroneOOD": {"baseline": "GRU", "node": "NO-AugH", "laplace": "LaplaceNet"},
    "NanoDroneRunSplit": {"baseline": "GRU", "node": "NO-AugH", "laplace": "LaplaceNet"},
}
# The submitted full-data RLC comparison uses the HPO-checked fraction runs.
rlc_full_source = "data_fraction" if EVALUATION_SOURCE == "thesis_results" else "training"
rlc_full_models = ["GRU", "NO-Obs", "NO-Prior", "NO-Aug0", "NO-AugH", "NO-Lat"]
baseline_models = ["FC", "LSTM", "GRU", "TCN"]
node_models = ["NO-Obs", "NO-Aug0", "NO-AugH", "NO-Lat"]
evidence = dict(zip(datasets, ["E1", "E2a", "E2b", "E3a", "E3a", "E3a", "E3a"]))
state_coverage = {dataset: ("full" if dataset in ["RLCDataset", "NanoDroneOOD", "NanoDroneRunSplit"] else "partial")
                  for dataset in datasets}


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def rmse(data, split):
    block = ((data.get("metrics") or {}).get(split) or {}).get("global") or {}
    value = block.get("rmse")
    if value is None:
        mse = block.get("mse", data.get(f"residuals_{split}"))
        value = math.sqrt(mse) if isinstance(mse, (int, float)) and mse >= 0 else None
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 else float("nan")


def read_result(source, dataset, model, seed, percent, expected_n, source_kind, training_root=None):
    """Read one planned cell, retaining missing/failed cells and their provenance."""
    source = Path(source)
    model = model_key(model)
    data = read_json(source)
    run_path = model_directory(training_path if training_root is None else training_root, dataset, model) / f"Seed_{seed}"
    status = [data.get("evaluation_status") or {}, data.get("status") or {}]
    status += [(data.get("metrics") or {}).get(split, {}).get("status") or {} for split in ["train", "val", "test"]]
    success = bool(data) and data.get("success") is not False and not any(s.get("success") is False for s in status)
    test_rmse, val_rmse = rmse(data, "test"), rmse(data, "val")
    success = success and math.isfinite(test_rmse)
    markers = set()
    if not success:
        failure = read_json(run_path / "failure.json") if source_kind == "training" else {}
        for item in status + [failure.get("status", failure)]:
            text = json.dumps(item).lower()
            solver = item.get("failure_type") == "solver_failure" or "underflow in dt" in text
            if solver:
                markers.add("N")
            if not solver and any(word in text for word in ["diverg", "nan", "non-finite", "nonfinite", "exceeded max_abs_prediction"]):
                markers.add("D")
    return {
        "dataset": dataset, "model": model, "seed": seed, "percent": percent,
        "success": success, "test_rmse": test_rmse if success else float("nan"),
        "val_rmse": val_rmse if success else float("nan"), "expected_n": expected_n,
        "failure_markers": ",".join(sorted(markers)), "source_path": str(source), "source_kind": source_kind,
        "run_path": str(run_path), "best_hparams_path": str(run_path.parent / "best_hparams_seed_42.json"),
        "model_config_path": str(run_path / "model_config.yaml"), "training_info_path": str(run_path / "training_info.yaml"),
        "training_info": data.get("training_info"), "model_config": data.get("model_config"),
    }


def read_training_result(dataset, model, seed, expected_n, training_root=None):
    root = Path(training_path if training_root is None else training_root)
    model = model_key(model)
    run = model_directory(root, dataset, model) / f"Seed_{seed}"
    source = run / "summary.json"
    summary = read_json(source)
    if summary.get("success") is False or any((summary.get(key) or {}).get("success") is False for key in ["status", "evaluation_status"]):
        return read_result(source, dataset, model, seed, 100, expected_n, "training", root)
    for filename in ["evaluation_results.json", "summary.json"]:
        candidate = run / filename
        data = read_json(candidate)
        if data:
            source = candidate
            if math.isfinite(rmse(data, "test")) or data.get("success") is False or any((data.get(key) or {}).get("success") is False for key in ["status", "evaluation_status"]):
                break
    return read_result(source, dataset, model, seed, 100, expected_n, "training", root)


def read_rlc_full_result(model, seed, expected_n, training_root=None, data_fraction_root=None):
    root = Path(training_path if training_root is None else training_root)
    fractions_root = Path(data_fraction_path if data_fraction_root is None else data_fraction_root)
    model = model_key(model)
    source = fractions_root / "RLCDataset/raw" / f"{model}_seed{seed}_100pct.json"
    row = read_result(source, "RLCDataset", model, seed, 100, expected_n, "data_fraction", root)
    if not source.exists() or not row["success"]:
        return row
    data = read_json(source)
    model_dir = model_directory(root, "RLCDataset", model)
    hparams = read_json(model_dir / "best_hparams_seed_42.json")
    reference_path = model_dir / "Seed_42/model_config.yaml"
    if not hparams or not reference_path.exists():
        raise ValueError(f"Missing reference HPO/config for {source}")
    reference = yaml.safe_load(reference_path.read_text(encoding="utf-8"))
    meta = data.get("meta") or {}
    if (meta.get("dataset") != "RLCDataset" or model_key(meta.get("model")) != model
            or model_key(meta.get("result_model", model)) != model or meta.get("seed") != seed
            or meta.get("train_fraction") != 1.0 or not meta.get("train_full_size")
            or meta.get("train_subset_size") != meta.get("train_full_size")):
        raise ValueError(f"Inconsistent RLC full-data metadata: {source}")
    if data.get("best_hyperparameters") != hparams:
        raise ValueError(f"RLC HPO mismatch: {source}")
    config = data.get("model_config")
    if meta.get("reused_existing_model") is True:
        config = yaml.safe_load((model_dir / f"Seed_{seed}/model_config.yaml").read_text(encoding="utf-8"))
    actual = (config or {}).get("model") or {}
    if any(actual.get(key) != value for key, value in hparams.items()):
        raise ValueError(f"RLC model config does not match HPO: {source}")
    for section, keys in [("dataset", ["history_length", "forecast_length", "stride", "dim_x", "dim_y"]),
                          ("experiment", ["num_epochs", "batch_size"])]:
        expected = reference.get(section) or {}
        actual = (config or {}).get(section) or {}
        if any(key in expected and actual.get(key) != expected[key] for key in keys):
            raise ValueError(f"RLC {section} config differs: {source}")
    row["model_config"] = config
    return row


def read_main_results(dataset_names=None, model_names=None, training_root=None, data_fraction_root=None, rlc_source=None, seed_values=None):
    training_root = Path(training_path if training_root is None else training_root)
    data_fraction_root = Path(data_fraction_path if data_fraction_root is None else data_fraction_root)
    rlc_source = rlc_full_source if rlc_source is None else rlc_source
    seed_values = seeds if seed_values is None else seed_values
    if rlc_source not in ["data_fraction", "training"]:
        raise ValueError("rlc_full_source must be 'data_fraction' or 'training'")
    records = []
    for dataset in datasets if dataset_names is None else dataset_names:
        for model in algorithms if model_names is None else model_names:
            model = model_key(model)
            if model == "NO-Prior" and dataset != "RLCDataset":
                continue
            planned = [persistence_seeds.get(dataset, 42)] if model == "PersistenceBaseline" else seed_values
            for seed in planned:
                if dataset == "RLCDataset" and model in rlc_full_models and rlc_source == "data_fraction":
                    row = read_rlc_full_result(model, seed, len(planned), training_root, data_fraction_root)
                else:
                    row = read_training_result(dataset, model, seed, len(planned), training_root)
                records.append(row)
    return pd.DataFrame(records)


def read_data_fraction_results():
    records = []
    for dataset in datasets:
        for role, model in selected_models[dataset].items():
            model = model_key(model)
            planned = fraction_seed_overrides.get((dataset, model), fraction_seeds)
            for percent in fractions:
                for seed in planned:
                    source = data_fraction_path / dataset / "raw" / f"{model}_seed{seed}_{percent:03d}pct.json"
                    row = read_result(source, dataset, model, seed, percent, len(planned), "data_fraction")
                    row["role"] = role
                    records.append(row)
    return pd.DataFrame(records)


def aggregate_results(records):
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=["dataset", "model", "percent", "n", "expected_n", "failure_markers",
                                     "val_rmse_mean", "val_rmse_std", "test_rmse_mean", "test_rmse_std"])
    keys = ["dataset", "model", "percent"]
    if frame.duplicated(keys + ["seed"]).any():
        raise ValueError("Duplicate dataset/model/seed/fraction results")
    rows = []
    for key, group in frame.groupby(keys, sort=False):
        if group.expected_n.nunique() != 1:
            raise ValueError(f"Conflicting planned seed count: {key}")
        valid = group[group.success & group.test_rmse.map(math.isfinite)]
        row = dict(zip(keys, key))
        row.update(n=len(valid), expected_n=int(group.expected_n.iloc[0]),
                   failure_markers=",".join(sorted({m for value in group.failure_markers for m in value.split(",") if m})))
        for split in ["val", "test"]:
            values = valid[f"{split}_rmse"].dropna()
            row[f"{split}_rmse_mean"] = values.mean()
            row[f"{split}_rmse_std"] = values.std(ddof=1) if len(values) > 1 else (0.0 if len(values) == 1 else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def model_label(model):
    return "Laplace-Net" if model == "LaplaceNet" else MODEL_TABLE_LABELS.get(model, model)


def format_number(value, digits=2):
    if not math.isfinite(value):
        return "--"
    if value == 0:
        return "0"
    if abs(value) < .001 or abs(value) >= 1000:
        mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
        return rf"{mantissa}\,10^{{{int(exponent)}}}"
    decimals = max(0, digits - 1 - math.floor(math.log10(abs(value))))
    return f"{value:.{decimals}f}"


def metric_cell(row, bold=False, italic=False):
    mean = format_number(row.test_rmse_mean)
    if bold:
        mean = rf"\mathbf{{{mean}}}"
    elif italic:
        mean = rf"\mathit{{{mean}}}"
    result = rf"${mean} \pm {format_number(row.test_rmse_std, 3)}$"
    if row.failure_markers:
        result += rf"\textsuperscript{{{row.failure_markers}}}"
    if row.n < row.expected_n:
        result += f" [{int(row.n)}/{int(row.expected_n)}]"
    return result


def write_table(path, headers, rows, caption, label):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [r"\begin{table}[!htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begingroup\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.08}",
             r"\begin{adjustbox}{max width=\columnwidth}", r"\begin{tabular}{@{}" + "l" * len(headers) + r"@{}}",
             r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    lines += [" & ".join(map(str, row)) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\endgroup", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def comparison_tables(aggregates):
    compact, observability = [], []
    for dataset in datasets:
        frame = aggregates[(aggregates.dataset == dataset) & (aggregates.percent == 100)].set_index("model")
        best = frame.test_rmse_mean.min()
        row = {"dataset": dataset, "best_overall_rmse": best}
        for role, models in [("baseline", ["PersistenceBaseline"] + baseline_models), ("node", node_models), ("laplace", ["LaplaceNet"])]:
            candidates = frame.loc[frame.index.intersection(models)].dropna(subset=["test_rmse_mean"])
            if candidates.empty:
                row.update({f"{role}_model": "--", f"{role}_rmse": float("nan"), f"{role}_delta_percent": float("nan")})
                continue
            model = candidates.test_rmse_mean.idxmin()
            mean = candidates.loc[model, "test_rmse_mean"]
            row.update({f"{role}_model": model, f"{role}_rmse": mean, f"{role}_delta_percent": 100 * (mean / best - 1) if best > 0 else float("nan")})
        compact.append(row)
        baseline = frame.loc[frame.index.intersection(baseline_models)].dropna(subset=["test_rmse_mean"])
        node = frame.loc[frame.index.intersection(node_models)].dropna(subset=["test_rmse_mean"])
        if not baseline.empty and not node.empty and "NO-Obs" in frame.index:
            b, n = baseline.test_rmse_mean.idxmin(), node.test_rmse_mean.idxmin()
            bm, nm, obs = frame.loc[b, "test_rmse_mean"], frame.loc[n, "test_rmse_mean"], frame.loc["NO-Obs", "test_rmse_mean"]
            observability.append({"dataset": dataset, "state_coverage": state_coverage[dataset], "baseline_model": b,
                                  "node_model": n, "baseline_rmse": bm, "node_rmse": nm, "no_obs_rmse": obs,
                                  "no_obs_gap_percent": 100 * (obs / nm - 1) if nm > 0 else float("nan"),
                                  "node_baseline_delta_percent": 100 * (nm / bm - 1) if bm > 0 else float("nan")})
    return pd.DataFrame(compact), pd.DataFrame(observability)


def write_accuracy_tables(aggregates, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    compact, observability = comparison_tables(aggregates)
    for kind, selected in [("synthetic", synthetic_datasets), ("real", real_datasets)]:
        full = aggregates[aggregates.dataset.isin(selected)]
        stem = f"training_full_{kind}_table"
        full.to_csv(destination / f"{stem}.csv", index=False)
        rows = []
        for model in algorithms:
            cells = [model_label(model)]
            for dataset in selected:
                frame = full[full.dataset == dataset].set_index("model")
                ranked = frame.test_rmse_mean.dropna().sort_values().index.tolist()
                cells.append(metric_cell(frame.loc[model], model in ranked[:1], model in ranked[1:2]))
            rows.append(cells)
        write_table(destination / f"{stem}.tex", ["Model"] + [f"{DATASET_LABELS[d]} ({evidence[d]})" for d in selected], rows,
                    r"Test RMSE (mean $\pm$ sample standard deviation). Best means are bold, second-best italic. D/N denote divergence/numerical failures; incomplete coverage is shown as successful/planned seeds.",
                    f"tab:eval_pb1_{kind}")
        sub = compact[compact.dataset.isin(selected)]
        stem = f"training_compact_{kind}_table"
        sub.to_csv(destination / f"{stem}.csv", index=False)
        rows = []
        for _, row in sub.iterrows():
            cells = [f"{DATASET_LABELS[row.dataset]} ({evidence[row.dataset]})"]
            for role in ["baseline", "node", "laplace"]:
                mean = row[f"{role}_rmse"]
                number = format_number(mean)
                if mean == row.best_overall_rmse:
                    number = rf"\mathbf{{{number}}}"
                cells.append(f"{model_label(row[f'{role}_model'])}: ${number}$ ({row[f'{role}_delta_percent']:+.3g}\\%)")
            rows.append(cells)
        write_table(destination / f"{stem}.tex", ["Dataset", "Best baseline", "Best NODE variant", "Laplace-Net"], rows,
                    "Best model per family by mean test RMSE; percentages are relative to the best overall mean.", f"tab:eval_pb1_compact_{kind}")
    stem = "training_node_baseline_observability_table"
    observability.to_csv(destination / f"{stem}.csv", index=False)
    rows = []
    for coverage in ["partial", "full"]:
        for _, row in observability[observability.state_coverage == coverage].iterrows():
            rows.append([coverage, DATASET_LABELS[row.dataset], f"{model_label(row.baseline_model)}: ${format_number(row.baseline_rmse)}$",
                         f"{model_label(row.node_model)}: ${format_number(row.node_rmse)}$", f"${format_number(row.no_obs_rmse)}$",
                         f"{row.no_obs_gap_percent:+.3g}\\%", f"{row.node_baseline_delta_percent:+.3g}\\%"])
    write_table(destination / f"{stem}.tex", ["State coverage", "Dataset", "Best baseline", "Best NODE", "NO-Obs", "NO-Obs gap", r"$\Delta_{\rm NODE-baseline}$"], rows,
                "Mean test RMSE by state coverage. Gaps compare NO-Obs to the best NODE and the best NODE to the best discrete baseline.", "tab:eval_pb1_node_baseline_observability")


def plot_data_efficiency(aggregates, destination):
    import matplotlib.pyplot as plt
    from Scripts.utils.evaluation import configure_plot, save_figure
    configure_plot()
    styles = {"baseline": ("#4d4d4d", "s", "Baseline"), "node": ("#1f77b4", "^", "NODE"), "laplace": ("#D55E00", "o", "Laplace-Net")}
    for kind, selected, shape in [("synthetic", synthetic_datasets, (1, 3)), ("real", real_datasets, (2, 2))]:
        fig, axes = plt.subplots(*shape, figsize=(5.9, 2.6 if kind == "synthetic" else 4.7), squeeze=False)
        for axis, dataset in zip(axes.flat, selected):
            for role, model in selected_models[dataset].items():
                frame = aggregates[(aggregates.dataset == dataset) & (aggregates.model == model)].sort_values("percent")
                frame = frame[frame.percent.isin(fractions[::2] + [fractions[-1]])]
                color, marker, label = styles[role]
                axis.errorbar(frame.percent, frame.test_rmse_mean, yerr=frame.test_rmse_std,
                              color=color, marker=marker, markersize=3, linewidth=1.1, capsize=1.7, label=label)
            axis.set(title=DATASET_LABELS[dataset], xlabel="Training data (%)", ylabel="Test RMSE", xticks=[5, 20, 40, 60, 80, 100])
            axis.grid(alpha=.25)
            axis.tick_params(labelsize=6.5)
            axis.text(.5, -.39, f"Baseline: {model_label(selected_models[dataset]['baseline'])} | NODE: {model_label(selected_models[dataset]['node'])}",
                      transform=axis.transAxes, ha="center", fontsize=5.9)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        fig.subplots_adjust(left=.09, right=.99, top=.91, bottom=.34 if kind == "synthetic" else .18, wspace=.43, hspace=.8)
        save_figure(fig, Path(destination) / f"data_fraction_{kind}_data_efficiency_rmse")


def main(dry_run=False):
    records = read_main_results()
    curves = read_data_fraction_results()
    main_stats, curve_stats = aggregate_results(records), aggregate_results(curves)
    print(f"Full data ({EVALUATION_SOURCE}): {int(records.success.sum())}/{len(records)} successful cells; RLC source={rlc_full_source}")
    print(f"Data fractions: {int(curves.success.sum())}/{len(curves)} successful cells. Output: {output_path}")
    if dry_run:
        return
    table_path = output_path / "tables"
    table_path.mkdir(parents=True, exist_ok=True)
    records.to_csv(table_path / "main_seed_results.csv", index=False)
    main_stats.to_csv(table_path / "main_aggregate.csv", index=False)
    curves.to_csv(table_path / "data_fraction_seed_results.csv", index=False)
    curve_stats.to_csv(table_path / "data_fraction_aggregate.csv", index=False)
    write_accuracy_tables(main_stats, table_path)
    plot_data_efficiency(curve_stats, output_path / "plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Read and validate results, without writing tables or plots.")
    main(dry_run=parser.parse_args().dry_run)
