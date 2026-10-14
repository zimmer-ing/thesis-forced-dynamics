"""Rebuild appendix tables D.1–D.7 from prepared data and saved runs.

Edit the paths and lists below. No dataset or model is constructed. Statistical
tables reuse the same small readers as the main research-question scripts.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Constants import DATA_PATH, EVALUATION_RESULTS_PATH, EVALUATION_SOURCE, OUTPUT_PATH
from Scripts.utils.naming import dataset_label, model_table_label, hyperparameter_label, model_key, model_directory
from src.datasets import DATASET_REGISTRY

data_root = DATA_PATH
config_root = ROOT / "configs"
training_root = EVALUATION_RESULTS_PATH / "full_data_hpo"
data_fraction_root = EVALUATION_RESULTS_PATH / "data_fraction"
rlc_full_source = "data_fraction" if EVALUATION_SOURCE == "thesis_results" else "training"
# D.5 uses the same central source selection in Evaluate_RQ1_3.py.
output_dir = OUTPUT_PATH / "pdf/Evaluation" / EVALUATION_SOURCE / "Appendix"
datasets = ["SMDSystem", "RLCDataset", "IndPenSim", "CascadedTanks",
            "IndustrialRobotForward", "NanoDroneOOD", "NanoDroneRunSplit"]
algorithms = ["FC", "LSTM", "GRU", "TCN",
              "NO-Obs", "NO-Aug0", "NO-AugH", "NO-Lat", "LaplaceNet"]
seeds = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def pair_directory(dataset, model):
    return model_directory(training_root, dataset, model)


def dataset_windows():
    """D.1: count windows from prepared train CSVs, using their time columns."""
    rows = []
    for dataset in datasets + ["FractionalBenchmarkForcing"]:
        config = yaml.safe_load((config_root / "datasets" / f"{dataset}.yaml").read_text())
        time_column, multiplier = "time_s", 1
        if dataset == "SMDSystem":
            files, time_column = sorted((data_root / "SMD_Dataset/train").glob("*.csv")), "time"
        elif dataset == "RLCDataset":
            files, time_column = [data_root / "4th_order_RLC/train/train_data.csv.gz"], "time"
        elif dataset == "IndPenSim":
            batches = list(range(1, 11)) + list(range(30, 41)) + list(range(61, 71))
            files = [data_root / f"IndPenSim/minmax_scaled/Batch_{batch}.csv" for batch in batches]
            time_column, multiplier = "Time (h)", 3600
        elif dataset == "FractionalBenchmarkForcing":
            files = sorted((data_root / "Fractional_Data/fractional_benchmark_data/forcing/train").glob("traj_*.csv"))
            time_column = "t"
        else:
            directory = (data_root / "IndustrialRobot/forward_minmax" if dataset == "IndustrialRobotForward"
                         else data_root / "NonlinearBenchmarks" / dataset / "minmax")
            files = [directory / name for name in DATASET_REGISTRY[dataset].TRAIN_FILES]
        if not files:
            raise FileNotFoundError(f"No prepared training files for {dataset}")
        history, forecast = int(config["history_length"]), config.get("forecast_length")
        count, first_time = 0, None
        for path in files:
            time = pd.read_csv(path, usecols=[time_column])[time_column].to_numpy()
            if config.get("sample_mode") == "full_trajectory":
                forecast = len(time) - history
                count += 1
            else:
                stride = int(config["stride"])
                if stride == -1:
                    stride = int(forecast)
                count += max(0, (len(time) - history - int(forecast)) // stride + 1)
            if first_time is None:
                # Saved datasets use float32 times; preserve their rounded spacing.
                first_time = np.asarray(time[:history + int(forecast)], dtype=np.float32).astype(float)
        differences = np.abs(np.diff(first_time))
        differences = differences[np.isfinite(differences) & (differences > 1e-12)]
        dt = float(np.median(differences)) * multiplier
        if dt >= 1 and abs(dt - round(dt)) < .1:
            dt = float(round(dt))
        batch = int(config["batch_size"])
        rows.append({"dataset": dataset, "dt": dt, "n_history": history, "n_forecast": int(forecast),
                     "history_time": history * dt if history > 1 else np.nan,
                     "forecast_time": forecast * dt, "batch_size": batch,
                     "train_samples": count, "train_batches": math.ceil(count / batch)})
    return pd.DataFrame(rows)


def hyperparameters():
    """D.2: prefer the search space stored with each selected checkpoint."""
    rows = []
    for dataset in datasets:
        for model in algorithms + (["NO-Prior"] if dataset == "RLCDataset" else []):
            model = model_key(model)
            pair = pair_directory(dataset, model)
            best_path = pair / "best_hparams_seed_42.json"
            if not best_path.is_file():
                raise FileNotFoundError(best_path)
            best = json.loads(best_path.read_text())
            snapshot = pair / "Seed_42/config_experiment.yaml"
            if snapshot.is_file():
                config = yaml.safe_load(snapshot.read_text())["model"]
            else:
                snapshot = config_root / "models" / f"{model}.yaml"
                config = yaml.safe_load(snapshot.read_text())
                config.update(config.pop("dataset_overrides", {}).get(dataset, {}))
            for parameter, value in best.items():
                entry = config.get(parameter, {})
                space = entry.get("search_space", {}) if isinstance(entry, dict) else {}
                if space.get("type") == "categorical":
                    search = ", ".join(str(choice) for choice in space["choices"])
                elif "low" in space:
                    search = f"[{space['low']}, {space['high']}]"
                else:
                    search = str(entry.get("default", "fixed") if isinstance(entry, dict) else entry)
                scale = "log" if space.get("log") or space.get("type") == "loguniform" else str(space.get("step", "--" if space else "fixed"))
                rows.append({"dataset": dataset, "model": model, "parameter": parameter,
                             "search_space": search, "step_scale": scale, "value": value,
                             "config_source": str(snapshot.relative_to(ROOT)) if snapshot.is_relative_to(ROOT) else str(snapshot)})
    return pd.DataFrame(rows)


def training_epochs(records):
    """D.3: best epoch from the exact full-budget cohort used in D.4."""
    rows = []
    for record in records.to_dict("records"):
        record["model"] = model_key(record["model"])
        if record["model"] not in algorithms or record["seed"] not in seeds:
            continue
        info = {}
        if record["source_kind"] == "data_fraction":
            payload = json.loads(Path(record["source_path"]).read_text())
            if not payload.get("meta", {}).get("reused_existing_model"):
                info = payload.get("training_info", {})
        if not info:
            run = pair_directory(record["dataset"], record["model"]) / f"Seed_{record['seed']}"
            path = run / "training_info.yaml"
            if path.is_file():
                info = yaml.safe_load(path.read_text()) or {}
            if info.get("best_epoch") is None:
                losses = run / "training_logs/val_losses.yaml"
                if losses.is_file():
                    values = np.asarray(yaml.safe_load(losses.read_text()), dtype=float)
                    if len(values) and np.isfinite(values).any():
                        info["best_epoch"] = int(np.nanargmin(values)) + 1
            if info.get("total_epochs") is None and (run / "config_experiment.yaml").is_file():
                info["total_epochs"] = yaml.safe_load((run / "config_experiment.yaml").read_text())["experiment"]["num_epochs"]
        if info.get("best_epoch") is not None:
            rows.append({"dataset": record["dataset"], "model": record["model"], "seed": record["seed"],
                         "best_epoch": info["best_epoch"], "total_epochs": info.get("total_epochs")})
    frame = pd.DataFrame(rows)
    return frame.groupby(["dataset", "model"], sort=False).agg(
        best_epoch_mean=("best_epoch", "mean"), best_epoch_std=("best_epoch", "std"),
        n=("best_epoch", "count"), final_epochs=("total_epochs", "max")).reset_index()


def checkpoint_counts(path, model):
    """Count CPU checkpoint tensors, excluding BatchNorm buffers and ODE scale."""
    import torch
    model = model_key(model)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    sizes = {name: tensor.numel() for name, tensor in state.items()
             if hasattr(tensor, "numel") and not name.endswith(("running_mean", "running_var", "num_batches_tracked"))
             and name != "ODE.scaling_factor"}
    if model == "LaplaceNet":
        prefix, role = "H_s_net.", "dynamic response operator"
    elif model.startswith("NO-"):
        prefix, role = "ODE.", "ODE right-hand side"
    elif model in {"LSTM", "GRU"}:
        prefix, role = "decoder.", "recurrent decoder"
    else:
        prefix, role = "net.", "fixed-horizon predictor"
    total = sum(sizes.values())
    core = sum(size for name, size in sizes.items() if name.startswith(prefix))
    return total, core, role


def parameter_counts():
    """D.6/D.7: architecture sizes of the selected Seed-42 checkpoints."""
    from Scripts.Evaluation.RQs.Evaluate_RQ2_3 import algorithms_by_dataset
    rows, core_rows = [], []
    for dataset in datasets:
        for model in algorithms:
            run = pair_directory(dataset, model) / "Seed_42"
            metadata = run / "parameter_counts.json"
            if metadata.is_file():
                counts = json.loads(metadata.read_text())
                total = counts.get("model_num_parameters", counts["total"])
            else:
                info = run / "training_info.yaml"
                total = (yaml.safe_load(info.read_text()) or {}).get("model_num_parameters") if info.is_file() else None
                if total is None:
                    total, _, _ = checkpoint_counts(run / "model_checkpoints/best_model.pth", model)
            rows.append({"dataset": dataset, "model": model, "parameters": total})
    for dataset, models in algorithms_by_dataset.items():
        for model in models:
            run = pair_directory(dataset, model) / "Seed_42"
            metadata = run / "parameter_counts.json"
            counts = json.loads(metadata.read_text()) if metadata.is_file() else {}
            if counts.get("core") is not None:
                total, core, role = counts["total"], counts["core"], counts["role"]
            else:
                total, core, role = checkpoint_counts(run / "model_checkpoints/best_model.pth", model)
            core_rows.append({"dataset": dataset, "model": model, "total": total, "core": core,
                              "core_share_percent": 100 * core / total, "role": role})
    return pd.DataFrame(rows), pd.DataFrame(core_rows)


def write_table(stem, caption, headers, rows):
    """Compact booktabs/longtable source, rendered by the shared TeX preview."""
    hparams = stem == "appendix_hyperparameter_tables"
    widths = ([1.4, 1.8, 1.05] + [1.35] * 7 if hparams
              else [2.0] + [max(.6, 12 / (len(headers) - 1))] * (len(headers) - 1))
    labels = {
        "appendix_dataset_window_table": "tab:appx_evaluation_window_lengths",
        "appendix_hyperparameter_tables": "tab:appx_hparams_overview",
        "appendix_final_training_epochs_table": "tab:appx_final_training_epochs",
        "appendix_full_budget_training": "tab:appendix_full_budget_pb1",
        "appendix_full_budget_fractional": "tab:appendix_full_budget_fractional",
        "appendix_parameter_counts": "tab:appx_p5_parameter_counts",
        "rq2_3_parameter_core_table": "tab:rq2_3_p5_parameter_core",
    }
    if hparams:
        headers = headers[1:]
    header_breaks = {
        "Step/scale": r"Step/\\scale", "IndPenSim": r"IndPen\\Sim",
        "Cascaded Tanks": r"Cascaded\\Tanks", "Industrial Robot": r"Industrial\\Robot",
        "NanoDrone 3": r"Nano\\Drone 3", "NanoDrone 4": r"Nano\\Drone 4",
    }
    headers = [r"\makecell[l]{" + header_breaks[h] + "}" if h in header_breaks else h for h in headers]
    heading = [r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    lines = [r"\begingroup", r"\scriptsize", r"\setlength{\tabcolsep}{2pt}",
             r"\renewcommand{\arraystretch}{1.08}",
             r"\begin{longtable}{" + " ".join(r">{\raggedright\arraybackslash}" + f"p{{{width:.2f}cm}}" for width in widths) + "}",
             r"\caption{" + caption + r"}\label{" + labels[stem] + r"}\\"]
    lines += heading + [r"\endfirsthead", r"\caption[]{" + caption + r" (continued)}\\"]
    lines += heading + [r"\endhead"]
    last_model = None
    for row in rows:
        if hparams:
            if row[0] != last_model:
                lines += [r"\addlinespace[2pt]", r"\multicolumn{10}{l}{\textbf{" + row[0] + r"}} \\*"]
                last_model = row[0]
            row = row[1:]
            values = {"LeakyReLU": r"\makecell[l]{Leaky\\ReLU}", r"leaky\_relu": r"\makecell[l]{leaky\_\\relu}"}
            row = row[:3] + [values.get(str(cell), cell) for cell in row[3:]]
        lines.append(" & ".join(str(cell) for cell in row) + r" \\")
    lines += [r"\bottomrule", r"\end{longtable}", r"\endgroup"]
    (output_dir / f"{stem}.tex").write_text("\n".join(lines) + "\n")


def write_budget_table(summary, stem, caption, dataset_order, model_order):
    rows = []
    for dataset in dataset_order:
        group = summary[summary.dataset == dataset]
        for model in model_order:
            if model == "NO-Prior" and dataset != "RLCDataset":
                continue
            match = group[group.model == model]
            nodes = group[group.model.isin(["NO-Obs", "NO-Obs-t", "NO-Aug0", "NO-Aug0-t", "NO-AugH", "NO-Lat"])]
            label = model_table_label(model)
            markers = ""
            for split, marker in [("val", "V"), ("test", "T")]:
                options = nodes.dropna(subset=[f"{split}_rmse_mean"]).sort_values([f"{split}_rmse_mean", "model"])
                if not options.empty and options.iloc[0].model == model:
                    markers += marker
            if markers:
                label += r"\textsuperscript{" + markers + "}"
            cells = [dataset_label(dataset), label]
            if match.empty:
                rows.append(cells + ["--", "--", "N/A"])
                continue
            item = match.iloc[0]
            for split in ["val", "test"]:
                mean, std = item[f"{split}_rmse_mean"], item[f"{split}_rmse_std"]
                value = "--" if pd.isna(mean) else f"{mean:.3g}" + (r" $\pm$ " + f"{std:.3g}" if pd.notna(std) else "")
                if pd.notna(mean) and math.isclose(mean, group[f"{split}_rmse_mean"].min(), rel_tol=0, abs_tol=1e-15):
                    value = r"\textbf{" + value + "}"
                cells.append(value)
            marker = str(item.get("failure_markers", ""))
            cells.append(f"{int(item.n)}/{int(item.expected_n)}" + (r"\textsuperscript{" + marker + "}" if marker else ""))
            rows.append(cells)
    write_table(stem, caption + ". V/T: best NODE by validation/test mean; D/N: divergence/numerical solver failure.", ["Dataset", "Model", "Validation RMSE", "Test RMSE", "$n/N$"], rows)


def main(dry_run=False):
    if dry_run:
        print(f"Source: {EVALUATION_SOURCE}\nPrepared data: {data_root}\nConfigurations: {config_root}\nTraining: {training_root}\nData fractions: {data_fraction_root}\nRLC full source: {rlc_full_source}\nOutput: {output_dir}")
        print("D.1 windows; D.2 HPO; D.3 epochs; D.4/D.5 full-budget metrics; D.6/D.7 parameter counts. No model/dataset constructors.")
        return 0
    from Scripts.Evaluation.RQs import Evaluate_RQ1_1_RQ2_2 as main_results
    from Scripts.Evaluation.RQs import Evaluate_RQ1_3 as fractional
    try:
        main_records = main_results.read_main_results(dataset_names=datasets, model_names=["PersistenceBaseline"] + algorithms + ["NO-Prior"],
                                                       training_root=training_root, data_fraction_root=data_fraction_root, rlc_source=rlc_full_source, seed_values=seeds)
        main_summary = main_results.aggregate_results(main_records)
        # A deterministic one-seed baseline has no sample standard deviation.
        main_summary.loc[main_summary.n <= 1, ["val_rmse_std", "test_rmse_std"]] = np.nan
        fractional_records, _ = fractional.read_results()
        fractional_summary = fractional.aggregate_results(fractional_records)
        windows, hparams, epochs = dataset_windows(), hyperparameters(), training_epochs(main_records)
        parameters, core = parameter_counts()
    except FileNotFoundError as error:
        missing = error.filename or str(error)
        print(f"Error: Appendix skipped; required input is missing: {missing}", file=sys.stderr)
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    for stem, frame in [("appendix_dataset_window_table", windows), ("appendix_hyperparameter_tables", hparams),
                        ("appendix_final_training_epochs", epochs), ("appendix_full_budget_training", main_summary),
                        ("appendix_full_budget_fractional", fractional_summary),
                        ("parameter_counts", parameters), ("parameter_core", core)]:
        frame.to_csv(output_dir / f"{stem}.csv", index=False)
    window_rows = [[dataset_label(r.dataset) if r.dataset != "FractionalBenchmarkForcing" else "Fractional benchmark",
                    f"{r.dt:.4g}", r.n_history, r.n_forecast,
                    "initial value" if pd.isna(r.history_time) else f"{r.history_time:.4g}",
                    f"{r.forecast_time:.4g}", r.batch_size, r.train_samples, r.train_batches]
                   for r in windows.itertuples()]
    write_table("appendix_dataset_window_table", "Dataset windows and batches", ["Dataset", "dt (s)", "H", "F", "H dt (s)", "F dt (s)", "Batch", "Windows", "Batches"], window_rows)
    hparam_rows = []
    for (model, parameter), group in hparams.groupby(["model", "parameter"], sort=False):
        spaces = list(dict.fromkeys(group.search_space))
        scales = list(dict.fromkeys(group.step_scale))
        row = [model_table_label(model), hyperparameter_label(parameter), " / ".join(spaces).replace("_", r"\_"), " / ".join(scales).replace("_", r"\_")]
        for dataset in datasets:
            values = group[group.dataset == dataset].value
            value = values.iloc[0] if len(values) else "--"
            row.append(f"{value:.3g}" if isinstance(value, (float, np.floating)) else str(value).replace("_", r"\_"))
        hparam_rows.append(row)
    write_table("appendix_hyperparameter_tables", "Search spaces and selected hyperparameters", ["Model", "Parameter", "Search space", "Step/scale"] + [dataset_label(d) for d in datasets], hparam_rows)
    epoch_rows = [["Final epochs"] + [str(int(epochs[epochs.dataset == d].final_epochs.max())) for d in datasets]]
    for model in algorithms:
        row = [model_table_label(model)]
        for dataset in datasets:
            selected = epochs[(epochs.dataset == dataset) & (epochs.model == model)]
            if selected.empty:
                row.append("--")
                continue
            r = selected.iloc[0]
            value = f"{r.best_epoch_mean:.0f}" + r" $\pm$ " + f"{r.best_epoch_std:.0f}"
            if r.n != len(seeds):
                value += f" [{int(r.n)}/{len(seeds)}]"
            row.append(value)
        epoch_rows.append(row)
    write_table("appendix_final_training_epochs_table", "Final training and best validation epochs", ["Model"] + [dataset_label(d) for d in datasets], epoch_rows)
    write_budget_table(main_summary, "appendix_full_budget_training", "Full-budget benchmark results", datasets, ["PersistenceBaseline"] + algorithms + ["NO-Prior"])
    write_budget_table(fractional_summary, "appendix_full_budget_fractional", "Fractional full-budget results (four seeds)", fractional.datasets, fractional.variants)
    parameter_rows = []
    for model in algorithms:
        row = [model_table_label(model)]
        for dataset in datasets:
            value = f"{int(parameters[(parameters.dataset == dataset) & (parameters.model == model)].iloc[0].parameters):,}"
            ranking = main_summary[(main_summary.dataset == dataset) & main_summary.model.isin(algorithms)].dropna(subset=["test_rmse_mean"]).sort_values(["test_rmse_mean", "model"])
            if len(ranking) and model == ranking.iloc[0].model:
                value = r"\textbf{" + value + "}"
            elif len(ranking) > 1 and model == ranking.iloc[1].model:
                value = r"\underline{" + value + "}"
            for family, mark in [(algorithms[:4], r"\dagger"), (algorithms[4:8], r"\ddagger")]:
                selected = ranking[ranking.model.isin(family)]
                if len(selected) and model == selected.iloc[0].model:
                    value += "$^{" + mark + "}$"
            row.append(value)
        parameter_rows.append(row)
    write_table("appendix_parameter_counts", r"Trainable parameters. Bold/underlined: first/second by test RMSE; $\dagger$/$\ddagger$: best baseline/NODE.", ["Model"] + [dataset_label(d) for d in datasets], parameter_rows)
    core_rows = [[dataset_label(r.dataset), model_table_label(r.model), f"{r.total:,}", f"{r.core:,}", f"{r.core_share_percent:.1f}", r.role] for r in core.itertuples()]
    write_table("rq2_3_parameter_core_table", "Dynamic core of runtime models", ["Dataset", "Model", "Total", "Core", r"Core (\%)", "Role"], core_rows)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    raise SystemExit(main(parser.parse_args().dry_run))
