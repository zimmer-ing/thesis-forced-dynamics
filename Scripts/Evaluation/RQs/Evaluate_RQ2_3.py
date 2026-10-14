"""Plot measured inference times and tabulate their longest-horizon endpoints."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Constants import EVALUATION_RESULTS_PATH, EVALUATION_SOURCE, NEW_RESULTS_PATH, OUTPUT_PATH
from Scripts.utils.naming import model_key

output_dir = OUTPUT_PATH / "pdf/Evaluation" / EVALUATION_SOURCE / "RQ2_3"
runtime_csv = (NEW_RESULTS_PATH / "resources/runtime_01/runtime_measurements.csv"
               if EVALUATION_SOURCE == "new_runs" else None)
duration_metadata = None  # Optional endpoint JSON with dataset, forecast_steps, history_length, duration_s.
selection_path = EVALUATION_RESULTS_PATH / "resources/source_selection.json"
runtime_path = EVALUATION_RESULTS_PATH / "resources"
duration_path = EVALUATION_RESULTS_PATH / "resources/forecast_durations.json"
datasets = ["SMDSystem", "IndPenSim", "CascadedTanks", "IndustrialRobotForward"]
algorithms_by_dataset = {
    "SMDSystem": ["LaplaceNet", "NO-AugH", "LSTM"],
    "IndPenSim": ["LaplaceNet", "NO-Lat", "GRU"],
    "CascadedTanks": ["LaplaceNet", "NO-AugH", "GRU"],
    "IndustrialRobotForward": ["LaplaceNet", "NO-Aug0", "FC"],
}
dataset_colors = {"SMDSystem": "#0072B2", "IndPenSim": "#D55E00",
                  "CascadedTanks": "#009E73", "IndustrialRobotForward": "#CC79A7"}


def read_timing_rows(path):
    rows, seen = [], set()
    with Path(path).open(newline="") as handle:
        for source in csv.DictReader(handle):
            if source.get("status") != "ok" or source.get("measurement") != "inference_time":
                continue
            if source.get("axis") not in {"forecast_steps", "samples"}:
                continue
            row = dict(source)
            for field in ("history_length", "forecast_steps", "samples", "seed"):
                row[field] = int(row[field])
            row["time_s_median"] = float(row["time_s_median"])
            if not math.isfinite(row["time_s_median"]) or row["time_s_median"] <= 0:
                raise ValueError(f"Invalid latency in {path}: {source}")
            if min(row[field] for field in ("history_length", "forecast_steps", "samples")) < 1:
                raise ValueError(f"Invalid measurement coordinate in {path}")
            if row["axis"] == "forecast_steps" and row["samples"] != 1:
                raise ValueError("Forecast-axis measurements require a batch size of one")
            key = tuple(row[field] for field in ("dataset", "model", "axis", "history_length", "forecast_steps", "samples", "seed"))
            if key in seen:
                raise ValueError(f"Duplicate timing coordinate in {path}: {key}")
            seen.add(key)
            rows.append(row)
    return rows


def load_runtime_rows():
    if runtime_csv is not None:
        rows = read_timing_rows(runtime_csv)
    else:
        selection = json.loads(selection_path.read_text())
        reference = EVALUATION_RESULTS_PATH / selection["reference_csv"]
        if hashlib.sha256(reference.read_bytes()).hexdigest() != selection["reference_sha256"]:
            raise ValueError("The reference timing CSV has changed")
        references = [row for row in read_timing_rows(reference) if row["model"] != selection["replaces_model"]]
        replacements = []
        for run, hashes, expected_count, selected_datasets in (
            (selection["full_run"], selection["full_source_sha256"], selection["full_rows"], None),
            (selection["full_repeat"]["run"], selection["full_repeat"]["source_sha256"],
             selection["full_repeat"]["rows"], selection["full_repeat"]["datasets"]),
        ):
            run_dir = runtime_path / run
            for filename, expected_hash in hashes.items():
                if hashlib.sha256((run_dir / filename).read_bytes()).hexdigest() != expected_hash:
                    raise ValueError(f"Timing source changed: {run_dir / filename}")
            if json.loads((run_dir / "completion.json").read_text()).get("success") is not True:
                raise ValueError(f"Incomplete runtime run: {run}")
            new_rows = read_timing_rows(run_dir / "runtime_measurements.csv")
            if len(new_rows) != expected_count or any(row["model"] != selection["model"] for row in new_rows):
                raise ValueError(f"Unexpected model/count in runtime run: {run}")
            raw = json.loads((run_dir / "runtime_measurements.json").read_text())["rows"]
            for row in raw:
                if row["status"] != "ok" or len(row["time_s_individual"]) != 10:
                    raise ValueError(f"Invalid repeated measurement: {run}")
                if not math.isclose(median(row["time_s_individual"]), float(row["time_s_median"]), rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError(f"Invalid measurement median: {run}")
            if selected_datasets is not None:
                if {row["dataset"] for row in new_rows} != set(selected_datasets):
                    raise ValueError("The repeat must replace the declared datasets completely")
                fields = ("dataset", "model", "axis", "history_length", "forecast_steps", "samples", "seed")
                before = {tuple(row[field] for field in fields) for row in replacements if row["dataset"] in selected_datasets}
                after = {tuple(row[field] for field in fields) for row in new_rows}
                if before != after:
                    raise ValueError("The repeat does not cover the original coordinate grid")
                replacements = [row for row in replacements if row["dataset"] not in selected_datasets]
            replacements.extend(new_rows)
        rows = references + replacements
    rows = [dict(row, model=model_key(row["model"])) for row in rows]
    coordinates = [tuple(row[field] for field in ("dataset", "model", "axis", "history_length", "forecast_steps", "samples", "seed")) for row in rows]
    if len(coordinates) != len(set(coordinates)):
        raise ValueError("Duplicate timing coordinate after model-name normalization")
    rows = [row for row in rows if row["dataset"] in datasets
            and row["model"] in algorithms_by_dataset[row["dataset"]]]
    if not rows:
        raise ValueError("No successful timing measurements for the selected datasets and models")
    for dataset in datasets:
        for model in algorithms_by_dataset[dataset]:
            for axis in ("forecast_steps", "samples"):
                selected = [row for row in rows if (row["dataset"], row["model"], row["axis"]) == (dataset, model, axis)]
                if not selected:
                    print(f"No successful measurements: {dataset}/{model}/{axis}")
                    continue
                if len({row["seed"] for row in selected}) != 1 or len({row["history_length"] for row in selected}) != 1:
                    raise ValueError(f"Mixed seeds or history lengths in one curve: {dataset}/{model}/{axis}")
                if axis == "samples" and len({row["forecast_steps"] for row in selected}) != 1:
                    raise ValueError(f"Mixed forecast lengths on batch axis: {dataset}/{model}")
    return rows


def forecast_duration(dataset, history, forecast):
    metadata_path = duration_metadata if runtime_csv is not None else duration_path
    if metadata_path is not None:
        for record in json.loads(Path(metadata_path).read_text()):
            recorded_history = record.get("history_length")
            if (record["dataset"], recorded_history, record["forecast_steps"]) == (dataset, history, forecast):
                duration = float(record["duration_s"])
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError("Invalid saved forecast duration")
                return duration
        raise ValueError(f"No matching duration metadata for {dataset}, history={history}, forecast={forecast}")

    import numpy as np
    import yaml
    from Constants import DATA_PATH
    from src.datasets import DATASET_REGISTRY

    if dataset == "SMDSystem" and not (DATA_PATH / "SMD_Dataset/train/Sample1.csv").is_file():
        raise FileNotFoundError("Prepare the SMD dataset first, or supply duration_metadata; evaluation does not generate data.")
    config = yaml.safe_load((ROOT / "configs/datasets" / f"{dataset}.yaml").read_text())
    data_root = config.get("data_root")
    if isinstance(data_root, str) and not Path(data_root).is_absolute():
        config["data_root"] = str(ROOT / data_root)
    for key, value in {"history_length": history, "forecast_length": forecast, "stride": 10**9}.items():
        config[key] = config[f"{key}_test"] = value
        config.setdefault("split_overrides", {}).setdefault("test", {})[key] = value
    dataset_class = DATASET_REGISTRY[dataset]
    data = dataset_class(config, data_split="test", device="cpu")
    sample = data[0]
    history_time = sample["time_history"].numpy().reshape(-1)
    forecast_time = sample["time_forecast"].numpy().reshape(-1)
    timeline = np.concatenate([history_time, forecast_time])
    if len(history_time) != history or len(forecast_time) != forecast or not np.isfinite(timeline).all() or not (np.diff(timeline) > 0).all():
        raise ValueError("Invalid dataset time window for runtime endpoint")
    unit = str(dataset_class.get_channels()["time"]).lower()
    factor = 3600.0 if any(text in unit for text in ("(h)", "[h]", "hour")) else (
        60.0 if "(min)" in unit or "[min]" in unit else 0.001 if "(ms)" in unit or "[ms]" in unit else 1.0)
    return float(forecast_time[-1] - history_time[-1]) * factor


def endpoint_records(rows):
    rows = [dict(row, model=model_key(row["model"])) for row in rows]
    records = []
    for dataset in datasets:
        selected = [row for row in rows if row["dataset"] == dataset and row["axis"] == "forecast_steps" and row["samples"] == 1]
        if not selected:
            continue
        forecast = max(row["forecast_steps"] for row in selected)
        endpoints = [row for row in selected if row["forecast_steps"] == forecast]
        histories = {row["history_length"] for row in endpoints}
        if len(histories) != 1:
            raise ValueError(f"Different history lengths at the endpoint: {dataset}")
        history = histories.pop()
        record = dict(dataset=dataset, history_length=history, forecast_steps=forecast,
                      duration_s=forecast_duration(dataset, history, forecast))
        for model, role in zip(algorithms_by_dataset[dataset], ("laplace", "node", "baseline")):
            matching = [row for row in endpoints if row["model"] == model]
            if len(matching) > 1:
                raise ValueError(f"Ambiguous endpoint: {dataset}/{model}")
            if matching:
                record[role + "_model"] = model
                record[role + "_ms"] = 1000 * matching[0]["time_s_median"]
        if "laplace_ms" in record and "node_ms" in record:
            record["node_over_laplace"] = record["node_ms"] / record["laplace_ms"]
        records.append(record)
    return records


def plot_runtime(rows):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter
    from Scripts.utils.naming import MODEL_MARKERS, dataset_label, model_table_label

    fig, axes = plt.subplots(1, 2, figsize=(5.9, 4.1), sharey=True)
    handles = {}
    for axis, name, field, title in zip(axes, ("forecast_steps", "samples"), ("forecast_steps", "samples"), ("Forecast steps", "Batch size")):
        axis.set(title=title, xlabel=title, xscale="log", yscale="log")
        axis.grid(True, which="both", color="#e6e6e6", linewidth=0.55)
        axis.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        axis.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.yaxis.set_minor_formatter(NullFormatter())
        for dataset in datasets:
            for model, linestyle, filled, width in zip(algorithms_by_dataset[dataset], ("-", "--", ":"), (True, False, False), (1.55, 1.35, 1.45)):
                values = sorted([row for row in rows if (row["dataset"], row["model"], row["axis"]) == (dataset, model, name)], key=lambda row: row[field])
                if not values:
                    continue
                label = f"{dataset_label(dataset)}: {model_table_label(model)}"
                if model in {"FC-dt", "FC"}:
                    label += r"$^{\ast}$"
                color = dataset_colors[dataset]
                line = axis.plot([row[field] for row in values], [row["time_s_median"] * 1000 for row in values],
                                 color=color, linestyle=linestyle, marker=MODEL_MARKERS.get(model, "o"),
                                 markersize=3.9, markerfacecolor=color if filled else "white", markeredgecolor=color,
                                 markeredgewidth=0.75, linewidth=width, alpha=0.95, label=label)[0]
                handles.setdefault(label, line)
    axes[0].set_ylabel("Inference time [ms]")
    items = list(handles.items())
    items = [item for column in range(3) for item in items[column::3]]
    fig.legend(handles=[item[1] for item in items], labels=[item[0] for item in items], loc="lower center",
               bbox_to_anchor=(0.5, 0.025), ncol=3, frameon=False, columnspacing=0.8, handlelength=1.6, fontsize=7.8)
    fig.tight_layout(rect=(0, 0.27, 1, 1))
    return fig


def write_endpoint_table(records, path):
    from Scripts.utils.naming import dataset_label, model_table_label

    def number(value, digits=2):
        rounded = float(f"{value:.{digits}g}")
        decimals = max(0, digits - 1 - math.floor(math.log10(abs(rounded)))) if rounded else 0
        return f"{rounded:.{decimals}f}"

    body = []
    for record in records:
        cells = [dataset_label(record["dataset"]), number(record["duration_s"], 3)]
        for role in ("laplace", "node", "baseline"):
            cell = number(record[role + "_ms"]) if role + "_ms" in record else "--"
            if role != "laplace" and role + "_model" in record:
                cell += f" ({model_table_label(record[role + '_model'])})"
            cells.append(cell)
        ratio = record.get("node_over_laplace")
        cells.append("--" if ratio is None else f"{ratio:.1f}" if ratio < 10 else f"{ratio:.0f}")
        body.append(" & ".join(cells) + r" \\")
    Path(path).write_text("\n".join([
        r"\begin{table}[h]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
        r"\caption[Inference times at the longest measured horizons]{Median inference time at the longest measured forecast horizon for each dataset, with batch size one. Missing model measurements are shown as --.}",
        r"\label{tab:rq2_3_p5_runtime_endpoints}", r"\begin{adjustbox}{max width=\columnwidth}",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Dataset & Horizon [s] & Laplace-Net [ms] & NODE [ms] & Baseline [ms] & Speed-up \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}", "",
    ]), encoding="utf-8")


def main(dry_run=False):
    print(f"Runtime source: {runtime_csv if runtime_csv is not None else selection_path}")
    print(f"Output: {output_dir}")
    if dry_run:
        return 0
    from Scripts.utils.evaluation import configure_plot, save_figure

    rows = load_runtime_rows()
    endpoints = endpoint_records(rows)
    configure_plot()
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, records in (("runtime_measurements.csv", rows), ("rq2_3_runtime_endpoints.csv", endpoints)):
        with (output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for record in records for key in record}))
            writer.writeheader()
            writer.writerows(records)
    (output_dir / "rq2_3_runtime_endpoints.json").write_text(json.dumps(endpoints, indent=2) + "\n")
    write_endpoint_table(endpoints, output_dir / "rq2_3_runtime_endpoints.tex")
    save_figure(plot_runtime(rows), output_dir / "rq2_3_inference_scaling")
    print(f"Evaluated {len(rows)} successful measurements and {len(endpoints)} dataset endpoints.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show editable input/output paths without reading datasets, measuring or writing.")
    raise SystemExit(main(dry_run=parser.parse_args().dry_run))
