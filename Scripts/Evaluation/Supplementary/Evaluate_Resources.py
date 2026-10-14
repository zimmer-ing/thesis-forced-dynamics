"""Measure checkpoint inference times for the dissertation resource comparison."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Constants import NEW_RESULTS_PATH
from Scripts.utils.naming import model_directory


datasets = ["SMDSystem", "IndPenSim", "CascadedTanks", "IndustrialRobotForward"]
algorithms_by_dataset = {
    "SMDSystem": ["LaplaceNet", "NO-AugH", "LSTM"],
    "IndPenSim": ["LaplaceNet", "NO-Lat", "GRU"],
    "CascadedTanks": ["LaplaceNet", "NO-AugH", "GRU"],
    "IndustrialRobotForward": ["LaplaceNet", "NO-Aug0", "FC"],
}
forecast_steps = [10, 25, 50, 100, 200, 400, 600, 800, 1000]
sample_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]
window_lengths = {
    "SMDSystem": {"history_length": 50, "forecast_length": 500},
    "IndPenSim": {"history_length": 50, "forecast_length": 200},
    "CascadedTanks": {"history_length": 10, "forecast_length": 150},
    "IndustrialRobotForward": {"history_length": 50, "forecast_length": 150},
}
seed = 42
run_name = "runtime_01"
results_root = NEW_RESULTS_PATH / "full_data_hpo"
checkpoint_root = None
output_root = NEW_RESULTS_PATH / "resources"


def checkpoint_directory(dataset_name, model_name):
    run = model_directory(results_root, dataset_name, model_name) / f"Seed_{seed}"
    if checkpoint_root is None or (run / "model_checkpoints/best_model.pth").is_file():
        return run
    archived = checkpoint_root / run.relative_to(results_root)
    # Keep the selected cohort; never resolve an alternative alias in the archive.
    for filename in ("model_config.yaml", "summary.json", "evaluation_results.json", "evaluation_results_copy.json"):
        current, saved = run / filename, archived / filename
        if filename == "model_config.yaml" or current.is_file():
            if not current.is_file() or not saved.is_file() or current.read_bytes() != saved.read_bytes():
                return run
    return archived if (archived / "model_checkpoints/best_model.pth").is_file() else run


def main(dry_run=False):
    output_dir = output_root / run_name
    missing = []
    for dataset_name in datasets:
        for model_name in algorithms_by_dataset[dataset_name]:
            checkpoint_dir = checkpoint_directory(dataset_name, model_name)
            for filename in ("model_config.yaml", "model_checkpoints/best_model.pth"):
                if not (checkpoint_dir / filename).is_file():
                    missing.append(str(checkpoint_dir / filename))
    if output_dir.exists():
        missing.append(f"Output already exists; choose a new run_name: {output_dir}")

    if not dry_run:
        if missing:
            print("\n".join(missing), file=sys.stderr)
            return 1
        import torch
        from src.experiments.MeasureResources import MeasureResources

        if not torch.cuda.is_available():
            print("CUDA is required; no CPU timing run will be started.", file=sys.stderr)
            return 1
        output_dir.mkdir(parents=True, exist_ok=False)

    rows = []
    total_points = 0
    for dataset_name in datasets:
        for model_name in algorithms_by_dataset[dataset_name]:
            checkpoint_dir = checkpoint_directory(dataset_name, model_name)
            horizons = list(forecast_steps)
            batches = list(sample_counts)
            if dataset_name == "SMDSystem":
                horizons = [value for value in horizons if value <= 400]
            if model_name == "LaplaceNet" and dataset_name in {"SMDSystem", "CascadedTanks"}:
                horizons = [20 if value < 20 else value for value in horizons]
            if model_name == "LaplaceNet" and dataset_name == "IndPenSim":
                batches = [value for value in batches if value <= 512]
            if model_name == "FC":
                horizons = [window_lengths[dataset_name]["forecast_length"]]
            total_points += len(horizons) + len(batches)
            print(f"{dataset_name}/{model_name}: Seed_{seed}, windows={window_lengths[dataset_name]}, "
                  f"forecast_steps={horizons}, sample_counts={batches}")
            if dry_run:
                continue

            experiment = MeasureResources(
                dataset_name, model_name, seed=seed, hash="Resources",
                results_path_override=output_dir / dataset_name / model_name,
            )
            try:
                experiment.config["dataset"].update(window_lengths[dataset_name])
                measured_rows = experiment.measure_runtime(
                    checkpoint_dir, forecast_steps=horizons, sample_counts=batches,
                    batch_transforms=model_name == "LaplaceNet", sample_index=0,
                    repeats=10, warmup_repeats=5, timing_sleep_s=0.1, cleanup_sleep_s=0.25,
                )
                for row in measured_rows:
                    row["dataset"] = dataset_name
                rows.extend(measured_rows)
            finally:
                experiment.finish()
            with (output_dir / "runtime_measurements.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
                writer.writeheader()
                writer.writerows(rows)
            (output_dir / "runtime_measurements.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")

    print(f"{total_points} measurement points; output: {output_dir}")
    if dry_run:
        if missing:
            print("Missing prerequisites:\n" + "\n".join(missing))
        print("Dry-run only. CUDA and dataset loading are checked when measurements are started.")
        return 0
    failed = sum(row["status"] != "ok" for row in rows)
    print(f"Saved {len(rows)} measurements; failed: {failed}")
    return 1 if failed or len(rows) != total_points else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show the editable measurement loops without writing or running models.")
    raise SystemExit(main(dry_run=parser.parse_args().dry_run))
