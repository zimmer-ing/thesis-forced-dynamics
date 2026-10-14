import argparse
import json
import math
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from Constants import NEW_RESULTS_PATH
from Scripts.utils.naming import model_directory

datasets = [
    "SMDSystem", "RLCDataset", "IndPenSim", "CascadedTanks",
    "IndustrialRobotForward", "NanoDroneOOD", "NanoDroneRunSplit",
]
algorithms = {
    "SMDSystem": ["LSTM", "NO-AugH", "LaplaceNet"],
    "RLCDataset": ["GRU", "NO-Obs", "NO-Prior", "NO-Aug0", "NO-AugH", "LaplaceNet"],
    "IndPenSim": ["GRU", "NO-Lat", "LaplaceNet"],
    "CascadedTanks": ["GRU", "NO-AugH", "LaplaceNet"],
    "IndustrialRobotForward": ["FC", "NO-Aug0", "LaplaceNet"],
    "NanoDroneOOD": ["GRU", "NO-AugH", "LaplaceNet"],
    "NanoDroneRunSplit": ["GRU", "NO-AugH", "LaplaceNet"],
}
primary_seed = 42
seeds = [42, 1, 2, 3]
fractions = [.05, .10, .15, .20, .30, .40, .50, .60, .70, .80, .90, 1.0]
training_path = NEW_RESULTS_PATH / "full_data_hpo"
results_path = NEW_RESULTS_PATH / "data_fraction"


def completed(path):
    try:
        result = json.loads(path.read_text())
        values = [result.get(f"residuals_{split}") for split in ("train", "val", "test")]
        return (result.get("success") is True
                and result.get("evaluation_status", {}).get("success") is True
                and all(isinstance(value, (int, float)) and math.isfinite(value) for value in values))
    except (OSError, ValueError, AttributeError):
        return False


def load_hparams(dataset, algorithm):
    model_path = model_directory(training_path, dataset, algorithm)
    base_path = model_path / f"Seed_{primary_seed}"
    if not base_path.is_dir():
        raise FileNotFoundError(f"Missing training run: {base_path}")
    paths = [model_path / f"best_hparams_seed_{primary_seed}.json",
             base_path / "hyperparameter_tuning/best_hyperparameters.yaml"]
    for path in paths:
        if path.is_file():
            params = yaml.safe_load(path.read_text())
            if not isinstance(params, dict):
                raise ValueError(f"Expected hyperparameters in {path}")
            return params, path
    raise FileNotFoundError(f"Missing training hyperparameters: {paths}")


def main(dry_run=False):
    parameters = {}
    errors = []
    for fraction in fractions:
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("fractions must be in (0, 1].")
    for dataset in dict.fromkeys(datasets):
        for algorithm in dict.fromkeys(algorithms[dataset]):
            if algorithm == "NO-Prior" and dataset != "RLCDataset":
                raise ValueError("NO-Prior requires RLCDataset.")
            try:
                parameters[dataset, algorithm] = load_hparams(dataset, algorithm)
            except (OSError, ValueError, yaml.YAMLError) as exc:
                errors.append(f"{dataset}/{algorithm}: {exc}")
    if errors:
        raise ValueError("Missing data-fraction prerequisites before training:\n" + "\n".join(errors))
    if not dry_run:
        from src.experiments import PredictionPerformance
        from src.utils.failure_handling import classify_exception

    failed = False
    for dataset in dict.fromkeys(datasets):
        for algorithm in dict.fromkeys(algorithms[dataset]):
            best_hparams, hparams_path = parameters[dataset, algorithm]
            print(f"{dataset}/{algorithm}: fractions {fractions}, seeds {seeds}; 100% is retrained")
            print(f"Hyperparameters: {hparams_path}\nResults: {results_path / dataset / 'raw'}")
            if dry_run:
                continue
            exp = None
            dataloaders = None
            try:
                for fraction in dict.fromkeys(fractions):
                    for seed in dict.fromkeys(seeds):
                        tag = f"{int(round(fraction * 100)):03d}pct"
                        target = results_path / dataset / "raw" / f"{algorithm}_seed{seed}_{tag}.json"
                        if completed(target):
                            continue
                        result = {
                            "meta": {
                                "dataset": dataset, "model": algorithm, "result_model": algorithm,
                                "seed": seed, "primary_seed_for_hparam_tuning": primary_seed,
                                "source_protocol": "model_training",
                                "train_fraction": fraction, "reused_existing_model": False,
                                "train_subset_size": None,
                                "train_full_size": None, "update_budget": None,
                            },
                            "best_hyperparameters": best_hparams,
                        }
                        try:
                            if exp is None:
                                exp = PredictionPerformance(dataset, algorithm, seed=primary_seed, hash="data_fraction",
                                    results_path_override=results_path / "_work" / dataset / algorithm)
                            if dataloaders is None:
                                dataloaders = exp.load_data(return_dataloader=True, shuffle_train=False)
                            full_size = len(dataloaders["train"].dataset)
                            result["meta"].update({"train_full_size": full_size,
                                "train_subset_size": full_size if abs(fraction - 1.0) < 1e-12 else max(1, int(round(full_size * fraction)))})
                            result.update(exp.run_data_fraction(best_hparams, fraction, seed, dataloaders=dataloaders))
                            result["meta"].update({
                                "train_subset_size": result["train_subset_size"],
                                "train_full_size": result["train_full_size"],
                                "update_budget": result["training_info"].get("update_budget"),
                            })
                        except Exception as exc:
                            failure = classify_exception(exc, stage=f"data_fraction/{dataset}/{algorithm}/seed{seed}/{tag}")
                            result.update({"success": False,
                                "training_info": None, "model_config": None,
                                "metrics": {}, "evaluation_status": failure,
                                "residuals_train": None, "residuals_val": None, "residuals_test": None})
                            print(f"Failed {dataset}/{algorithm}/seed{seed}/{tag}: {exc}")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(json.dumps(result, indent=2) + "\n")
                        if not completed(target):
                            failed = True
            finally:
                if exp is not None:
                    exp.finish()
    return int(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edit datasets, algorithms, seeds, fractions and paths above; then retrain with existing hyperparameters.")
    parser.add_argument("--dry-run", action="store_true", help="Check training prerequisites and show the runs without training or writing files.")
    args = parser.parse_args()
    raise SystemExit(main(dry_run=args.dry_run))
