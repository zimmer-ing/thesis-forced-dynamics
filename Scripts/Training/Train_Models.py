import argparse
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from Constants import CONFIGS_PATH, NEW_RESULTS_PATH, THESIS_RESULTS_PATH
from Scripts.utils.naming import model_directory
from src.utils.model_names import model_key

datasets = [
    "SMDSystem", "RLCDataset", "IndPenSim", "CascadedTanks",
    "IndustrialRobotForward", "NanoDroneOOD", "NanoDroneRunSplit",
]
algorithms = [
    "FC", "LSTM", "GRU", "TCN",
    "LaplaceNet", "NO-Obs", "NO-Aug0", "NO-AugH", "NO-Lat",
]
rlc_algorithms = ["NO-Prior"]
primary_seed = 42
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
final_gpus = None  # None: all visible GPUs; []: sequential; [0, 1]: selected visible GPUs.
results_path = NEW_RESULTS_PATH / "full_data_hpo"


def completed(path):
    try:
        result = json.loads(path.read_text())
        values = [result.get(f"residuals_{split}") for split in ("train", "val", "test")]
        return (result.get("success") is not False
                and all((result.get(key) or {}).get("success") is not False
                        for key in ("evaluation_status", "status"))
                and all(isinstance(value, (int, float)) and math.isfinite(value) for value in values))
    except (OSError, ValueError, AttributeError):
        return False


def load_hparams(model_path, return_path=False):
    import yaml

    paths = [model_path / f"best_hparams_seed_{primary_seed}.json",
             model_path / f"Seed_{primary_seed}/hyperparameter_tuning/best_hyperparameters.yaml"]
    for path in paths:
        if path.is_file():
            params = json.loads(path.read_text()) if path.suffix == ".json" else yaml.safe_load(path.read_text())
            if not isinstance(params, dict) or not params or any(not isinstance(key, str) for key in params):
                raise ValueError(f"Expected a nonempty hyperparameter dictionary in {path}")
            json.dumps(params, allow_nan=False)
            return (params, path) if return_path else params
    raise FileNotFoundError(f"Missing hyperparameters: {paths}")


def check_results_path(path, source):
    path, source = Path(path).resolve(), Path(source).resolve()
    if path.is_relative_to(source) or source.is_relative_to(path):
        raise ValueError(f"Results must be separate from the source results: {source}")


def train_final_seed(dataset, algorithm, seed, best_hparams, run_path, gpu=None):
    if gpu is not None:
        import torch
        torch.cuda.set_device(gpu)
    from src.experiments import PredictionPerformance
    from src.utils.failure_handling import classify_exception

    exp = None
    failed = False
    try:
        if completed(run_path / "summary.json"):
            print(f"Already completed: {run_path}")
            return
        if run_path.exists():
            raise FileExistsError(f"Incomplete run exists: {run_path}. Choose a new results_path to rerun.")
        print(f"Final training {dataset}/{algorithm}/Seed_{seed} on {'default device' if gpu is None else f'GPU {gpu}'}", flush=True)
        exp = PredictionPerformance(dataset, algorithm, seed=seed, hash="model_training",
                                    results_path_override=run_path)
        model = exp.final_training(best_hparams)
        result = exp.performance_measurement(model)
        exp.write_results(result, "summary")
        failed = not completed(run_path / "summary.json")
    except Exception as exc:
        failed = True
        print(f"Failed {dataset}/{algorithm}/Seed_{seed}: {exc}", flush=True)
        if exp is not None:
            exp.write_results({"success": False, "status": classify_exception(exc, stage="model_training/final_seed")}, "summary")
    finally:
        if exp is not None:
            exp.finish()
    if failed:
        raise SystemExit(1)


def main(dry_run=False, dataset=None, model=None, seed_values=None, output_path=None, hparams_from=None):
    selected_datasets = datasets if dataset is None else [dataset]
    selected_seeds = list(dict.fromkeys([primary_seed] + seeds if seed_values is None else seed_values))
    if not selected_seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in selected_seeds):
        raise ValueError("seeds must contain integers between 0 and 2**32 - 1.")
    if hparams_from is None and primary_seed not in selected_seeds:
        raise ValueError(f"A new HPO requires seed {primary_seed}; use --hparams-from to train other seeds only.")
    output_path = Path(results_path if output_path is None else output_path)
    protected_paths = [THESIS_RESULTS_PATH] + ([] if hparams_from is None else [Path(hparams_from)])
    for protected in protected_paths:
        check_results_path(output_path, protected)

    runs = []
    for dataset_name in dict.fromkeys(selected_datasets):
        selected_models = ([model] if model is not None else
                           algorithms + (rlc_algorithms if dataset_name == "RLCDataset" else []))
        for algorithm in dict.fromkeys(selected_models):
            for name in (dataset_name, algorithm):
                if not name or name in (".", "..") or Path(name).name != name or "\\" in name:
                    raise ValueError(f"Invalid dataset or model name: {name}")
            if model_key(algorithm) == "NO-Prior" and dataset_name != "RLCDataset":
                raise ValueError("NO-Prior requires RLCDataset.")
            for config_path in (CONFIGS_PATH / "datasets" / f"{dataset_name}.yaml",
                                CONFIGS_PATH / "models" / f"{model_key(algorithm)}.yaml"):
                if not config_path.is_file():
                    raise FileNotFoundError(f"Missing configuration: {config_path}")
            model_path = model_directory(output_path, dataset_name, algorithm)
            for protected in protected_paths:
                check_results_path(model_path, protected)
                for seed in selected_seeds:
                    check_results_path(model_path / f"Seed_{seed}", protected)
            best_hparams = hparams_path = None
            if hparams_from is not None:
                source_path = model_directory(hparams_from, dataset_name, algorithm)
                check_results_path(model_path, source_path)
                best_hparams, hparams_path = load_hparams(source_path, return_path=True)
                if (model_path / f"best_hparams_seed_{primary_seed}.json").exists() or any(model_path.glob("Seed_*")):
                    if load_hparams(model_path) != best_hparams:
                        raise ValueError(f"Hyperparameters differ from existing runs in {model_path}. Choose a new results path.")
            runs.append((dataset_name, algorithm, model_path, best_hparams, hparams_path))

    if not dry_run:
        import torch
        from src.experiments import PredictionPerformance
        from src.utils.failure_handling import classify_exception
        from src.datasets import DATASET_REGISTRY
        from src.models import MODEL_REGISTRY
        for dataset_name, algorithm, *_ in runs:
            if dataset_name not in DATASET_REGISTRY or algorithm not in MODEL_REGISTRY:
                raise ValueError(f"Dataset or model is not registered: {dataset_name}/{algorithm}")
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        gpus = list(range(gpu_count)) if final_gpus is None else list(final_gpus)
        if len(set(gpus)) != len(gpus) or any(type(gpu) is not int or gpu < 0 or gpu >= gpu_count for gpu in gpus):
            raise ValueError(f"final_gpus must contain unique visible GPU indices from 0 to {gpu_count - 1}; use [] for sequential training.")

    failed = False
    for dataset, algorithm, model_path, best_hparams, hparams_path in runs:
        run_path = model_path / f"Seed_{primary_seed}"
        additional_seeds = [seed for seed in selected_seeds if seed != primary_seed]
        if hparams_from is None:
            print(f"{dataset}/{algorithm}: HPO + final seed {primary_seed}, then seeds {additional_seeds}")
        else:
            print(f"{dataset}/{algorithm}: final training only, seeds {selected_seeds}")
            print(f"Hyperparameters: {hparams_path}")
        print(f"Results: {model_path}")
        print(f"Final GPUs: {'all visible GPUs automatically' if final_gpus is None else final_gpus}")
        if dry_run:
            continue

        if hparams_from is None:
            exp = None
            try:
                if completed(run_path / "summary.json"):
                    best_hparams = load_hparams(model_path)
                    print(f"Already completed: {run_path}")
                else:
                    if run_path.exists():
                        raise FileExistsError(f"Incomplete run exists: {run_path}. Choose a new results_path to rerun.")
                    exp = PredictionPerformance(dataset, algorithm, seed=primary_seed, hash="model_training",
                                                results_path_override=run_path)
                    result = exp.run_experiment()
                    best_hparams = exp.load_best_hparams()
                    exp.write_results(result, "summary")
                    (model_path / f"best_hparams_seed_{primary_seed}.json").write_text(
                        json.dumps(best_hparams, indent=2) + "\n")
                    if not completed(run_path / "summary.json"):
                        failed = True
            except Exception as exc:
                failed = True
                print(f"Failed {dataset}/{algorithm}/Seed_{primary_seed}: {exc}")
                if exp is not None:
                    exp.write_results({"success": False, "status": classify_exception(exc, stage="model_training/primary")}, "summary")
                continue
            finally:
                if exp is not None:
                    exp.finish()
        else:
            try:
                target_hparams = model_path / f"best_hparams_seed_{primary_seed}.json"
                if not target_hparams.is_file():
                    model_path.mkdir(parents=True, exist_ok=True)
                    target_hparams.write_text(json.dumps(best_hparams, indent=2, allow_nan=False) + "\n")
                if primary_seed in selected_seeds:
                    train_final_seed(dataset, algorithm, primary_seed, best_hparams, run_path)
            except (Exception, SystemExit) as exc:
                failed = True
                print(f"Failed {dataset}/{algorithm}: {exc}")
                continue

        pending_seeds = []
        for seed in additional_seeds:
            if completed(model_path / f"Seed_{seed}" / "summary.json"):
                print(f"Already completed: {model_path / f'Seed_{seed}'}")
            else:
                pending_seeds.append(seed)

        if not gpus:
            for seed in pending_seeds:
                try:
                    train_final_seed(dataset, algorithm, seed, best_hparams, model_path / f"Seed_{seed}")
                except SystemExit:
                    failed = True
            continue

        context = mp.get_context("spawn")
        for start in range(0, len(pending_seeds), len(gpus)):
            processes = []
            round_seeds = pending_seeds[start:start + len(gpus)]
            try:
                for gpu, seed in zip(gpus, round_seeds):
                    run_path = model_path / f"Seed_{seed}"
                    process = context.Process(target=train_final_seed,
                        args=(dataset, algorithm, seed, best_hparams, run_path, gpu))
                    process.start()
                    processes.append((process, seed))
                for process, seed in processes:
                    process.join()
                    if process.exitcode != 0 or not completed(model_path / f"Seed_{seed}" / "summary.json"):
                        failed = True
                        print(f"Failed final process: {dataset}/{algorithm}/Seed_{seed}, exit={process.exitcode}", flush=True)
            except BaseException:
                for process, _ in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join()
                raise
    return int(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HPO and final trainings, or retrain with saved hyperparameters.")
    parser.add_argument("--dry-run", action="store_true", help="Show datasets, models, seeds and paths without training or writing files.")
    parser.add_argument("--dataset", help="Run one dataset instead of the list above.")
    parser.add_argument("--model", help="Run one model instead of the list above.")
    parser.add_argument("--seeds", type=int, nargs="+", help="Train exactly these seeds; new HPO requires seed 42.")
    parser.add_argument("--results-path", type=Path, help="Root for the new result directories.")
    parser.add_argument("--hparams-from", type=Path, help="Read saved hyperparameters from ROOT/<dataset>/<model>; skip HPO.")
    args = parser.parse_args()
    raise SystemExit(main(dry_run=args.dry_run, dataset=args.dataset, model=args.model,
                          seed_values=args.seeds, output_path=args.results_path, hparams_from=args.hparams_from))
