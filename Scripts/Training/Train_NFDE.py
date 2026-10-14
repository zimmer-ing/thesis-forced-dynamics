"""Fractional benchmark: edit the lists and switches below, then run this file.

The default trains the two fixed-alpha NFDE variants. Historical NODE/I-LSTM
references and their data-fraction curves are optional, separate sections.
Use --dry-run to display the selected experiments without creating files.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Constants import NEW_RESULTS_PATH
from Scripts.utils.naming import (
    model_key, variant_key, variant_directory, variant_result_name,
)


# Experiment settings: these are the lists and switches to edit.
datasets = [
    "FractionalBenchmarkReconstruction",
    "FractionalBenchmarkExtrapolation",
    "FractionalBenchmarkForcing",
]

variants = [
    {"label": "NFDE", "model": "NFDE",
     "overrides": {"model": {"include_time": False}}},
    {"label": "NFDE-t", "model": "NFDE",
     "overrides": {"model": {"include_time": True}}},
]
# NFDE fixes alpha=0.7 and learn_alpha=False in the model itself.

reference_variants = [
    {"label": "NO-Obs", "model": "NO-Obs",
     "overrides": {"model": {"include_time": False, "fallback_on_nan_dt": True}}},
    {"label": "NO-Obs-t", "model": "NO-Obs",
     "overrides": {"model": {"include_time": True, "fallback_on_nan_dt": True}}},
    {"label": "NO-Aug0", "model": "NO-Aug",
     "overrides": {"model": {"include_time": False, "augmentation_type": "zero", "fallback_on_nan_dt": True}}},
    {"label": "NO-Aug0-t", "model": "NO-Aug",
     "overrides": {"model": {"include_time": True, "augmentation_type": "zero", "fallback_on_nan_dt": True}}},
    {"label": "I-LSTM", "model": "I-LSTM", "overrides": {"model": {}}},
    # Uncomment to include the supplementary FC/TCN references:
    # {"label": "FC-dt", "model": "FC-dt",
    #  "overrides": {"dataset": {"forecast_length": 400}, "model": {}}},
    # {"label": "TCN", "model": "TCN", "overrides": {"model": {}}},
]

primary_seed = 42
seeds_full = [42, 1, 2, 3]
seeds_reduced = [42, 1, 2, 3]
seeds_references = [42, 1, 2, 3]
fractions = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90]  # Percent, excluding 100.
reference_fractions = [10, 25, 50, 100]

# Only these reference pairs used the dense grid in the dissertation.
dense_reference_variants = {
    "FractionalBenchmarkReconstruction": ["NO-Aug0", "I-LSTM", "FC-dt"],
    "FractionalBenchmarkExtrapolation": ["NO-Aug0-t", "TCN"],
    "FractionalBenchmarkForcing": ["NO-Aug0", "I-LSTM"],
}

run_hpo = True
run_full = True
run_reduced = True
run_references = False
run_reference_curves = False

nfde_results = NEW_RESULTS_PATH / "fractional/nfde"
reference_results = NEW_RESULTS_PATH / "fractional/references"
reference_checkpoint_root = None
reference_curve_results = NEW_RESULTS_PATH / "fractional/reference_curves"


def save_result(path, result):
    """Write the existing JSON format, including finite-or-null numeric values."""
    def json_value(value):
        if isinstance(value, dict):
            return {str(key): json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_value(item) for item in value]
        if hasattr(value, "detach"):
            return json_value(value.detach().cpu().tolist())
        if hasattr(value, "tolist"):
            return json_value(value.tolist())
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(result), indent=2) + "\n")


def successful_result(path):
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text())
        return (result.get("success", True) is True
                and (result.get("evaluation_status") or {}).get("success", True) is True
                and math.isfinite(float(result["metrics"]["test"]["global"]["rmse"])))
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def write_summary(output_dir, historical=False):
    """Rebuild the reader's tables; retain other historical reference rows."""
    rows = []
    if historical:
        summary_path = output_dir / "summary_table.json"
        existing = json.loads(summary_path.read_text())["rows"] if summary_path.is_file() else []
        pairs = {(row["dataset"], variant_key(row["variant"])) for row in existing}
        pairs.update((path.parent.name, variant_key(path.name))
                     for path in output_dir.glob("FractionalBenchmark*/*") if path.is_dir())
        for dataset, variant in sorted(pairs):
            directory = variant_directory(output_dir, dataset, variant)
            previous = [row for row in existing
                        if (row["dataset"], variant_key(row["variant"])) == (dataset, variant)]
            by_seed = {}
            if directory.is_dir():
                # A canonical pair is rebuilt solely from its own seed files.
                # Legacy summaries can contain rows whose detailed files were not archived.
                if directory.name != variant:
                    by_seed = {int(row["seed"]): dict(row, variant=variant) for row in previous
                               if row["variant"] in (directory.name, variant)}
            elif previous:
                # Preserve untouched summary-only pairs, choosing one naming cohort.
                selected_name = variant if any(row["variant"] == variant for row in previous) else previous[0]["variant"]
                by_seed = {int(row["seed"]): dict(row, variant=variant) for row in previous
                           if row["variant"] == selected_name}
            for seed_dir in sorted(directory.glob("Seed_*")):
                path = seed_dir / "evaluation_results.json"
                if not path.is_file():
                    path = seed_dir / "evaluation_results_copy.json"
                if not path.is_file():
                    continue
                result = json.loads(path.read_text())
                metrics = result.get("metrics", {}).get("test", {}).get("global", {})
                seed = int(seed_dir.name[5:])
                by_seed[seed] = {
                    "dataset": dataset, "variant": variant, "seed": seed,
                    "test_mse": metrics.get("mse"), "test_rmse": metrics.get("rmse"),
                    "test_mae": metrics.get("mae"), "test_max_ae": metrics.get("max_ae", metrics.get("maxae")),
                }
            rows.extend(by_seed.values())
        csv_path = output_dir / "summary_table.csv"
    else:
        seen = set()
        for directory in sorted(output_dir.glob("FractionalBenchmark*")):
            names = {variant_key(path.name.split("_seed", 1)[0]) for path in directory.glob("*_seed*.json")}
            for variant in sorted(names):
                selected_name = variant_result_name(output_dir, directory.name, variant)
                for path in sorted(directory.glob(f"{selected_name}_seed*.json")):
                    result = json.loads(path.read_text())
                    meta = result.get("meta", {})
                    coordinate = (directory.name, variant, meta.get("seed"), meta.get("train_fraction"))
                    if coordinate in seen:
                        raise ValueError(f"Duplicate fractional summary coordinate: {coordinate}")
                    seen.add(coordinate)
                    metrics = result.get("metrics", {})
                    test = metrics.get("test", {}).get("global", {})
                    rows.append({
                        "dataset": meta.get("dataset"), "variant": variant_key(meta.get("variant")), "model": model_key(meta.get("model")),
                        "seed": meta.get("seed"), "fraction": meta.get("train_fraction"),
                        "fraction_tag": meta.get("train_fraction_tag"), "train_subset_size": meta.get("train_subset_size"),
                        "train_full_size": meta.get("train_full_size"), "success": result.get("success"),
                        "reused_existing_model": meta.get("reused_existing_model"),
                        "train_rmse": metrics.get("train", {}).get("global", {}).get("rmse"),
                        "val_rmse": metrics.get("val", {}).get("global", {}).get("rmse"),
                        "test_rmse": test.get("rmse"), "test_mae": test.get("mae"),
                        "test_maxae": test.get("maxae", test.get("max_ae")),
                        "failure_type": (result.get("evaluation_status") or {}).get("failure_type"),
                    })
        summary_path = output_dir / "summary/summary.json"
        csv_path = output_dir / "summary/summary.csv"
    save_result(summary_path, {"rows": rows})
    if rows:
        with csv_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def reference_checkpoint_directory(run):
    if reference_checkpoint_root is None or (run / "model_checkpoints/best_model.pth").is_file():
        return run
    archived = reference_checkpoint_root / run.relative_to(reference_results)
    # Only the identical selected cohort may supply an archived checkpoint.
    for filename in ("model_config.yaml", "summary.json", "evaluation_results.json", "evaluation_results_copy.json"):
        current, saved = run / filename, archived / filename
        if filename == "model_config.yaml" or current.is_file():
            if not current.is_file() or not saved.is_file() or current.read_bytes() != saved.read_bytes():
                return run
    return archived if (archived / "model_checkpoints/best_model.pth").is_file() else run


def main(dry_run=False):
    # The following list only chooses which visible variant settings to use.
    curve_variants = variants if run_hpo or run_full or run_reduced else []
    if run_references or run_reference_curves:
        curve_variants = curve_variants + reference_variants

    # Check all required HPO files before creating an experiment. A dry run
    # prints the same selections and missing inputs without importing training.
    missing = []
    for dataset in datasets:
        for variant in curve_variants:
            label = variant["label"]
            is_nfde = variant in variants
            supplemental = label in ("FC-dt", "TCN")
            if dataset == "FractionalBenchmarkExtrapolation" and label == "FC-dt":
                continue
            if is_nfde or supplemental:
                root = nfde_results if is_nfde else reference_curve_results
                hparams_path = variant_directory(root / "_hparams", dataset, label) / "best_hparams_seed_42.json"
                hpo_allowed = run_hpo
            else:
                hparams_path = variant_directory(reference_results, dataset, label) / "best_hparams_seed_42.json"
                hpo_allowed = run_references and run_hpo
            if hparams_path.is_file():
                try:
                    hparams = json.loads(hparams_path.read_text())
                    if not isinstance(hparams, dict) or not hparams:
                        raise ValueError("expected a non-empty JSON mapping")
                except (OSError, ValueError) as exc:
                    missing.append(f"Invalid HPO parameters: {hparams_path}: {exc}")
            elif not hpo_allowed:
                missing.append(f"Missing HPO parameters: {hparams_path}")
            if label == "I-LSTM" and run_reference_curves:
                for seed in seeds_references:
                    run_dir = variant_directory(reference_results, dataset, label) / f"Seed_{seed}"
                    checkpoint = reference_checkpoint_directory(run_dir) / "model_checkpoints/best_model.pth"
                    existing_result = run_dir / "evaluation_results.json"
                    if not existing_result.is_file():
                        existing_result = run_dir / "evaluation_results_copy.json"
                    if not checkpoint.is_file() and (not run_references or successful_result(existing_result)):
                        missing.append(f"Missing historical I-LSTM checkpoint: {checkpoint}")
            if dry_run:
                print(f"{dataset} / {label}: HPO {'available' if hparams_path.is_file() else 'run' if hpo_allowed else 'missing'}")
                if is_nfde:
                    if run_full:
                        print(f"  full: fractions=[100], seeds={seeds_full}")
                    if run_reduced:
                        print(f"  reduced: fractions={fractions}, seeds={seeds_reduced}")
                else:
                    if run_references:
                        print(f"  reference final training: seeds={seeds_references}")
                    if run_reference_curves:
                        grid = fractions + [100] if label in dense_reference_variants.get(dataset, []) else reference_fractions
                        print(f"  reference curves: fractions={grid}, seeds={seeds_references}")
    if dry_run:
        for message in missing:
            print(message)
        return 0
    if missing:
        raise FileNotFoundError("\n".join(missing))

    from src.experiments import PredictionPerformance
    from src.utils.failure_handling import classify_exception

    failed_cells = []

    # Historical reference models: HPO plus Seed 42 final training, then the
    # other three seeds. FC/TCN use the data-fraction training section below.
    if run_references:
        for dataset in datasets:
            for variant in reference_variants:
                label = variant["label"]
                if label in ("FC-dt", "TCN"):
                    continue
                pair_dir = variant_directory(reference_results, dataset, label)
                hparams_path = pair_dir / "best_hparams_seed_42.json"
                best_hparams = json.loads(hparams_path.read_text()) if hparams_path.is_file() else None
                if best_hparams is None and primary_seed not in seeds_references:
                    raise ValueError("Reference HPO needs Seed 42 in seeds_references.")
                ordered_seeds = [primary_seed] + [seed for seed in seeds_references if seed != primary_seed]
                for seed in ordered_seeds:
                    if seed not in seeds_references:
                        continue
                    run_dir = pair_dir / f"Seed_{seed}"
                    result_path = run_dir / "evaluation_results.json"
                    if not result_path.is_file():
                        result_path = run_dir / "evaluation_results_copy.json"
                    if successful_result(result_path):
                        if best_hparams is None:
                            raise FileNotFoundError(f"Restore the missing HPO file before changing a completed reference run: {hparams_path}")
                        print(f"Already completed: {dataset}/{label}/Seed_{seed}")
                        continue
                    experiment = PredictionPerformance(
                        dataset, variant["model"], seed=seed, variant_name=label,
                        config_overrides=variant["overrides"], results_path_override=run_dir,
                    )
                    try:
                        if best_hparams is None:
                            best_hparams = experiment.hyperparameter_tuning()
                            save_result(hparams_path, best_hparams)
                        model = experiment.final_training(best_hyperparameters=best_hparams)
                        result = experiment.performance_measurement(model)
                        save_result(run_dir / "evaluation_results_copy.json", result)
                        save_result(run_dir / "run_manifest.json", {
                            "run_name": "fractional_references", "dataset": dataset,
                            "variant_label": label, "model_name": variant["model"], "seed": seed,
                            "config_overrides": variant["overrides"],
                            "best_hyperparameters_used_for_final_training": best_hparams,
                        })
                        if not successful_result(run_dir / "evaluation_results_copy.json"):
                            raise RuntimeError(f"Reference evaluation failed: {dataset}/{label}/Seed_{seed}")
                    finally:
                        experiment.finish()
        write_summary(reference_results, historical=True)

    # NFDE and reference curves: one base experiment and one set of full
    # train/val/test loaders per dataset/variant, reused for all its cells.
    for dataset in datasets:
        for variant in curve_variants:
            label = variant["label"]
            is_nfde = variant in variants
            supplemental = label in ("FC-dt", "TCN")
            if dataset == "FractionalBenchmarkExtrapolation" and label == "FC-dt":
                continue
            if not is_nfde and not run_reference_curves and not (supplemental and run_references):
                continue
            root = nfde_results if is_nfde else reference_curve_results
            hparams_root = root / "_hparams" if is_nfde or supplemental else reference_results
            hparams_path = variant_directory(hparams_root, dataset, label) / "best_hparams_seed_42.json"
            result_label = variant_result_name(root, dataset, label)

            if is_nfde:
                selected_fractions = ([100] if run_full else []) + (fractions if run_reduced else [])
            elif run_reference_curves:
                selected_fractions = fractions + [100] if label in dense_reference_variants.get(dataset, []) else reference_fractions
            else:
                selected_fractions = [100]  # Explicit supplementary FC/TCN reference runs.

            if not hparams_path.is_file():
                experiment = PredictionPerformance(
                    dataset, variant["model"], seed=primary_seed, variant_name=label,
                    config_overrides=variant["overrides"],
                    results_path_override=root / "_hpo_runs" / dataset / label / "Seed_42",
                )
                try:
                    best_hparams = experiment.hyperparameter_tuning()
                    save_result(hparams_path, best_hparams)
                finally:
                    experiment.finish()
            else:
                best_hparams = json.loads(hparams_path.read_text())
            if not selected_fractions:
                continue

            # Avoid even loading datasets if this entire selected pair is done.
            pending = False
            for percent in selected_fractions:
                seeds = seeds_full if is_nfde and percent == 100 else seeds_reduced if is_nfde else seeds_references
                for seed in seeds:
                    if not successful_result(root / dataset / f"{result_label}_seed{seed}_{percent:03d}pct.json"):
                        pending = True
            if not pending:
                print(f"Already completed: {dataset}/{label}")
                continue

            experiment = PredictionPerformance(
                dataset, variant["model"], seed=primary_seed, variant_name=label,
                config_overrides=variant["overrides"], console_mode=True,
                results_path_override=root / "_runtime" / dataset / label / "base",
            )
            try:
                dataloaders = experiment.load_data()
                for percent in selected_fractions:
                    seeds = seeds_full if is_nfde and percent == 100 else seeds_reduced if is_nfde else seeds_references
                    for seed in seeds:
                        result_path = root / dataset / f"{result_label}_seed{seed}_{percent:03d}pct.json"
                        if successful_result(result_path):
                            continue
                        print(f"Training {dataset}/{label}: seed={seed}, data={percent}%")
                        checkpoint_dir = None
                        try:
                            if label == "I-LSTM" and percent == 100:
                                run_dir = variant_directory(reference_results, dataset, label) / f"Seed_{seed}"
                                checkpoint_dir = reference_checkpoint_directory(run_dir)
                                checkpoint = checkpoint_dir / "model_checkpoints/best_model.pth"
                                if not checkpoint.is_file():
                                    raise FileNotFoundError(f"Missing historical I-LSTM checkpoint: {checkpoint}")
                                reference = PredictionPerformance(
                                    dataset, variant["model"], seed=seed, variant_name=label,
                                    config_overrides=variant["overrides"], results_path_override=run_dir,
                                    create_results_dir=False, console_mode=True,
                                )
                                try:
                                    reference.path_results = checkpoint_dir
                                    try:
                                        model = reference.load_best_model()
                                    finally:
                                        reference.path_results = run_dir
                                    model.to(reference.device)
                                    model.eval()
                                    evaluation = reference.evaluate_prediction_splits(model, dataloaders, splits=["train", "val", "test"])
                                    metrics = evaluation["metrics"]
                                    result = {
                                        "success": bool(evaluation["evaluation_status"].get("success", False)),
                                        "metrics": metrics, "evaluation_status": evaluation["evaluation_status"],
                                        "model_config": None,
                                        "training_info": {
                                            "source": "existing_model_predict_only",
                                            "seed": seed, "model_name": variant["model"], "dataset_name": dataset,
                                            "batch_size": int(experiment.config["experiment"]["batch_size"]),
                                        },
                                        "train_subset_size": len(dataloaders["train"].dataset),
                                        "train_full_size": len(dataloaders["train"].dataset),
                                        "residuals_train": reference._metric_or_none(metrics, "train", "global", "mse"),
                                        "residuals_val": reference._metric_or_none(metrics, "val", "global", "mse"),
                                        "residuals_test": reference._metric_or_none(metrics, "test", "global", "mse"),
                                    }
                                finally:
                                    reference.finish()
                            else:
                                result = experiment.run_data_fraction(
                                    best_hparams, percent / 100, seed,
                                    dataloaders=dataloaders, record_training_losses=True,
                                )
                        except Exception as exc:
                            failure = classify_exception(exc, stage=f"fractional_data_fraction/{dataset}/{label}/seed{seed}/{percent:03d}pct")
                            full_size = len(dataloaders["train"].dataset)
                            result = {
                                "success": False, "metrics": {}, "evaluation_status": failure,
                                "model_config": None, "training_info": None,
                                "residuals_train": None, "residuals_val": None, "residuals_test": None,
                                "train_subset_size": max(1, round(full_size * percent / 100)), "train_full_size": full_size,
                            }
                        result["meta"] = {
                            "protocol": "fractional_data_fraction", "dataset": dataset, "variant": label,
                            "model": variant["model"], "seed": seed, "primary_seed_for_hparam_tuning": primary_seed,
                            "train_fraction": percent / 100,
                            "train_fraction_tag": f"{percent:03d}pct",
                            "train_subset_size": result.pop("train_subset_size"), "train_full_size": result.pop("train_full_size"),
                            "reused_existing_model": checkpoint_dir is not None,
                            "config_overrides": variant["overrides"],
                        }
                        result["best_hyperparameters"] = best_hparams
                        save_result(result_path, result)
                        write_summary(root)
                        if not successful_result(result_path):
                            failed_cells.append(str(result_path))
                            print(f"Cell failed; continuing with the remaining cells: {result_path}")
            finally:
                experiment.finish()
    if failed_cells:
        raise RuntimeError("Fractional evaluations failed:\n" + "\n".join(failed_cells))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show the selected experiments without training or writing files.")
    args = parser.parse_args()
    raise SystemExit(main(dry_run=args.dry_run))
