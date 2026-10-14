from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Constants import NEW_RESULTS_PATH
from src.datasets import DATASET_REGISTRY  # noqa: E402


def load_simple_yaml(path: Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip('"').strip("'")
        if value.lower() in {"true", "false"}:
            parsed: Any = value.lower() == "true"
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
        config[key.strip()] = parsed
    return config


def global_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
    diff = (y_pred.double() - y_true.double())
    abs_diff = diff.abs()
    mse = (diff**2).mean()
    denom = y_true.double().abs().clamp_min(1e-12)
    return {
        "mse": float(mse.item()),
        "rmse": float(torch.sqrt(mse).item()),
        "mae": float(abs_diff.mean().item()),
        "maxae": float(abs_diff.max().item()),
        "mape": float((abs_diff / denom).mean().item() * 100.0),
    }


def sequence_channel_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, Any]:
    y_true = y_true.double()
    y_pred = y_pred.double()
    if y_true.ndim == 2:
        y_true = y_true.unsqueeze(-1)
        y_pred = y_pred.unsqueeze(-1)
    elif y_true.ndim > 3:
        time_steps = y_true.shape[-2]
        channels = y_true.shape[-1]
        y_true = y_true.reshape(-1, time_steps, channels)
        y_pred = y_pred.reshape(-1, time_steps, channels)

    diff = y_pred - y_true
    mae_sc = diff.abs().mean(dim=1)
    rmse_sc = torch.sqrt((diff**2).mean(dim=1))

    channels = y_true.shape[-1]
    std_y = y_true.reshape(-1, channels).std(dim=0, unbiased=False)
    nrmse_sc = torch.zeros_like(rmse_sc)
    nonzero = std_y > 0
    if bool(nonzero.any()):
        nrmse_sc[:, nonzero] = rmse_sc[:, nonzero] / std_y[nonzero]

    median_mae_c = mae_sc.median(dim=0).values
    median_rmse_c = rmse_sc.median(dim=0).values
    p95_rmse_c = torch.quantile(rmse_sc, 0.95, dim=0)
    q95 = torch.quantile(rmse_sc, 0.95, dim=0, keepdim=True)
    mask = rmse_sc >= q95
    top5p_rmse_mean_c = (rmse_sc * mask).sum(dim=0) / mask.sum(dim=0).clamp_min(1)
    median_nrmse_c = nrmse_sc.median(dim=0).values

    per_channel = {
        "median_mae": median_mae_c.detach().cpu().tolist(),
        "median_rmse": median_rmse_c.detach().cpu().tolist(),
        "p95_rmse": p95_rmse_c.detach().cpu().tolist(),
        "top5p_rmse_mean": top5p_rmse_mean_c.detach().cpu().tolist(),
        "median_nrmse": median_nrmse_c.detach().cpu().tolist(),
    }
    overall = {key: float(torch.as_tensor(value).mean().item()) for key, value in per_channel.items()}
    return {"per_channel": per_channel, "overall": overall}


def evaluate_split(dataset_name: str, config: dict[str, Any], split: str) -> dict[str, Any]:
    dataset_class = DATASET_REGISTRY[dataset_name]
    dataset = dataset_class(config, data_split=split, device="cpu")
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 128)),
        shuffle=False,
        collate_fn=dataset_class.collate_fn,
    )

    y_true_parts = []
    y_pred_parts = []
    for batch in loader:
        y_true = batch["response_forecast"].detach().cpu().float()
        y_last = batch["response_history"][:, -1:, :].detach().cpu().float()
        y_pred = y_last.repeat(1, y_true.shape[1], 1)
        y_true_parts.append(y_true)
        y_pred_parts.append(y_pred)

    y_true_all = torch.cat(y_true_parts, dim=0)
    y_pred_all = torch.cat(y_pred_parts, dim=0)
    return {
        "global": global_metrics(y_true_all, y_pred_all),
        "sequence_channel": sequence_channel_metrics(y_true_all, y_pred_all),
        "status": {
            "success": True,
            "stage": f"evaluation/{split}",
            "failure_type": None,
            "exception_type": None,
            "message": None,
            "details": {"split": split, "n_sequences": len(dataset)},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="CascadedTanks")
    parser.add_argument("--seed", default="0")
    parser.add_argument("--output-root", type=Path, default=NEW_RESULTS_PATH / "full_data_hpo")
    args = parser.parse_args()

    config = load_simple_yaml((PROJECT_ROOT / "configs/datasets") / f"{args.dataset}.yaml")
    metrics = {split: evaluate_split(args.dataset, config, split) for split in ["train", "val", "test"]}
    summary = {
        "meta": {
            "dataset": args.dataset,
            "model": "PersistenceBaseline",
            "seed": args.seed,
            "hash": "persistence",
            "note": "Repeats the last response-history value over the full forecast horizon.",
        },
        "metrics": metrics,
        "evaluation_status": {
            "success": True,
            "n_splits": 3,
            "n_failed_splits": 0,
            "failed_splits": [],
            "split_status": {split: metrics[split]["status"] for split in metrics},
        },
        "residuals_train": metrics["train"]["global"]["mse"],
        "residuals_val": metrics["val"]["global"]["mse"],
        "residuals_test": metrics["test"]["global"]["mse"],
    }

    run_dir = args.output_root / args.dataset / "PersistenceBaseline" / f"Seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "evaluation_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "config_experiment.yaml").write_text(
        "\n".join(f"{key}: {value}" for key, value in config.items()) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {run_dir / 'summary.json'}")
    print(f"test_rmse={metrics['test']['global']['rmse']:.6f}")


if __name__ == "__main__":
    main()
