import Constants as const
from src.experiments.experiments_base import ExperimentBase
from torch.utils.data import DataLoader, Subset
from copy import deepcopy
import math
import time
import torch
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve
import logging
from src.utils.failure_handling import (
    NumericalFailureError,
    classify_exception,
    ensure_finite_tensor,
    finite_tree_summary,
    get_failure_policy,
    success_record,
)


class PredictionPerformance(ExperimentBase):

    def __init__(self, dataset, model, seed=None, hash=None, console_mode=False, **kwargs):

        assert dataset is not None, "dataset must be provided"
        assert model is not None, "model must be provided"

        super().__init__(dataset, model, seed, hash, console_mode, **kwargs)

    def run_experiment(self, default_parameters=False):
        self.logger.info(
            "Running Prediction Performance Evaluation Experiment with model: "
            + self.model_name + " and dataset: " + self.dataset_name
        )
        if default_parameters:
            best_hyperparameters = None
        else:
            best_hyperparameters = self.hyperparameter_tuning()

        model = self.final_training(best_hyperparameters)
        results = self.performance_measurement(model)

        return results

    def run_data_fraction(self, best_hparams, fraction, seed, dataloaders=None, record_training_losses=False):
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0, 1].")
        if dataloaders is None:
            dataloaders = self.load_data(return_dataloader=True, shuffle_train=False)

        dataset_train = dataloaders["train"].dataset
        batch_size = int(self.config["experiment"]["batch_size"])
        epochs = int(self.config["experiment"]["num_epochs"])
        if abs(fraction - 1.0) < 1e-12:
            subset = dataset_train
        else:
            subset_size = max(1, int(round(len(dataset_train) * fraction)))
            indices = np.random.default_rng(seed).permutation(len(dataset_train))[:subset_size].tolist()
            subset = Subset(dataset_train, indices)
        train_loader = DataLoader(
            subset, batch_size=batch_size, shuffle=True, collate_fn=self.dataset_class.collate_fn,
        )

        self.set_seed(seed)
        model, model_config = self.get_untrained_model(deepcopy(best_hparams) if record_training_losses else best_hparams)
        model.pretraining(train_loader)
        best_val_loss = float("inf")
        best_state = None
        best_epoch = 0
        train_loss_at_best = None
        train_losses = []
        val_losses = []
        start = time.time() if record_training_losses else None

        for epoch in range(epochs):
            train_loss = model.train_step(train_loader, return_loss=True)
            if record_training_losses:
                train_loss = float(train_loss.detach().cpu().item()) if isinstance(train_loss, torch.Tensor) else float(train_loss)
            val_loss = model.validate_step(dataloaders["val"])
            if record_training_losses:
                val_loss = float(val_loss.detach().cpu().item()) if isinstance(val_loss, torch.Tensor) else float(val_loss)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_epoch = epoch + 1
                if record_training_losses:
                    train_loss_at_best = train_loss

            if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
                self.logger.info(
                    f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.3e}, Val Loss: {val_loss:.3e}"
                )

        if best_state is not None:
            model.load_state_dict({key: value.to(self.device) for key, value in best_state.items()})

        training_info = {
            "total_epochs": int(epochs),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "seed": int(seed),
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "batch_size": batch_size,
            "source": "trained_full_data" if abs(fraction - 1.0) < 1e-12 else "retrained_subset",
        }
        if record_training_losses:
            training_info.update({
                "train_loss_at_best_epoch": float(train_loss_at_best) if train_loss_at_best is not None else None,
                "last_epoch_train_loss": train_losses[-1] if train_losses else None,
                "last_epoch_val_loss": val_losses[-1] if val_losses else None,
                "mean_train_loss": float(np.mean(train_losses)) if train_losses else None,
                "mean_val_loss": float(np.mean(val_losses)) if val_losses else None,
                "wall_clock_training_seconds": float(time.time() - start),
                "source": "retrained_data_fraction",
            })
        else:
            reference_batches = max(1, math.ceil(len(dataset_train) / batch_size))
            train_batches = max(1, math.ceil(len(subset) / batch_size))
            training_info["update_budget"] = {
                "mode": "constant_epochs", "base_epochs": epochs, "batch_size": batch_size,
                "train_full_size": len(dataset_train), "train_subset_size": len(subset),
                "reference_batches_per_epoch": reference_batches, "train_batches_per_epoch": train_batches,
                "target_optimizer_updates": epochs * reference_batches, "requested_epochs": epochs,
                "used_epochs": epochs, "planned_optimizer_updates": epochs * train_batches,
                "max_epochs_per_cell": None, "epochs_capped": False,
            }

        evaluation = self.evaluate_prediction_splits(model, dataloaders, splits=["train", "val", "test"])
        metrics = evaluation["metrics"]
        return {
            "success": bool(evaluation["evaluation_status"].get("success", False)),
            "model_config": model_config,
            "training_info": training_info,
            "metrics": metrics,
            "evaluation_status": evaluation["evaluation_status"],
            "residuals_train": self._metric_or_none(metrics, "train", "global", "mse"),
            "residuals_val": self._metric_or_none(metrics, "val", "global", "mse"),
            "residuals_test": self._metric_or_none(metrics, "test", "global", "mse"),
            "train_subset_size": len(subset),
            "train_full_size": len(dataset_train),
        }

    def load_data(self, return_dataloader=True, shuffle_train=True, batch_size=None):
        """
        Loads the data for the experiment.

        Args:
            return_dataloader (bool, optional): If True, return DataLoader objects.
                If False, return raw dataset objects. Defaults to True.
            shuffle_train (bool, optional): Ignored here (we do not shuffle in this experiment),
                kept for API compatibility. Defaults to True.
            batch_size (int, optional): Batch size for the DataLoader. If None, the default
                batch size from the experiment config is used.

        Returns:
            dict: Dictionary with keys {"train", "val", "test"} containing either
                  DataLoaders or datasets, depending on `return_dataloader`.
        """
        train_dataset = self.dataset_class(self.config['dataset'], data_split='train', device=self.device)
        val_dataset = self.dataset_class(self.config['dataset'], data_split='val', device=self.device)
        test_dataset = self.dataset_class(self.config['dataset'], data_split='test', device=self.device)

        if return_dataloader:
            if batch_size is None:
                batch_size = self.config['experiment']['batch_size']

            # For this experiment we do not want to shuffle the data (sequence-level metrics)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self.dataset_class.collate_fn
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self.dataset_class.collate_fn
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self.dataset_class.collate_fn
            )
            return {"train": train_loader, "val": val_loader, "test": test_loader}

        return {"train": train_dataset, "val": val_dataset, "test": test_dataset}

    def _stack_batch_list(self, batch_list, name: str):
        if isinstance(batch_list, torch.Tensor):
            return batch_list

        if isinstance(batch_list, (list, tuple)):
            tensors = []
            for i, b in enumerate(batch_list):
                if b is None:
                    continue
                if isinstance(b, torch.Tensor):
                    tensors.append(b)
                else:
                    raise TypeError(
                        f"{name}: expected Tensor or None in batch list, "
                        f"got {type(b)} at index {i}"
                    )

            if len(tensors) == 0:
                raise ValueError(f"{name}: no tensors found in batch list")

            stacked = torch.cat(tensors, dim=0)
            self.logger.info(
                f"{name}: stacked {len(tensors)} batches -> shape {stacked.shape}"
            )
            return stacked

        self.logger.warning(
            f"{name}: unexpected type {type(batch_list)}, trying to convert to tensor"
        )
        return torch.as_tensor(batch_list, device=self.device)

    @staticmethod
    def _empty_global_metrics():
        return {"mse": None, "rmse": None, "mae": None, "maxae": None}

    @staticmethod
    def _empty_sequence_channel_metrics():
        return {
            "per_channel": {
                "median_mae": [],
                "median_rmse": [],
                "p95_rmse": [],
                "top5p_rmse_mean": [],
                "median_nrmse": [],
            },
            "overall": {
                "median_mae": None,
                "median_rmse": None,
                "p95_rmse": None,
                "top5p_rmse_mean": None,
                "median_nrmse": None,
            },
        }

    @staticmethod
    def _metric_or_none(metrics_by_split, split: str, scope: str, metric: str):
        try:
            return metrics_by_split[split][scope][metric]
        except (KeyError, TypeError):
            return None

    def _compute_global_metrics(self, y_true, y_pred):
        """
        Compute simple global regression metrics over all elements
        (aggregating over sequences, time steps and channels).

        Returns:
            dict with keys: {"mse", "rmse", "mae", "maxae"}
        """
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.as_tensor(y_true, device=self.device)
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.as_tensor(y_pred, device=self.device)

        y_true = y_true.double()
        y_pred = y_pred.double()

        diff = y_pred - y_true
        mse = (diff ** 2).mean()
        rmse = torch.sqrt(mse)
        mae = diff.abs().mean()
        maxae = diff.abs().max()

        return {
            "mse": mse.detach().cpu().item(),
            "rmse": rmse.detach().cpu().item(),
            "mae": mae.detach().cpu().item(),
            "maxae": maxae.detach().cpu().item(),
        }

    def _compute_sequence_channel_metrics(self, y_true, y_pred):
        """
        Compute MAE, RMSE, NRMSE per sequence and channel, then aggregate
        over sequences (for a single seed).

        Assumptions:
            - y_true, y_pred have shape (N, T, C) or (N, T)
              * N = number of sequences
              * T = time steps
              * C = channels / dimensions
            - Time dimension is the second-to-last, channel dimension is the last.

        Returns:
            dict with:
            {
                "per_channel": { ... lists, length = C ... },
                "overall":     { ... scalars (mean over channels) ... }
            }
        """

        if not isinstance(y_true, torch.Tensor):
            y_true = torch.as_tensor(y_true, device=self.device)
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.as_tensor(y_pred, device=self.device)

        y_true = y_true.double()
        y_pred = y_pred.double()

        # Shape handling:
        # (N, T) -> (N, T, 1)
        if y_true.ndim == 2:
            self.logger.warning(
                f"_compute_sequence_channel_metrics: got 2D y_true {y_true.shape}, "
                f"assuming single channel and unsqueezing"
            )
            y_true = y_true.unsqueeze(-1)
            y_pred = y_pred.unsqueeze(-1)
        # If more than 3 dims: flatten everything except (T, C) into "sequence"
        elif y_true.ndim > 3:
            self.logger.warning(
                f"_compute_sequence_channel_metrics: got >3D y_true {y_true.shape}, "
                f"flattening to (N, T, C)"
            )
            T = y_true.shape[-2]
            C = y_true.shape[-1]
            y_true = y_true.view(-1, T, C)
            y_pred = y_pred.view(-1, T, C)

        assert y_true.ndim == 3, f"Expected shape (N, T, C) or (N, T), got {y_true.shape}"
        N, T, C = y_true.shape
        self.logger.info(f"Metrics use N={N}, T={T}, C={C} (channels)")

        diff = y_pred - y_true  # (N, T, C)

        # 1) Sequence × channel errors (mean over time)
        mae_sc = diff.abs().mean(dim=1)               # (N, C)
        rmse_sc = torch.sqrt((diff ** 2).mean(dim=1)) # (N, C)

        # NRMSE: RMSE_sc / std(y_·,c) (std over all sequences and time steps per channel)
        y_flat = y_true.reshape(-1, C)                # (N*T, C)
        std_y = y_flat.std(dim=0, unbiased=False)     # (C,)

        nrmse_sc = torch.zeros_like(rmse_sc)
        nonzero = std_y > 0
        if nonzero.any():
            nrmse_sc[:, nonzero] = rmse_sc[:, nonzero] / std_y[nonzero]

        # 2) Aggregation over sequences (seed-level, per channel)

        # Median MAE/RMSE over sequences
        median_mae_c = mae_sc.median(dim=0).values    # (C,)
        median_rmse_c = rmse_sc.median(dim=0).values  # (C,)

        # P95 RMSE over sequences
        p95_rmse_c = torch.quantile(rmse_sc, 0.95, dim=0)  # (C,)

        # Tail metric: mean of top 5% RMSE (CVaR_95-like)
        q95 = torch.quantile(rmse_sc, 0.95, dim=0, keepdim=True)  # (1, C)
        mask = rmse_sc >= q95                                     # (N, C) bool
        top5p_rmse_mean_c = (rmse_sc * mask).sum(dim=0) / mask.sum(dim=0).clamp_min(1)

        # Median NRMSE over sequences
        median_nrmse_c = nrmse_sc.median(dim=0).values           # (C,)

        # Aggregation over channels (mean over channels as overall summary)
        overall = {
            "median_mae":      median_mae_c.mean().item(),
            "median_rmse":     median_rmse_c.mean().item(),
            "p95_rmse":        p95_rmse_c.mean().item(),
            "top5p_rmse_mean": top5p_rmse_mean_c.mean().item(),
            "median_nrmse":    median_nrmse_c.mean().item(),
        }

        per_channel = {
            "median_mae":      median_mae_c.detach().cpu().tolist(),
            "median_rmse":     median_rmse_c.detach().cpu().tolist(),
            "p95_rmse":        p95_rmse_c.detach().cpu().tolist(),
            "top5p_rmse_mean": top5p_rmse_mean_c.detach().cpu().tolist(),
            "median_nrmse":    median_nrmse_c.detach().cpu().tolist(),
        }

        return {
            "per_channel": per_channel,
            "overall": overall,
        }

    def evaluate_prediction_splits(self, model, dataloader, splits=None):
        """Evaluate splits and record numerical failures instead of hiding them."""

        if splits is None:
            splits = ["train", "val", "test"]

        policy = get_failure_policy(self.config)
        prediction_threshold = (
            policy["max_abs_prediction"]
            if policy["non_finite_is_failure"] and policy["divergent_prediction_is_failure"]
            else None
        )

        metrics_by_split = {}
        split_status = {}

        for split in splits:
            self.logger.info(f"Predicting on {split} split...")
            try:
                split_results = model.predict(dataloader[split], samples_only=True)

                y_true_batches = split_results["response_forecast"]
                y_pred_batches = split_results["predictions"]

                y_true = self._stack_batch_list(y_true_batches, name=f"{split}/response_forecast")
                y_pred = self._stack_batch_list(y_pred_batches, name=f"{split}/predictions")

                ensure_finite_tensor(
                    y_pred,
                    f"{split}/predictions",
                    f"evaluation/{split}",
                    max_abs_threshold=prediction_threshold,
                )

                global_metrics = self._compute_global_metrics(y_true, y_pred)
                seq_metrics = self._compute_sequence_channel_metrics(y_true, y_pred)

                metric_summary = finite_tree_summary(
                    {"global": global_metrics, "sequence_channel": seq_metrics},
                    prefix=split,
                )
                if not metric_summary["all_finite"]:
                    raise NumericalFailureError(
                        "divergence_failure",
                        f"evaluation/{split}",
                        f"{split} metrics contain non-finite values.",
                        metric_summary,
                    )

                status = success_record(
                    f"evaluation/{split}",
                    split=split,
                    n_sequences=int(y_true.shape[0]) if hasattr(y_true, "shape") else None,
                )
                metrics_by_split[split] = {
                    "global": global_metrics,
                    "sequence_channel": seq_metrics,
                    "status": status,
                }
                split_status[split] = status

            except Exception as exc:
                failure = classify_exception(exc, stage=f"evaluation/{split}")
                failure["details"] = {
                    **failure.get("details", {}),
                    "split": split,
                    "failure_policy": policy,
                }
                self.logger.warning(
                    f"Prediction evaluation failed on {split} split with "
                    f"{failure['failure_type']}: {failure['message']}"
                )
                metrics_by_split[split] = {
                    "global": self._empty_global_metrics(),
                    "sequence_channel": self._empty_sequence_channel_metrics(),
                    "status": failure,
                }
                split_status[split] = failure

        failed_splits = [split for split, status in split_status.items() if not status.get("success", False)]
        evaluation_status = {
            "success": len(failed_splits) == 0,
            "n_splits": int(len(split_status)),
            "n_failed_splits": int(len(failed_splits)),
            "failed_splits": failed_splits,
            "split_status": split_status,
            "failure_policy": policy,
        }

        return {
            "metrics": metrics_by_split,
            "evaluation_status": evaluation_status,
        }

    def performance_measurement(self, model):
        """
        Evaluate the performance of the model on train/val/test splits.

        Uses model.predict(..., samples_only=True), which returns a dict whose
        values are lists of batch tensors (one Tensor per DataLoader batch).

        For each split, compute:
            - Global metrics (MSE, RMSE, MAE, MaxAE)
            - Sequence×channel-based metrics:
                * per-channel: median MAE/RMSE, P95 RMSE, tail metric, median NRMSE
                * overall: mean over channels

        Returns:
            dict: results structured by split and metric type.
        """
        self.logger.info("Evaluating prediction performance (sequence-wise metrics)...")
        dataloader = self.load_data(return_dataloader=True, shuffle_train=False)

        evaluation = self.evaluate_prediction_splits(model, dataloader, splits=["train", "val", "test"])
        metrics_by_split = evaluation["metrics"]

        res_dict = {
            "meta": {
                "dataset": self.dataset_name,
                "model": self.model_name,
                "seed": self.seed,
                "hash": self.config_hash,
            },
            "metrics": metrics_by_split,
            "evaluation_status": evaluation["evaluation_status"],
        }

        # Backward compatibility:
        # Store MSE per split under the old keys "residuals_train/val/test"
        # so that existing analysis scripts expecting the old structure
        # still continue to work without modification.
        # These values correspond to the previous MSELoss over the full split.
        res_dict["residuals_train"] = self._metric_or_none(metrics_by_split, "train", "global", "mse")
        res_dict["residuals_val"]   = self._metric_or_none(metrics_by_split, "val", "global", "mse")
        res_dict["residuals_test"]  = self._metric_or_none(metrics_by_split, "test", "global", "mse")

        self.write_results(res_dict, 'evaluation_results')

        return res_dict


class TestPredictionPerformance(PredictionPerformance):

    def __init__(self, dataset, model, seed=None, hash=None, console_mode=False, **kwargs):

        assert dataset is not None, "dataset must be provided"
        assert model is not None, "model must be provided"

        super().__init__(dataset, model, seed, hash, console_mode, **kwargs)


if __name__ == "__main__":

    dataset = "SMDSystem"
    model = "ModularNeuralLaplace"
    exp = TestPredictionPerformance(dataset, model)
    results = exp.run_experiment()
