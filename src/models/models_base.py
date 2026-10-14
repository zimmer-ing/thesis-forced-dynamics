import torch
import torch.nn as nn
from torch.optim import Adam
from abc import ABC, abstractmethod
import json
import Constants as const
from tqdm import tqdm
from ..utils.helpers import ensure_sequential_dataloader, concatenate_batches
from ..utils.failure_handling import (
    ensure_finite_number,
    ensure_finite_tensor,
    get_failure_policy,
)

import warnings
import logging
from typing import Callable, Tuple, Optional, Dict, List

class BaseRegressionModel(nn.Module, ABC):
    """
    Base class for regression models.

    Provides the common training, validation, prediction, and save/load methods.
    Subclasses implement forward() and can override the loss or optimizer parameters.

    Attributes:
        config (dict): Configuration used to prepare training.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        loss_fn (torch.nn.Module): Loss function, initialized as MSELoss.
    """

    def __init__(self, logger=None, **kwargs) -> None:
        """Initializes the base regression model and its logger."""
        super(BaseRegressionModel, self).__init__()
        if logger is None:
            print("No logger provided, creating a default logger.")
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            logger.info("No logger provided, creating a default logger.")
        self.logger = logger
        self.device = torch.device("cpu")
        self._device = self.device


    def prepare_training(self, config: dict) -> None:
        """Stores the configuration and initializes the Adam optimizer and MSE loss."""
        self.config = config
        self.optimizer = self._initialize_optimizer()
        self.loss_fn = nn.MSELoss()

    def pretraining(self, data_loader: torch.utils.data.DataLoader) -> None:
        """Optional pretraining hook; subclasses can implement it for the given data loader."""
        pass


    @abstractmethod
    def forward(self,excitation_history: torch.Tensor,response_history: torch.Tensor,
                time_history: torch.Tensor,
                excitation_forecast: torch.Tensor,
                time_forecast: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to be implemented by subclasses.

        Args:
            excitation_history (torch.Tensor): Known forcing history, shape (B, H, dim_x).
            response_history (torch.Tensor): Observed response history, shape (B, H, dim_y).
            time_history (torch.Tensor): History times, shape (B, H, 1).
            excitation_forecast (torch.Tensor): Known forcing over the forecast, shape (B, F, dim_x).
            time_forecast (torch.Tensor): Forecast times, shape (B, F, 1).

        Returns:
            torch.Tensor: Predicted responses, shape (B, F, dim_y).
        """

        pass

    def _infer_device(self) -> torch.device:
        """Return the device of the first parameter or CPU if unavailable."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        explicit_device = kwargs.get("device", args[0] if args else None)
        self._device = explicit_device if explicit_device is not None else self._infer_device()
        self.device = self._device
        return module

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Returns the parameters passed to the optimizer; subclasses can override this selection."""
        return list(self.parameters())

    def calculate_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates the configured loss between predictions and response targets."""
        return self.loss_fn(predictions, targets)

    def _initialize_optimizer(self) -> Adam:
        """Initializes Adam with config['model']['learning_rate'], or 0.001 if not specified."""

        lr = self.config.get("model", {}).get("learning_rate")

        if lr is None:
            warnings.warn("Learning rate not specified in configuration. Using default value of 0.001.", UserWarning)
            lr = 0.001
        return Adam(self.trainable_parameters(), lr=lr)

    def _max_abs_prediction_threshold(self) -> Optional[float]:
        policy = get_failure_policy(getattr(self, "config", None))
        if not policy["non_finite_is_failure"] or not policy["divergent_prediction_is_failure"]:
            return None
        return policy["max_abs_prediction"]

    def _max_abs_loss_threshold(self) -> Optional[float]:
        policy = get_failure_policy(getattr(self, "config", None))
        if not policy["non_finite_is_failure"]:
            return None
        return policy["max_abs_loss"]

    def train_step(self, data_loader: torch.utils.data.DataLoader, return_loss: bool = False) -> Optional[float]:
        """
        Train the model for one epoch on the given data loader.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader with training data.
            return_loss (bool, optional): If True, return the mean of the batch losses.

        Returns:
            Optional[float]: Mean batch loss if return_loss is True, otherwise None.
        """
        losses = []
        self.train()
        for batch in tqdm(data_loader, desc="Iteration Training Set", disable=not const.VERBOSE_BATCHES):
            excitation_history = batch['excitation_history']
            response_history = batch['response_history']
            excitation_forecast = batch['excitation_forecast']
            response_forecast = batch['response_forecast']
            time_history = batch['time_history']
            time_forecast = batch['time_forecast']
            self.optimizer.zero_grad()
            predictions = self(excitation_history, response_history, time_history, excitation_forecast, time_forecast)
            ensure_finite_tensor(
                predictions,
                "training/predictions",
                "training",
                max_abs_threshold=self._max_abs_prediction_threshold(),
            )
            loss = self.calculate_loss(predictions, response_forecast)
            ensure_finite_tensor(
                loss,
                "training/loss",
                "training",
                max_abs_threshold=self._max_abs_loss_threshold(),
            )
            loss.backward()
            if return_loss:
                losses.append(loss.item())
            self.optimizer.step()
        if return_loss:
            return sum(losses) / len(losses)

    def predict(self, data_loader: torch.utils.data.DataLoader, samples_only: bool = False) -> dict:
        """
        Predicts on the given data loader in sequential order.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader with prediction data.
            samples_only (bool): If True, keep a list of batches for each result key.
                Otherwise, concatenate batches and flatten batch/time axes of tensor data.

        Returns:
            dict: All batch fields and 'predictions'. Tensor results are stored on CPU.
        """
        results = {}

        data_loader = ensure_sequential_dataloader(data_loader)
        self.eval()
        target_device = self._infer_device()

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Iteration Prediction Set", disable=not const.VERBOSE_BATCHES):
                if not results:
                    results = {key: [] for key in batch.keys()}
                    results["predictions"] = []


                batch_on_device = {}
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        value_on_device = value.to(target_device)
                        batch_on_device[key] = value_on_device
                        results[key].append(value_on_device.detach().cpu())
                    else:
                        batch_on_device[key] = value
                        results[key].append(value)

                y_hat= self(
                    batch_on_device.get("excitation_history"),
                    batch_on_device.get("response_history"),
                    batch_on_device.get("time_history"),
                    batch_on_device.get("excitation_forecast"),
                    batch_on_device.get("time_forecast")
                )
                ensure_finite_tensor(
                    y_hat,
                    "prediction/predictions",
                    "prediction",
                    max_abs_threshold=self._max_abs_prediction_threshold(),
                )
                results["predictions"].append(y_hat.detach().cpu())


        if samples_only:
            return results


        for key in results.keys():
            results[key] = concatenate_batches(results[key])

        return results

    def validate_step(self, data_loader: torch.utils.data.DataLoader) -> float:
        """
        Validate the model on the given data loader.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader with validation data.

        Returns:
            float: Mean of the validation batch losses.
        """
        losses = []
        self.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Iteration Validation Set", disable=not const.VERBOSE_BATCHES):
                excitation_history = batch['excitation_history']
                response_history = batch['response_history']
                excitation_forecast = batch['excitation_forecast']
                response_forecast = batch['response_forecast']
                time_history = batch['time_history']
                time_forecast = batch['time_forecast']
                predictions = self(excitation_history, response_history, time_history, excitation_forecast,
                                   time_forecast)
                ensure_finite_tensor(
                    predictions,
                    "validation/predictions",
                    "validation",
                    max_abs_threshold=self._max_abs_prediction_threshold(),
                )
                loss = self.calculate_loss(predictions, response_forecast)
                ensure_finite_tensor(
                    loss,
                    "validation/loss",
                    "validation",
                    max_abs_threshold=self._max_abs_loss_threshold(),
                )
                losses.append(loss.item())
            if len(losses) == 0:
                raise ValueError("Validation set is empty.")
            val_loss = sum(losses) / len(losses)
            ensure_finite_number(
                val_loss,
                "validation/mean_loss",
                "validation",
                max_abs_threshold=self._max_abs_loss_threshold(),
            )
            return val_loss

    def save_model(self, path: str) -> None:
        """Saves the model state dictionary to the given path."""
        torch.save(self.state_dict(), path)

    def load_model(self, path: str) -> None:
        """Loads the model state dictionary and switches to evaluation mode."""
        self.load_state_dict(torch.load(path, weights_only=True))
        self.eval()

    @staticmethod
    def save_config(config: dict, path: str) -> None:
        """Saves the configuration as JSON."""
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)

    @staticmethod
    def load_config(path: str) -> dict:
        """Loads and returns the configuration from a JSON file."""
        with open(path, 'r') as f:
            config = json.load(f)
        return config


