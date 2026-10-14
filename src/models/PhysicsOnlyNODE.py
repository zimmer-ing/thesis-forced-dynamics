import warnings
from typing import Optional

import torch

from src.models.NeuralODE_base import BaseNODEModel, TimeInterpolation
from src.models.physics import PHYSICS_MODEL_REGISTRY


class PhysicsOnlyODE(torch.nn.Module):
    """
    ODE right-hand side backed only by a registered physics model.

    """

    def __init__(self, config: dict, interp_mode: str = "linear"):
        super().__init__()

        dataset_name = config["dataset"]["name"]
        prior_level = config["model"].get("physics_prior_level", "Greybox")
        physics_model_name = f"{dataset_name}_{prior_level}"
        if physics_model_name not in PHYSICS_MODEL_REGISTRY:
            raise ValueError(
                f"Physics model '{physics_model_name}' not found in PHYSICS_MODEL_REGISTRY."
            )

        physics_cls = PHYSICS_MODEL_REGISTRY[physics_model_name]
        self.physics_ODE = physics_cls(config)
        self.interpolator = TimeInterpolation(interp_mode=interp_mode)

    def set_x(self, t: torch.Tensor, x: torch.Tensor):
        self.interpolator.set_interpolation_data(t, x)

    def interpolate_x(self, t_query: torch.Tensor) -> torch.Tensor:
        return self.interpolator.interpolate_x(t_query)

    def forward(self, t: torch.Tensor, y: torch.Tensor, x: torch.Tensor = None) -> torch.Tensor:
        if x is None:
            x = self.interpolate_x(t)
        return self.physics_ODE(t, y, x)

    def get_physical_parameters(self):
        if hasattr(self.physics_ODE, "get_physical_parameters_summary"):
            return self.physics_ODE.get_physical_parameters_summary()
        return {}


class BasePhysicsOnlyNODE(BaseNODEModel):
    """NODE integration with a physics-only right-hand side."""

    def __init__(self, config: dict, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)
        self.config = config
        self.dim_z = self.dim_y

        model_cfg = config["model"]
        interp_mode = str(model_cfg.get("interp_mode", "linear")).lower()
        if interp_mode in ("zero", "zoh"):
            interp_mode = "zero"
        elif interp_mode != "linear":
            raise ValueError(f"Unknown interp_mode '{interp_mode}', use 'zero' or 'linear'.")

        self.ODE = PhysicsOnlyODE(config, interp_mode=interp_mode).to(self.device)
        self.physics_ODE = self.ODE.physics_ODE

        train_physics_params = bool(model_cfg.get("train_physics_params", True))
        if not train_physics_params:
            for param in self.physics_ODE.parameters():
                param.requires_grad_(False)

        self.prepare_training(config)

    def _trainable_physics_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]

    def _has_trainable_physics_parameters(self) -> bool:
        return bool(self._trainable_physics_parameters())

    def _initialize_optimizer(self):
        lr = float(self.config["model"].get("learning_rate", 1e-3))
        opt_name = str(self.config["model"].get("optimizer", "adam")).lower()
        params = self._trainable_physics_parameters()

        if not params:
            warnings.warn(
                "PhysicsOnlyNODE has no trainable physics parameters. "
                "Training steps will be skipped.",
                UserWarning,
            )
            self._dummy_optimizer_param = torch.nn.Parameter(torch.zeros(1, device=self.device))
            params = [self._dummy_optimizer_param]

        if opt_name == "adam":
            return torch.optim.Adam(params, lr=lr)
        if opt_name == "lbfgs":
            return torch.optim.LBFGS(
                params,
                lr=lr,
                max_iter=int(self.config["model"].get("max_iter", 45)),
                history_size=int(self.config["model"].get("history_size", 15)),
            )
        raise ValueError(f"Unknown optimizer '{opt_name}'. Use 'adam' or 'lbfgs'.")

    def train_step(self, data_loader: torch.utils.data.DataLoader, return_loss: bool = False) -> Optional[float]:
        if not self._has_trainable_physics_parameters():
            loss = self.validate_step(data_loader)
            return loss if return_loss else None

        self.train()
        optimizer_name = str(self.config["model"].get("optimizer", "adam")).lower()
        total_loss = 0.0
        num_samples = 0

        if optimizer_name == "lbfgs":
            def closure():
                self.optimizer.zero_grad()
                loss_sum = 0.0
                count = 0
                for batch in data_loader:
                    excitation_history = batch["excitation_history"].to(self.device)
                    response_history = batch["response_history"].to(self.device)
                    excitation_forecast = batch["excitation_forecast"].to(self.device)
                    response_forecast = batch["response_forecast"].to(self.device)
                    time_history = batch["time_history"].to(self.device)
                    time_forecast = batch["time_forecast"].to(self.device)

                    batch_size = response_forecast.size(0)
                    preds = self(
                        excitation_history,
                        response_history,
                        time_history,
                        excitation_forecast,
                        time_forecast,
                    )
                    loss = self.calculate_loss(preds, response_forecast)
                    loss_sum = loss_sum + loss * batch_size
                    count += batch_size

                if count == 0:
                    return torch.tensor(0.0, device=self.device, requires_grad=True)
                avg_loss = loss_sum / count
                avg_loss.backward()
                nonlocal total_loss, num_samples
                total_loss = float(loss_sum.detach().cpu())
                num_samples = count
                return avg_loss

            final_loss_tensor = self.optimizer.step(closure)
            if num_samples == 0:
                return float(final_loss_tensor.detach().cpu()) if return_loss else None
            return (total_loss / num_samples) if return_loss else None

        if optimizer_name == "adam":
            for batch in data_loader:
                self.optimizer.zero_grad()
                excitation_history = batch["excitation_history"].to(self.device)
                response_history = batch["response_history"].to(self.device)
                excitation_forecast = batch["excitation_forecast"].to(self.device)
                response_forecast = batch["response_forecast"].to(self.device)
                time_history = batch["time_history"].to(self.device)
                time_forecast = batch["time_forecast"].to(self.device)

                batch_size = response_forecast.size(0)
                preds = self(
                    excitation_history,
                    response_history,
                    time_history,
                    excitation_forecast,
                    time_forecast,
                )
                loss = self.calculate_loss(preds, response_forecast)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * batch_size
                num_samples += batch_size

            if num_samples == 0:
                raise ValueError("Training set is empty.")
            return (total_loss / num_samples) if return_loss else None

        raise ValueError(f"Unknown optimizer '{optimizer_name}'.")

    def get_physical_parameters(self):
        return self.ODE.get_physical_parameters()


class PhysicsOnlyFixedStepNODE(BasePhysicsOnlyNODE):
    def __init__(self, config, logger=None, **kwargs):
        config["model"].setdefault("solver", "rk4")
        super().__init__(config, logger=logger, **kwargs)
        assert self._is_fixed_step, (
            f"Solver '{self.solver}' is not a fixed-step solver. "
            f"Choose one of: {sorted(self._fixed_step_solvers)}"
        )


class PhysicsOnlyAdaptiveNODE(BasePhysicsOnlyNODE):
    def __init__(self, config, logger=None, **kwargs):
        config["model"].setdefault("solver", "dopri5")
        super().__init__(config, logger=logger, **kwargs)
        assert self._is_adaptive, (
            f"Solver '{self.solver}' is not an adaptive solver. "
            f"Choose one of: {sorted(self._adaptive_solvers)}"
        )
