import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint as odeint_normal
from torchdiffeq import odeint_adjoint as odeint_adjoint
from src.models.models_base import BaseRegressionModel

import warnings


class TimeInterpolation:
    """
    Time-series interpolation for NODE-style models with per-sample time grids.

    Supports:
        - Zero-Order Hold ('zero' / 'zoh')
        - Linear interpolation ('linear')

    Stores:
        self._t_grid:   [batch, T]
        self._x_series: [batch, T, dim_x]
    """
    def __init__(self, interp_mode: str = "linear"):
        self._init_interpolator(interp_mode)

    def _init_interpolator(self, interp_mode: str):
        """
        Initialize interpolation mode and internal buffers.
        """
        mode = interp_mode.lower()
        if mode in ("zero", "zoh"):
            self._interp_mode = "zero"
        elif mode == "linear":
            self._interp_mode = "linear"
        else:
            raise ValueError(f"Unknown interp_mode '{interp_mode}', use 'zero' or 'linear'.")
        self._t_grid = None    # [B, T]
        self._x_series = None  # [B, T, dim_x]

    def set_interpolation_data(self, t: torch.Tensor, x: torch.Tensor):
        """
        Store per-sample time grids and known forcing series.

        Args:
            t: time stamps, one of
                   - [B, T]       (per-sample times)
                   - [B, T, 1]
                   - [T]          (shared grid, will be broadcast to all samples)
            x: known forcing series of shape [B, T, dim_x].
        """
        if x.dim() != 3:
            raise ValueError(f"x must be [B, T, dim_x]. Got {x.shape} instead.")

        B, T, dim_x = x.shape

        if t.dim() == 3 and t.size(0) == B and t.size(2) == 1:
            # [B, T, 1] -> [B, T]
            t_grid = t[:, :, 0]
        elif t.dim() == 2 and t.size(0) == B and t.size(1) == T:
            # [B, T]
            t_grid = t
        elif t.dim() == 1 and t.size(0) == T:
            # shared grid [T] -> broadcast to all samples
            t_grid = t.unsqueeze(0).expand(B, -1)
        else:
            raise ValueError(
                f"Unsupported t shape {t.shape} for x shape {x.shape} "
                "(expected [B,T], [B,T,1] or [T])."
            )

        # Check sorted rows (monotone non-decreasing per sample)
        if not torch.all(t_grid[:, 1:] >= t_grid[:, :-1]):
            warnings.warn("Some rows in t_grid are not sorted. Interpolation may be incorrect.")

        self._t_grid = t_grid
        self._x_series = x

    def interpolate_x(self, t_query: torch.Tensor) -> torch.Tensor:
        """
        Interpolate x(t_query) for all samples in the batch.

        Assumes t_query is a scalar (0-D tensor), as used by torchdiffeq.odeint.

        Returns:
            Tensor of shape [B, dim_x].
        """
        if self._t_grid is None or self._x_series is None:
            raise RuntimeError("Interpolation buffers not set. Call set_interpolation_data(...) first.")

        if t_query.dim() != 0:
            raise ValueError(
                f"t_query must be a scalar (0-D tensor). Got shape {t_query.shape}."
            )

        if self._interp_mode == "linear":
            return self._interp_linear(t_query)
        else:  # "zero"
            return self._interp_zoh(t_query)

    # ------------------------------------------------------------------
    # Zero-order hold (per-sample)
    # ------------------------------------------------------------------

    def _interp_zoh(self, t_query: torch.Tensor) -> torch.Tensor:
        """
        Zero-order hold per sample:
        For each sample b, take the last time index k with t_grid[b, k] <= t_query.
        """
        t = self._t_grid          # [B, T]
        x = self._x_series        # [B, T, dim_x]
        B, T, dim_x = x.shape

        # mask_le[b, j] = True if t[b, j] <= t_query
        mask_le = t <= t_query    # [B, T]

        # "searchsorted right": number of elements <= t_query in each row
        idx_right = mask_le.sum(dim=1)  # [B], values in [0, T]

        # ZOH uses idx = idx_right - 1, clamped
        idx = (idx_right - 1).clamp(min=0, max=T - 1)  # [B]

        idx_exp = idx.view(B, 1, 1).expand(B, 1, dim_x)
        x_interp = torch.gather(x, 1, idx_exp).squeeze(1)  # [B, dim_x]
        return x_interp

    # ------------------------------------------------------------------
    # Linear interpolation (per-sample)
    # ------------------------------------------------------------------

    def _interp_linear(self, t_query: torch.Tensor) -> torch.Tensor:
        """
        Piecewise linear interpolation per sample.

        For each sample b:
            idx_right[b] = number of elements <= t_query in row b
            -> if idx_right==0: before first sample, clamp to first
               if idx_right==T: after last sample, clamp to last
               else interpolate between (idx_right-1, idx_right)
        """
        t = self._t_grid          # [B, T]
        x = self._x_series        # [B, T, dim_x]
        B, T, dim_x = x.shape

        # mask_le[b, j] = True if t[b, j] <= t_query
        mask_le = t <= t_query     # [B, T]
        idx_right = mask_le.sum(dim=1)  # [B], in [0, T]

        # Boundary masks
        mask_before = (idx_right == 0)      # t_query < first time
        mask_after = (idx_right == T)       # t_query >= last time

        # Effective right indices for inner samples
        idx_right_eff = idx_right.clone().clamp(min=1, max=T - 1)  # [B]
        idx_left_eff = (idx_right_eff - 1).clamp(min=0)            # [B]

        # Gather x_left and x_right: [B, dim_x]
        idx_left_exp = idx_left_eff.view(B, 1, 1).expand(B, 1, dim_x)
        idx_right_exp = idx_right_eff.view(B, 1, 1).expand(B, 1, dim_x)
        x_left = torch.gather(x, 1, idx_left_exp).squeeze(1)
        x_right = torch.gather(x, 1, idx_right_exp).squeeze(1)

        # Gather t_left and t_right: [B]
        t_left = torch.gather(t, 1, idx_left_eff.view(B, 1)).squeeze(1)
        t_right = torch.gather(t, 1, idx_right_eff.view(B, 1)).squeeze(1)

        # Compute interpolation weights w[b] in [0,1]
        denom = (t_right - t_left)
        # avoid divide-by-zero
        denom_safe = torch.where(denom.abs() < 1e-14, torch.ones_like(denom), denom)
        w = ((t_query - t_left) / denom_safe).view(B, 1)  # [B, 1]

        x_interp_lin = x_left + (x_right - x_left) * w    # [B, dim_x]

        # Clamp boundary cases
        x_first = x[:, 0, :]   # [B, dim_x]
        x_last = x[:, -1, :]   # [B, dim_x]

        # Start from linear interior result, then override where needed
        x_interp = x_interp_lin
        x_interp = torch.where(mask_before.view(B, 1), x_first, x_interp)
        x_interp = torch.where(mask_after.view(B, 1), x_last, x_interp)

        return x_interp


# Neural ODE Base Classes


class BaseODE(nn.Module):
    """
    Base class for ODE networks.

    Subclasses must implement the forward(t, y, x=None) method.
    """
    def __init__(self, interp_mode, logger, scaling_dim=None, **kwargs):
        self.interpolator = TimeInterpolation(interp_mode)
        super().__init__()
        self.logger = logger
        scaling_dim = int(scaling_dim or 1)
        self.register_buffer("scaling_factor", torch.ones(scaling_dim), persistent=True)

    def forward(self, t: torch.Tensor, y: torch.Tensor, x: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the derivative dy/dt at time t.

        Args:
            t (torch.Tensor): Current time (scalar or [batch, 1]).
            y (torch.Tensor): Current state [batch, dim_y].
            x (torch.Tensor, optional): Input data [batch, dim_x].
        Returns:
            torch.Tensor: dy/dt of shape [batch, dim_y].
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    def set_x(self, t: torch.Tensor, x: torch.Tensor):
        """
        Register the time grid and known forcing series for interpolation.

        Args:
            t: time grid, shapes supported by TimeInterpolation
            x: known forcing series [B, T, dim_x]
        """
        self.interpolator.set_interpolation_data(t, x)

    def interpolate_x(self, t_query: torch.Tensor) -> torch.Tensor:
        """Interpolate the known forcing at t_query using TimeInterpolation."""
        return self.interpolator.interpolate_x(t_query)

    # ------------------------------------------------------------------
    # Scaling utilities
    # ------------------------------------------------------------------

    def set_scaling_factor(self, scaling_factor: torch.Tensor):
        if not torch.is_tensor(scaling_factor):
            scaling_factor = torch.as_tensor(scaling_factor)
        scaling_factor = scaling_factor.detach().clone().to(
            device=self.scaling_factor.device,
            dtype=self.scaling_factor.dtype,
        )
        if scaling_factor.dim() == 0:
            scaling_factor = scaling_factor.view(1)
        self.scaling_factor = scaling_factor
        self.logger.info("Scaling factor of the ODE adjusted")

    def get_scaling_factor(self):
        return self.scaling_factor


class BlackBoxODE(BaseODE):
    """
    ODE network that computes the derivative of the latent state z as a function of time,
    the latent state, and optionally the known forcing x.

    Parameters:
        dim_x (int): Input dimension of x.
        dim_z (int): Dimension of the latent state z.
        hidden_size (int): Hidden layer size.
        n_layers_hidden (int): Number of hidden layers.
        activation (str): Activation function ('relu', 'leaky_relu', 'tanh', 'elu', 'sigmoid', or 'none').
        include_time (bool): Whether to include time t as an input.
        homogeneous (bool): If True, the input x is not included separately.
        interp_mode (str): Interpolation mode selection; 'zero' or 'linear'.
    """

    def __init__(self, dim_x, dim_z, hidden_size, n_layers_hidden=3, activation='relu',
                 include_time=True, homogeneous=False, interp_mode='zero',logger=None,**kwargs):
        super().__init__(interp_mode, logger, scaling_dim=dim_z, **kwargs)
        self.include_time = include_time
        self.homogeneous = homogeneous

        # Determine input dimension: optionally include time, x and z or only z if homogeneous
        input_dim = 0
        if include_time:
            input_dim += 1
        if not homogeneous:
            input_dim += (dim_x + dim_z)
        else:
            input_dim += dim_z

        # Create fully-connected layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_size))
        for _ in range(n_layers_hidden - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, dim_z))

        # Set activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'none':
            self.activation = lambda x: x
        else:
            raise ValueError("Unsupported activation function")

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0.0)

        # explicitly initialize last layer to zero
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    # ------------------------------------------------------------------
    # ODE RHS
    # ------------------------------------------------------------------

    def forward(self, t: torch.Tensor, y: torch.Tensor, x: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the derivative of the output y.

        Args:
            t (torch.Tensor): Current time. For odeint this is usually a scalar (0-D tensor).
            y (torch.Tensor): Current latent state [batch, dim_z].
            x (torch.Tensor, optional): Input data. If not provided, it is interpolated from stored history.

        Returns:
            torch.Tensor: dy/dt of shape [batch, dim_z].
        """
        # Handle the known forcing x.
        if not self.homogeneous:
            if x is None:
                x = self.interpolate_x(t)  # [batch, dim_x]
            temp = torch.cat([y, x], dim=-1)
        else:
            temp = y

        # Optionally include time as feature
        if self.include_time:
            if t.dim() == 0:
                # odeint case: scalar time -> broadcast to [batch, 1]
                t_feat = t.view(1, 1).expand(y.size(0), 1)
            elif t.dim() == 1:
                # e.g. pretraining: [N] -> [N,1]
                t_feat = t.unsqueeze(-1)
            else:
                # assume already [batch, 1] or compatible
                t_feat = t
            temp = torch.cat([t_feat, temp], dim=-1)

        # Pass through MLP
        for layer in self.layers[:-1]:
            temp = self.activation(layer(temp))
        z_out = self.layers[-1](temp)
        return z_out * self.scaling_factor


# ======================================================================
# Base NODE model: shared solver logic, pretraining & forward
# ======================================================================

class BaseNODEModel(BaseRegressionModel):
    """
    Base class for Neural ODE regression models.

    Handles:
        - common config checks (dims, device)
        - solver configuration (fixed-step vs adaptive)
        - pretraining (optional, via config)
        - time grid construction
        - ODE integration and forward pass

    Subclasses must:
        - construct `self.ODE` in __init__, with signature ODE(t, y, x=None)
        - call `self.prepare_training(config)` after `self.ODE` is set
    """

    def __init__(self, config, logger=None, **kwargs):
        super().__init__(logger=logger, **kwargs)
        self.config = config

        # Basic config checks
        assert self.config['dataset']['dim_x'] is not None, "dim_x must be provided in config"
        assert self.config['dataset']['dim_y'] is not None, "dim_y must be provided in config"
        assert self.config['chosen_device'] is not None, "'chosen_device' must be provided in config"
        assert self.config['model']['adjoint'] in [True, False], "'adjoint' must be True or False"

        self.device = self.config['chosen_device']
        self.dim_x = self.config['dataset']['dim_x']
        self.dim_y = self.config['dataset']['dim_y']
        self.adjoint = self.config['model']['adjoint']

        if self.adjoint:
            self.odeint = odeint_adjoint
        else:
            self.odeint = odeint_normal

        # Solver configuration
        self.solver = self.config['model'].get('solver', 'rk4')
        self.rtol = self.config['model'].get('rtol', 1e-4)
        self.atol = self.config['model'].get('atol', 1e-7)
        self.intermediate_timesteps = self.config['model'].get('intermediate_timesteps', 1)
        # Optional fallback for adaptive solvers; disabled by default for clean evaluation.
        self.fallback_on_nan_dt = self.config['model'].get('fallback_on_nan_dt', False)
        self.fallback_solver = self.config['model'].get('fallback_solver', 'rk4')
        self.fallback_step_size = float(config['model'].get('fallback_step_size', 1e-6))
        assert self.fallback_step_size > 0.0, "fallback_step_size must be positive."

        # Supported solvers
        self._fixed_step_solvers = {
            'euler',
            'midpoint',
            'rk4',
            'explicit_adams',
            'implicit_adams',
        }
        self._adaptive_solvers = {
            'dopri8',
            'dopri5',
            'bosh3',
            'fehlberg2',
            'adaptive_heun',
        }

        all_supported = self._fixed_step_solvers.union(self._adaptive_solvers)
        assert self.solver in all_supported, (
            f"Unknown solver '{self.solver}'. Supported: {sorted(all_supported)}"
        )

        # For convenience
        self._is_fixed_step = self.solver in self._fixed_step_solvers
        self._is_adaptive = self.solver in self._adaptive_solvers

    # ------------------------------------------------------------------
    # Pretraining
    # ------------------------------------------------------------------

    def pretraining(self, data_loader: torch.utils.data.DataLoader) -> None:
        """
        Optional pretraining for NODE models.

        If pretraining_scaling is enabled, computes output scaling factors from
        finite-difference gradients of the observed response. Gradient pretraining
        additionally requires dim_z == dim_y and pretraining_epochs > 0.
        """
        model_cfg = self.config['model']

        if not model_cfg.get('pretraining_scaling', False):
            self.logger.info("Pretraining skipped as per configuration.")
            return

        self.logger.info("Computing gradient-based scaling factors for the ODE...")

        grad_batches = []
        for batch in data_loader:
            response_forecast = batch['response_forecast'].to(self.device)
            time_forecast = batch['time_forecast'].to(self.device)
            excitation_forecast = batch['excitation_forecast'].to(self.device)

            batch_size, seq_len, dim_y = response_forecast.shape
            assert dim_y == self.dim_y, "response_forecast last dimension must equal dim_y."

            dt = time_forecast[:, 1:, :] - time_forecast[:, :-1, :]  # [B, seq_len-1, 1]
            dy = response_forecast[:, 1:, :] - response_forecast[:, :-1, :]
            dy_dt = dy / dt  # [B, seq_len-1, dim_y]

            grad_batches.append((
                excitation_forecast[:, :-1, :],  # [B, seq_len-1, dim_x]
                response_forecast[:, :-1, :],    # [B, seq_len-1, dim_y]
                time_forecast[:, :-1, :],        # [B, seq_len-1, 1]
                dy_dt                             # [B, seq_len-1, dim_y]
            ))

        # maximum gradient per output dimension (dim_y)
        all_grads = torch.cat([g for _, _, _, g in grad_batches], dim=0)  # [B*(T-1), dim_y]
        max_grads, _ = torch.max(torch.abs(all_grads).view(-1, self.dim_y), dim=0)
        max_grads = max_grads.to(self.device)  # [dim_y]

        # Build scaling factor for the full latent state dim_z
        dim_z = getattr(self, "dim_z", self.dim_y)
        if dim_z < self.dim_y:
            raise ValueError(f"dim_z ({dim_z}) must be >= dim_y ({self.dim_y}).")

        scaling_factor = torch.ones(dim_z, device=self.device)
        scaling_factor[:self.dim_y] = max_grads  # observed part scaled, augmented part = 1

        # Apply scaling factor if supported by the ODE
        if hasattr(self.ODE, "set_scaling_factor"):
            self.ODE.set_scaling_factor(scaling_factor)
        else:
            self.logger.warning("ODE has no 'set_scaling_factor' method. Scaling factor not applied.")

        # If latent dimension != output dimension, skip gradient pretraining
        if dim_z != self.dim_y:
            self.logger.info(
                "dim_z != dim_y (e.g. augmented or latent NODE). "
                "Skipping gradient pretraining; applied scaling only."
            )
            return

        if model_cfg.get('pretraining_epochs', 0) <= 0:
            self.logger.info("pretraining_epochs <= 0. Skipping gradient pretraining.")
            return

        assert model_cfg.get('pretraining_lr') is not None, "pretraining_lr must be set for pretraining."

        self.logger.info("Starting gradient pretraining of the ODE network...")

        optimizer = torch.optim.Adam(self.ODE.parameters(), lr=model_cfg['pretraining_lr'])
        epochs = model_cfg['pretraining_epochs']
        loss_fn = nn.MSELoss()
        self.ODE.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for excitation_forecast, response_forecast, time_forecast, dy_dt in grad_batches:
                dy_dt_flat = dy_dt.reshape(-1, self.dim_y)

                pred_dy_dt = self.ODE(
                    time_forecast.reshape(-1, 1),
                    response_forecast.reshape(-1, self.dim_y),
                    excitation_forecast.reshape(-1, excitation_forecast.shape[2])
                )

                loss = loss_fn(pred_dy_dt, dy_dt_flat)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(grad_batches)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                self.logger.info(f"Pretraining Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.6f}")
        self.logger.info("Pretraining completed.")

    # ------------------------------------------------------------------
    # Solver call
    # ------------------------------------------------------------------

    def call_odeint(self, z0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Helper function to call odeint with the correct parameters.

        Uses either a fixed-step or an adaptive integration method depending on
        `self.solver`.

        With fallback_on_nan_dt enabled, an adaptive solver AssertionError
        containing 'underflow in dt' triggers a fixed-step fallback using
        self.fallback_step_size.
        """
        t_vec = t[0, :, 0]  # [T]

        if self._is_fixed_step:
            # Standard fixed-step integration: derive a reasonable step size from the data.
            dt = t[:, 1:, :] - t[:, :-1, :]  # [B, T-1, 1]
            dt_min = dt.min()
            if dt_min <= 0:
                raise ValueError("Non-positive minimal time step encountered.")

            steps = int(self.intermediate_timesteps)
            if steps < 1:
                steps = 1
            step_size = (dt_min / steps).item()

            z = self.odeint(
                self.ODE,
                z0,
                t_vec,
                method=self.solver,
                options={"step_size": step_size},
            )

        else:
            # Adaptive solver with optional fallback.
            try:
                z = self.odeint(
                    self.ODE,
                    z0,
                    t_vec,
                    method=self.solver,
                    rtol=self.rtol,
                    atol=self.atol,
                )
            except AssertionError as e:
                msg = str(e)
                # If fallback is disabled or this is not the dt-underflow error, re-raise.
                if (not getattr(self, "fallback_on_nan_dt", False)) or ("underflow in dt" not in msg):
                    raise

                # Fallback: fixed-step solver with absolute step size.
                h = float(self.fallback_step_size)
                self.logger.warning(
                    f"Adaptive solver '{self.solver}' failed with '{msg}'. "
                    f"Falling back to fixed-step '{self.fallback_solver}' "
                    f"with absolute step_size h={h:.3e}."
                )

                z = self.odeint(
                    self.ODE,
                    z0,
                    t_vec,
                    method=self.fallback_solver,
                    options={"step_size": h},
                )

        # [T, B, dim_z] -> [B, T-1, dim_z], drop initial state
        z = z.permute(1, 0, 2)[:, 1:, :]
        return z

    # ------------------------------------------------------------------
    # Forward pass (shared)
    # ------------------------------------------------------------------

    def forward(self,
                excitation_history: torch.Tensor,
                response_history: torch.Tensor,
                time_history: torch.Tensor,
                excitation_forecast: torch.Tensor,
                time_forecast: torch.Tensor,
                return_latent: bool = False) -> torch.Tensor:
        """
        Performs a forward pass through the Neural ODE model.

        Args:
            excitation_history: [batch, H, dim_x]
            response_history:   [batch, H, dim_y]
            time_history:       [batch, H, 1]
            excitation_forecast:[batch, F, dim_x]
            time_forecast:      [batch, F, 1]

        Returns:
            [batch, F, dim_y], or (y, z) if return_latent=True.
        """
        batch_size = excitation_history.size(0)

        # Initial state y0: last history response
        y0 = response_history[:, -1, :]  # [B, dim_y]
        z0 = y0  # latent == output for now

        # Relative time grid t [B, F+1, 1]
        last_hist_time = time_history[:, -1, :]   # [B, 1]
        first_pred_time = time_forecast[:, 0, :]  # [B, 1]
        dt_since_hist = first_pred_time - last_hist_time

        rel_time = time_forecast - time_forecast[:, 0:1, :] + dt_since_hist.unsqueeze(1)
        if rel_time.dim() == 2:
            rel_time = rel_time.unsqueeze(-1)

        t = torch.cat(
            [torch.zeros(batch_size, 1, 1, device=self.device), rel_time],
            dim=1
        )  # [B, F+1, 1]

        # Inputs x(t): last history input + forecast inputs
        x_last_hist = excitation_history[:, -1:, :]  # [B,1,dim_x]
        x_all = torch.cat([x_last_hist, excitation_forecast], dim=1)  # [B,F+1,dim_x]

        # Configure ODE's interpolation data (assumes self.ODE.set_x exists)
        if hasattr(self.ODE, "set_x"):
            self.ODE.set_x(t, x_all)

        # Integrate ODE
        z = self.call_odeint(z0, t)  # [B, F, dim_z]

        # Here dim_z == dim_y, so projection is identity
        y = z[:, :, :self.dim_y]

        if return_latent:
            return y, z
        return y


if __name__ == "__main__":

    ##### Check TimeInterpolation with per-sample time grids #####
    import matplotlib.pyplot as plt
    # Example: two samples with different time grids
    # Sample 1: t = [0, 1, 3, 4], x = t^2
    # Sample 2: t = [0, 1, 2, 3], x = t^2 + 1

    t_grid = torch.tensor([
        [0.0, 1.0, 3.0, 4.0],
        [0.0, 1.0, 2.0, 3.0],
    ])  # [B=2, T=4]

    x_vals_1 = t_grid[0] ** 2          # sample 1
    x_vals_2 = t_grid[1] ** 2 + 1.0    # sample 2

    x_series = torch.stack([x_vals_1, x_vals_2], dim=0)  # [B=2, T=4]
    x_series = x_series.unsqueeze(-1)  # [B, T, dim_x=1]

    # Create interpolators
    interp_zoh = TimeInterpolation(interp_mode="zero")
    interp_lin = TimeInterpolation(interp_mode="linear")

    interp_zoh.set_interpolation_data(t_grid, x_series)
    interp_lin.set_interpolation_data(t_grid, x_series)

    # Query points (e.g., solver times) - dense, evenly spaced axis
    t_query_vals = torch.linspace(0.0, 4.0, steps=101)  # finer resolution

    zoh_results = []
    lin_results = []

    for t_q in t_query_vals:
        zoh_results.append(interp_zoh.interpolate_x(t_q))  # [B, 1]
        lin_results.append(interp_lin.interpolate_x(t_q))

    zoh_results = torch.stack(zoh_results, dim=0)  # [N, B, 1]
    lin_results = torch.stack(lin_results, dim=0)  # [N, B, 1]

    # Convert to numpy for plotting
    t_query_np = t_query_vals.numpy()
    zoh_np = zoh_results.squeeze(-1).numpy()  # [N, B]
    lin_np = lin_results.squeeze(-1).numpy()  # [N, B]
    t_grid_np = t_grid.numpy()
    x_np = x_series.squeeze(-1).numpy()       # [B, T]

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

    sample_titles = ["Sample 1", "Sample 2"]
    mode_titles = ["Zero-Order Hold", "Linear interpolation"]

    for b in range(2):
        # ZOH
        ax = axes[0, b]
        # interpolated values: red scatter
        ax.scatter(t_query_np, zoh_np[:, b], s=15)
        # support points: blue scatter
        ax.scatter(t_grid_np[b], x_np[b], s=40)
        ax.set_title(f"{sample_titles[b]} - {mode_titles[0]}")
        ax.set_ylabel("x(t)")
        ax.grid(True)

        # Linear
        ax = axes[1, b]
        # interpolated values: red scatter
        ax.scatter(t_query_np, lin_np[:, b], s=15)
        # support points: blue scatter
        ax.scatter(t_grid_np[b], x_np[b], s=40)
        ax.set_title(f"{sample_titles[b]} - {mode_titles[1]}")
        ax.set_xlabel("t")
        ax.set_ylabel("x(t)")
        ax.grid(True)

    fig.suptitle("Per-sample interpolation check: different time grids per sample")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
