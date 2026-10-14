import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.models_base import BaseRegressionModel
from src.models.NeuralODE_base import TimeInterpolation
from src.models.FDEint.FixedStepFDEint_with_forcing import FDEint



class BlackBoxFDE(nn.Module):
    """
    RHS network for a Neural Fractional Differential Equation D^alpha y(t) = f_theta(t, y(t), u(t))
    """

    def __init__(
        self,
        dim_u,
        dim_y,
        hidden_size,
        n_layers_hidden=3,
        activation="relu",
        include_time=True,
        homogeneous=False,
        interp_mode="linear",
        logger=None,
        **kwargs,
    ):
        super().__init__()
        self.logger = logger
        self.include_time = include_time
        self.homogeneous = homogeneous
        self.interpolator = TimeInterpolation(interp_mode)

        input_dim = 0
        if include_time:
            input_dim += 1
        if not homogeneous:
            input_dim += dim_u + dim_y
        else:
            input_dim += dim_y

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_size))
        for _ in range(n_layers_hidden - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, dim_y))

        if activation == "relu":
            self.activation = F.relu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "elu":
            self.activation = F.elu
        elif activation == "sigmoid":
            self.activation = torch.sigmoid
        elif activation == "none":
            self.activation = lambda x: x
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0.0)

        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def set_u(self, t: torch.Tensor, u: torch.Tensor):
        self.interpolator.set_interpolation_data(t, u)

    def interpolate_u(self, t_query: torch.Tensor) -> torch.Tensor:
        return self.interpolator.interpolate_x(t_query)

    def forward(self, t: torch.Tensor, y: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:

        if not self.homogeneous:
            if u is None:
                if t.dim() == 0:
                    u = self.interpolate_u(t)
                else:
                    raise ValueError("u is None but t is not scalar. Provide u explicitly or call from solver.")
            temp = torch.cat([y, u], dim=-1)
        else:
            temp = y

        if self.include_time:
            if t.dim() == 0:
                t_feat = t.view(1, 1).expand(y.size(0), 1)
            elif t.dim() == 1:
                t_feat = t.unsqueeze(-1)
            else:
                t_feat = t
            temp = torch.cat([t_feat, temp], dim=-1)

        for layer in self.layers[:-1]:
            temp = self.activation(layer(temp))
        out = self.layers[-1](temp)
        return out




class BaseNFDEModel(BaseRegressionModel):
    """
    Base class for Neural Fractional Differential Equation regression models.

    """

    def __init__(self, config, logger=None, **kwargs):
        super().__init__(logger=logger, **kwargs)
        self.config = config

        assert self.config["dataset"]["dim_x"] is not None, "dim_x must be provided in config"
        assert self.config["dataset"]["dim_y"] is not None, "dim_y must be provided in config"
        assert self.config["chosen_device"] is not None, "'chosen_device' must be provided in config"

        self.device = self.config["chosen_device"]
        self.dim_u = self.config["dataset"]["dim_x"]
        self.dim_y = self.config["dataset"]["dim_y"]

        model_cfg = self.config["model"]
        self.dtype = getattr(torch, model_cfg.get("dtype", "float32"))
        self.h = model_cfg.get("h", None)

        assert "alpha" in model_cfg, "For NFDE, model.alpha must be provided"
        alpha_value = float(model_cfg["alpha"])
        assert 0.0 < alpha_value <= 1.0, "alpha must be in (0,1]"

        self.learn_alpha = model_cfg.get("learn_alpha", False)
        if self.learn_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha_value, dtype=self.dtype, device=self.device))
        else:
            self.register_buffer("alpha", torch.tensor(alpha_value, dtype=self.dtype, device=self.device))

    def call_fdeint(self, y0: torch.Tensor, t: torch.Tensor, u_all: torch.Tensor) -> torch.Tensor:
        """
        Solve NFDE on forecast window.
        """
        y = FDEint(
            f=self.FDE,
            t=t,
            y0=y0,
            alpha=self.alpha,
            h=self.h,
            dtype=self.dtype,
            DEBUG=False,
            return_internals=False,
            u=u_all,
            u_fn=None,
        )
        return y[:, 1:, :]

    def forward(
        self,
        excitation_history: torch.Tensor,
        response_history: torch.Tensor,
        time_history: torch.Tensor,
        excitation_forecast: torch.Tensor,
        time_forecast: torch.Tensor,
        return_latent: bool = False,
    ):

        batch_size = excitation_history.size(0)

        y0 = response_history[:, -1, :].to(self.device, dtype=self.dtype)

        last_hist_time = time_history[:, -1, :]
        first_pred_time = time_forecast[:, 0, :]
        dt_since_hist = first_pred_time - last_hist_time

        rel_time = time_forecast - time_forecast[:, 0:1, :] + dt_since_hist.unsqueeze(1)
        if rel_time.dim() == 2:
            rel_time = rel_time.unsqueeze(-1)

        t = torch.cat(
            [torch.zeros(batch_size, 1, 1, device=self.device, dtype=self.dtype), rel_time.to(self.device, dtype=self.dtype)],
            dim=1,
        )

        u_last_hist = excitation_history[:, -1:, :].to(self.device, dtype=self.dtype)
        u_all = torch.cat([u_last_hist, excitation_forecast.to(self.device, dtype=self.dtype)], dim=1)

        if hasattr(self.FDE, "set_u"):
            self.FDE.set_u(t, u_all)

        y = self.call_fdeint(y0, t, u_all)

        if return_latent:
            return y, y
        return y


class NFDEModel(BaseNFDEModel):
    """
    Callable NFDE model in torch.
    """

    def __init__(self, config, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)

        model_cfg = self.config["model"]

        assert model_cfg["hidden_size_FDE"] > 0, "hidden_size_FDE must be > 0"
        assert model_cfg["n_layers_hidden_FDE"] > 0, "n_layers_hidden_FDE must be > 0"
        assert model_cfg["activation_FDE"] in ["relu", "leaky_relu", "tanh", "elu", "sigmoid", "none"]
        assert model_cfg["include_time"] in [True, False]
        assert model_cfg["homogeneous"] in [True, False]

        self.hidden_size = model_cfg["hidden_size_FDE"]
        self.n_layers_hidden = model_cfg["n_layers_hidden_FDE"]
        self.activation = model_cfg["activation_FDE"]
        self.include_time = model_cfg["include_time"]
        self.homogeneous = model_cfg["homogeneous"]
        interp_mode = model_cfg.get("interp_mode", "linear")

        self.FDE = BlackBoxFDE(
            dim_u=self.dim_u,
            dim_y=self.dim_y,
            hidden_size=self.hidden_size,
            n_layers_hidden=self.n_layers_hidden,
            activation=self.activation,
            include_time=self.include_time,
            homogeneous=self.homogeneous,
            interp_mode=interp_mode,
            logger=self.logger,
        ).to(self.device)

        self.prepare_training(config)


class NFDEFixedAlphaModel(NFDEModel):
    """NFDE with fixed at alpha=0.7."""



    def __init__(self, config, logger=None, fixed_alpha = 0.7, **kwargs):
        config["model"]["alpha"] = self.FIXED_ALPHA=fixed_alpha
        config["model"]["learn_alpha"] = False
        super().__init__(config, logger=logger, **kwargs)
