import torch
import torch.nn as nn

from src.models.NeuralODE_base import BaseNODEModel, BlackBoxODE


def _normalize_augmentation_type(augmentation_type: str) -> str:
    normalized = augmentation_type.lower()
    aliases = {
        "learned": "history_encoder",
        "encoder": "history_encoder",
        "history": "history_encoder",
        "history_encoder": "history_encoder",
        "zero": "zero",
        "zeros": "zero",
    }
    if normalized not in aliases:
        raise ValueError(
            "augmentation_type must be one of "
            "['history_encoder', 'zero']"
        )
    return aliases[normalized]


class HistoryRNNEncoder(nn.Module):
    """RNN-based encoder (LSTM or GRU)."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        rnn_type: str = "lstm",
        device="cpu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.rnn_type = rnn_type.lower()

        assert self.rnn_type in ["lstm", "gru"]

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU

        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
            bias=True,
        ).to(self.device)

    def init_hidden(self, batch_size: int,device=None):
        if device is None:
            device = self.device
        if self.rnn_type == "lstm":
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            return (h0, c0)
        else:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            return h0

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        batch_size = input_seq.size(0)
        hidden0 = self.init_hidden(batch_size,device=input_seq.device)
        _, hidden = self.rnn(input_seq, hidden0)

        if self.rnn_type == "lstm":
            h_T = hidden[0][-1]
        else:
            h_T = hidden[-1]

        return h_T


def _forward_with_history_encoder(
    self,
    excitation_history: torch.Tensor,
    response_history: torch.Tensor,
    time_history: torch.Tensor,
    excitation_forecast: torch.Tensor,
    time_forecast: torch.Tensor,
    return_latent: bool = False,
) -> torch.Tensor:
    """
    Forward pass for augmented Neural ODE models with a history encoder.
    """

    batch_size = excitation_history.size(0)
    z0 = _build_augmented_initial_state(
        self,
        excitation_history=excitation_history,
        response_history=response_history,
        time_history=time_history,
    )


    last_hist_time = time_history[:, -1, :]
    first_pred_time = time_forecast[:, 0, :]
    dt_since_hist = first_pred_time - last_hist_time

    rel_time = time_forecast - time_forecast[:, 0:1, :] + dt_since_hist.unsqueeze(1)
    if rel_time.dim() == 2:
        rel_time = rel_time.unsqueeze(-1)

    t = torch.cat(
        [torch.zeros(batch_size, 1, 1, device=self.device), rel_time],
        dim=1,
    )


    x_last_hist = excitation_history[:, -1:, :]
    x_all = torch.cat([x_last_hist, excitation_forecast], dim=1)

    if hasattr(self.ODE, "set_x"):
        self.ODE.set_x(t, x_all)

    # Integrate ODE in augmented state space
    z = self.call_odeint(z0, t)

    # Project to observed output
    y = z[:, :, :self.dim_y]

    if return_latent:
        return y, z
    return y


def _build_augmented_initial_state(
    self,
    excitation_history: torch.Tensor,
    response_history: torch.Tensor,
    time_history: torch.Tensor,
) -> torch.Tensor:
    y0 = response_history[:, -1, :]

    if self.augmentation_type == "zero":
        u0 = torch.zeros(
            y0.size(0),
            self.augmented_dim,
            device=y0.device,
            dtype=y0.dtype,
        )
    else:
        time_history_zero = time_history - time_history[:, 0:1, :]
        encoder_input = torch.cat(
            (excitation_history, response_history, time_history_zero),
            dim=-1,
        )
        h_T = self.encoder(encoder_input)
        u0 = self.fc_u0(h_T)

    return torch.cat([y0, u0], dim=-1)


class NODEfixedStepModel(BaseNODEModel):
    """Fixed-step Neural ODE with augmented state and history-based augmentation."""

    def __init__(self, config,logger=None, **kwargs):
        model_cfg = config['model']

        if 'solver' not in model_cfg:
            model_cfg['solver'] = 'rk4'

        super().__init__(config,logger=logger, **kwargs)

        assert self._is_fixed_step, (
            f"Solver '{self.solver}' is not a fixed-step solver. "
            f"Choose one of: {sorted(self._fixed_step_solvers)}"
        )

        assert model_cfg['hidden_size_ODE'] > 0
        assert model_cfg['n_layers_hidden_ODE'] > 0
        assert model_cfg['activation_ODE'] in ['relu', 'leaky_relu', 'tanh', 'elu', 'sigmoid', 'none']
        assert model_cfg['include_time'] in [True, False]
        assert model_cfg['homogeneous'] in [True, False]
        assert model_cfg.get('intermediate_timesteps', 1) > 0

        # Augmented state dimension: dim_z = dim_y + augmented_dim
        self.augmented_dim = model_cfg.get('augmented_dim_ODE', 1)
        assert self.augmented_dim >= 1
        self.dim_z = self.dim_y + self.augmented_dim

        self.hidden_size = model_cfg['hidden_size_ODE']
        self.n_layers_hidden = model_cfg['n_layers_hidden_ODE']
        self.activation = model_cfg['activation_ODE']
        self.include_time = model_cfg['include_time']
        self.homogeneous = model_cfg['homogeneous']
        self.augmentation_type = _normalize_augmentation_type(
            model_cfg.get('augmentation_type', 'history_encoder')
        )
        interp_mode = model_cfg.get('interp_mode', 'linear')

        self.ODE = BlackBoxODE(
            dim_x=self.dim_x,
            dim_z=self.dim_z,
            hidden_size=self.hidden_size,
            n_layers_hidden=self.n_layers_hidden,
            activation=self.activation,
            include_time=self.include_time,
            homogeneous=self.homogeneous,
            interp_mode=interp_mode,
            logger=self.logger,
        ).to(self.device)

        if self.augmentation_type == "history_encoder":
            encoder_hidden_size = model_cfg.get('hidden_size_encoder', self.hidden_size)
            encoder_num_layers = model_cfg.get('num_layers_encoder', 1)
            rnn_type = model_cfg.get('rnn_type', 'lstm')

            encoder_input_dim = self.dim_x + self.dim_y + 1

            self.encoder = HistoryRNNEncoder(
                input_dim=encoder_input_dim,
                hidden_size=encoder_hidden_size,
                num_layers=encoder_num_layers,
                rnn_type=rnn_type,
                device=self.device,
            )

            self.fc_u0 = nn.Linear(encoder_hidden_size, self.augmented_dim).to(self.device)
        else:
            self.encoder = None
            self.fc_u0 = None

        self.prepare_training(config)

    def forward(
        self,
        excitation_history: torch.Tensor,
        response_history: torch.Tensor,
        time_history: torch.Tensor,
        excitation_forecast: torch.Tensor,
        time_forecast: torch.Tensor,
        return_latent: bool = False,
    ) -> torch.Tensor:
        return _forward_with_history_encoder(
            self,
            excitation_history,
            response_history,
            time_history,
            excitation_forecast,
            time_forecast,
            return_latent=return_latent,
        )


class NODEadaptiveModel(BaseNODEModel):
    """Adaptive-step Neural ODE with augmented state and history-based augmentation."""

    def __init__(self, config,logger=None, **kwargs):
        model_cfg = config['model']

        if 'solver' not in model_cfg:
            model_cfg['solver'] = 'dopri5'

        super().__init__(config,logger=logger, **kwargs)

        assert self._is_adaptive, (
            f"Solver '{self.solver}' is not an adaptive solver. "
            f"Choose one of: {sorted(self._adaptive_solvers)}"
        )

        assert model_cfg['hidden_size_ODE'] > 0
        assert model_cfg['n_layers_hidden_ODE'] > 0
        assert model_cfg['activation_ODE'] in ['relu', 'leaky_relu', 'tanh', 'elu', 'sigmoid', 'none']
        assert model_cfg['include_time'] in [True, False]
        assert model_cfg['homogeneous'] in [True, False]

        self.augmented_dim = model_cfg.get('augmented_dim_ODE', 1)
        assert self.augmented_dim >= 1
        self.dim_z = self.dim_y + self.augmented_dim

        self.hidden_size = model_cfg['hidden_size_ODE']
        self.n_layers_hidden = model_cfg['n_layers_hidden_ODE']
        self.activation = model_cfg['activation_ODE']
        self.include_time = model_cfg['include_time']
        self.homogeneous = model_cfg['homogeneous']
        self.augmentation_type = _normalize_augmentation_type(
            model_cfg.get('augmentation_type', 'history_encoder')
        )
        interp_mode = model_cfg.get('interp_mode', 'linear')

        self.ODE = BlackBoxODE(
            dim_x=self.dim_x,
            dim_z=self.dim_z,
            hidden_size=self.hidden_size,
            n_layers_hidden=self.n_layers_hidden,
            activation=self.activation,
            include_time=self.include_time,
            homogeneous=self.homogeneous,
            interp_mode=interp_mode,
            logger=self.logger,
        ).to(self.device)

        if self.augmentation_type == "history_encoder":
            encoder_hidden_size = model_cfg.get('hidden_size_encoder', self.hidden_size)
            encoder_num_layers = model_cfg.get('num_layers_encoder', 1)
            rnn_type = model_cfg.get('rnn_type', 'lstm')

            encoder_input_dim = self.dim_x + self.dim_y + 1

            self.encoder = HistoryRNNEncoder(
                input_dim=encoder_input_dim,
                hidden_size=encoder_hidden_size,
                num_layers=encoder_num_layers,
                rnn_type=rnn_type,
                device=self.device,
            )

            self.fc_u0 = nn.Linear(encoder_hidden_size, self.augmented_dim).to(self.device)
        else:
            self.encoder = None
            self.fc_u0 = None

        self.prepare_training(config)

    def forward(
        self,
        excitation_history: torch.Tensor,
        response_history: torch.Tensor,
        time_history: torch.Tensor,
        excitation_forecast: torch.Tensor,
        time_forecast: torch.Tensor,
        return_latent: bool = False,
    ) -> torch.Tensor:
        return _forward_with_history_encoder(
            self,
            excitation_history,
            response_history,
            time_history,
            excitation_forecast,
            time_forecast,
            return_latent=return_latent,
        )
