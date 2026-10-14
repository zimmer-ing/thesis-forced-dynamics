from .models_base import BaseRegressionModel
import torch
import torch.nn as nn
import torch.nn.functional as F


def _config_value(value, default):
    if isinstance(value, dict) and "default" in value:
        return value["default"]
    return default if value is None else value


class FCSimple(BaseRegressionModel):
    """
    Simple feed-forward baseline:
    """

    def __init__(self, config, logger=None, **kwargs):
        super().__init__(logger=logger, **kwargs)


        assert config['dataset']['dim_x'] is not None, "dim_x must be provided in config"
        assert config['dataset']['dim_y'] is not None, "dim_y must be provided in config"
        assert config['dataset']['history_length'] is not None, "history_length must be provided in config"
        assert config['dataset']['forecast_length'] is not None, "forecast_length must be provided in config"
        assert config['chosen_device'] is not None, "'chosen_device' must be provided in config"


        self.input_dim = int(config['dataset']['dim_x'])
        self.output_dim = int(config['dataset']['dim_y'])
        self.hist_len = int(config['dataset']['history_length'])
        self.fore_len = int(config['dataset']['forecast_length'])

        model_cfg = config.get('model', {})
        self.hidden_width = int(_config_value(model_cfg.get('hidden_width'), 512))  
        self.hidden_layers = int(_config_value(model_cfg.get('hidden_layers'), 2))
        activation_name = _config_value(model_cfg.get('activation'), 'ReLU')
        self.activation_fn = getattr(nn, activation_name)
        self.use_dt = bool(_config_value(model_cfg.get('use_dt'), True))
        self.use_layernorm = bool(_config_value(model_cfg.get('layernorm'), False))
        self.response_only_history = bool(
            _config_value(model_cfg.get('response_only_history'), False)
        )

        self.device = torch.device(config['chosen_device'])

        history_input_dim = self.output_dim * self.hist_len
        if not self.response_only_history:
            history_input_dim += self.input_dim * self.hist_len + self.hist_len
        self.ff_input_dim = (
            history_input_dim
            + self.input_dim * self.fore_len
            + self.fore_len
        )
        self.ff_output_dim = self.output_dim * self.fore_len


        self.net = self._make_mlp(
            in_dim=self.ff_input_dim,
            hidden_width=self.hidden_width,
            hidden_layers=self.hidden_layers,
            out_dim=self.ff_output_dim,
            activation=self.activation_fn,
            layernorm=self.use_layernorm
        )


        self.prepare_training(config)

    def forward(self,
                excitation_history: torch.Tensor,
                response_history: torch.Tensor,
                time_history: torch.Tensor,
                excitation_forecast: torch.Tensor,
                time_forecast: torch.Tensor) -> torch.Tensor:

        B = excitation_history.shape[0]
        Tf = excitation_forecast.shape[1]


        if time_history.dim() == 2:
            time_history = time_history.unsqueeze(-1)
        if time_forecast.dim() == 2:
            time_forecast = time_forecast.unsqueeze(-1)

        if self.use_dt:
            time_fore_features = torch.cat((time_history[:, -1].unsqueeze(1), time_forecast), dim=1).diff(dim=1)
            if not self.response_only_history:
                time_hist_features = F.pad(
                    time_history.diff(dim=1),
                    (0, 0, 1, 0),
                    mode="constant",
                    value=0,
                )
        else:
            time_fore_features = time_forecast - time_forecast[:, 0:1, :]
            if not self.response_only_history:
                time_hist_features = time_history - time_history[:, 0:1, :]

        history_features = [response_history.reshape(B, -1)]
        if not self.response_only_history:
            history_features = [
                excitation_history.reshape(B, -1),
                response_history.reshape(B, -1),
                time_hist_features.reshape(B, -1),
            ]
        feat = torch.cat(
            [
                *history_features,
                excitation_forecast.reshape(B, -1),
                time_fore_features.reshape(B, -1),
            ],
            dim=-1,
        )

        y_flat = self.net(feat)
        y = y_flat.view(B, Tf, self.output_dim)
        return y

    @staticmethod
    def _make_mlp(in_dim, hidden_width, hidden_layers, out_dim, activation, layernorm=False):
        layers = []
        last = in_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(last, hidden_width))
            if layernorm:
                layers.append(nn.LayerNorm(hidden_width))
            layers.append(activation())
            last = hidden_width
        layers.append(nn.Linear(last, out_dim))
        return nn.Sequential(*layers)
