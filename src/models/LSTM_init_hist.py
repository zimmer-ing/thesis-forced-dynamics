from .models_base import BaseRegressionModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMMarkov(BaseRegressionModel):
    """
    LSTM  model for forecasting with history informed initialization of the initial state
    """

    def __init__(self, config, logger=None, **kwargs):
        super().__init__(logger=logger, **kwargs)

        # Basic config checks
        assert config["dataset"]["dim_x"] is not None, "dim_x must be provided in config"
        assert config["dataset"]["dim_y"] is not None, "dim_y must be provided in config"
        assert config["model"]["hidden_size"] is not None, "hidden_size must be provided in config"
        assert config["model"]["num_layers"] is not None, "num_layers must be provided in config"
        assert config["chosen_device"] is not None, "'chosen_device' must be provided in config"

        # Core dimensions
        self.input_dim = config["dataset"]["dim_x"]
        self.output_dim = config["dataset"]["dim_y"]
        self.hidden_size = config["model"]["hidden_size"]
        self.num_layers = config["model"]["num_layers"]
        self.device = torch.device(config["chosen_device"])

        self.dropout = config["model"].get("dropout", 0.0)
        if self.num_layers == 1:
            self.dropout = 0.0
            config["model"]["dropout"] = 0.0

        decoder_input_dim = self.input_dim + 1

        self.lstm = nn.LSTM(
            input_size=decoder_input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
            bias=True,
        ).to(self.device)


        self.linear = nn.Linear(self.hidden_size, self.output_dim).to(self.device)

        init_input_dim = self.input_dim + self.output_dim + 1
        init_hidden_size = config["model"].get("init_hidden_size", self.hidden_size)
        init_activation = config["model"].get("init_activation", "tanh").lower()

        # Choose activation
        if init_activation == "relu":
            act = F.relu
        elif init_activation == "leaky_relu":
            act = F.leaky_relu
        elif init_activation == "elu":
            act = F.elu
        elif init_activation == "tanh":
            act = torch.tanh
        elif init_activation == "none":
            act = lambda x: x
        else:
            raise ValueError(f"Unsupported init_activation '{init_activation}'.")


        self.init_activation = act
        self.init_fc1 = nn.Linear(init_input_dim, init_hidden_size).to(self.device)
        self.init_fc2 = nn.Linear(init_hidden_size, 2 * self.hidden_size).to(self.device)
        self.prepare_training(config)

    def _compute_initial_state(
        self,
        excitation_history: torch.Tensor,
        response_history: torch.Tensor,
        time_history: torch.Tensor,
    ):

        time_history_zero = time_history - time_history[:, 0:1, :]
        t_last = time_history_zero[:, -1, :]

        x_last = excitation_history[:, -1, :]
        y_last = response_history[:, -1, :]

        s0 = torch.cat([x_last, y_last, t_last], dim=-1)


        h = self.init_fc1(s0)
        h = self.init_activation(h)
        init_vec = self.init_fc2(h)


        h0 = init_vec[:, :self.hidden_size]
        c0 = init_vec[:, self.hidden_size:]


        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = c0.unsqueeze(0).repeat(self.num_layers, 1, 1)

        return h0.contiguous(), c0.contiguous()

    def forward(
        self,
        excitation_history: torch.Tensor,
        response_history: torch.Tensor,
        time_history: torch.Tensor,
        excitation_forecast: torch.Tensor,
        time_forecast: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        """
        batch_size = excitation_history.size(0)


        h0, c0 = self._compute_initial_state(
            excitation_history,
            response_history,
            time_history,
        )


        time_forecast_zero = time_forecast - time_forecast[:, 0:1, :]
        decoder_input = torch.cat([excitation_forecast, time_forecast_zero], dim=-1)
        output, _ = self.lstm(decoder_input, (h0, c0))
        predictions = self.linear(output)

        return predictions
