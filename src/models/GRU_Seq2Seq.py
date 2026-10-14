from .models_base import BaseRegressionModel
import torch
import torch.nn as nn


class GRUSeq2Seq(BaseRegressionModel):
    """
    GRU Sequence-to-Sequence model for forecasting.

    This mirrors LSTMSeq2Seq, but uses GRU cells in encoder and decoder.
    """

    def __init__(self, config, logger=None, **kwargs):
        super(GRUSeq2Seq, self).__init__(logger=logger, **kwargs)

        assert config["dataset"]["dim_x"] is not None, "dim_x must be provided in config"
        assert config["dataset"]["dim_y"] is not None, "dim_y must be provided in config"
        assert config["model"]["hidden_size"] is not None, "hidden_size must be provided in config"
        assert config["model"]["num_layers"] is not None, "num_layers must be provided in config"
        assert config["chosen_device"] is not None, "'chosen_device' must be provided in config"

        self.input_dim = config["dataset"]["dim_x"]
        self.output_dim = config["dataset"]["dim_y"]
        self.hidden_size = config["model"]["hidden_size"]
        self.num_layers = config["model"]["num_layers"]
        self.device = torch.device(config["chosen_device"])
        self.dropout = config.get("model", {}).get("dropout", 0.0)
        response_only_history = config.get("model", {}).get("response_only_history", False)
        if isinstance(response_only_history, dict):
            response_only_history = response_only_history.get("default", False)
        self.response_only_history = bool(response_only_history)
        if config["model"]["num_layers"] == 1:
            self.dropout = config["model"]["dropout"] = 0.0

        self.history_input_dim = (
            self.output_dim
            if self.response_only_history
            else self.input_dim + self.output_dim + 1
        )
        self.encoder = Encoder(
            self.num_layers,
            self.hidden_size,
            self.history_input_dim,
            self.dropout,
            self.device,
        )
        self.decoder = Decoder(
            self.num_layers,
            self.hidden_size,
            self.input_dim + 1,
            self.output_dim,
            self.dropout,
            self.device,
        )

        self.prepare_training(config)

    def forward(
        self,
        excitation_history: torch.Tensor,
        response_history: torch.Tensor,
        time_history: torch.Tensor,
        excitation_forecast: torch.Tensor,
        time_forecast: torch.Tensor,
    ) -> torch.Tensor:
        if self.response_only_history:
            input_encoder = response_history
        else:
            time_history_zero = time_history - time_history[:, 0, :].unsqueeze(1)
            input_encoder = torch.cat(
                (excitation_history, response_history, time_history_zero),
                dim=-1,
            )
        hidden = self.encoder(input_encoder)

        time_forecast_zero = time_forecast - time_forecast[:, 0, :].unsqueeze(1)
        input_decoder = torch.cat((excitation_forecast, time_forecast_zero), dim=-1)
        output_decoder, _ = self.decoder(input_decoder, hidden)

        return output_decoder


class Encoder(nn.Module):
    def __init__(self, num_layers, hidden_size, input_dim, dropout=0, device=torch.device("cpu")):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bias=True,
        ).to(device)

    def forward(self, input_seq):
        _, hidden = self.gru(input_seq)
        return hidden


class Decoder(nn.Module):
    def __init__(self, num_layers, hidden_size, input_dim, output_dim, dropout=0, device=torch.device("cpu")):
        super(Decoder, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bias=True,
        ).to(device)
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_dim).to(device)

    def forward(self, input_seq, hidden):
        output, hidden = self.gru(input_seq, hidden)
        output = self.linear(output)
        return output, hidden
