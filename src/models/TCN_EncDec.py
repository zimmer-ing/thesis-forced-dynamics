# SPDX-License-Identifier: MIT
# Copyright (c) 2018 CMU Locus Lab
# See licenses/TCN-MIT.txt for the original license.

from .models_base import BaseRegressionModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

############################################################################################################
# Implementation of the Temporal Convolutional Network (TCN) from the paper
# "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" by Bai et al.
# Code adapted from https://github.com/locuslab/TCN/blob/2f8c2b817050206397458dfd1f5a25ce8a32fe65/TCN/tcn.py

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """A residual block used in Temporal Convolutional Networks."""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # First dilated convolution layer
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)  # Ensure causality by chomping
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second dilated convolution layer
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # Ensure causality by chomping
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Combine layers in a sequential block
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # Downsample if the input and output channel sizes differ
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """Initialize convolution weights with normal distribution."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        Passes the input through the block with residual connection.
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """A multi-layer Temporal Convolutional Network (TCN).

    This network consists of multiple TemporalBlock layers, each with increasing dilation
    to capture longer temporal dependencies.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, dilation_base=2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = dilation_base ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                        padding=(kernel_size-1) * dilation_size, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):

        return self.network(x)


class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network (TCN) Encoder.

    The encoder processes the input time series (forcing + response history)
    and extracts a latent representation that summarizes the past information.
    """

    def __init__(self, num_inputs, num_channels, latent_dim, kernel_size=2, dropout=0.2):
        super().__init__()
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_size, dropout)
        self.projection = nn.Linear(num_channels[-1], latent_dim)

    def forward(self, x):

        out = self.tcn(x)
        context = out[:, :, -1]
        return self.projection(context)


class TCNDecoder(nn.Module):


    def __init__(self, num_inputs, num_channels, latent_dim, outputs,kernel_size=2, dropout=0.2):
        super().__init__()
        self.tcn = TemporalConvNet(num_inputs + latent_dim, num_channels, kernel_size, dropout)
        self.fc = nn.Conv1d(num_channels[-1], outputs, kernel_size=1)

    def forward(self, x, context):


        context_expanded = context.unsqueeze(2).expand(-1, -1, x.shape[2])
        x_dec = torch.cat([x, context_expanded], dim=1)
        return self.fc(self.tcn(x_dec))


class TCNEncoderDecoder(BaseRegressionModel):
    """
    Callable TCN-based Sequence-to-Sequence Model .
    """

    def __init__(self, config, logger=None, **kwargs):
        super(TCNEncoderDecoder, self).__init__(logger=logger, **kwargs)

        # Validate required configuration keys
        required_keys = {
            'dataset': ['dim_x', 'dim_y'],
            'model': ['encoder_hidden_size', 'decoder_hidden_size', 'num_layers', 'latent_dim', 'dropout'],
            'chosen_device': None
        }
        for section, keys in required_keys.items():
            if keys is None:
                assert section in config, f"'{section}' must be provided in config"
            else:
                assert section in config, f"'{section}' section missing in config"
                for key in keys:
                    assert key in config[section], f"'{key}' must be provided in config['{section}']"


        self.input_dim = config['dataset']['dim_x']
        self.output_dim = config['dataset']['dim_y']


        self.encoder_hidden_size = config['model']['encoder_hidden_size']
        self.decoder_hidden_size = config['model']['decoder_hidden_size']
        self.num_layers = config['model']['num_layers']
        self.latent_dim = config['model']['latent_dim']
        self.dropout = config['model'].get('dropout', 0.2)
        response_only_history = config['model'].get('response_only_history', False)
        if isinstance(response_only_history, dict):
            response_only_history = response_only_history.get('default', False)
        self.response_only_history = bool(response_only_history)

        self.device = torch.device(config['chosen_device'])

        self.history_input_dim = (
            self.output_dim
            if self.response_only_history
            else self.input_dim + self.output_dim + 1
        )
        self.encoder = TCNEncoder(num_inputs=self.history_input_dim,
                                  num_channels=[self.encoder_hidden_size] * self.num_layers,
                                  latent_dim=self.latent_dim,
                                  dropout=self.dropout)


        self.decoder = TCNDecoder(num_inputs=self.input_dim + 1,
                                  num_channels=[self.decoder_hidden_size] * self.num_layers,
                                  latent_dim=self.latent_dim,
                                  outputs=self.output_dim,
                                  dropout=self.dropout)

        self.prepare_training(config)

    def forward(self, excitation_history: torch.Tensor,
                response_history: torch.Tensor,
                time_history: torch.Tensor,
                excitation_forecast: torch.Tensor,
                time_forecast: torch.Tensor) -> torch.Tensor:


        batch_size, hist_seq_len, _ = excitation_history.shape
        _, fore_seq_len, _ = excitation_forecast.shape


        if self.response_only_history:
            input_encoder = response_history
        else:
            time_history_zero = time_history - time_history[:, 0, :].unsqueeze(1)
            input_encoder = torch.cat(
                (excitation_history, response_history, time_history_zero),
                dim=-1,
            )
        input_encoder = input_encoder.permute(0, 2, 1)


        context = self.encoder(input_encoder)

        time_forecast_zero = time_forecast - time_forecast[:, 0, :].unsqueeze(1)
        input_decoder = torch.cat((excitation_forecast, time_forecast_zero), dim=-1)
        input_decoder = input_decoder.permute(0, 2, 1)
        output = self.decoder(input_decoder, context)

        return output.permute(0, 2, 1)
