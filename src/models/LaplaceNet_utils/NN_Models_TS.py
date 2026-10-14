import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class InitialStateNN(nn.Module):
    """
    A neural network to approximate the initial-state polynomial P(s) from history.
    The GRU state is also passed to the transfer-function network.
    """

    def __init__(self,
                 dim_m,
                 dim_history,
                 order_dim=2,
                 RNN_hidden_dim=64,
                 RNN_num_layers=1):
        """
        Args:
            dim_m (int): Number of components of P(s).
            dim_history (int): Number of features in the history sequence.
            order_dim (int): Number of polynomial coefficients per component.
            RNN_hidden_dim (int): Size of the GRU hidden state.
            RNN_num_layers (int): Number of GRU layers.
        """
        super(InitialStateNN, self).__init__()

        self.dim_m = dim_m
        self.order_dim = order_dim

        self.RNN_encoder = nn.GRU(
            input_size=dim_history,
            hidden_size=RNN_hidden_dim,
            num_layers=RNN_num_layers,
            batch_first=True,
        )
        self.P_net = nn.Linear(RNN_hidden_dim, order_dim*dim_m)


    @staticmethod
    def polynomial_function(coefficients, s_queries):
        """
        Computes a polynomial for the given s-values.

        Args:
            coefficients (Tensor): Coefficients in ascending order, (B, T, F, order).
            s_queries (Tensor): Complex query points, (B, T, 1, ilt_terms)
                                or (B, T, ilt_terms).

        Returns:
            Tensor: Polynomial values, (B, T, ilt_terms, F).
        """
        if s_queries.dim() == 3:
            s_queries = s_queries.unsqueeze(2)

        order = coefficients.shape[-1]
        B, T, F, _ = coefficients.shape
        results = torch.zeros(B, T, F, s_queries.shape[-1],
                              device=coefficients.device, dtype=s_queries.dtype)

        for i in range(order):
            c = coefficients[..., i].unsqueeze(-1)
            results += c * (s_queries ** i)

        return results.permute(0, 1, 3, 2)


    def forward(self,  s_queries: Tensor,history_sequence: Tensor,):
        """
        Encodes the history and evaluates the initial-state polynomial P(s).

        Args:
            s_queries (Tensor): Complex query points, (B, T, 1, ilt_terms).
            history_sequence (Tensor): History sequence, (B, history_len, dim_history).

        Returns:
            tuple: P(s), (B, T, ilt_terms, dim_m), and the final GRU output,
                   (B, RNN_hidden_dim).
        """

        batch_dim, time_dim, features ,ilt_terms_dim = s_queries.shape

        out, _ = self.RNN_encoder(history_sequence)
        P = self.P_net(out[:, -1, :]).view(batch_dim,self.dim_m,self.order_dim).unsqueeze(1).expand(-1, time_dim, -1, -1)

        return self.polynomial_function(P, s_queries),out[:, -1, :]

class TransferFunctionNN(nn.Module):
    """
    Neural network to approximate the complex transfer function H(s),
    conditioned on the encoded history.
    """

    def __init__(self,
                 dim_terms_ilt=1,
                 dim_m=2,
                 dim_n=2,
                 hidden_dim=64,
                 state_dim=64,
                 num_hidden_layers=3,
                 activation_fn=F.relu,
                 process_all_terms_as_features=False):
        """
        Args:
            dim_terms_ilt (int): Number of ILT query points per time step.
            dim_m (int): Number of input components of H(s).
            dim_n (int): Number of output components of H(s).
            hidden_dim (int): Size of the hidden layers.
            state_dim (int): Size of the encoded history state.
            num_hidden_layers (int): Number of hidden layers.
            activation_fn (callable): Activation after each hidden layer.
            process_all_terms_as_features (bool): Use all ILT terms as features
                                                 in the same network evaluation.
        """
        super(TransferFunctionNN, self).__init__()
        self.activation_fn = activation_fn
        self.dim_latent_state = dim_m
        self.dim_response = dim_n
        self.process_all_terms_as_features = process_all_terms_as_features

        if process_all_terms_as_features:
            input_dim = 2 * dim_terms_ilt+state_dim
            output_dim = 2 * (dim_m * dim_n * dim_terms_ilt)

        else:
            input_dim = 2 +state_dim
            output_dim = 2

        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim)
                ) for i in range(num_hidden_layers)
            ]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, grid: Tensor,state:Tensor) -> Tensor:
        """
        Computes H(s) from the grid and the encoded history state.
        The following shapes use process_all_terms_as_features=True.

        Args:
            grid (Tensor): Real grid of time and ILT term indices, (B, T, ilt_terms, 2).
            state (Tensor): Encoded history, (B, state_dim).

        Returns:
            Tensor: Complex H(s), (B, T, ilt_terms, dim_n, dim_m).
        """
        assert len(grid.shape) == 4, "Expecting a 2D grid in dims [ batch, time, ILT terms, 2]."
        batch_dim, time_dim, ilt_terms_dim,_ = grid.shape
        x=grid

        if self.process_all_terms_as_features:
            x = x.view(batch_dim, time_dim, -1)

        state=state.unsqueeze(1).expand(-1,time_dim,-1)
        x=torch.cat([state,x],dim=-1)

        for layer in self.hidden_layers:
            x = self.activation_fn(layer(x))
        x = self.output_layer(x)

        real_part, imag_part = torch.chunk(x, 2, dim=-1)
        s_complex = torch.view_as_complex(torch.stack((real_part, imag_part), dim=-1))


        return s_complex.view(batch_dim, time_dim, ilt_terms_dim, self.dim_response, self.dim_latent_state)
