import torch
import torch.nn as nn


class ILTFourier(nn.Module):
    r"""
    Inverse Laplace Transform (ILT) using a Fourier-series approximation.
    Computes the Laplace query points and reconstructs time-domain trajectories.
    """

    def __init__(
            self,
            ilt_reconstruction_terms=33,
            alpha=1.0e-3,
            tol=None,
            scale=2.0,
            f_max=None,
            shift=0.0,
            torch_float_dtype=torch.float32,
            torch_complex_dtype=torch.complex64,
            device=torch.device('cpu')
    ):
        r"""
        Args:
            ilt_reconstruction_terms (int): Number of Fourier reconstruction terms.
            alpha (float): Real shift of the Bromwich contour.
            tol (float): Contour tolerance; defaults to 10 * alpha.
            scale (float): Time scaling, T = scale * (time_axis + shift).
            f_max (float): Sets scale to (ilt_reconstruction_terms - 1) / (2 * f_max)
                           when provided.
            shift (float): Offset added to reconstruction times.
            torch_float_dtype (torch.dtype): Real-valued dtype, default torch.float32.
            torch_complex_dtype (torch.dtype): Complex dtype, default torch.complex64.
            device (torch.device): Initial device for the reconstruction terms.
        """
        super().__init__()
        self.ilt_reconstruction_terms = ilt_reconstruction_terms
        self.alpha = alpha
        self.tol = tol if tol is not None else 10 * alpha
        if f_max is not None:
            self.scale = (ilt_reconstruction_terms- 1) / (2 * f_max)
        else:
            self.scale = scale
        self.shift = shift
        self.torch_float_dtype = torch_float_dtype
        self.torch_complex_dtype = torch_complex_dtype

        # Buffers follow the module's device.
        k = torch.arange(
            ilt_reconstruction_terms,
            dtype=self.torch_float_dtype,
            device=device,
        )
        self.register_buffer("k", k,persistent=False)
        self.register_buffer(
            "imag_k_pi",
            (1j * torch.pi * self.k.view(1, -1)).to(self.torch_complex_dtype),
        persistent=False)


    def get_query_points(self, time_axis):
        r"""
        Generate Laplace query points s = alpha - log(tol) / T + i * pi * k / T,
        where T = scale * (time_axis + shift).

        Args:
            time_axis (Tensor): Time points, (T,), (B, T), or (B, T, 1).
                                time_axis + shift must be positive.

        Returns:
            Tensor: Complex query points with an added ILT term dimension:
                    (T, ilt_reconstruction_terms), (B, T, ilt_reconstruction_terms),
                    or (B, T, 1, ilt_reconstruction_terms). With one ILT term,
                    the final singleton dimension is removed.
        """

        min_t = torch.min(time_axis + self.shift).item()
        assert min_t > 0, "Time axis must start with non-zero time."
        t = (time_axis+self.shift).unsqueeze(-1).to(dtype=self.torch_complex_dtype)
        T = (t * self.scale).to(self.torch_complex_dtype)
        tol_tensor = torch.tensor(self.tol, dtype=self.torch_complex_dtype, device=time_axis.device)
        real = self.alpha - torch.log(tol_tensor) / T
        imag=(self.imag_k_pi / T).to(self.torch_complex_dtype)

        s = real + imag
        return s.squeeze(-1)

    def exponential_scaling(self, time_axis):
        r"""
        Compute exp(sigma * t) / (scale * t), with t = time_axis + shift
        and sigma the real part of the Laplace query points.

        Args:
            time_axis (Tensor): Time points, (T,), (B, T), or (B, T, 1).

        Returns:
            Tensor: Real scaling factors with the same shape as time_axis.
        """

        query_s=self.get_query_points(time_axis)
        time_axis = time_axis + self.shift
        sigma=query_s.real[...,0]
        T=(time_axis*self.scale).to(self.torch_float_dtype)
        exp_sigma_t = torch.exp(sigma * time_axis)
        scaling_factor = (1 / T) * exp_sigma_t
        return scaling_factor.to(self.torch_float_dtype)

    def reconstruct(self, Y_s, time_axis, exp_scaling=True):
        r"""
        Reconstruct time-domain trajectories with the Fourier ILT.

        Args:
            Y_s (Tensor): Laplace representation, (B, T, ilt_reconstruction_terms, F).
            time_axis (Tensor): Positive times, (B, T) or (B, T, 1).
                                A vector (T,) can be used for a single batch item.
            exp_scaling (bool): Apply exponential reconstruction scaling.

        Returns:
            Tensor: Real reconstructed trajectories, (B, T, F).
        """
        assert len(Y_s.shape)==4 , "Y_s must be of shape (BatchSize, SeqLen, ilt_reconstruction_terms, d_obs)"
        assert time_axis.min()>0, "time_axis must start with non-zero time"

        Y_s = Y_s.permute(0, 1, 3, 2).to(dtype=self.torch_complex_dtype)

        # The zero-frequency term has half weight.
        first_term = Y_s[:, :, :, 0] / 2.0

        exp_k = torch.exp(self.imag_k_pi[:, 1:]/self.scale)
        summation_terms = torch.sum((Y_s[:, :, :, 1:] * exp_k).real, dim=-1)


        if exp_scaling:
            scaling_factor = self.exponential_scaling(time_axis)
            reconstructed = scaling_factor.view(Y_s.size(0), Y_s.size(1), 1) * (first_term + summation_terms)
        else:
            reconstructed = first_term + summation_terms

        return reconstructed.real

    def generate_time_grid(self, time_points: int, ilt_terms: int=0,value_range=[-1,1]):
        r"""
        Generate the grid of time and ILT term indices used by the network.

        Args:
            time_points (int): Number of time points, greater than one.
            ilt_terms (int): Number of terms, greater than one; zero uses
                             ilt_reconstruction_terms.
            value_range (list): Lower and upper limits for both grid coordinates.

        Returns:
            Tensor: Real grid, (time_points, ilt_terms, 2).
        """
        if ilt_terms==0:
            ilt_terms=self.ilt_reconstruction_terms

        device = self.imag_k_pi.device
        time_indices = (
            torch.arange(time_points, device=device)
            /(time_points-1)*(value_range[1]-value_range[0])+value_range[0]
        )
        ilt_indices = (
            torch.arange(ilt_terms, device=device)
            /(ilt_terms-1)*(value_range[1]-value_range[0])+value_range[0]
        )
        grid = torch.cartesian_prod(time_indices, ilt_indices)


        return grid.view(time_points , ilt_terms, 2).to(self.imag_k_pi.device).to(self.torch_float_dtype)

    def forward(self, Y_s: torch.Tensor, time_axis: torch.Tensor):
        r"""
        Reconstruct trajectories from Y_s, (B, T, ilt_reconstruction_terms, F).
        time_axis is (B, T) or (B, T, 1); the output is (B, T, F).
        """
        return self.reconstruct(Y_s, time_axis)


    def inverse_transform_LP_func(self, Y_s_transformed, time_axis):
        r"""
        Remove exponential reconstruction scaling from Laplace-domain values.

        Args:
            Y_s_transformed (Tensor): Values, (B, T, ilt_terms, F1, F2)
                                      or (B, T, ilt_terms, F1).
            time_axis (Tensor): Time points, (B, T) or (B, T, 1), before adding shift.

        Returns:
            Tensor: Scaled values, (B, T, ilt_terms, F1, F2).
                    A four-dimensional input receives a final dimension of size one.
        """
        time_axis = time_axis + self.shift
        batch_size = Y_s_transformed.size(0)
        timesteps = Y_s_transformed.size(1)
        ILT_terms = Y_s_transformed.size(2)
        features1 = Y_s_transformed.size(3)
        if len(Y_s_transformed.shape)==4:
            Y_s_transformed = Y_s_transformed.unsqueeze(-1)
        features2=Y_s_transformed.size(4)
        Y_s_transformed = Y_s_transformed.permute(0, 2, 3, 4, 1)

        # Inverse of exp(sigma * t) / (scale * t).
        scaling_factor = (
                self.scale
                * time_axis
                * self.tol ** (1 / self.scale)
                * torch.exp(-self.alpha * time_axis)
        )
        scaling_factor =scaling_factor.view(batch_size,1,1,1,timesteps).repeat(1,ILT_terms,features1,features2,1)
        Y_s = (Y_s_transformed * scaling_factor)
        return Y_s.permute(0,4,1,2,3)
