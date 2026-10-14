from src.models.models_base import BaseRegressionModel
import torch
import torch.nn.functional as F

from .LaplaceNet_utils.InverseLaplaceTransform import ILTFourier
from .LaplaceNet_utils.LaplaceTransform import discrete_laplace_transform,laplace_from_fft
from .LaplaceNet_utils.NN_Models_TS import InitialStateNN, TransferFunctionNN


from typing import Optional
from ..utils.helpers import ensure_sequential_dataloader, concatenate_batches,dl_is_shuffle,dataloader_hash


class LaplaceNet(BaseRegressionModel):
    """
    A modular neural Laplace model for system identification.

    Transforms the known future forcing to X(s), estimates the transfer function
    H(s) and the initial-state polynomial P(s) from the history, and computes
    Y(s) = H(s) (B X(s) + P(s)). The inverse Laplace transform reconstructs the
    response for each forecast segment.

    The dataset config defines the input and output dimensions. The model config
    defines the transforms, networks and number of forecast segments (steps).
    response_only_history selects response history or the complete history of
    known forcing, response and relative time for the history encoder.
    """

    _VALID_TRANSFORM_MODES = {"sample", "batch", "chunk"}

    def __init__(self, config, logger=None, **kwargs):
        """
        Initializes the transforms, networks and trainable input matrix B.

        Args:
            config (dict): Dataset and model settings, including chosen_device.
            logger: Optional logger passed to BaseRegressionModel.
            **kwargs: Additional arguments passed to BaseRegressionModel.
        """
        super(LaplaceNet, self).__init__(logger=logger, **kwargs)

        assert config['dataset']['dim_x'] is not None, "dim_x must be provided in config"
        assert config['dataset']['dim_y'] is not None, "dim_y must be provided in config"
        assert config['chosen_device'] is not None, "'chosen_device' must be provided in config"
        for key in ["ilt_terms", "ilt_alpha", "ilt_scale", "ilt_shift",
                    "tf_hidden_dim", "tf_num_hidden_layers", "tf_activation",
                     "is_order_dim", "is_RNN_hidden_dim", "is_RNN_num_layers",
                    "learning_rate","laplace_scaling_factor","steps"]:
            assert key in config["model"], f"'{key}' must be provided in model config"
        assert config["model"]["LP_trafo_type"] in ["FFLT", "DLT"], "LP_trafo_type must be either 'FFLT' or 'DLT'"
        self.input_dim = config['dataset']["dim_x"]
        self.output_dim = config['dataset']["dim_y"]
        self.device = torch.device(config["chosen_device"])

        config_model = config["model"]
        self.ilt_terms = config_model["ilt_terms"]
        self.ilt_alpha = config_model["ilt_alpha"]
        self.ilt_scale = config_model["ilt_scale"]
        self.ilt_shift = config_model["ilt_shift"]
        self.LP_trafo_type = config_model["LP_trafo_type"]
        self.use_laplace_scaling = config_model.get("use_laplace_scaling", True)
        self.grid_limits = config_model.get("grid_limits", [-1, 1])
        self.laplace_scaling_factor=config_model["laplace_scaling_factor"]
        self.steps=config_model["steps"]
        self.response_only_history = bool(config_model.get("response_only_history", True))
        self.dlt_mode = config_model.get("dlt_mode", "batch")
        self.fflt_mode = config_model.get("fflt_mode", "batch")
        self.transform_chunk_size = int(config_model.get("transform_chunk_size", 1000))

        for transform, mode in (("DLT", self.dlt_mode), ("FFLT", self.fflt_mode)):
            if mode not in self._VALID_TRANSFORM_MODES:
                raise ValueError(
                    f"Invalid {transform} mode '{mode}'. "
                    f"Expected one of {sorted(self._VALID_TRANSFORM_MODES)}."
                )
        if self.transform_chunk_size <= 0:
            raise ValueError("transform_chunk_size must be positive.")

        self.float_dtype = torch.float32
        self.complex_dtype = torch.complex64
        self.dl_hash_train = -1
        self.dl_hash_val = -1

        self.learning_rate = config_model["learning_rate"]


        self.ILT = ILTFourier(
            ilt_reconstruction_terms=self.ilt_terms,
            alpha=self.ilt_alpha,
            scale=self.ilt_scale,
            shift=self.ilt_shift,
            device=self.device
        )

        self.dim_n=self.input_dim
        self.dim_m=self.output_dim
        self.B = torch.nn.Parameter(torch.randn(self.dim_m ,self.dim_n), requires_grad=True)

        self.H_s_net = TransferFunctionNN(
            dim_terms_ilt=self.ilt_terms,
            dim_m=self.dim_m,
            dim_n=self.dim_m,
            hidden_dim=config_model["tf_hidden_dim"],
            num_hidden_layers=config_model["tf_num_hidden_layers"],
            state_dim=config_model["is_RNN_hidden_dim"],
            activation_fn=getattr(F, config_model.get("tf_activation", "tanh")),
            process_all_terms_as_features=config_model.get("tf_process_all_terms", True)
        )

        self.history_input_dim = (
            self.output_dim
            if self.response_only_history
            else self.input_dim + self.output_dim + 1
        )
        self.P_s_net = InitialStateNN(
            dim_m=self.dim_m,
            dim_history=self.history_input_dim,
            order_dim=config_model["is_order_dim"],
            RNN_hidden_dim=config_model["is_RNN_hidden_dim"],
            RNN_num_layers=config_model["is_RNN_num_layers"]
        )


        self.to(self.device)
        self.prepare_training(config)

    def _history_encoder_input(
            self,
            excitation_history: torch.Tensor,
            response_history: torch.Tensor,
            time_history: torch.Tensor) -> torch.Tensor:
        """Build the configured history sequence for the initial-state encoder."""
        if self.response_only_history:
            return response_history

        if time_history.dim() == 2:
            time_history = time_history.unsqueeze(-1)
        time_history_zero = time_history - time_history[:, 0:1, :]
        return torch.cat(
            (excitation_history, response_history, time_history_zero),
            dim=-1,
        )

    @staticmethod
    def _step_bounds(forecast_len: int, steps: int) -> list[tuple[int, int]]:
        """Return a balanced, contiguous partition of the forecast horizon."""
        forecast_len = int(forecast_len)
        steps = int(steps)
        if steps < 1:
            raise ValueError("steps must be at least 1.")
        if forecast_len < 2 * steps:
            raise ValueError(
                "Each Laplace-Net step requires at least two forecast samples; "
                f"got forecast_len={forecast_len} and steps={steps}."
            )
        return [
            (i * forecast_len // steps, (i + 1) * forecast_len // steps)
            for i in range(steps)
        ]

    def forward(self,
                excitation_history: torch.Tensor,
                response_history: torch.Tensor,
                time_history: torch.Tensor,
                excitation_forecast: torch.Tensor,
                time_forecast: torch.Tensor,
                X_s=None) -> torch.Tensor:
        """
        Forward pass for the modular neural Laplace model.

        Args:
            excitation_history (torch.Tensor): Known forcing history (batch, hist_seq_len, input_dim).
            response_history (torch.Tensor): Historical response signal (batch, hist_seq_len, output_dim).
            time_history (torch.Tensor): Time stamps for the historical signal (batch, hist_seq_len, 1).
            excitation_forecast (torch.Tensor): Known future forcing (batch, forecast_seq_len, input_dim).
            time_forecast (torch.Tensor): Time stamps for the forecast period (batch, forecast_seq_len, 1).
            X_s (list, optional): Precomputed X(s) for each forecast segment,
                                  with shape (batch, segment_len, ilt_terms, input_dim).

        Returns:
            torch.Tensor: Reconstructed forecast response signal (batch, forecast_seq_len, output_dim).
        """
        device = self._infer_device()
        batch_size, forecast_len, _ = excitation_forecast.shape
        step_bounds = self._step_bounds(forecast_len, self.steps)


        # 1. Compute X(s) for the known future forcing using DLT or FFLT.
        if X_s is None:
            X_s=self.calculate_X_S_steps(excitation_forecast,time_forecast)


        history_encoder_iter = self._history_encoder_input(
            excitation_history,
            response_history,
            time_history,
        )
        output=torch.zeros(batch_size, forecast_len, self.output_dim, device=device)
        for i, (start, end) in enumerate(step_bounds):

            # 2. Generate the grid for the transfer-function network.
            grid = self.ILT.generate_time_grid(end-start, self.ilt_terms, self.grid_limits)
            grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

            time_forecast_step = time_forecast[:, start:end, :]
            min_dt = time_forecast_step.diff(1, dim=1).min(dim=1).values.unsqueeze(-1)
            time_forecast_step = time_forecast_step - time_forecast_step[:, 0, :].unsqueeze(1) + min_dt
            act_X_s=X_s[i].to(device)
            query_s_forecast_step = self.ILT.get_query_points(time_forecast_step)
            # 3. Compute P(s) and the encoded history state.
            P_s,state = self.P_s_net(query_s_forecast_step, history_encoder_iter)
            # 4. Compute H(s) and apply Laplace scaling if enabled.
            H_s = self.H_s_net(grid,state)
            if self.use_laplace_scaling:
                H_s = self.ILT.inverse_transform_LP_func(H_s, time_forecast_step)/self.laplace_scaling_factor
            # 5. Combine B X(s) and P(s).
            B_complex = self.B.to(self.complex_dtype)
            B_Sum_P_X = torch.matmul(B_complex, act_X_s.unsqueeze(-1)) + P_s.to(act_X_s.dtype).unsqueeze(-1)
            # 6. Compute Y(s) = H(s) (B X(s) + P(s)).
            Y_s = torch.matmul(H_s, B_Sum_P_X).squeeze(-1)

            # 7. Reconstruct the response with the inverse Laplace transform.
            response_forecast_step = self.ILT.reconstruct(Y_s, time_forecast_step)
            output[:, start:end, :] = response_forecast_step

            if self.response_only_history:
                history_encoder_iter = response_forecast_step
            else:
                history_encoder_iter = self._history_encoder_input(
                    excitation_forecast[:, start:end, :],
                    response_forecast_step,
                    time_forecast[:, start:end, :],
                )

        return output


    def calculate_X_S_steps(self, excitation_forecast, time_forecast):
        """
        Calculates X(s) separately for each forecast segment using local time.

        Args:
            excitation_forecast (torch.Tensor): Known forcing (batch, forecast_len, input_dim).
            time_forecast (torch.Tensor): Time stamps (batch, forecast_len, 1).

        Returns:
            list: Complex transforms (batch, segment_len, ilt_terms, input_dim),
                  in forecast order, using the configured LP_trafo_type.
        """
        batch_size, forecast_len, _ = excitation_forecast.shape
        step_bounds = self._step_bounds(forecast_len, self.steps)
        X_s = []
        for start, end in step_bounds:
            time_forecast_step = time_forecast[:, start:end, :]
            min_dt = time_forecast_step.diff(1, dim=1).min(dim=1).values.unsqueeze(-1)
            time_forecast_step = time_forecast_step - time_forecast_step[:, 0, :].unsqueeze(1) + min_dt

            excitation_forecast_step = excitation_forecast[:, start:end, :]
            X_s_step = self.calculate_X_S(excitation_forecast_step, time_forecast_step, LT_type=self.LP_trafo_type,
                                          low_prec=True)
            X_s.append(X_s_step)
        return X_s

    def calculate_X_S(self, excitation_forecast, time_forecast, low_prec=True, LT_type="FFLT"):
        """
        Calculates the Laplace transform X(s) of the known forcing.

        Args:
            excitation_forecast (torch.Tensor): Known forcing (batch, seq_len, input_dim).
            time_forecast (torch.Tensor): Time stamps (batch, seq_len, 1).
            low_prec (bool): Use float32/complex64 rather than float64/complex128
                             for the transform calculation. Defaults to True.
            LT_type (str): Transform method, "FFLT" or "DLT". Defaults to "FFLT".

        Returns:
            torch.Tensor: Complex64 X(s), (batch, seq_len, ilt_terms, input_dim).
        """
        assert LT_type in ["FFLT", "DLT"], "LT_type must be either 'FFLT' or 'DLT'"
        if low_prec:
            dtype_float = torch.float32
            dtype_complex = torch.complex64
        else:
            dtype_float = torch.float64
            dtype_complex = torch.complex128

        with torch.no_grad():
            excitation_forecast = excitation_forecast.to(dtype_float)
            time_forecast = time_forecast.to(dtype_float)
            query_s_forecast = self.ILT.get_query_points(time_forecast).to(dtype_complex)
            if LT_type == "FFLT":
                X_s = laplace_from_fft(
                    excitation_forecast,
                    query_s_forecast,
                    time_forecast + self.ilt_shift,
                    mode=self.fflt_mode,
                    chunk_size=self.transform_chunk_size,
                )
            elif LT_type == "DLT":
                X_s = discrete_laplace_transform(
                    excitation_forecast,
                    query_s_forecast,
                    time_forecast + self.ilt_shift,
                    mode=self.dlt_mode,
                    chunk_size=self.transform_chunk_size,
                )
            else:
                raise ValueError("Unknown LT type")

        return X_s.to(self.complex_dtype)

    def train_step(self, data_loader: torch.utils.data.DataLoader, X_S=None, return_loss: bool = False) -> Optional[
        float]:
        """
        Trains the model for one epoch on the given data loader.
        For a sequential loader, X(s) is cached while the loader hash is unchanged.

        Args:
            data_loader (DataLoader): Training data.
            X_S (list, optional): Precomputed transforms per batch, each containing
                                  the list of forecast-segment transforms.
            return_loss (bool): Return the mean batch loss. Defaults to False.

        Returns:
            Optional[float]: Mean batch loss if requested, otherwise None.
        """
        losses = []
        self.train()
        if X_S is None:
            if not dl_is_shuffle(data_loader):
                hash = dataloader_hash(data_loader)
                if self.dl_hash_train != hash:
                    self.stored_X_S_train = []
                    for batch in data_loader:
                        excitation_forecast = batch['excitation_forecast']
                        time_forecast = batch['time_forecast']
                        X_S_batch = self.calculate_X_S_steps(excitation_forecast, time_forecast)
                        self.stored_X_S_train.append(X_S_batch)
                        X_S = self.stored_X_S_train
                    self.dl_hash_train = hash
                else:
                    X_S = self.stored_X_S_train

            else:
                X_S = [None] * len(data_loader)
        for batch, X_S_batch in zip(data_loader, X_S):
            excitation_history = batch['excitation_history']
            response_history = batch['response_history']
            excitation_forecast = batch['excitation_forecast']
            response_forecast = batch['response_forecast']
            time_history = batch['time_history']
            time_forecast = batch['time_forecast']
            self.optimizer.zero_grad()
            predictions = self(excitation_history, response_history, time_history, excitation_forecast, time_forecast,
                               X_S_batch, )
            loss = self.calculate_loss(predictions, response_forecast.to(self.float_dtype))
            loss.backward()

            if return_loss:
                losses.append(loss.item())
            self.optimizer.step()
        if return_loss:
            return sum(losses) / len(losses)

    def validate_step(self, data_loader: torch.utils.data.DataLoader, X_S=None) -> float:
        """
        Validates the model without updating its parameters.
        For a sequential loader, X(s) is cached while the loader hash is unchanged.

        Args:
            data_loader (DataLoader): Validation data.
            X_S (list, optional): Precomputed transforms per batch, each containing
                                  the list of forecast-segment transforms.

        Returns:
            float: Mean of the batch losses.
        """
        losses = []
        self.eval()
        if X_S is None:
            if not dl_is_shuffle(data_loader):
                hash = dataloader_hash(data_loader)
                if self.dl_hash_val != hash:
                    self.stored_X_S_val = []
                    for batch in data_loader:
                        excitation_forecast = batch['excitation_forecast']
                        time_forecast = batch['time_forecast']
                        X_S_batch = self.calculate_X_S_steps(excitation_forecast, time_forecast)
                        self.stored_X_S_val.append(X_S_batch)
                        X_S = self.stored_X_S_val
                    self.dl_hash_val = hash
                else:
                    X_S = self.stored_X_S_val

            else:
                X_S = [None] * len(data_loader)
        with torch.no_grad():
            for batch, X_S_batch in zip(data_loader, X_S):
                excitation_history = batch['excitation_history']
                response_history = batch['response_history']
                excitation_forecast = batch['excitation_forecast']
                response_forecast = batch['response_forecast']
                time_history = batch['time_history']
                time_forecast = batch['time_forecast']
                predictions = self(excitation_history, response_history, time_history, excitation_forecast,
                                   time_forecast, X_S_batch)
                loss = self.calculate_loss(predictions, response_forecast)
                losses.append(loss.item())
            if len(losses) == 0:
                raise ValueError("Validation set is empty.")
            return sum(losses) / len(losses)

    def predict(self, data_loader: torch.utils.data.DataLoader, X_S=None, samples_only: bool = False) -> dict:
        """
        Predicts responses in sequential data-loader order.

        Args:
            data_loader (DataLoader): Data for prediction.
            X_S (list, optional): Precomputed transforms in data-loader order.
            samples_only (bool): Keep the results as lists of batches.
                                 Defaults to False.

        Returns:
            dict: Input batch data and predictions. With samples_only=False,
                  tensor sequences are merged into (total_time_points, features).
        """
        results = {}
        target_device= self._infer_device()
        data_loader = ensure_sequential_dataloader(data_loader)
        self.eval()
        if X_S is None:
            X_S = [None] * len(data_loader)
        with torch.no_grad():
            for batch, X_S_batch in zip(data_loader, X_S):
                if not results:
                    results = {key: [] for key in batch.keys()}
                    results[
                        "predictions"] = []

                for key, value in batch.items():
                    results[key].append(value)

                if X_S_batch is not None:
                    X_S_batch.to(target_device)

                y_hat = self(
                    batch.get("excitation_history").to(target_device),
                    batch.get("response_history").to(target_device),
                    batch.get("time_history").to(target_device),
                    batch.get("excitation_forecast").to(target_device),
                    batch.get("time_forecast").to(target_device)
                    , X_S_batch
                )
                results["predictions"].append(y_hat)

        if samples_only:
            return results

        for key in results.keys():
            results[key] = concatenate_batches(results[key])

        return results
