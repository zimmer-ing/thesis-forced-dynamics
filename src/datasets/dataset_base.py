from abc import abstractmethod, ABC
import Constants as const
import json
from typing import List, Dict, Optional, Union, Any
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
import joblib


_DEFAULT_CONFIG_WARNINGS_EMITTED: set[str] = set()


def _warn_default_config_once(key: str, message: str) -> None:
    if key in _DEFAULT_CONFIG_WARNINGS_EMITTED:
        return
    _DEFAULT_CONFIG_WARNINGS_EMITTED.add(key)
    warnings.warn(message, UserWarning, stacklevel=3)


def _resolve_split_config_value(config: dict, key: str, data_split: Optional[str], default=None):
    """
    Resolve a config value with optional split-specific overrides.

    Supported patterns:
        - config[key]
        - config[f"{key}_{data_split}"]
        - config["split_overrides"][data_split][key]
    """
    split_overrides = config.get("split_overrides", {})
    if data_split and isinstance(split_overrides, dict):
        split_cfg = split_overrides.get(data_split, {})
        if isinstance(split_cfg, dict) and key in split_cfg:
            return split_cfg[key]

    if data_split:
        split_key = f"{key}_{data_split}"
        if split_key in config:
            return config[split_key]

    return config.get(key, default)

def save_scaler_details(scaler: Union[MinMaxScaler, StandardScaler],
                        scaled_columns: List[str],
                        exclude_columns: List[str],
                        output_dir: Path,
                        filename_base: str = "scaler_params") -> Dict[str, Path]:
    """
    Saves a fitted MinMaxScaler or StandardScaler as a pickle and JSON file.

    Args:
        scaler: The trained MinMaxScaler or StandardScaler object.
        scaled_columns: A list of column names that were transformed by the scaler.
                        The order must match the internal order of the scaler.
        exclude_columns: Column names that were not scaled.
        output_dir: Directory for the saved files.
        filename_base: Base name for the .pkl and .json files.

    Returns:
        Paths to the saved files under the keys 'pkl' and 'json'.

    Parameters are stored under 'scaling_details', with keys such as 'min',
    'scale', and 'mean'.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    scaler_pkl_path = output_dir / f"{filename_base}.pkl"
    scaler_json_path = output_dir / f"{filename_base}.json"

    n_features = 0
    if isinstance(scaler, MinMaxScaler) or isinstance(scaler, StandardScaler):
        if hasattr(scaler, 'n_features_in_'):
             n_features = scaler.n_features_in_
        elif hasattr(scaler, 'scale_'): # Older sklearn versions
             n_features = len(scaler.scale_)
        else:
             print("Warning: Could not verify the number of features in the scaler.")

        if n_features > 0 and len(scaled_columns) != n_features:
             raise ValueError(f"Number of scaled_columns ({len(scaled_columns)}) does not match "
                              f"the expected features of the scaler ({n_features}).")

    joblib.dump(scaler, scaler_pkl_path)

    details = {
        "exclude_columns": exclude_columns,
        "scaled_columns": scaled_columns,
        "scaling_details": {}
    }

    if isinstance(scaler, MinMaxScaler):
        details["scaler_type"] = "minmax"
        for i, col in enumerate(scaled_columns):
            details["scaling_details"][col] = {
                "min": float(scaler.min_[i]),
                "scale": float(scaler.scale_[i]),
                "data_min": float(scaler.data_min_[i]),
                "data_max": float(scaler.data_max_[i]),
                "data_range": float(scaler.data_range_[i]),
            }
    elif isinstance(scaler, StandardScaler):
        details["scaler_type"] = "standard"
        for i, col in enumerate(scaled_columns):
             details["scaling_details"][col] = {
                 "mean": float(scaler.mean_[i]) if hasattr(scaler, 'mean_') else 0.0,
                 "scale": float(scaler.scale_[i]) if hasattr(scaler, 'scale_') else 1.0,
                 "var": float(scaler.var_[i]) if hasattr(scaler, 'var_') else 1.0
             }
    else:
        raise TypeError("Scaler must be a MinMaxScaler or StandardScaler.")

    with open(scaler_json_path, "w") as fp:
        json.dump(details, fp, indent=4)

    print(f"Scaler saved as Pickle: {scaler_pkl_path}")
    print(f"Scaler details saved as JSON: {scaler_json_path}")

    return {"pkl": scaler_pkl_path, "json": scaler_json_path}


class MultiFileTimeSeriesDataset(Dataset):
    """
    Dataset class for managing prepared time series files for training, validation, or testing.

    Each file is loaded separately and split into history/forecast samples.
    Values are used as stored; saved scalers are loaded for inverse transformation.
    """

    def __init__(
        self,
        directory: str,
        files: List[str],
        config: dict,
        device: Optional[str] = 'cpu',
        data_split: Optional[str] = None
    ):
        """
        Initializes the dataset for a specific data split and loads scalers if available.

        Args:
            directory (str): Directory containing the prepared dataset files.
            files (List[str]): File names belonging to this split.
            config (dict): Configuration for the task and subsequence generation.
            device (str, optional): Device for the time, known forcing, and response tensors.
            data_split (str, optional): Split name for configuration overrides.
        """
        self.directory = Path(directory)
        self.config = config
        self.device = device
        self.files = files
        self.data_split = data_split
        self.scalers = self.load_scalers()
        self.timeseries_datasets = self.load_datasets()

        self.len_subsets = [len(dataset) for dataset in self.timeseries_datasets]
        self.cumulative_len_subsets = np.cumsum(self.len_subsets)

    def load_scaler_dict(self):
        """Loads scaler_params.json, or returns None if it cannot be loaded."""
        scaler_path=self.directory
        scaler_params = None
        try:
            with open(scaler_path / 'scaler_params.json', 'r') as f:
                scaler_params = json.load(f)
        except FileNotFoundError:
            if const.DEBUG:
                print(
                    f"Scaler parameters file 'scaler_params.json' not found in {scaler_path}. Skipping as no scalers are available.")
        except Exception as e:
            print(f"Error loading scalers: {e}")
        return scaler_params

    def load_scalers(self) -> Dict[str, any]:
        """
        Loads saved scaler parameters for the known forcing and response channels.

        Channel names come from get_channels(). Parameters are read from
        'scaling_details' using scale/min for min-max scaling or scale/mean/var
        for standard scaling, with or without a trailing underscore. Missing
        parameters use unit scale and zero offset.

        Returns:
            dict: Available scalers under 'excitation' and 'response'. These are
                used for inverse transformation, not for scaling loaded files.
        """
        scalers = {}
        scaler_path = self.directory
        try:
            with open(scaler_path / 'scaler_params.json', 'r') as f:
                scaler_params = json.load(f)

            scaler_type = scaler_params.get('scaler_type', 'none')
            if scaler_type == 'none':
                return scalers

            channels = self.get_channels()
            excitation_channels = channels['excitation']
            response_channels = channels['response']

            def configure_scaler_for_channels(scaler, channels, params_by_channel):
                scales = []
                mins = []
                means = []
                variances = []
                for channel in channels:
                    channel_params = params_by_channel.get(channel, {})
                    scales.append(channel_params.get('scale_', channel_params.get('scale', 1.0)))
                    mins.append(channel_params.get('min_', channel_params.get('min', 0.0)) if scaler_type == 'minmax' else None)
                    means.append(channel_params.get('mean_', channel_params.get('mean', 0.0)) if scaler_type == 'standard' else None)
                    variances.append(channel_params.get('var_', channel_params.get('var', 1.0)) if scaler_type == 'standard' else None)

                scaler.scale_ = np.array(scales)
                if scaler_type == 'minmax':
                    scaler.min_ = np.array(mins)
                elif scaler_type == 'standard':
                    scaler.mean_ = np.array(means)
                    scaler.var_ = np.array(variances)
                return scaler

            if scaler_type == 'minmax':
                excitation_scaler = MinMaxScaler()
            elif scaler_type == 'standard':
                excitation_scaler = StandardScaler()
            else:
                raise ValueError(f"Invalid scaler type: {scaler_type}")
            excitation_scaler = configure_scaler_for_channels(
                excitation_scaler, excitation_channels, scaler_params['scaling_details']
            )

            if scaler_type == 'minmax':
                response_scaler = MinMaxScaler()
            elif scaler_type == 'standard':
                response_scaler = StandardScaler()
            else:
                raise ValueError(f"Invalid scaler type: {scaler_type}")
            response_scaler = configure_scaler_for_channels(
                response_scaler, response_channels, scaler_params['scaling_details']
            )

            scalers['excitation'] = excitation_scaler
            scalers['response'] = response_scaler

        except FileNotFoundError:
            if const.DEBUG:
                print(f"Scaler parameters file 'scaler_params.json' not found in {scaler_path}. Skipping as no scalers are available.")
        except Exception as e:
            print(f"Error loading scalers: {e}")

        return scalers


    def load_datasets(self) -> List[Dataset]:
        """
        Loads one forecasting or reconstruction dataset per file, as selected by config['task'].
        """
        task = str(self.config.get("task", "forecasting")).lower()
        if task == "forecasting":
            dataset_cls = TimeSeriesForecastingDataset
        elif task == "reconstruction":
            dataset_cls = TimeSeriesReconstructionDataset
        else:
            raise ValueError(
                f"Unsupported dataset task '{task}'. Use 'forecasting' or 'reconstruction'."
            )

        datasets = []
        for file_name in self.files:
            data = self.load_file(self.directory / file_name)
            dataset = dataset_cls(
                data,
                self.config,
                self.get_channels(),
                device=self.device,
                data_split=self.data_split
            )
            datasets.append(dataset)
        return datasets

    def __len__(self):
        """Returns the total number of samples across all files."""
        return sum(self.len_subsets)

    def __getitem__(self, index: int):
        """Retrieves a sample by locating its file and local index."""
        dataset_index = np.argmax(self.cumulative_len_subsets > index)
        if dataset_index == 0:
            sample_index = index
        else:
            sample_index = index - self.cumulative_len_subsets[dataset_index - 1]
        return self.timeseries_datasets[dataset_index][sample_index]

    def inverse_transform(self, excitation, response, return_as_dataframe: bool = True):
        """
        Applies available inverse scalers to known forcing and response data.

        Args:
            excitation: Known forcing values with shape (N, number of forcing channels).
            response: Response values with shape (N, number of response channels).
            return_as_dataframe (bool): Return named DataFrames if True.

        Returns:
            tuple: Known forcing and response data. Inputs may be tensors, NumPy arrays,
                or DataFrames; channels without a saved scaler are left unchanged.
        """
        if isinstance(excitation, torch.Tensor):
            excitation = excitation.cpu().numpy()
        if isinstance(response, torch.Tensor):
            response = response.cpu().numpy()

        if 'excitation' in self.scalers:
            excitation = self.scalers['excitation'].inverse_transform(excitation)

        if 'response' in self.scalers:
            response = self.scalers['response'].inverse_transform(response)

        if return_as_dataframe:
            excitation_columns = self.get_channels()['excitation']
            response_columns = self.get_channels()['response']
            excitation = pd.DataFrame(excitation, columns=excitation_columns)
            response = pd.DataFrame(response, columns=response_columns)


        return excitation, response

    def get_channels(self) -> Dict[str, List[str]]:
        """Returns column names for time, known forcing ('excitation'), response, and auxiliary data."""
        return {
            "time": ["time"],
            "excitation": ["excitation"],
            "response": ["response"],
            "auxiliary": ["aux1", "aux2"]
        }

    def load_file(self, file_path: Path) -> pd.DataFrame:
        """Loads a CSV file. Subclasses can override this for other file formats."""
        return pd.read_csv(file_path)

    @staticmethod
    def collate_fn(batch):
        """Combines sample dictionaries, stacking tensors and collecting other values in lists."""
        collated = {}
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            collated[key] = torch.stack(values) if torch.is_tensor(values[0]) else values
        return collated

class TimeSeriesForecastingDataset(Dataset):
    """
    Dataset class for creating history and forecast subsequences from a single file.

    data_sequences contains sample dictionaries with time, known forcing, response,
    and auxiliary data. Prepared values are used without further scaling.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: dict,
        channels: Dict[str, List[str]],
        device: Optional[str] = 'cpu',
        data_split: Optional[str] = None
    ):
        """
        Initializes the dataset for time series forecasting.

        Args:
            data (pd.DataFrame): Prepared time series data.
            config (dict): Configuration for subsequence generation.
            channels (Dict[str, List[str]]): Column names for time, 'excitation'
                (known forcing), response, and optional auxiliary data.
            device (str, optional): Device for the time, known forcing, and response tensors.
            data_split (str, optional): Split name for configuration overrides.
        """
        self.config = config
        self.channels = channels
        self.device = device
        self.data_split = data_split
        self.data_sequences = self.create_subsequences(
            data,
            config,
            channels,
            device=self.device,
            data_split=self.data_split
        )

    def __len__(self) -> int:
        """Returns the number of subsequences in the dataset."""
        return len(self.data_sequences)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Returns a sample dictionary with history/forecast tensors and auxiliary data."""
        return self.data_sequences[index]

    @staticmethod
    def collate_fn(batch):
        """Combines sample dictionaries, stacking tensors and collecting other values in lists."""
        collated = {}
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            collated[key] = torch.stack(values) if torch.is_tensor(values[0]) else values
        return collated

    @staticmethod
    def _resolve_window_config(config: dict, data_split: Optional[str]) -> Dict[str, int]:
        """Resolves stride and history/forecast lengths, including split-specific overrides."""
        def _has_explicit_value(key: str) -> bool:
            if key in config:
                return True
            if data_split and f"{key}_{data_split}" in config:
                return True
            split_overrides = config.get("split_overrides", {})
            if data_split and isinstance(split_overrides, dict):
                split_cfg = split_overrides.get(data_split, {})
                if isinstance(split_cfg, dict) and key in split_cfg:
                    return True
            return False

        sample_mode = str(
            _resolve_split_config_value(config, 'sample_mode', data_split, 'sliding_windows')
        ).lower()

        stride = _resolve_split_config_value(config, 'stride', data_split, 1)
        if sample_mode != "full_trajectory" and not _has_explicit_value('stride'):
            _warn_default_config_once('stride', "Default value for 'stride' (1) is being used.")

        history_length = _resolve_split_config_value(config, 'history_length', data_split, 10)
        if not _has_explicit_value('history_length'):
            _warn_default_config_once(
                'history_length',
                "Default value for 'history_length' (10) is being used.",
            )

        if sample_mode == "full_trajectory":
            forecast_length = int(_resolve_split_config_value(config, 'forecast_length', data_split, 0))
        else:
            forecast_length = _resolve_split_config_value(config, 'forecast_length', data_split, None)
            if forecast_length is None:
                forecast_length = _resolve_split_config_value(config, 'reconstruction_length', data_split, 5)
                if not _has_explicit_value('forecast_length') and not _has_explicit_value('reconstruction_length'):
                    _warn_default_config_once(
                        'forecast_length',
                        "Default value for 'forecast_length'/'reconstruction_length' (5) is being used.",
                    )

        return {
            "stride": int(stride),
            "history_length": int(history_length),
            "forecast_length": int(forecast_length),
        }

    @staticmethod
    def _rolling_window_view(data: np.ndarray, window_length: int) -> np.ndarray:
        """Creates overlapping time windows while keeping the feature axis."""
        if data.ndim == 1:
            return np.lib.stride_tricks.sliding_window_view(data, window_shape=window_length)

        if data.ndim != 2:
            raise ValueError(f"Expected 1D or 2D array for rolling windows, got shape {data.shape}.")

        n_samples, n_features = data.shape
        if window_length > n_samples:
            raise ValueError("Window length is too long for the input data.")

        if n_features == 0:
            return np.zeros((n_samples - window_length + 1, window_length, 0), dtype=data.dtype)

        windows = np.lib.stride_tricks.sliding_window_view(
            data,
            window_shape=(window_length, n_features)
        )
        return windows.squeeze(axis=1)

    @staticmethod
    def _ensure_3d_windows(windows: np.ndarray) -> np.ndarray:
        """Adds a feature axis to windows from a single-channel series."""
        if windows.ndim == 2:
            return windows[:, :, None]
        return windows

    @classmethod
    def _build_sample_dicts(
        cls,
        time_windows: np.ndarray,
        excitation_windows: np.ndarray,
        response_windows: np.ndarray,
        aux_windows: np.ndarray,
        history_length: int,
        device: Optional[str],
        include_full_trajectory: bool = False
    ) -> List[Dict[str, Any]]:
        """Splits windows into history/forecast tensors and keeps forecast auxiliary values as arrays."""
        time_windows = cls._ensure_3d_windows(time_windows)
        excitation_windows = cls._ensure_3d_windows(excitation_windows)
        response_windows = cls._ensure_3d_windows(response_windows)
        aux_windows = cls._ensure_3d_windows(aux_windows)

        time_hist_seqs = torch.from_numpy(time_windows[:, :history_length].copy()).float().to(device)
        time_fore_seqs = torch.from_numpy(time_windows[:, history_length:].copy()).float().to(device)
        excitation_hist_seqs = torch.from_numpy(excitation_windows[:, :history_length].copy()).float().to(device)
        excitation_fore_seqs = torch.from_numpy(excitation_windows[:, history_length:].copy()).float().to(device)
        response_hist_seqs = torch.from_numpy(response_windows[:, :history_length].copy()).float().to(device)
        response_fore_seqs = torch.from_numpy(response_windows[:, history_length:].copy()).float().to(device)

        samples = []
        for idx in range(time_hist_seqs.shape[0]):
            sample = {
                "time_history": time_hist_seqs[idx],
                "time_forecast": time_fore_seqs[idx],
                "excitation_history": excitation_hist_seqs[idx],
                "excitation_forecast": excitation_fore_seqs[idx],
                "response_history": response_hist_seqs[idx],
                "response_forecast": response_fore_seqs[idx],
                "auxiliary": aux_windows[idx, history_length:, :]
            }

            if include_full_trajectory:
                sample["time_trajectory"] = torch.from_numpy(time_windows[idx].copy()).float().to(device)
                sample["excitation_trajectory"] = torch.from_numpy(excitation_windows[idx].copy()).float().to(device)
                sample["response_trajectory"] = torch.from_numpy(response_windows[idx].copy()).float().to(device)

            samples.append(sample)

        return samples

    @staticmethod
    def create_subsequences(
        data: pd.DataFrame,
        config: dict,
        channels: Dict[str, List[str]],
        device: Optional[str] = None,
        data_split: Optional[str] = None,
        include_full_trajectory: bool = False
    ):
        """
        Creates history and forecast subsequences from the prepared data.

        Args:
            data (pd.DataFrame): Input time series data.
            config (dict): Window lengths, stride, and sample_mode. A stride of
                -1 uses the forecast length; full_trajectory creates one sample
                with the observed history and the remaining trajectory as target.
            channels (Dict[str, List[str]]): Column names for time, 'excitation'
                (known forcing), response, and optional auxiliary data.
            device (str, optional): Device for the time, known forcing, and response tensors.
            data_split (str, optional): Split name for configuration overrides.
            include_full_trajectory (bool): Include full time, known forcing, and response windows.

        Returns:
            list: Sample dictionaries. History/forecast tensors have shape
                (sequence length, number of channels); auxiliary forecast values
                remain NumPy arrays.
        """
        window_cfg = TimeSeriesForecastingDataset._resolve_window_config(config, data_split)
        stride = window_cfg["stride"]
        history_length = window_cfg["history_length"]
        forecast_length = window_cfg["forecast_length"]
        total_length = history_length + forecast_length

        time_data = data[channels["time"]].values
        excitation_data = data[channels["excitation"]].values
        response_data = data[channels["response"]].values
        aux_data = data[channels.get("auxiliary", [])].values

        sample_mode = str(
            _resolve_split_config_value(config, 'sample_mode', data_split, 'sliding_windows')
        ).lower()

        if sample_mode == "full_trajectory":
            if history_length < 1:
                raise ValueError("For sample_mode='full_trajectory', history_length must be >= 1.")
            if history_length >= time_data.shape[0]:
                raise ValueError(
                    "For sample_mode='full_trajectory', history_length must be smaller than the trajectory length."
                )

            time_windows = np.expand_dims(time_data, axis=0)
            excitation_windows = np.expand_dims(excitation_data, axis=0)
            response_windows = np.expand_dims(response_data, axis=0)
            aux_windows = np.expand_dims(aux_data, axis=0)

            return TimeSeriesForecastingDataset._build_sample_dicts(
                time_windows=time_windows,
                excitation_windows=excitation_windows,
                response_windows=response_windows,
                aux_windows=aux_windows,
                history_length=history_length,
                device=device,
                include_full_trajectory=include_full_trajectory
            )

        if stride == -1:
            stride = forecast_length
        if total_length > time_data.shape[0]:
            raise ValueError("History and forecast lengths are too long for the input data.")

        time_windows = TimeSeriesForecastingDataset._rolling_window_view(time_data, total_length)[::stride]
        excitation_windows = TimeSeriesForecastingDataset._rolling_window_view(excitation_data, total_length)[::stride]
        response_windows = TimeSeriesForecastingDataset._rolling_window_view(response_data, total_length)[::stride]
        aux_windows = TimeSeriesForecastingDataset._rolling_window_view(aux_data, total_length)[::stride]

        return TimeSeriesForecastingDataset._build_sample_dicts(
            time_windows=time_windows,
            excitation_windows=excitation_windows,
            response_windows=response_windows,
            aux_windows=aux_windows,
            history_length=history_length,
            device=device,
            include_full_trajectory=include_full_trajectory
        )


class TimeSeriesReconstructionDataset(TimeSeriesForecastingDataset):
    """
    Reconstruction dataset with the same history/forecast interface as the forecasting dataset.

    In addition to the standard batch keys, each sample contains the full windowed trajectory:
        - time_trajectory
        - excitation_trajectory
        - response_trajectory

    This is useful for NODE/NFDE experiments that want both:
        - a standard observed-prefix / target-suffix split for training
        - access to the full trajectory for custom evaluation or plotting
    """

    @staticmethod
    def create_subsequences(
        data: pd.DataFrame,
        config: dict,
        channels: Dict[str, List[str]],
        device: Optional[str] = None,
        data_split: Optional[str] = None,
        include_full_trajectory: bool = True
    ):
        return TimeSeriesForecastingDataset.create_subsequences(
            data=data,
            config=config,
            channels=channels,
            device=device,
            data_split=data_split,
            include_full_trajectory=include_full_trajectory
        )
