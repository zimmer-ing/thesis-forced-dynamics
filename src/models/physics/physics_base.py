import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import warnings
import copy

# Constants for numerical stability
_SOFTPLUS_BETA = 1.0
_SOFTPLUS_THRESHOLD = 20.0

class BasePhysicsODEMinMax(nn.Module):
    """
    Simplified base class for physics-based ODE systems on MinMax scaled data.
    Handles (de-)scaling of variables and derivatives.
    Assumes learnable physical parameters might require positivity and applies
    softplus transformation during retrieval.
    Specific system dynamics must be implemented in 'physics_odes'.
    """

    @staticmethod
    def softplus(x: torch.Tensor) -> torch.Tensor:
        """Numerically stable softplus function."""
        return F.softplus(x, beta=_SOFTPLUS_BETA, threshold=_SOFTPLUS_THRESHOLD)

    @staticmethod
    def inverse_softplus(y: torch.Tensor) -> torch.Tensor:
        """Numerically stable inverse of the softplus function."""
        beta = _SOFTPLUS_BETA
        threshold = _SOFTPLUS_THRESHOLD
        # Clamp to avoid log(0) or log(negative) for numerical stability
        y_clamped = y.clamp(min=torch.finfo(y.dtype).eps)
        is_large = (y_clamped * beta) > threshold
        # Use y for large values, and log(expm1(beta*y))/beta for smaller values
        log_expm1_val = torch.log(torch.expm1(beta * y_clamped)) / beta
        return torch.where(is_large, y_clamped, log_expm1_val)


    def __init__(self, config: Dict, excitation_channels: List[str], response_channels: List[str], **kwargs):
        """
        Initializes the scaling factors and physical parameters.

        Args:
            config (Dict): model.scaler_params contains the min-max scaling,
                with data_min and data_max under scaling_details[channel].
                model.physics_params maps parameter names to fixed scalar values
                or dictionaries with initial_value, learnable, and learn_as_factor.
                Both learning flags default to False; learn_as_factor requires learnable.
            excitation_channels (List[str]): Names of known forcing channels (x).
            response_channels (List[str]): Names of response state channels (y).
            **kwargs: Additional fixed system parameters.
        """
        super().__init__()
        self.config = config
        self.params = kwargs

        if not isinstance(excitation_channels, list):
            raise TypeError("excitation_channels must be a list of strings.")
        if not isinstance(response_channels, list):
            raise TypeError("response_channels must be a list of strings.")
        if not config or 'model' not in config or 'scaler_params' not in config['model']:
             raise ValueError("Configuration dictionary 'config' must contain 'model' with 'scaler_params'.")

        scaler_params = config['model']['scaler_params']
        if scaler_params.get('scaler_type') != 'minmax':
            raise ValueError("Requires 'minmax' scaling.")
        scaling_details = scaler_params.get('scaling_details')
        if scaling_details is None:
            raise KeyError("'scaling_details' not found in 'scaler_params'.")

        def _extract_min_max(channel_list: List[str], details: Dict) -> Tuple[List[float], List[float]]:
            min_list, max_list = [], []
            for channel_name in channel_list:
                channel_data = details.get(channel_name)
                if channel_data is None: raise KeyError(f"Scaling details missing for channel: '{channel_name}'.")
                min_val = channel_data.get('data_min')
                max_val = channel_data.get('data_max')
                if min_val is None or max_val is None: raise ValueError(f"Min/Max missing for channel: '{channel_name}'.")
                min_list.append(min_val)
                max_list.append(max_val)
            return min_list, max_list

        tensor_dtype = torch.float32
        epsilon_val = torch.finfo(tensor_dtype).eps
        self.register_buffer('epsilon', torch.tensor(epsilon_val, dtype=tensor_dtype))

        # Known forcing (x)
        x_min_list, x_max_list = _extract_min_max(excitation_channels, scaling_details)
        self.register_buffer('x_min', torch.tensor(x_min_list, dtype=tensor_dtype))
        self.register_buffer('x_max', torch.tensor(x_max_list, dtype=tensor_dtype))
        self.register_buffer('x_range', (self.x_max - self.x_min).clamp(min=epsilon_val))

        # Response (y)
        y_min_list, y_max_list = _extract_min_max(response_channels, scaling_details)
        self.register_buffer('y_min', torch.tensor(y_min_list, dtype=tensor_dtype))
        self.register_buffer('y_max', torch.tensor(y_max_list, dtype=tensor_dtype))
        self.register_buffer('y_range', (self.y_max - self.y_min).clamp(min=epsilon_val))

        self._excitation_channels = excitation_channels
        self._response_channels = response_channels

        self.physical_params = nn.ParameterDict() # Stores transformed learnable parts (direct values or factors)
        self.fixed_physical_params = {}          # Stores fixed parts or base values for factor learning
        self.param_learning_style = {}           # Stores 'direct', 'factor', or 'fixed' for each param name

        if 'physics_params' in config['model']:
            physics_parameters_config = config['model']['physics_params']

            if not isinstance(physics_parameters_config, dict):
                raise TypeError("'parameters' in 'physics_params' must be a dictionary.")

            for name, param_config_entry in physics_parameters_config.items():
                initial_value_original = None
                learnable = False
                learn_as_factor = False

                if isinstance(param_config_entry, dict):
                    if 'initial_value' not in param_config_entry:
                        raise KeyError(f"Parameter '{name}' config missing 'initial_value'.")
                    initial_value_original = param_config_entry['initial_value']
                    learnable = param_config_entry.get('learnable', False)
                    learn_as_factor = param_config_entry.get('learn_as_factor', False)
                    if learn_as_factor and not learnable:
                        warnings.warn(f"Parameter '{name}' has 'learn_as_factor=True' but 'learnable=False'. Treating as fixed with initial_value.")
                        learn_as_factor = False
                        learnable = False

                elif isinstance(param_config_entry, (float, int)):
                    initial_value_original = param_config_entry
                    learnable = False
                    learn_as_factor = False
                else:
                    raise TypeError(f"Parameter '{name}' has invalid configuration type: {type(param_config_entry)}. Expected dict, float, or int.")

                initial_tensor_original = torch.tensor(initial_value_original, dtype=tensor_dtype)

                if learnable:
                    if learn_as_factor:
                        self.fixed_physical_params[name] = initial_tensor_original
                        # Learnable part is a factor, initialized so that softplus(transformed_factor) = 1.0 initially
                        factor_initial_value_target = torch.tensor(1.0, dtype=tensor_dtype)
                        transformed_factor = self.inverse_softplus(factor_initial_value_target)
                        self.physical_params[name] = nn.Parameter(transformed_factor)
                        self.param_learning_style[name] = 'factor'
                    else: # Direct learning (value itself is learned)
                        # Apply inverse_softplus for positivity constraint.
                        if initial_tensor_original <= 0:
                            warnings.warn(f"Initial value for learnable param '{name}' ({initial_value_original}) is non-positive. "
                                          "Applying inverse_softplus for positivity constraint. "
                                          "The effective initial value after softplus will be positive.")
                        transformed_value = self.inverse_softplus(initial_tensor_original)
                        self.physical_params[name] = nn.Parameter(transformed_value)
                        self.param_learning_style[name] = 'direct'
                else: # Fixed parameter (either from scalar or dict with learnable=False)
                    self.fixed_physical_params[name] = initial_tensor_original
                    self.param_learning_style[name] = 'fixed'
        else:
            warnings.warn("'physics_params' with 'parameters' key not found in config['model']. "
                          "No physical parameters initialized from the primary configuration structure.")


    def _get_physical_param(self, name: str) -> torch.Tensor:
        """
        Retrieves the value of a physical parameter.
        Applies softplus to learnable parts assuming they are stored
        in the inverse_softplus space for positivity, and handles factor-based learning.
        """
        learning_style = self.param_learning_style.get(name)
        # Determine device from a buffer or a parameter if available, else default to CPU
        if hasattr(self, 'y_min') and self.y_min is not None:
            device_to_use = self.y_min.device
        elif any(self.parameters()):
            device_to_use = next(self.parameters()).device
        else:
            device_to_use = torch.device("cpu")


        if learning_style == 'direct':
            return self.softplus(self.physical_params[name])
        elif learning_style == 'factor':
            # Learned as a factor: base_value * softplus(learned_factor)
            base_value = self.fixed_physical_params[name].to(device_to_use)
            factor = self.softplus(self.physical_params[name])
            return base_value * factor
        elif learning_style == 'fixed':
            return self.fixed_physical_params[name].to(device_to_use)
        elif name in self.params:
             param_val = self.params[name]
             if not isinstance(param_val, torch.Tensor):
                 param_val = torch.tensor(param_val, dtype=torch.float32)
             return param_val.to(device_to_use)
        else:
            raise KeyError(f"Physical parameter '{name}' not found or not configured. "
                           f"Check YAML under model.physics_params.parameters and initialization **kwargs. "
                           f"Available configured params: {list(self.param_learning_style.keys())}")

    def unscale(self, x_norm: torch.Tensor, y_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts normalized states and known forcing back to the physical space."""
        x = x_norm * self.x_range + self.x_min
        y = y_norm * self.y_range + self.y_min
        return x, y

    def rescale_deriv(self, dy_dt: torch.Tensor) -> torch.Tensor:
        """Converts physical derivatives to the normalized space."""
        return dy_dt / self.y_range

    def physics_odes(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the physical ODEs dy/dt = f(t, x, y, params).
        Subclasses implement the dynamics and use self._get_physical_param('name') for parameters.

        Args:
            t (torch.Tensor): Current time.
            x (torch.Tensor): Current known forcing in the *physical* space.
            y (torch.Tensor): Current state in the *physical* space.

        Returns:
            torch.Tensor: The derivative dy/dt in the *physical* space.
        """
        raise NotImplementedError("Subclasses must implement physics_odes!")

    def forward(self, t: torch.Tensor,  y_norm: torch.Tensor, x_norm: torch.Tensor,) -> torch.Tensor:
        """
        Function called by the ODE solver: unscales inputs and rescales the physical derivative.

        Args:
            t (torch.Tensor): Current time.
            y_norm (torch.Tensor): Normalized state at time t.
            x_norm (torch.Tensor): Normalized known forcing at time t.

        Returns:
            torch.Tensor: Normalized derivative dy_norm/dt, with the same shape as y_norm.
        """
        x_phys, y_phys = self.unscale(x_norm, y_norm)

        dy_dt_phys = self.physics_odes(t, x_phys, y_phys)

        dy_norm_dt = self.rescale_deriv(dy_dt_phys)

        return dy_norm_dt

    def get_physical_parameters_summary(self) -> Dict[str, float]:
        """
        Returns current physical parameter values as a dictionary of floats.

        Learnable values include the softplus transformation and any learned factor.
        Additional parameters from **kwargs are included; failed conversions yield NaN.
        """
        summary = {}

        for name in self.param_learning_style.keys():
            try:
                param_tensor = self._get_physical_param(name)
                summary[name] = param_tensor.item()
            except KeyError:
                warnings.warn(
                    f"Parameter '{name}' is in param_learning_style, but could not be retrieved via _get_physical_param.")
            except Exception as e:
                warnings.warn(f"Error retrieving parameter '{name}': {e}")
                summary[name] = float('nan')

        for name in self.params.keys():
            if name not in summary:
                try:
                    param_tensor = self._get_physical_param(name)
                    summary[name] = param_tensor.item()
                except Exception as e:
                    warnings.warn(f"Error retrieving kwarg-parameter '{name}': {e}")
                    summary[name] = float('nan')

        return summary




class BasePhysicsODENoScaler(BasePhysicsODEMinMax):
    """
    Variant of BasePhysicsODEMinMax that automatically creates an identity MinMax scaling
    (data_min = 0.0, data_max = 1.0) for all known forcing and response channels if no
    scaler_params are present in config['model'] or if a channel is missing in
    scaling_details.

    This allows using raw physical data (i.e., already in physical units) without
    precomputed dataset scalers, while still reusing the MinMax-based logic for
    unscale() and rescale_deriv().
    """

    def __init__(
        self,
        config: Dict,
        excitation_channels: List[str],
        response_channels: List[str],
        **kwargs,
    ):
        # Make a defensive copy of the config to avoid mutating the original dict
        cfg = copy.deepcopy(config) if isinstance(config, dict) else {"model": {}}

        model_cfg = cfg.get("model")
        if not isinstance(model_cfg, dict):
            model_cfg = {}
            cfg["model"] = model_cfg

        raw_scaler_params = model_cfg.get("scaler_params", None)
        if not isinstance(raw_scaler_params, dict):
            scaler_params = {}
            model_cfg["scaler_params"] = scaler_params
        else:
            scaler_params = raw_scaler_params

        scaler_params.setdefault("scaler_type", "minmax")

        raw_scaling_details = scaler_params.get("scaling_details", None)
        if not isinstance(raw_scaling_details, dict):
            scaling_details = {}
            scaler_params["scaling_details"] = scaling_details
        else:
            scaling_details = raw_scaling_details

        # For all known forcing and response channels, create identity scaling
        # if not already present in scaling_details.
        for ch in list(excitation_channels) + list(response_channels):
            if ch not in scaling_details:
                scaling_details[ch] = {
                    "data_min": 0.0,
                    "data_max": 1.0,
                }

        super().__init__(
            config=cfg,
            excitation_channels=excitation_channels,
            response_channels=response_channels,
            **kwargs,
        )
