from src.models.NeuralODE_base import BaseNODEModel, BlackBoxODE
import torch
import torch.nn as nn
from src.models.NeuralODE_base import BaseNODEModel, BaseODE
from src.models.physics import PHYSICS_MODEL_REGISTRY

class WhiteboxODE(BaseODE):

    def __init__(self, config: dict, interp_mode: str = "linear",logger=None,**kwargs):
        super().__init__(interp_mode,logger=logger,**kwargs)

        assert "dataset" in config and "name" in config["dataset"], \
            "Config must contain dataset.name for physics model lookup."

        physics_model_name = config["dataset"]["name"] + "_Greybox"
        if physics_model_name not in PHYSICS_MODEL_REGISTRY:
            raise ValueError(
                f"Physics model '{physics_model_name}' not found in PHYSICS_MODEL_REGISTRY."
            )
        PhysicsODEClass = PHYSICS_MODEL_REGISTRY[physics_model_name]

        # Instantiate physics ODE model (implements f(t, x, y, theta))
        self.physics_ODE = PhysicsODEClass(config)


    def get_physical_parameters(self):
        """
        Delegate to physics_ODE if it implements get_physical_parameters_summary().
        """
        if hasattr(self.physics_ODE, "get_physical_parameters_summary"):
            return self.physics_ODE.get_physical_parameters_summary()
        return {}

    def forward(self, t: torch.Tensor, y: torch.Tensor, x: torch.Tensor = None) -> torch.Tensor:
        """
        Standard RHS signature for torchdiffeq
        """
        if x is None:
            x = self.interpolate_x(t)  # [B, dim_x]

        # physics_ODE is assumed to implement: physics_ODE(t, y, x)
        dy_dt = self.physics_ODE(t, y, x)
        return dy_dt

class GreyboxWrapper(BaseODE):
    """
    Wrapper module that combines a physics-based ODE with a black-box correction.
    """

    def __init__(self, physics_ode: WhiteboxODE, blackbox_ode: BlackBoxODE,interp_mode: str = "linear",logger=None,**kwargs):
        super().__init__(interp_mode,logger=logger,**kwargs)
        self.physics_ode = physics_ode
        self.blackbox_ode = blackbox_ode

    def set_scaling_factor(self, scaling_factor: torch.Tensor):
        self.blackbox_ode.set_scaling_factor(scaling_factor)

    def get_scaling_factor(self):
        return self.blackbox_ode.get_scaling_factor()


    def forward(self, t: torch.Tensor, y: torch.Tensor, x: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the combined ODE RHS at time t.

        Args:
            t: current time [scalar]
            y: current state [B, dim_y]
            x: current excitation [B, dim_x] (optional; can be obtained via interpolation)

        Returns:
            dy/dt: combined ODE RHS [B, dim_y]
        """
        if x is None:
            x = self.interpolate_x(t)

        f_phys = self.physics_ode(t, y, x)
        f_bb = self.blackbox_ode(t, y, x)

        return f_phys + f_bb

# ======================================================================
# NODE models: fixed-step and adaptive
# ======================================================================

class NODEfixedStepModel(BaseNODEModel):
    """
    Fixed-step NODE model for regression using Neural ODEs.

    Uses one of the fixed-step solvers from torchdiffeq:
        - 'euler', 'midpoint', 'rk4', 'explicit_adams', 'implicit_adams'
    """

    def __init__(self, config,logger=None, **kwargs):

        model_cfg = config['model']
        if 'solver' not in model_cfg:
            model_cfg['solver'] = 'rk4'

        super().__init__(config,logger=logger, **kwargs)

        assert self._is_fixed_step, (
            f"Solver '{self.solver}' is not a fixed-step solver. "
            f"Choose one of: {sorted(self._fixed_step_solvers)}"
        )

        assert model_cfg['hidden_size_ODE'] > 0, "hidden_size_ODE must be > 0"
        assert model_cfg['n_layers_hidden_ODE'] > 0, "n_layers_hidden_ODE must be > 0"
        assert model_cfg['activation_ODE'] in ['relu', 'leaky_relu', 'tanh', 'elu', 'sigmoid', 'none']
        assert model_cfg['include_time'] in [True, False]
        assert model_cfg['homogeneous'] in [True, False]
        assert model_cfg.get('intermediate_timesteps', 1) > 0

        self.dim_z = self.dim_y
        self.hidden_size = model_cfg['hidden_size_ODE']
        self.n_layers_hidden = model_cfg['n_layers_hidden_ODE']
        self.activation = model_cfg['activation_ODE']
        self.include_time = model_cfg['include_time']
        self.homogeneous = model_cfg['homogeneous']


        interp_mode = model_cfg.get('interp_mode', 'linear')

        # Construct the ODE network
        self.ODE_bb = BlackBoxODE(
            dim_x=self.dim_x,
            dim_z=self.dim_z,
            hidden_size=self.hidden_size,
            n_layers_hidden=self.n_layers_hidden,
            activation=self.activation,
            include_time=self.include_time,
            homogeneous=self.homogeneous,
            interp_mode=interp_mode,
        ).to(self.device)

        self.ODE_Phy=WhiteboxODE(config,
                                 interp_mode=interp_mode,
                                 logger=logger,
                                 ).to(self.device)
        self.ODE = GreyboxWrapper(physics_ode=self.ODE_Phy,
                                  blackbox_ode=self.ODE_bb,
                                  interp_mode=interp_mode,
                                  logger=logger

                                  ).to(self.device)


        self.prepare_training(config)


class NODEadaptiveModel(BaseNODEModel):
    """
    Adaptive-step NODE model for regression using Neural ODEs.

    Uses one of the adaptive solvers from torchdiffeq:
        - 'dopri8', 'dopri5', 'bosh3', 'fehlberg2', 'adaptive_heun'
    """

    def __init__(self, config,logger=None, **kwargs):
        model_cfg = config['model']


        if 'solver' not in model_cfg:
            model_cfg['solver'] = 'dopri5'


        super().__init__(config,logger=logger, **kwargs)


        assert self._is_adaptive, (
            f"Solver '{self.solver}' is not an adaptive solver. "
            f"Choose one of: {sorted(self._adaptive_solvers)}"
        )


        assert model_cfg['hidden_size_ODE'] > 0, "hidden_size_ODE must be > 0"
        assert model_cfg['n_layers_hidden_ODE'] > 0, "n_layers_hidden_ODE must be > 0"
        assert model_cfg['activation_ODE'] in ['relu', 'leaky_relu', 'tanh', 'elu', 'sigmoid', 'none']
        assert model_cfg['include_time'] in [True, False]
        assert model_cfg['homogeneous'] in [True, False]

        self.dim_z = self.dim_y
        self.hidden_size = model_cfg['hidden_size_ODE']
        self.n_layers_hidden = model_cfg['n_layers_hidden_ODE']
        self.activation = model_cfg['activation_ODE']
        self.include_time = model_cfg['include_time']
        self.homogeneous = model_cfg['homogeneous']


        interp_mode = model_cfg.get('interp_mode', 'linear')

        # Construct the ODE network
        self.ODE_bb = BlackBoxODE(
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

        self.ODE_Phy = WhiteboxODE(config, interp_mode=interp_mode).to(self.device)
        self.ODE = GreyboxWrapper(physics_ode=self.ODE_Phy, blackbox_ode=self.ODE_bb).to(self.device)

        # Prepare for training (optimiser etc.)
        self.prepare_training(config)