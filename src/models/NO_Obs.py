from src.models.NeuralODE_base import BaseNODEModel, BlackBoxODE


class NODEfixedStepModel(BaseNODEModel):
    """
    Fixed-step NODE model for regression using Neural ODEs.

    Uses one of the fixed-step solvers from torchdiffeq:
        - 'euler', 'midpoint', 'rk4', 'explicit_adams', 'implicit_adams'
    """

    def __init__(self, config,logger=None, **kwargs):
        # Set a sensible default solver if none is provided
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
        self.ODE = BlackBoxODE(
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

        # Prepare for training (optimiser etc.)
        self.prepare_training(config)


class NODEadaptiveModel(BaseNODEModel):
    """
    Adaptive-step NODE model for regression using Neural ODEs.

    Uses one of the adaptive solvers from torchdiffeq:
        - 'dopri8', 'dopri5', 'bosh3', 'fehlberg2', 'adaptive_heun'
    """

    def __init__(self, config,logger=None, **kwargs):
        model_cfg = config['model']

        # Set a sensible default adaptive solver if none is provided
        if 'solver' not in model_cfg:
            model_cfg['solver'] = 'dopri5'

        # Initialise BaseNODEModel (sets dim_x, dim_y, device, solver, flags, ...)
        super().__init__(config,logger=logger, **kwargs)

        # Enforce that the chosen solver is adaptive
        assert self._is_adaptive, (
            f"Solver '{self.solver}' is not an adaptive solver. "
            f"Choose one of: {sorted(self._adaptive_solvers)}"
        )

        # Same BlackBox-specific checks as in fixed-step
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

        # Choose interpolation mode (default is linear)
        interp_mode = model_cfg.get('interp_mode', 'linear')

        # Construct the ODE network
        self.ODE = BlackBoxODE(
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

        # Prepare for training (optimiser etc.)
        self.prepare_training(config)