from .physics_base import BasePhysicsODENoScaler
import torch
from typing import Dict


class RLCCircuitWhiteboxPhysics(BasePhysicsODENoScaler):
    """
    White-box RLC circuit model.
    """

    def __init__(
        self,
        config: Dict,
        Nonlinear_C1: bool = True,
        Nonlinear_C2: bool = True,
        Nonlinear_L1: bool = True,
        Nonlinear_L2: bool = True,
        **kwargs,
    ):

        excitation_channels = ["v_IN", "i_OUT"]
        response_channels = ["v_OUT", "i_IN", "v_INT", "i_INT"]


        super().__init__(
            config=config,
            excitation_channels=excitation_channels,
            response_channels=response_channels,
            **kwargs,
        )

        self.nonlinear_C1 = Nonlinear_C1
        self.nonlinear_C2 = Nonlinear_C2
        self.nonlinear_L1 = Nonlinear_L1
        self.nonlinear_L2 = Nonlinear_L2

    def _nonlinear_capacitor(
        self,
        voltage: torch.Tensor,
        C: torch.Tensor,
        slope: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(
            voltage >= 0,
            C * torch.exp(-slope * voltage),
            C * torch.exp(slope * voltage),
        )

    def _nonlinear_inductor(
        self,
        current: torch.Tensor,
        L: torch.Tensor,
        slope_l: torch.Tensor,
    ) -> torch.Tensor:
        abs_i = current.abs()
        tanh_term = torch.tanh(slope_l * current)
        denom = slope_l * current

        denom_safe = torch.where(
            denom.abs() < self.epsilon,
            self.epsilon * torch.ones_like(denom),
            denom,
        )

        ratio = tanh_term / denom_safe
        L_eff = L * ratio


        L_eff = torch.where(abs_i < 1e-6, L, L_eff)
        return L_eff

    def physics_odes(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        x[..., 0] = v_IN
        x[..., 1] = i_OUT
        y[..., 0] = v_OUT
        y[..., 1] = i_IN
        y[..., 2] = v_INT
        y[..., 3] = i_INT
        """

        v_IN = x[..., 0]
        i_OUT = x[..., 1]


        v_OUT = y[..., 0]
        i_IN = y[..., 1]
        v_INT = y[..., 2]
        i_INT = y[..., 3]

        # Physical parameters (fixed or learnable, depending on the config)
        C1 = self._get_physical_param("C1")
        C2 = self._get_physical_param("C2")
        R1 = self._get_physical_param("R1")
        R2 = self._get_physical_param("R2")
        L1 = self._get_physical_param("L1")
        L2 = self._get_physical_param("L2")
        slope_c1 = self._get_physical_param("slope_c1")
        slope_c2 = self._get_physical_param("slope_c2")
        slope_l1 = self._get_physical_param("slope_l1")
        slope_l2 = self._get_physical_param("slope_l2")

        # Effective (possibly nonlinear) capacitances and inductances
        C1_eff = (
            self._nonlinear_capacitor(v_INT, C1, slope_c1)
            if self.nonlinear_C1
            else C1
        )
        C2_eff = (
            self._nonlinear_capacitor(v_OUT, C2, slope_c2)
            if self.nonlinear_C2
            else C2
        )

        L1_eff = (
            self._nonlinear_inductor(i_IN, L1, slope_l1)
            if self.nonlinear_L1
            else L1
        )
        L2_eff = (
            self._nonlinear_inductor(i_INT, L2, slope_l2)
            if self.nonlinear_L2
            else L2
        )

        # RLC equations
        d_v_INT_dt = (i_IN / C1_eff) - (i_INT / C1_eff)
        d_i_IN_dt = (v_IN / L1_eff) - ((R1 * i_IN) / L1_eff) - (v_INT / L1_eff)
        d_v_OUT_dt = (i_INT / C2_eff) - (i_OUT / C2_eff)
        d_i_INT_dt = (v_INT / L2_eff) - ((R2 * i_INT) / L2_eff) - (v_OUT / L2_eff)

        dy_dt = torch.stack(
            [d_v_OUT_dt, d_i_IN_dt, d_v_INT_dt, d_i_INT_dt],
            dim=-1,
        )
        return dy_dt

class RLCCircuitGreyboxPhysics(RLCCircuitWhiteboxPhysics):
    """
    Grey-box RLC circuit model.

    """

    def __init__(
        self,
        config: Dict,
        Nonlinear_C1: bool = False,
        Nonlinear_C2: bool = False,
        Nonlinear_L1: bool = False,
        Nonlinear_L2: bool = False,
        **kwargs,
    ):
        super().__init__(
            config=config,
            Nonlinear_C1=Nonlinear_C1,
            Nonlinear_C2=Nonlinear_C2,
            Nonlinear_L1=Nonlinear_L1,
            Nonlinear_L2=Nonlinear_L2,
            **kwargs,
        )



    def physics_odes(
            self,
            t: torch.Tensor,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> torch.Tensor:
        """
        x[..., 0] = v_IN
        x[..., 1] = i_OUT
        y[..., 0] = v_OUT
        y[..., 1] = i_IN
        y[..., 2] = v_INT
        y[..., 3] = i_INT
        """

        v_IN = x[..., 0]
        i_OUT = x[..., 1]


        v_OUT = y[..., 0]
        i_IN = y[..., 1]
        v_INT = y[..., 2]
        i_INT = y[..., 3]

        # Physical parameters (fixed or learnable, depending on the config)
        C1 = self._get_physical_param("C1")
        C2 = self._get_physical_param("C2")
        R1 = self._get_physical_param("R1")
        R2 = self._get_physical_param("R2")
        L1 = self._get_physical_param("L1")
        L2 = self._get_physical_param("L2")
        slope_c1 = self._get_physical_param("slope_c1")
        slope_c2 = self._get_physical_param("slope_c2")
        slope_l1 = self._get_physical_param("slope_l1")
        slope_l2 = self._get_physical_param("slope_l2")

        # Effective (possibly nonlinear) capacitances and inductances
        C1_eff = (
            self._nonlinear_capacitor(v_INT, C1, slope_c1)
            if self.nonlinear_C1
            else C1
        )
        C2_eff = (
            self._nonlinear_capacitor(v_OUT, C2, slope_c2)
            if self.nonlinear_C2
            else C2
        )

        L1_eff = (
            self._nonlinear_inductor(i_IN, L1, slope_l1)
            if self.nonlinear_L1
            else L1
        )
        L2_eff = (
            self._nonlinear_inductor(i_INT, L2, slope_l2)
            if self.nonlinear_L2
            else L2
        )

        # Original RLC equations in the physical domain (physical units)

        d_i_IN_dt = (v_IN / L1_eff) - ((R1 * i_IN) / L1_eff) - (v_INT / L1_eff)
        d_v_OUT_dt = (i_INT / C2_eff) - (i_OUT / C2_eff)
        d_i_INT_dt = torch.zeros_like(d_i_IN_dt)#(v_INT / L2_eff) - ((R2 * i_INT) / L2_eff) - (v_OUT / L2_eff)
        d_v_INT_dt = torch.zeros_like(d_i_IN_dt)#(i_IN / C1_eff) - (i_INT / C1_eff)

        dy_dt = torch.stack(
            [d_v_OUT_dt, d_i_IN_dt, d_v_INT_dt, d_i_INT_dt],
            dim=-1,
        )
        return dy_dt