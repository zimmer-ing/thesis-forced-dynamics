import unittest

import torch

from src.models.FDEint.FixedStepFDEint_with_forcing import FDEint


class LinearStateRHS(torch.nn.Module):
    def forward(self, _t, y, _u=None):
        return y


class FixedStepFDEintTest(unittest.TestCase):
    def test_rhs_history_uses_state_from_matching_time_step(self):
        """Check the PECE values for y'=y, alpha=1 and h=0.1."""
        t = torch.tensor([[[0.0], [0.1], [0.2]]], dtype=torch.float64)
        y0 = torch.tensor([[1.0]], dtype=torch.float64)
        alpha = torch.tensor(1.0, dtype=torch.float64)

        result = FDEint(
            f=LinearStateRHS(),
            t=t,
            y0=y0,
            alpha=alpha,
            dtype=torch.float64,
        )

        expected = torch.tensor([[[1.0], [1.105], [1.221025]]], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
