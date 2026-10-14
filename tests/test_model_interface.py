import logging
from pathlib import Path
import unittest

import torch
import yaml

from src.models import MODEL_REGISTRY


MODEL_NAMES = ["FC", "GRU"]
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "models"


class ModelInterfaceTest(unittest.TestCase):
    def test_forward_and_backward(self):
        for model_name in MODEL_NAMES:
            with open(CONFIG_PATH / f"{model_name}.yaml") as file:
                model_config = yaml.safe_load(file)
            model_config = {
                key: value["default"] if isinstance(value, dict) and "default" in value else value
                for key, value in model_config.items()
            }
            config = {
                "chosen_device": "cpu",
                "dataset": {"dim_x": 2, "dim_y": 3, "history_length": 4, "forecast_length": 6},
                "model": model_config,
            }
            for batch_size in (1, 3):
                with self.subTest(model=model_name, batch_size=batch_size):
                    torch.manual_seed(42)
                    model = MODEL_REGISTRY[model_name](config, logger=logging.getLogger(__name__))
                    time = torch.arange(10, dtype=torch.float32).view(1, 10, 1) * 0.1
                    time = time.repeat(batch_size, 1, 1)
                    prediction = model(
                        torch.randn(batch_size, 4, 2),
                        torch.randn(batch_size, 4, 3),
                        time[:, :4],
                        torch.randn(batch_size, 6, 2),
                        time[:, 4:],
                    )
                    self.assertEqual(tuple(prediction.shape), (batch_size, 6, 3))
                    self.assertTrue(torch.isfinite(prediction).all().item())
                    loss = model.calculate_loss(prediction, torch.randn_like(prediction))
                    loss.backward()
                    gradients = [p.grad for p in model.trainable_parameters() if p.grad is not None]
                    self.assertTrue(gradients, "No trainable parameter received a gradient.")
                    self.assertTrue(all(torch.isfinite(grad).all().item() for grad in gradients))
                    self.assertTrue(any(torch.count_nonzero(grad).item() for grad in gradients))


if __name__ == "__main__":
    unittest.main()
