import csv
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader

from src.datasets.dataset_base import MultiFileTimeSeriesDataset


class ExampleDataset(MultiFileTimeSeriesDataset):
    def __init__(self, directory, config, data_split):
        directory = Path(directory) / data_split
        files = [file.name for file in sorted(directory.glob("*.csv"))]
        super().__init__(directory, files, config, device="cpu", data_split=data_split)

    @staticmethod
    def get_channels():
        return {"time": "time", "excitation": ["input_1", "input_2"],
                "response": ["output"], "auxiliary": []}


class DatasetInterfaceTest(unittest.TestCase):
    def test_windows_channels_and_splits(self):
        config = {"history_length": 3, "forecast_length": 2, "stride": 2}
        with tempfile.TemporaryDirectory() as directory:
            for split_index, split in enumerate(("train", "val", "test")):
                folder = Path(directory) / split
                folder.mkdir()
                for file_index in range(2):
                    base = -0.9 + split_index * 0.6 + file_index * 0.3
                    with open(folder / f"sample_{file_index}.csv", "w", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(["time", "input_1", "input_2", "output"])
                        for row in range(8):
                            value = base + row * 0.05
                            writer.writerow([row * 0.1, value, -value, 2 * value])

                dataset = ExampleDataset(directory, config, data_split=split)
                self.assertEqual(len(dataset), 4)
                for index in range(len(dataset)):
                    with self.subTest(split=split, window=index):
                        sample = dataset[index]
                        base = -0.9 + split_index * 0.6 + (index // 2) * 0.3
                        start = (index % 2) * 2
                        values = base + torch.arange(start, start + 5) * 0.05
                        inputs = torch.stack((values, -values), dim=1)
                        torch.testing.assert_close(sample["excitation_history"], inputs[:3])
                        torch.testing.assert_close(sample["excitation_forecast"], inputs[3:])
                        torch.testing.assert_close(sample["response_history"], 2 * values[:3, None])
                        torch.testing.assert_close(sample["response_forecast"], 2 * values[3:, None])
                        time = torch.arange(start, start + 5, dtype=torch.float32)[:, None] * 0.1
                        torch.testing.assert_close(sample["time_history"], time[:3])
                        torch.testing.assert_close(sample["time_forecast"], time[3:])
                loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
                batch = next(iter(loader))
                self.assertEqual(tuple(batch["excitation_history"].shape), (2, 3, 2))
                self.assertEqual(tuple(batch["response_forecast"].shape), (2, 2, 1))


if __name__ == "__main__":
    unittest.main()
