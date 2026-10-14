import Constants as const
import copy
import math
from pathlib import Path
import yaml
from src.experiments.experiments_base import ExperimentBase
from src.utils.model_names import model_key
from torch.utils.data import DataLoader
import torch
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve
import logging
import time
import os, time, statistics, gc, torch, psutil
from contextlib import contextmanager
from typing import Callable, Dict, List
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except (ImportError, pynvml.NVMLError):
    HAS_NVML = False




class MeasureResources(ExperimentBase):

    def __init__(self, dataset,model,seed=None,hash=None,console_mode=False, **kwargs):

        assert dataset is not None, "dataset must be provided"
        assert model is not None, "model must be provided"
        super().__init__(dataset,model,seed,hash,console_mode, **kwargs)

    def run_experiment(self,default_parameters=False,best_hyperparameters=None):


        self.logger.info("Running Measure Performance Experiment with model: "+
              self.model_name+ " and dataset: "+ self.dataset_name)
        self.logger.info("Experiment type: " + type(self).__name__)
        self.logger.info("Experiment seed: " + str(self.seed))
        self.logger.info("Experiment hash: " + str(self.config_hash))

        if default_parameters:
            best_hyperparameters = None
        else:
            if best_hyperparameters is None:
                self.logger.info("No hyperparameters provided, starting hyperparameter tuning.")
                best_hyperparameters = self.hyperparameter_tuning()
        model,config=self.get_untrained_model(best_hyperparameters)
        results=self.performance_measurement(model)

        return results

    def _runtime_config(self, checkpoint_dir, batch_transforms):
        with (Path(checkpoint_dir) / "model_config.yaml").open() as handle:
            config = yaml.safe_load(handle)
        config = copy.deepcopy(config)
        if batch_transforms:
            model_config = config["model"]
            if model_key(model_config.get("name")) != "LaplaceNet" or model_config.get("response_only_history") is not False:
                raise ValueError("Batch runtime requires a saved full-history LaplaceNet configuration")
            model_config.update(dlt_mode="batch", fflt_mode="batch")
        config["chosen_device"] = self.device
        return config

    def _runtime_load_model(self, checkpoint_dir, batch_transforms):
        config = self._runtime_config(checkpoint_dir, batch_transforms)
        model, _ = self.get_untrained_model(config=config)
        checkpoint = Path(checkpoint_dir) / "model_checkpoints/best_model.pth"
        state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model = model.to(self.device)
        model.eval()
        return model, checkpoint

    def _runtime_sample(self, history_length, forecast_length, sample_index):
        config = copy.deepcopy(self.config["dataset"])
        config.update(history_length=history_length, forecast_length=forecast_length,
                      history_length_test=history_length, forecast_length_test=forecast_length)
        config.setdefault("split_overrides", {}).setdefault("test", {}).update(
            history_length=history_length, forecast_length=forecast_length)
        dataset = self.dataset_class(config, data_split="test", device=self.device)
        if len(dataset) == 0:
            raise ValueError(f"Empty test dataset for forecast={forecast_length}")
        return dataset[min(sample_index, len(dataset) - 1)]

    def _runtime_batch(self, sample, forecast_length, sample_count):
        full_length = int(sample["time_forecast"].shape[0])
        if forecast_length > full_length:
            raise ValueError("The real sample is shorter than the requested forecast window")
        sliced = {}
        for key, value in sample.items():
            if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == full_length:
                sliced[key] = value[:forecast_length].clone()
            elif torch.is_tensor(value):
                sliced[key] = value.clone()
            else:
                sliced[key] = value
        loader = DataLoader([sliced] * sample_count, batch_size=sample_count, shuffle=False,
                            collate_fn=self.dataset_class.collate_fn)
        return next(iter(loader))

    def _runtime_forward(self, model, batch):
        model.eval()
        prediction = model(batch["excitation_history"], batch["response_history"], batch["time_history"],
                           batch["excitation_forecast"], batch["time_forecast"])
        return {"output_shape": "x".join(str(int(dim)) for dim in prediction.shape),
                "output_numel": int(prediction.numel())}

    def _runtime_sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _runtime_cleanup(self, sleep_s):
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            self._runtime_sync()
        if sleep_s > 0:
            time.sleep(sleep_s)

    def _runtime_time_point(self, checkpoint_dir, batch, batch_transforms,
                            repeats, warmup_repeats, timing_sleep_s, cleanup_sleep_s):
        times = []
        model = None
        try:
            model, _ = self._runtime_load_model(checkpoint_dir, batch_transforms)
            warmup_start = time.monotonic()
            warmup_duration_s = 0.0
            for _ in range(warmup_repeats):
                self._runtime_sync()
                with torch.inference_mode():
                    self._runtime_forward(model, batch)
                self._runtime_sync()
                warmup_duration_s = time.monotonic() - warmup_start
            for repeat_index in range(repeats):
                self._runtime_sync()
                start = time.perf_counter()
                with torch.inference_mode():
                    payload = self._runtime_forward(model, batch)
                self._runtime_sync()
                times.append(time.perf_counter() - start)
                if timing_sleep_s > 0 and repeat_index + 1 < repeats:
                    time.sleep(timing_sleep_s)
            return {
                "time_s_individual": times,
                "time_s_mean": statistics.fmean(times), "time_s_median": statistics.median(times),
                "time_s_std": statistics.pstdev(times) if len(times) > 1 else 0.0,
                "repeats": repeats, "warmup_repeats": warmup_repeats, "warmup_completed": warmup_repeats,
                "warmup_duration_s": warmup_duration_s, "warmup_min_s": 0.0,
                "forwards_per_repeat": 1, "timing_sleep_s": timing_sleep_s, **payload,
            }
        finally:
            del model
            self._runtime_cleanup(cleanup_sleep_s)

    def measure_runtime(self, checkpoint_dir, forecast_steps, sample_counts, batch_transforms=False,
                        sample_index=0, repeats=10, warmup_repeats=5,
                        timing_sleep_s=0.1, cleanup_sleep_s=0.25):
        """Measure direct test-set forwards of a freshly loaded checkpoint at each point.

        Forecast scaling uses one real window per horizon. Batch scaling repeats
        one real window at the configured forecast length. Loading, batching,
        warm-up and pauses are outside the timed forward intervals.
        """
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Runtime measurements require CUDA; refusing a CPU timing run")
        if repeats < 1 or warmup_repeats < 0 or sample_index < 0 or timing_sleep_s < 0 or cleanup_sleep_s < 0:
            raise ValueError("Invalid timing repetitions, sample index or pauses")
        forecasts = sorted(set(int(value) for value in forecast_steps))
        batches = sorted(set(int(value) for value in sample_counts))
        if not forecasts or not batches or min(forecasts + batches) < 1:
            raise ValueError("Forecast steps and sample counts must be positive and nonempty")
        checkpoint_dir = Path(checkpoint_dir)
        config = self._runtime_config(checkpoint_dir, batch_transforms)
        if not (checkpoint_dir / "model_checkpoints/best_model.pth").is_file():
            raise FileNotFoundError(f"Checkpoint missing: {checkpoint_dir}")
        model_config = config["model"]
        laplace_model = model_key(model_config.get("name")) == "LaplaceNet"
        if laplace_model:
            segments = model_config.get("steps", 1)
            if not isinstance(segments, int) or isinstance(segments, bool) or segments < 1:
                raise ValueError("The trained segment count must be a positive integer")
            minimum = max(2, 2 * segments)
            forecasts = sorted({max(minimum, value) for value in forecasts})
        history = int(self.config["dataset"].get("history_length", 10))
        batch_forecast = int(self.config["dataset"].get("forecast_length", 1))
        if laplace_model and batch_forecast < minimum:
            raise ValueError("Batch-axis horizon is too short for the trained Laplace segments")
        model_name = model_key(self.model_name)
        if model_name in {"FC", "FC-dt"}:
            forecasts = [batch_forecast]

        samples = {}
        for forecast in sorted({batch_forecast, *forecasts}):
            try:
                samples[forecast] = self._runtime_sample(history, forecast, sample_index)
            except Exception as exc:
                self.logger.warning(f"Runtime sample failed for forecast={forecast}: {exc}")
                samples[forecast] = {"_error": {"error_type": type(exc).__name__}}

        rows = []
        for axis, values in (("forecast_steps", forecasts), ("samples", batches)):
            for value in values:
                forecast = value if axis == "forecast_steps" else batch_forecast
                count = 1 if axis == "forecast_steps" else value
                row = {"dataset": self.dataset_name, "model": model_name, "seed": self.seed,
                       "measurement": "inference_time", "axis": axis, "split": "test",
                       "history_length": history, "forecast_steps": forecast, "samples": count}
                sample = samples[forecast]
                if "_error" in sample:
                    rows.append({**row, "status": "failed", **sample["_error"]})
                    continue
                batch = None
                try:
                    batch = self._runtime_batch(sample, forecast, count)
                    payload = self._runtime_time_point(checkpoint_dir, batch, batch_transforms,
                                                       repeats, warmup_repeats, timing_sleep_s, cleanup_sleep_s)
                    shape = tuple(int(value) for value in payload["output_shape"].split("x"))
                    if len(shape) != 3 or shape[:2] != (count, forecast):
                        raise ValueError("Model output shape differs from the requested runtime coordinate")
                    if laplace_model:
                        payload.update(checkpoint_segments=segments, runtime_segments=segments,
                                       transform_execution="batch" if batch_transforms else "saved_configuration",
                                       checkpoint_model=model_key(model_config["name"]))
                    rows.append({**row, "status": "ok", **payload})
                except Exception as exc:
                    self.logger.warning(f"Runtime measurement failed for {axis}={value}: {exc}")
                    rows.append({**row, "status": "failed", "error_type": type(exc).__name__})
                finally:
                    del batch
                    self._runtime_cleanup(cleanup_sleep_s)
        return rows



    def performance_measurement(self, model):
        res_dict = {}
        dataloaders = self.load_data(shuffle_train=False)
        res_benchmark = self.benchmark_one_epoch(model, dataloaders['train'], self.device, n_runs=self.config['experiment']['n_runs'])
        res_dict['benchmark'] = res_benchmark
        res_dict['parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)



        return res_dict

    def _measure_time(self, model, train_loader, device, seed):
        self.set_seed(seed)
        net = model.to(device)

        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

        train_start = time.perf_counter()
        net.train_step(train_loader, return_loss=False)
        torch.cuda.synchronize(device)
        train_end=time.perf_counter()
        duration_train = train_end - train_start
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

        inference_start = time.perf_counter()
        net.validate_step(train_loader)
        torch.cuda.synchronize(device)
        inference_end=time.perf_counter()
        duration_inference = inference_end - inference_start
        return duration_train,duration_inference

    def _measure_memory(self, model, train_loader, device, seed):
        self.set_seed(seed)
        net = model.to(device)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        net.train_step(train_loader)
        torch.cuda.synchronize(device)
        train_mem= torch.cuda.max_memory_allocated(device)/1024**2 # convert to MB


        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        net.validate_step(train_loader)
        torch.cuda.synchronize(device)
        inference_mem = torch.cuda.max_memory_allocated(device)/1024**2 # convert to MB
        return train_mem, inference_mem

    def benchmark_one_epoch(self,
                            model,
                            train_loader,
                            device,
                            n_runs: int = 5):

        base_seed = self.config['experiment']['seed']
        times_train = []
        times_inference = []
        mems_train = []
        mems_inference = []

        for i in range(n_runs):
            seed = base_seed + i
            self.logger.info(f"Benchmarking run {i+1}/{n_runs} with seed {seed}")
            duration_train, duration_inference = self._measure_time(model, train_loader, device, seed)
            mem_train, mem_inference = self._measure_memory(model, train_loader, device, seed)

            times_train.append(duration_train)
            times_inference.append(duration_inference)
            mems_train.append(mem_train)
            mems_inference.append(mem_inference)
        mean_time_train = np.mean(times_train)
        std_time_train = np.std(times_train)
        mean_time_inference = np.mean(times_inference)
        std_time_inference = np.std(times_inference)
        mean_mem_train = np.mean(mems_train)
        std_mem_train = np.std(mems_train)
        mean_mem_inference = np.mean(mems_inference)
        std_mem_inference = np.std(mems_inference)
        results = {
            'train_times': {
                'runs': times_train,
                'mean': mean_time_train,
                'std': std_time_train
            },
            'inference_times': {
                'runs': times_inference,
                'mean': mean_time_inference,
                'std': std_time_inference
            },
            'train_mems': {
                'runs': mems_train,
                'mean': mean_mem_train,
                'std': std_mem_train
            },
            'inference_mems': {
                'runs': mems_inference,
                'mean': mean_mem_inference,
                'std': std_mem_inference
            }
        }
        return results


