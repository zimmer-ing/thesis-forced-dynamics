import yaml
import torch
import warnings
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import Constants as CONST
from pathlib import Path
from src.models import MODEL_REGISTRY
from src.datasets import DATASET_REGISTRY
import hashlib
import optuna
import logging
from multiprocessing import cpu_count
from datetime import datetime
import numpy as np
import random
import json
import math
import time
from multiprocessing import Lock, Condition, Manager
import gc
import os
import psutil
import subprocess
from src.utils.helpers import get_pytorch_nvidia_gpu_mapping, get_best_available_gpu,release_gpu
from src.utils.failure_handling import (
    classify_exception,
    ensure_finite_number,
    get_failure_policy,
)
from src.utils.model_names import MODEL_ALIASES

import multiprocessing as mp
import functools
from multiprocessing import Queue
import logging
import logging.handlers
from optuna.trial import TrialState
import sys
import atexit
import pandas as pd
from collections import Counter
import copy
import shutil




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

import re, yaml, ast

_NUM_RE = re.compile(r'^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$')

def _deep_cast(o):
    if isinstance(o, dict):
        return {k: _deep_cast(v) for k,v in o.items()}
    if isinstance(o, list):
        return [_deep_cast(v) for v in o]
    if isinstance(o, str) and _NUM_RE.match(o.strip()):
        try:
            return ast.literal_eval(o)
        except Exception:
            return o
    return o

def deep_update(dest: dict, src: dict) -> dict:
    """
    Recursively merge `src` into `dest` and return the updated dictionary.
    Nested dictionaries are merged; other values are overwritten.
    """
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dest.get(key), dict):
            deep_update(dest[key], value)
        else:
            dest[key] = value
    return dest


def _normalise_gpu_slot_counts(raw_slot_counts, gpu_ids, default_slots: int) -> dict:
    """Build a physical NVIDIA-SMI GPU -> parallel slot count mapping."""
    default_slots = max(1, int(default_slots))
    if raw_slot_counts is None:
        return {int(gpu_id): default_slots for gpu_id in gpu_ids}

    if not isinstance(raw_slot_counts, dict):
        raise ValueError(
            "hyperparameter_tuning.gpu_slot_counts must be a mapping like "
            "{0: 1, 1: 1, 4: 2}."
        )

    slot_counts = {}
    for gpu_id in gpu_ids:
        gpu_id_int = int(gpu_id)
        value = raw_slot_counts.get(gpu_id_int, raw_slot_counts.get(str(gpu_id_int), default_slots))
        value = int(value)
        if value > 0:
            slot_counts[gpu_id_int] = value
    if not slot_counts:
        raise ValueError("No usable GPU slots configured for hyperparameter tuning.")
    return slot_counts


def _cuda_memory_budget_gb(config: dict | None) -> float | None:
    experiment_cfg = (config or {}).get("experiment", {})
    raw_limit = experiment_cfg.get("gpu_memory_limit_gb")
    if raw_limit is None:
        raw_limit = experiment_cfg.get("gpu_memory_limit_GB")
    if raw_limit is None:
        return None
    limit_gb = float(raw_limit)
    if limit_gb <= 0:
        return None
    return limit_gb


def _apply_cuda_memory_budget(config: dict | None, device: torch.device, logger, context: str) -> dict | None:
    """Limit one PyTorch process to the configured single-GPU memory budget."""
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        return None

    limit_gb = _cuda_memory_budget_gb(config)
    if limit_gb is None:
        return None

    device_index = torch.device(device).index
    if device_index is None:
        device_index = torch.cuda.current_device()

    total_bytes = torch.cuda.get_device_properties(device_index).total_memory
    limit_bytes = limit_gb * (1024 ** 3)
    fraction = min(max(limit_bytes / total_bytes, 0.0), 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)

    total_gib = total_bytes / (1024 ** 3)
    logger.info(
        f"{context} CUDA memory budget: {limit_gb:.2f} GiB "
        f"on cuda:{device_index} ({fraction:.3f} of {total_gib:.2f} GiB device memory)."
    )
    return {
        "gpu_memory_limit_gb": float(limit_gb),
        "gpu_memory_fraction": float(fraction),
        "gpu_total_gib": float(total_gib),
    }


class ConfigManager:
    """
    Manages the loading and combination of configurations for datasets, models,
    experiments, and their combination rules.
    """

    model_aliases = MODEL_ALIASES

    def __init__(self):
        """
        Initializes the configuration manager and sets the paths for each config type.
        """
        self.base_path = Path(CONST.CONFIGS_PATH)
        self.dataset_path = self.base_path / 'datasets'
        self.model_path = self.base_path / 'models'
        self.experiment_path = self.base_path / 'experiments'
        self.physics_path = self.model_path / 'physics'
        self.combination_rules_path = self.base_path / 'combination_rules.yaml'
        self.combination_rules = self._load_combination_rules()




    def _load_config(self, config_path):
        """
        Loads a YAML configuration file and converts numeric strings to numbers.

        Args:
            config_path (Path): Path to the configuration file.

        Returns:
            dict: The loaded configuration, or an empty dictionary for an empty file.
        """
        if not config_path.exists():
            logging.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        try:
            with open(config_path, 'r') as f:
                config = _deep_cast(yaml.safe_load(f))
            return config if config is not None else {}
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file {config_path}: {e}")
            raise ValueError(f"Error parsing YAML file {config_path}: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading {config_path}: {e}")
            raise

    def load_dataset_config(self, dataset_name):
        """
        Loads the configuration for a specific dataset.
        If 'scaler' is 'minmax', loads the scaler parameters through the dataset class.

        Args:
            dataset_name (str): The registered name of the dataset.

        Returns:
            tuple: Dataset configuration and scaler parameters. The scaler parameters
                are None if no supported scaler is configured.
        """
        config_path = self.dataset_path / f"{dataset_name}.yaml"
        dataset_config = self._load_config(config_path)
        dataset_config['name']=dataset_name

        dataset_config['scaler_params_content'] = None
        scaler_type = dataset_config.get('scaler',None)
        scaler_params = None
        logging.info(f"Dataset '{dataset_name}' uses '{scaler_type}' scaler.")
        if scaler_type == "minmax":
            logging.info(f"Dataset '{dataset_name}' uses '{scaler_type}' scaler. Attempting to load parameters.")
            dataset=DATASET_REGISTRY[dataset_name](dataset_config)
            scaler_params=dataset.load_scaler_dict()


        elif scaler_type is not None and scaler_type != "none":
            logging.warning(
                f"Unsupported scaler type '{scaler_type}' specified for dataset '{dataset_name}'. No parameters will be loaded.")

        return dataset_config,scaler_params

    def load_model_config(self, model_name: str, dataset_name: str, scaler_params) -> dict:
        """
        Loads the model configuration and applies its dataset-specific overrides.
        Adds the scaler parameters and, if 'has_physics' is true, loads the dataset's
        base physics configuration and any model-specific physics overrides.

        Args:
            model_name (str): The registered name of the model.
            dataset_name (str): The registered name of the dataset.
            scaler_params (dict or None): Scaler parameters loaded by the dataset.

        Returns:
            dict: Model configuration, including scaler and optional physics parameters.
        """
        config_name = self.model_aliases.get(model_name, model_name)
        model_config_path = self.model_path / f"{config_name}.yaml"
        model_config = self._load_config(model_config_path)
        dataset_overrides = model_config.pop('dataset_overrides', {})
        deep_update(model_config, dataset_overrides.get(dataset_name, {}))

        model_config['scaler_params'] = scaler_params
        model_config['name'] = model_name

        if model_config.get('has_physics', False):
            logging.info(f"Model '{model_name}' requires physics parameters.")

            #load the physics base config first (must be there)
            physics_config_filename = dataset_name + '_base' + ".yaml"
            physics_config_path = self.physics_path / physics_config_filename
            if not physics_config_path.exists():
                err_str=f"'has_physics' is true for model '{model_name}', but 'physics_config_file' key is missing in {str(physics_config_path)}."
                logging.error(err_str)
                raise ValueError(err_str)

            logging.info(f"Loading physics config from: {physics_config_path}")

            try:
                physics_params_config = self._load_config(physics_config_path)
                model_config['physics_params'] = physics_params_config
                logging.info(f"Successfully loaded and integrated physics parameters for dataset '{dataset_name}'.")
            except FileNotFoundError:
                logging.error(
                    f"Physics configuration file specified in '{model_name}.yaml' not found: {physics_config_path}")
                raise
            except Exception as e:
                logging.error(f"Failed to load or integrate physics config {physics_config_path}: {e}")
                raise

            ### now check if there is a specific physics config for this dataset+model. If so, load and overwrite the base physics config


            physics_config_filename = dataset_name+'_'+config_name+".yaml"
            physics_config_path = self.physics_path / physics_config_filename

            if physics_config_path.exists():
                logging.info(f"Model specific physics config found for dataset '{dataset_name}' and model '{model_name}'. Loading from: {physics_config_path}")
                try:
                    physics_params_config = self._load_config(physics_config_path)
                    deep_update(model_config['physics_params'], physics_params_config)
                    logging.info(f"Successfully updated physics parameters for dataset '{dataset_name}' and model '{model_name}'.")
                except Exception as e:
                    logging.error(f"Failed to load or integrate model-specific physics config {physics_config_path}: {e}")
                    raise



        return model_config



    def load_experiment_config(self, experiment_name,dataset_name):
        """
        Loads the base experiment configuration and applies dataset-specific overrides.

        Args:
            experiment_name (str): Name of the experiment configuration.
            dataset_name (str): The registered name of the dataset.

        Returns:
            dict: Configuration for the experiment on this dataset.
        """
        config_path = Path(self.experiment_path , f"{experiment_name}.yaml")
        raw_conf= self._load_config(config_path)
        if 'base' in raw_conf:
            conf=raw_conf['base']
        else:
            conf=raw_conf

        if dataset_name in raw_conf:
            dataset_conf=raw_conf[dataset_name]
            print("Dataset config found. Updating the experiment config")
            deep_update(conf,dataset_conf)
        return conf

    def _load_combination_rules(self):
        """
        Loads the combination rules, or returns an empty dictionary if no file exists.
        """
        if self.combination_rules_path.exists():
            return self._load_config(self.combination_rules_path)
        return {}

    def apply_combination_rules(self, dataset_name, model_name, experiment_name, dataset_config, model_config):
        """
        Applies the rules for the dataset-model-experiment combination.
        The supported feature rule limits the dataset's x_features in place.

        Args:
            dataset_name (str): The name of the dataset.
            model_name (str): The name of the model.
            experiment_name (str): The name of the experiment.
            dataset_config (dict): Configuration for the dataset.
            model_config (dict): Configuration for the model.

        Returns:
            tuple: The adjusted dataset and model configurations.
        """
        if (dataset_name in self.combination_rules and
                model_name in self.combination_rules[dataset_name] and
                experiment_name in self.combination_rules[dataset_name][model_name]):

            rules = self.combination_rules[dataset_name][model_name][experiment_name]
            if "adjust_x_features" in rules:
                limit = rules["adjust_x_features"]
                dataset_config["x_features"] = dataset_config["x_features"][:limit]
            if "custom_processing" in rules:
                logging.info(f"Applying custom processing for {model_name} on {dataset_name} in experiment {experiment_name}")

        return dataset_config, model_config

    def load_full_config(self, dataset_name, model_name, experiment_name):
        """
        Loads and combines the dataset, model, and experiment configurations.
        Applies the combination rules before returning the full configuration.

        Args:
            dataset_name (str): The registered name of the dataset.
            model_name (str): The registered name of the model.
            experiment_name (str): Name of the experiment configuration.

        Returns:
            dict: Configuration with 'dataset', 'model', and 'experiment' sections.
        """
        dataset_config,scaler_params = self.load_dataset_config(dataset_name)
        model_config = self.load_model_config(model_name,dataset_name,scaler_params)
        experiment_config = self.load_experiment_config(experiment_name,dataset_name)

        adjusted_dataset_config, adjusted_model_config = self.apply_combination_rules(
            dataset_name, model_name, experiment_name, dataset_config, model_config
        )

        full_config = {
            "dataset": adjusted_dataset_config,
            "model": adjusted_model_config,
            "experiment": experiment_config
        }
        return full_config


def build_optuna_search_space(config):

    """
    Builds an Optuna-compatible search space from the model configuration.

    Args:
        config (dict): Full configuration containing the 'model' section.

    Returns:
        dict: Parameter names mapped to functions that suggest values for a trial.
    """
    def create_suggestion_function(key, space):
        """
        Creates the suggestion function for the given parameter and search-space type.
        """
        if space["type"] == "int":
            return lambda trial: trial.suggest_int(key, int(space["low"]), int(space["high"]),
                                                   step=int(space.get("step", 1)))
        elif space["type"] == "float":
            return lambda trial: trial.suggest_float(key, float(space["low"]), float(space["high"]),
                                                     step=space.get("step", None))
        elif space["type"] == "categorical":
            return lambda trial: trial.suggest_categorical(key, space["choices"])
        elif space["type"] == "loguniform":
            return lambda trial: trial.suggest_float(key, float(space["low"]), float(space["high"]), log=True)
        elif space["type"] == "uniform":
            return lambda trial: trial.suggest_uniform(key, float(space["low"]), float(space["high"]))
        else:
            raise ValueError(f"Unknown hyperparameter type: {space['type']}")

    config_model = config['model']
    search_space = {}

    for key, value in config_model.items():

        if isinstance(value, dict) and "search_space" in value:
            space = value["search_space"]
            search_space[key] = create_suggestion_function(key, space)

    return search_space

class ExperimentBase(ABC):
    """
    Abstract base class for running an experiment for a model. Provides the structure
    for configuration, data loading, hyperparameter tuning, training, and evaluation.
    """

    def __init__(
        self,
        dataset: str,
        model: str,
        seed=None,
        hash=None,
        console_mode=False,
        results_path_override=None,
        create_results_dir: bool = True,
        config_overrides: dict | None = None,
        variant_name: str | None = None,
        **kwargs,
    ):
        """
        Initializes the experiment configuration, result path, logging, and device.
        Looks up the model and dataset classes in their registries.

        Args:
            dataset (str): The registered name of the dataset.
            model (str): The registered name of the model.
            seed (int, optional): Random seed. Uses the experiment config or 42 if omitted.
            hash (str, optional): Existing run hash. Otherwise generated from the config.
            console_mode (bool, optional): Logs only to the console when True.
            results_path_override (Path or str, optional): Custom results directory.
            create_results_dir (bool, optional): Creates the result directory and saves
                the config. If False, keeps an existing saved config.
            config_overrides (dict, optional): Nested overrides applied before hashing.
            variant_name (str, optional): Label appended to the default result folder.
            **kwargs: Additional arguments accepted by subclasses.
        """
        self.model_name = model
        self.dataset_name = dataset
        self.variant_name = variant_name
        self.console_mode=console_mode
        self.config_manager = ConfigManager()
        experiment_name=type(self).__name__
        self.config = self.config_manager.load_full_config(dataset, model, experiment_name)
        if config_overrides:
            deep_update(self.config, copy.deepcopy(config_overrides))
        if seed is not None:
            self.seed = seed
            self.set_seed(seed)
            config_seed = self.config['experiment'].get('seed')
            if config_seed is not None and config_seed != seed:
                warn_msg = "Seed from config (" + str(
                    config_seed) + ") overwritten by seed from experiment(" + str(seed) + ")"
                warnings.warn(warn_msg)
        elif 'seed' in self.config['experiment']:
            self.seed = self.config['experiment']['seed']
            self.set_seed(self.seed)
        else:
            self.seed = 42
            self.set_seed(self.seed)
            warnings.warn("Seed not found in config for experiment, using default seed 42")
        if hash is None:
            config_hash = self.generate_hash_from_config()
        else:
            config_hash = hash
        self.config_hash=config_hash
        if results_path_override is not None:
            self.path_results = Path(results_path_override)
        else:
            run_name = model + '_' + dataset
            if self.variant_name:
                run_name = run_name + '_' + str(self.variant_name)
            self.path_results = Path(CONST.RESULTS_PATH, experiment_name, run_name,'Seed_'+str(self.seed), config_hash)
        if create_results_dir:
            self.path_results.mkdir(parents=True, exist_ok=True)
        if create_results_dir and Path(self.path_results, "timestamps.txt").exists():
            Path(self.path_results, "timestamps.txt").unlink()
        # Rebinding an existing run for read-only analysis should not require
        # a multiprocessing manager or socket-based queue listener.
        self.analysis_mode = results_path_override is not None and not create_results_dir
        if self.analysis_mode:
            self.manager = None
            self.log_queue = None
        else:
            self.manager = Manager()
            self.log_queue = self.manager.Queue()

        self.logger,self.queue_listener = self.__setup_logging()
        self._log_time("Start Experiment")
        atexit.register(self.__stop_logging)


        self.model_class,self.dataset_class=self.initialize(model,dataset)




        # Persist the config only for newly created result directories.
        # When rebinding an experiment to an existing copied run for analysis,
        # we must not overwrite the original saved config artifacts.
        config_save_path = Path(self.path_results, "config_experiment.yaml")
        if create_results_dir or not config_save_path.exists():
            with open(config_save_path, 'w') as file:
                yaml.dump(self.config, file)

    def path(self):
        """
        Returns the path to the experiment results.
        """
        return self.path_results

    @staticmethod
    def load_config_from_run(run_path: Path) -> dict:

        """
        Loads config_experiment.yaml from the given run directory.

        Args:
            run_path (Path): Path to the experiment run.

        Returns:
            dict: The saved experiment configuration.
        """
        config_path = Path(run_path, "config_experiment.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"config_experiment.yaml not found at {config_path}")

        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    @staticmethod
    def extract_hyperparameter_search_space(config: dict) -> dict:

        """
        Extracts the model's search-space definitions from a full configuration.

        Args:
            config (dict): Full experiment configuration.

        Returns:
            dict: Hyperparameter names mapped to their search-space definitions.
        """
        search_space = {}
        for param_name, param_config in config.get("model", {}).items():
            if isinstance(param_config, dict) and "search_space" in param_config:
                search_space[param_name] = param_config["search_space"]
        return search_space

    def __setup_logging(self):
        # Clear all existing handlers to prevent duplicate logs
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)


        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        logger = logging.getLogger(f"{self.__class__.__name__}_{self.dataset_name}_{self.model_name}_{self.config_hash}")

        if self.log_queue is None:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            logger.addHandler(console_handler)
            logger.propagate = False
            logging_level = logging.DEBUG if CONST.DEBUG else logging.INFO
            logger.setLevel(logging_level)
            logger.debug("Logger initialized in direct mode")
            return logger, None

        queue_listener = logging.handlers.QueueListener(self.log_queue, console_handler)
        queue_listener.start()

        # make sure QueueHandler is only added once
        if not any(isinstance(h, logging.handlers.QueueHandler) for h in logger.handlers):
            queue_handler = logging.handlers.QueueHandler(self.log_queue)
            logger.addHandler(queue_handler)

        logging_level = logging.DEBUG if CONST.DEBUG else logging.INFO
        logger.setLevel(logging_level)

        logger.debug("Logger initialized in queue mode")

        return logger, queue_listener

    def __del__(self):
        self.__stop_logging()

    def finish(self):
        """
        Stops the experiment's logging listener and closes its handlers.
        """
        self.__stop_logging()

    def __stop_logging(self):
        if getattr(self, "_logging_stopped", False):
            return
        self._logging_stopped = True

        if hasattr(self, 'queue_listener') and self.queue_listener is not None:
            try:
                self.logger.info("Stopping Logging (QueueListener)...")
                self.queue_listener.stop()
            except Exception as e:
                print(f"Warning: Exception while stopping QueueListener: {e}")

        if hasattr(self, 'log_queue') and self.log_queue is not None:
            try:
                self.log_queue.put_nowait(None)
            except Exception:
                pass

        self.queue_listener = None
        self.log_queue = None

    def print_experiment_path(self):
        print(f"Experiment path: {self.path_results}")

    def generate_hash_from_config(self) -> str:
        """
        Returns an eight-character hash of the configuration to identify the experiment.
        """
        hash_string = json.dumps(self.config, sort_keys=True)

        return hashlib.sha256(hash_string.encode('utf-8')).hexdigest()[:8]
    @staticmethod
    def set_seed(seed):
        set_seed(seed)
    def chose_device(self):
        assert self.config['experiment']['wanted_device'] in ['cuda', 'cpu',
                                                              'mps'], "device must be either 'cuda', 'cpu' or 'mps'"
        if self.config['experiment']['wanted_device'] == 'cuda' and torch.cuda.is_available():
            device_obj = torch.device('cuda')
            self.config['chosen_device'] = 'cuda'
            self.logger.info("CUDA available, using GPU")
            self.cuda_memory_budget = _apply_cuda_memory_budget(
                self.config,
                device_obj,
                self.logger,
                "Experiment process",
            )
        elif self.config['experiment']['wanted_device'] == 'cuda' and not torch.cuda.is_available():
            device_obj = torch.device('cpu')
            self.config['chosen_device'] = 'cpu'
            self.logger.info("CUDA not available, using CPU")
            self.cuda_memory_budget = None
        elif self.config['experiment']['wanted_device'] == 'mps' and torch.backends.mps.is_available():
            device_obj = torch.device('mps')
            self.config['chosen_device'] = 'mps'
            self.logger.info("MPS available, using MPS")
            self.cuda_memory_budget = None
        elif self.config['experiment']['wanted_device'] == 'mps' and not torch.backends.mps.is_available():
            device_obj = torch.device('cpu')
            self.config['chosen_device'] = 'cpu'
            self.logger.info("MPS not available, using CPU")
            self.cuda_memory_budget = None
        else:
            device_obj = torch.device('cpu')
            self.config['chosen_device'] = 'cpu'
            self.logger.info("Using CPU")
            self.cuda_memory_budget = None
        self.device = device_obj

    def initialize(self, model_name, dataset_name):
        """
        Looks up the model and dataset classes and selects the experiment device.

        Args:
            model_name (str): The registered name of the model.
            dataset_name (str): The registered name of the dataset.

        Returns:
            tuple: The model and dataset classes.
        """
        self.logger.info("Initializing model and dataset")
        assert model_name in MODEL_REGISTRY, f"Model {model_name} not found in registry. Please add to src.model.__init__.py"
        assert dataset_name in DATASET_REGISTRY, f"Dataset {dataset_name} not found in registry. Please add to src.datasets.__init__.py"
        assert 'wanted_device' in self.config['experiment'], "device not found in experiment config"

        self.chose_device()
        return MODEL_REGISTRY[model_name],DATASET_REGISTRY[dataset_name]

    def check_hyperparameter_search(self, start_dashboard=False):
        """
        Checks the best 5% of completed trials, with at least five trials when available.
        Warns when their hyperparameters are near the search-space boundaries.

        Args:
            start_dashboard (bool, optional): Starts the Optuna dashboard when True.
        """
        from optuna.trial import FrozenTrial

        path_db = Path(self.path_results, "hyperparameter_tuning","optuna_study.db")
        study_names = optuna.study.get_all_study_names(storage="sqlite:///" + str(path_db))
        self.logger.info(f"Evaluating Trials for {self.model_name} on {self.dataset_name}")
        self.logger.info(f"Available studies: {study_names}")
        if study_names:
            correct_study_name = study_names[0]
        else:
            raise ValueError("No study found in the database!")

        study = optuna.load_study(
            study_name=correct_study_name,
            storage="sqlite:///" + str(path_db)
        )

        search_space = build_optuna_search_space(self.config)

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        error_trails=[t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        pruned_trails=[t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

        self.logger.info(f"Number of completed trials: {len(completed_trials)} of {len(study.trials)} total trials.")
        self.logger.info(f"Number of pruned trials: {len(pruned_trails)} of {len(study.trials)} total trials.")
        self.logger.info(f"Number of error trials: {len(error_trails)} of {len(study.trials)} total trials.")

        if not completed_trials:
            self.logger.warning("No completed trials found.")
            return

        # Take the top 5% trials (at least 5)
        sorted_trials = sorted(completed_trials, key=lambda t: t.value)
        top_n = max(5, int(len(sorted_trials) * 0.05))
        top_trials = sorted_trials[:top_n]

        self.logger.info(f"Analyzing the top {top_n} trials:")
        self.logger.info(f"The top trials are: {[t.number for t in top_trials]}")
        param_counts = {param: Counter() for param in search_space.keys()}

        warnings_dict = {}

        for trial in top_trials:
            for param_name, param_value in trial.params.items():
                if param_name in search_space:
                    space = self.config['model'][param_name]['search_space']

                    if space["type"] in ["int", "float", "loguniform", "uniform"]:
                        low, high = float(space["low"]), float(space["high"])

                    elif space["type"] == "categorical":
                        choices = space["choices"]

                        if all(isinstance(v, (int, float)) for v in choices):
                            low, high = float(min(choices)), float(max(choices))
                            space["low"], space["high"] = low, high
                        else:
                            continue
                    else:
                        continue

                    param_counts[param_name][param_value] += 1

                    threshold = (high - low) * 0.10  # 10% of the range as threshold
                    lower_diff = abs(param_value - low)
                    upper_diff = abs(param_value - high)

                    if lower_diff < threshold or upper_diff < threshold:
                        if param_name not in warnings_dict:
                            warnings_dict[param_name] = []

                        warning_msg = f"Trial {trial.number} - Value: {param_value:.5f} (Range: {low} - {high})"
                        if lower_diff < threshold:
                            warning_msg += f" ⚠ Near Lower Bound ({low})"
                        if upper_diff < threshold:
                            warning_msg += f" ⚠ Near Upper Bound ({high})"

                        warnings_dict[param_name].append(warning_msg)

        self.logger.info("\n--- Best Hyperparameter Values ---\n")
        summary_data = []
        for param_name, counts in param_counts.items():
            most_common = counts.most_common(top_n)
            values_str = ", ".join([f"{val} ({count}x)" for val, count in most_common])
            summary_data.append([param_name, values_str])

        df_summary = pd.DataFrame(summary_data, columns=["Hyperparameter", "Top Values"])

        self.logger.info(df_summary.to_string(index=False))

        if warnings_dict:
            self.logger.info("\n⚠ Hyperparameter Warnings ⚠")
            for param, warnings_list in warnings_dict.items():
                self.logger.info(f"\n🔹 {param}:")
                for warning in warnings_list:
                    self.logger.info(f"  {warning}")

        self.logger.info("\nHyperparameter search analysis completed.")

        if start_dashboard:
            try:
                subprocess.run(
                    ["optuna-dashboard", f"sqlite:///{path_db}"],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error starting Optuna Dashboard: {e}")

    @staticmethod
    def run_trial(config, model_class_name, dataset_class_name, default_config, study_db_path, gpu_mapping,
                  lock, completed_trials, first_trial_nr,log_queue,seed,gpu_lock,gpu_status):
        """
        Runs a single Optuna trial in an isolated process.
        Records the best validation loss in the study and releases the GPU slot.

        Args:
            config (dict): Full configuration containing the hyperparameter search space.
            model_class_name (str): The registered model name.
            dataset_class_name (str): The registered dataset name.
            default_config (dict): Full configuration with default model values.
            study_db_path (Path or str): Path to the Optuna SQLite database.
            gpu_mapping (dict): NVIDIA-SMI GPU IDs mapped to PyTorch GPU IDs.
            lock: Lock for updating the study and completed trials.
            completed_trials: Shared list of completed trial numbers.
            first_trial_nr (int): First trial number of this tuning run.
            log_queue: Shared queue for worker logs.
            seed (int): Base seed; each trial uses seed + trial number.
            gpu_lock: Lock for reserving and releasing GPU slots.
            gpu_status: Shared dictionary of available slots per GPU.

        Returns:
            float or None: Last validation loss on success, NaN on failure, or None
                if the trial is pruned. The study objective is the best validation loss.
        """
        logger = logging.getLogger("WorkerProcess")
        if CONST.DEBUG:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO
        logger.setLevel(log_level)

        if not any(isinstance(h, logging.handlers.QueueHandler) for h in logger.handlers):
            queue_handler = logging.handlers.QueueHandler(log_queue)
            logger.addHandler(queue_handler)

        try:
            hyperparameters = None  # make sure hyperparameters is defined
            trial = None
            trial_id = -1  # Initialize trial_id to avoid reference before assignment
            nv_gpu_id = None
            memory_budget = None

            hyperparameter_config = config['experiment']['hyperparameter_tuning']
            if hyperparameter_config['use_pruner']:
                pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=hyperparameter_config['min_epochs'],
                                                                reduction_factor=hyperparameter_config[
                                                                    'reduction_factor'])
            else:
                pruner = optuna.pruners.NopPruner()
            # Load Optuna study (each process must reconnect separately)
            study = optuna.load_study(
                study_name=f"Hyperparameter Optimization for {model_class_name} on {dataset_class_name}",
                storage=f"sqlite:///{study_db_path}",
                pruner=pruner,
            )

            with lock:
                trial = study.ask()
            trial_id = trial.number
            study.sampler = optuna.samplers.TPESampler(seed=seed+trial_id)
            set_seed(seed+trial_id)
            logger.debug(f"Starting trial {trial_id} seed {seed+trial_id}")
            logger.info(f"Starting trial {trial_id} process with PID {os.getpid()}")


            if torch.cuda.is_available() and default_config['chosen_device'] == 'cuda':
                nv_gpu_id = get_best_available_gpu(gpu_lock,gpu_status,logger)
                pt_gpu_id = gpu_mapping.get(nv_gpu_id, 0)  # Default to GPU 0 if mapping fails
                torch.cuda.set_device(pt_gpu_id)
                device = torch.device(f"cuda:{pt_gpu_id}")
                memory_budget = _apply_cuda_memory_budget(
                    default_config,
                    device,
                    logger,
                    f"HPO trial {trial_id}",
                )
                logger.debug(f"Trial {trial_id} running on NVIDIA-SMI GPU {nv_gpu_id} (PyTorch GPU cuda:{pt_gpu_id})")
            else:
                device = torch.device('cpu')

            if default_config['chosen_device'] == 'mps':
                device = torch.device('mps')


            if default_config['chosen_device'] == 'cuda':
                logger.info(f"Trial {trial_id} running on device: {device} which is NVIDIA-SMI GPU {nv_gpu_id}")
            else:
                logger.info(f"Trial {trial_id} running on device: {device}")



            search_space = build_optuna_search_space(config)
            hyperparameters = {key: suggest_fn(trial) for key, suggest_fn in search_space.items()}
            logger.debug(f"Trial {trial_id} sampled hyperparameters: {hyperparameters}")

            config = copy.deepcopy(default_config) #make sure we have a copy in any case
            config['model'].update(hyperparameters)

            model_class = MODEL_REGISTRY[model_class_name]
            dataset_class = DATASET_REGISTRY[dataset_class_name]
            model = model_class(config, logger=logger).to(device)

            dataset_train = dataset_class(config['dataset'], data_split='train', device=device)
            dataset_val = dataset_class(config['dataset'], data_split='val', device=device)

            batch_size = config['experiment']['batch_size']
            if batch_size<len(dataset_train):
                shuffle=True
            else:
                shuffle=False
            train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle,
                                      collate_fn=dataset_class.collate_fn)
            val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=dataset_class.collate_fn)

            total_epochs = config['experiment']['hyperparameter_tuning']['num_epochs']
            min_val_loss = float("inf")
            model.pretraining(train_loader)
            for epoch in range(total_epochs):
                model.train_step(train_loader)
                val_loss = model.validate_step(val_loader)
                ensure_finite_number(
                    val_loss,
                    "hpo/validation_loss",
                    "hyperparameter_tuning",
                    max_abs_threshold=get_failure_policy(config)["max_abs_loss"],
                )
                if val_loss < min_val_loss:
                    min_val_loss = val_loss #track the
                if epoch % max(1, total_epochs // 50) == 0 or epoch == total_epochs - 1:
                    with lock:
                        trial.report(val_loss, epoch)

                if epoch % max(1, total_epochs // 10) == 0 or epoch == total_epochs - 1:
                    logger.debug(f"Trail {trial_id} in Epoch {epoch + 1}/{total_epochs}: Validation Loss: {val_loss:.5f}")


                if hyperparameter_config['use_pruner']:
                    if trial.should_prune():

                        logger.debug(f"Trial {trial_id} pruned at epoch {epoch}.")
                        if CONST.DEBUG and trial.number - 1 not in completed_trials and trial.number > first_trial_nr:
                            logger.debug(f"Trial {trial_id} waiting for Trial {trial_id - 1}...")
                        # Wait until the previous trial finishes before proceeding
                        while trial.number - 1 not in completed_trials and trial.number > first_trial_nr:
                            if CONST.DEBUG:
                                time.sleep(3)
                                logger.debug(f"Trial {trial_id} still waiting for Trial {trial_id - 1}...")
                            time.sleep(0.5)

                        logger.info(f"Trial {trial_id} pruned with val_loss: {val_loss}")
                        if torch.cuda.is_available() and default_config['chosen_device'] == 'cuda':
                            release_gpu(gpu_lock,gpu_status,nv_gpu_id)
                        with lock:
                            study.tell(trial, state=TrialState.PRUNED)
                            completed_trials.append(trial.number)
                        return



            # Ensure trials finish in the correct order, accounting for first_trial_nr
            logger.debug(f"Trial {trial_id} is ready now. Check if prior trails are ready")

            if CONST.DEBUG and trial.number - 1 not in completed_trials and trial.number > first_trial_nr:
                logger.debug(f"Trial {trial_id} waiting for Trial {trial_id - 1}...")
            # Wait until the previous trial finishes before proceeding
            while trial.number - 1 not in completed_trials and trial.number > first_trial_nr:
                if CONST.DEBUG:
                    time.sleep(3)
                    logger.debug(f"Trial {trial_id} still waiting for Trial {trial_id - 1}...")
                time.sleep(0.5)

            logger.info(f"Trial {trial_id} completed successfully with best val_loss: {min_val_loss}")
            if torch.cuda.is_available() and default_config['chosen_device'] == 'cuda':
                release_gpu(gpu_lock, gpu_status, nv_gpu_id)
            with lock:
                trial.set_user_attr("numerical_status", "success")
                trial.set_user_attr("failure_type", None)
                trial.set_user_attr("completed_epochs", int(total_epochs))
                trial.set_user_attr("best_val_loss", float(min_val_loss))
                trial.set_user_attr("fallback_on_nan_dt", bool(config["model"].get("fallback_on_nan_dt", False)))
                if memory_budget is not None:
                    trial.set_user_attr("gpu_memory_limit_gb", memory_budget["gpu_memory_limit_gb"])
                    trial.set_user_attr("gpu_memory_fraction", memory_budget["gpu_memory_fraction"])
                    trial.set_user_attr("gpu_total_gib", memory_budget["gpu_total_gib"])
                study.tell(trial, float(min_val_loss))
                completed_trials.append(trial.number)
            return float(val_loss)

        except Exception as e:
            failure = classify_exception(e, stage="hyperparameter_tuning")
            logger.error(
                f"Trial {trial_id} with hyperparameters {hyperparameters} failed "
                f"with {failure['failure_type']}: {e}",
                exc_info=True,
            )
            time.sleep(1)
            if trial is not None and CONST.DEBUG and trial.number - 1 not in completed_trials and trial.number > first_trial_nr:
                logger.debug(f"Trial {trial_id} waiting for Trial {trial_id - 1}...")
            while trial is not None and trial.number - 1 not in completed_trials and trial.number > first_trial_nr:
                if CONST.DEBUG:
                    time.sleep(3)
                    logger.debug(f"Trial {trial_id} still waiting for Trial {trial_id - 1}...")
                time.sleep(0.5)  # Wait until the previous trial finishes before proceeding
            if torch.cuda.is_available() and default_config['chosen_device'] == 'cuda' and nv_gpu_id is not None:
                release_gpu(gpu_lock, gpu_status, nv_gpu_id)
            if trial is not None:
                with lock:
                    trial.set_user_attr("numerical_status", "failed")
                    trial.set_user_attr("failure_type", failure["failure_type"])
                    trial.set_user_attr("failure_stage", failure["stage"])
                    trial.set_user_attr("failure_exception_type", failure["exception_type"])
                    trial.set_user_attr("failure_message", failure["message"])
                    trial.set_user_attr("failure_details", failure["details"])
                    trial.set_user_attr("fallback_on_nan_dt", bool((config or {}).get("model", {}).get("fallback_on_nan_dt", False)))
                    if memory_budget is not None:
                        trial.set_user_attr("gpu_memory_limit_gb", memory_budget["gpu_memory_limit_gb"])
                        trial.set_user_attr("gpu_memory_fraction", memory_budget["gpu_memory_fraction"])
                        trial.set_user_attr("gpu_total_gib", memory_budget["gpu_total_gib"])
                    if hyperparameters is not None:
                        trial.set_user_attr("sampled_hyperparameters", hyperparameters)
                    study.tell(trial, state=TrialState.FAIL)
                    completed_trials.append(trial.number)
            logger.error(f"Trial {trial_id} failed and process ended.")
            return np.nan
        finally:
            logger.debug(f"Stopping QueueListener in trial {trial_id}")
            logging.getLogger().handlers.clear()

    @staticmethod
    def write_hyperparameter_failure_summary(study, base_path: Path) -> dict:
        """Write a compact failure report for the Optuna study."""

        rows = []
        for trial in study.get_trials(deepcopy=False):
            attrs = trial.user_attrs or {}
            state_name = trial.state.name
            failure_type = attrs.get("failure_type")
            numerical_status = attrs.get("numerical_status")

            if state_name == "COMPLETE" and numerical_status is None:
                numerical_status = "success"
            elif state_name == "FAIL" and numerical_status is None:
                numerical_status = "failed"
            if state_name == "FAIL" and failure_type is None:
                failure_type = "unclassified_failure"

            rows.append(
                {
                    "number": trial.number,
                    "state": state_name,
                    "value": trial.value,
                    "numerical_status": numerical_status,
                    "failure_type": failure_type,
                    "failure_stage": attrs.get("failure_stage"),
                    "failure_exception_type": attrs.get("failure_exception_type"),
                    "failure_message": attrs.get("failure_message"),
                    "fallback_on_nan_dt": attrs.get("fallback_on_nan_dt"),
                    "gpu_memory_limit_gb": attrs.get("gpu_memory_limit_gb"),
                    "gpu_memory_fraction": attrs.get("gpu_memory_fraction"),
                    "gpu_total_gib": attrs.get("gpu_total_gib"),
                    "completed_epochs": attrs.get("completed_epochs"),
                }
            )

        df = pd.DataFrame(rows)
        csv_path = Path(base_path, "hyperparameter_failure_summary.csv")
        df.to_csv(csv_path, index=False)

        if df.empty:
            summary = {
                "n_trials": 0,
                "n_complete": 0,
                "n_failed": 0,
                "n_pruned": 0,
                "failure_rate": None,
                "failure_type_counts": {},
                "state_counts": {},
            }
        else:
            state_counts = df["state"].value_counts(dropna=False).to_dict()
            failed_mask = df["state"].eq("FAIL") | df["numerical_status"].eq("failed")
            failure_type_counts = (
                df.loc[failed_mask, "failure_type"]
                .fillna("unclassified_failure")
                .value_counts(dropna=False)
                .to_dict()
            )
            n_trials = int(len(df))
            n_failed = int(failed_mask.sum())
            summary = {
                "n_trials": n_trials,
                "n_complete": int(state_counts.get("COMPLETE", 0)),
                "n_failed": n_failed,
                "n_pruned": int(state_counts.get("PRUNED", 0)),
                "failure_rate": float(n_failed / n_trials) if n_trials else None,
                "failure_type_counts": {str(k): int(v) for k, v in failure_type_counts.items()},
                "state_counts": {str(k): int(v) for k, v in state_counts.items()},
                "csv_path": str(csv_path),
            }

        with open(Path(base_path, "hyperparameter_failure_summary.json"), "w") as file:
            json.dump(summary, file, indent=2)

        return summary

    def hyperparameter_tuning(self):

        """
        Runs or resumes an Optuna study using separate worker processes.
        Limits parallel jobs to the configured GPU slots or available CPU cores.
        GPU workers reserve an available slot before training.

        Returns:
            dict: The best hyperparameters found during tuning.
        """
        hyperparameter_config = self.config['experiment']['hyperparameter_tuning']
        n_trials = hyperparameter_config['n_trials']
        gpu_mapping = get_pytorch_nvidia_gpu_mapping() if torch.cuda.is_available() else {}
        if self.device.type == 'cuda' and gpu_mapping:
            hpo_gpu_ids = sorted(gpu_mapping.keys())
        else:
            hpo_gpu_ids = list(range(torch.cuda.device_count()))

        if self.device.type=='cuda':
            jobs_per_GPU=hyperparameter_config.get('jobs_per_gpu',2)
            gpu_slot_counts = _normalise_gpu_slot_counts(
                hyperparameter_config.get('gpu_slot_counts'),
                hpo_gpu_ids,
                jobs_per_GPU,
            )
            max_jobs=sum(gpu_slot_counts.values())
            n_jobs=hyperparameter_config.get('n_jobs', torch.cuda.device_count())
            if hyperparameter_config.get('n_jobs', torch.cuda.device_count()) > max_jobs:
                self.logger.info(f"Reducing n_jobs from {hyperparameter_config.get('n_jobs', torch.cuda.device_count())} to {max_jobs} to avoid GPU memory conflicts")
                n_jobs=max_jobs
            elif hyperparameter_config.get('n_jobs', torch.cuda.device_count()) ==0:
                self.logger.error("n_jobs is set to 0, setting it to 1")
                hyperparameter_config['n_jobs']=1
                n_jobs=1
            elif hyperparameter_config.get('n_jobs', torch.cuda.device_count()) <0:
                self.logger.info(f"Performing max configured GPU slots in total: {max_jobs}")
                n_jobs = max_jobs
        else:
            gpu_slot_counts = {}
            if hyperparameter_config.get('n_jobs', cpu_count()) <0:
                self.logger.info(f"Performing max trials per CPU which is {cpu_count()}")
                n_jobs = cpu_count()
            elif hyperparameter_config.get('n_jobs', cpu_count()) ==0:
                self.logger.error("n_jobs is set to 0, setting it to 1")
                hyperparameter_config['n_jobs']=1
                n_jobs=1
            else:
                n_jobs = min(hyperparameter_config.get('n_jobs', cpu_count()), cpu_count())
        self.logger.info(f"Using {n_jobs} parallel jobs for hyperparameter tuning")

        # Load the dataset once to trigger generation/download from the main process
        warmup_dataset = self.dataset_class(self.config['dataset'], data_split='train', device=self.device)
        # Immediately release the warm-up dataset to avoid lingering GPU allocations
        del warmup_dataset
        if torch.cuda.is_available() and self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

        base_path = Path(self.path_results, 'hyperparameter_tuning')
        base_path.mkdir(parents=True, exist_ok=True)
        db_path = Path(base_path, "optuna_study.db")


        self.logger.info(f"Starting hyperparameter tuning with {n_trials} trials and {n_jobs} parallel jobs")
        self._log_time("Start Hyperparameter Tuning")

        if hyperparameter_config['use_pruner']:
            pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=hyperparameter_config['min_epochs'],
                                                            reduction_factor=hyperparameter_config['reduction_factor'])
        else:
            pruner = optuna.pruners.NopPruner()

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            study_name=f"Hyperparameter Optimization for {self.model_name} on {self.dataset_name}",
            storage=f"sqlite:///{db_path}",
            load_if_exists=True,
            pruner=pruner
        )
        time.sleep(1)  # Ensure the study is created before proceeding
        # Identify the next trial number in case the study is restarted mid-way
        existing_trials = study.get_trials()
        first_trial_number = max([trial.number for trial in existing_trials] or [0]) + 1

        with Manager() as manager:
            completed_trials = manager.list([-1])  # Shared list tracking completed trials.
            lock = manager.Lock()  # Lock for synchronizing GPU selection across processes.
            if self.device.type == 'cuda' and gpu_mapping:
                self.logger.info(
                    "HPO GPU slots use NVIDIA-SMI IDs from CUDA/PyTorch UUID mapping: "
                    + ", ".join(str(gpu_id) for gpu_id in hpo_gpu_ids)
                )
                self.logger.info(
                    "HPO GPU slot counts: "
                    + ", ".join(f"{gpu_id}:{slots}" for gpu_id, slots in sorted(gpu_slot_counts.items()))
                )
            gpu_status = manager.dict(gpu_slot_counts)
            gpu_lock = manager.Lock()

            run_trial_partial = functools.partial(
                self.run_trial,
                config=self.config,
                model_class_name=self.model_name,
                dataset_class_name=self.dataset_name,
                default_config=self.get_model_defaults(),
                study_db_path=db_path,
                gpu_mapping=gpu_mapping,
                lock=lock,
                completed_trials=completed_trials,
                first_trial_nr=first_trial_number,
                log_queue=self.log_queue,
                seed=self.seed,
                gpu_lock=gpu_lock,
                gpu_status=gpu_status

            )

            # Each trial runs in a separate process to prevent GPU memory conflicts.
            with mp.get_context('spawn').Pool(processes=n_jobs, maxtasksperchild=1) as pool:
                results = []
                num_new_trials = n_trials - len(study.trials)

                # Start new trials one by one with a short delay to avoid GPU congestion.
                for i in range(num_new_trials):
                    time.sleep(5)
                    self.logger.debug(f"Main process is starting trial index {i}")

                    r = pool.apply_async(run_trial_partial)

                    results.append(r)

                for r in results:
                    val_loss = r.get()
                    self.logger.debug(f"Main process got result val_loss={val_loss}")

            pool.close()
            pool.join()

        self._log_time("End Hyperparameter Tuning")
        trials_df = study.trials_dataframe()
        trials_df.to_csv(Path(base_path, "hyperparameter_search.csv"), index=False)
        failure_summary = self.write_hyperparameter_failure_summary(study, base_path)
        self.logger.info(f"Hyperparameter failure summary: {failure_summary}")

        # Save best hyperparameters. If all trials failed, the failure summary
        # above is still available for diagnosis before Optuna raises here.
        best_params = study.best_params
        self.logger.info(f"Best hyperparameters found: {best_params}")
        with open(Path(base_path, "best_hyperparameters.yaml"), 'w') as file:
            yaml.dump(best_params, file)

        return best_params
    def _log_time(self, name):
        self.logger.info(f'{name}: {datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    def load_best_hparams(self):
        if not Path(self.path_results, "hyperparameter_tuning", "best_hyperparameters.yaml").exists():
            raise FileNotFoundError(f"Best hyperparameters not found at {Path(self.path_results, 'hyperparameter_tuning', 'best_hyperparameters.yaml')}")
        with open(Path(self.path_results, "hyperparameter_tuning", "best_hyperparameters.yaml"), 'r') as file:
            best_hyperparameters = yaml.safe_load(file)
        return best_hyperparameters

    def get_untrained_model(self, best_hyperparameters=None,config=None):
        """
        Creates an untrained model using the given config or the model defaults.
        Updates the model configuration with best_hyperparameters when provided.

        Args:
            best_hyperparameters (dict, optional): Parameters to apply to the model config.
            config (dict, optional): Full configuration used to initialize the model.

        Returns:
            tuple: The model on the experiment device and its configuration.
        """
        if config is  None:
            self.logger.info("No base config provided. Using default config.")
            config = self.get_model_defaults()
        if best_hyperparameters is not None:
            self.logger.info("Updating config with best hyperparameters found during tuning")
            config['model'].update(best_hyperparameters)

        return self.model_class(config, logger=self.logger).to(self.device),config

    def final_training(self, best_hyperparameters=None,seed=None,epochs=None):
        """
        Trains the model and tracks training and validation loss across epochs.
        Saves and returns the model with the best validation loss.

        Args:
            best_hyperparameters (dict, optional): Parameters applied to the model defaults.
            seed (int, optional): Resets the random seed before training. If omitted,
                keeps the experiment's current random state.
            epochs (int, optional): Number of epochs; otherwise uses the experiment config.

        Returns:
            model: The trained model with the best validation performance.
        """
        self.logger.info("Starting final training")
        self._log_time('Final Training Start')
        log_every = 1 if CONST.DEBUG else 10
        if seed is not None:
            self.set_seed(seed)
            warn_msg = "Seed from config (" + str(
                self.config['experiment']['seed']) + ") overwritten by seed from experiment(" + str(seed) + ")"
            warnings.warn(warn_msg)
            self.logger.info(f"Using seed {seed} for final training")

        dataloaders = self.load_data(shuffle_train=False)
        self.model,config = self.get_untrained_model(best_hyperparameters)

        with open(Path(self.path_results, "model_config.yaml"), 'w') as file:
            yaml.dump(config, file)

        self.model.pretraining(dataloaders['train'])

        train_losses = []
        val_losses = []
        model_checkpoints = []  # Stores (state_dict, val_loss, epoch) tuples
        if epochs is not None:
            total_epochs = epochs
            self.logger.info(f"Overwriting number of epochs to {epochs} for final training")
        else:
            total_epochs = self.config['experiment']['num_epochs']

        best_val_loss = float("inf")
        train_loss_at_best_epoch = 0

        best_checkpoint = (copy.deepcopy(self.model.state_dict()), float("inf"), 0)
        start_time = time.time()
        for epoch in range(total_epochs):
            train_loss = self.model.train_step(dataloaders['train'], return_loss=True)
            train_losses.append(train_loss)

            val_loss = self.model.validate_step(dataloaders['val'])
            val_losses.append(val_loss)

            # Save best model state_dict with associated validation loss and epoch number
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                train_loss_at_best_epoch=train_loss
                best_checkpoint=(copy.deepcopy(self.model.state_dict()), val_loss, epoch + 1)
            if (epoch+1)%log_every == 0 or epoch == total_epochs - 1:
                self.logger.info(f"Epoch {epoch + 1}/{total_epochs}: Train Loss: {train_loss:.3e}, Val Loss: {val_loss:.3e}")

        training_duration = time.time() - start_time

        best_state_dict, best_val_loss, best_epoch = best_checkpoint
        self.model.load_state_dict(best_state_dict)

        best_model_path = Path(self.path_results, 'model_checkpoints', 'best_model.pth')
        best_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), best_model_path)

        with open(Path(self.path_results, 'model_checkpoints', "best_model_info.txt"), 'w') as file:
            file.write(f"Model trained for {total_epochs} epochs.\n")
            file.write(f"Best model found at epoch: {best_epoch}\n")
            file.write(f"Best model validation loss: {best_val_loss:.4e}\n")
            file.write(f"Best model training loss: {train_loss_at_best_epoch:.4e}\n")
            file.write(f"Training loss at last epoch: {train_losses[-1]:.4e}\n")
            file.write(f"Validation loss at last epoch: {val_losses[-1]:.4e}\n")
        self.logger.info(f"Best model saved at epoch {best_epoch} with validation loss {best_val_loss:.4e}")

        # Helper quantities for statistics / overfitting analysis
        epoch_of_first_min_val = int(np.argmin(val_losses) + 1)
        last_train_loss = float(train_losses[-1])
        last_val_loss = float(val_losses[-1])
        val_minus_train_at_best = float(best_val_loss - train_loss_at_best_epoch)
        val_minus_train_last = float(last_val_loss - last_train_loss)
        if best_val_loss > 0:
            relative_overfitting_last_vs_best = float((last_val_loss - best_val_loss) / best_val_loss)
        else:
            relative_overfitting_last_vs_best = None

        model_num_parameters = int(sum(p.numel() for p in self.model.parameters()))

        training_info = {
            'total_epochs': int(total_epochs),
            'best_epoch': int(best_epoch),
            'epoch_of_first_min_val_loss': int(epoch_of_first_min_val),

            'seed': int(self.seed) if hasattr(self, 'seed') else None,
            'config_hash': getattr(self, 'config_hash', None),
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,

            'device': str(self.device),
            'batch_size': int(self.config['experiment']['batch_size']),
            'model_num_parameters': model_num_parameters,
            'gpu_memory_limit_gb': (
                float(self.cuda_memory_budget["gpu_memory_limit_gb"])
                if getattr(self, "cuda_memory_budget", None) else None
            ),
            'gpu_memory_fraction': (
                float(self.cuda_memory_budget["gpu_memory_fraction"])
                if getattr(self, "cuda_memory_budget", None) else None
            ),
            'gpu_total_gib': (
                float(self.cuda_memory_budget["gpu_total_gib"])
                if getattr(self, "cuda_memory_budget", None) else None
            ),

            'best_val_loss': float(best_val_loss),
            'train_loss_at_best_epoch': float(train_loss_at_best_epoch),
            'last_epoch_train_loss': last_train_loss,
            'last_epoch_val_loss': last_val_loss,
            'val_minus_train_at_best': val_minus_train_at_best,
            'val_minus_train_last': val_minus_train_last,
            'relative_overfitting_last_vs_best': relative_overfitting_last_vs_best,

            'mean_train_loss': float(np.mean(train_losses)),
            'std_train_loss': float(np.std(train_losses)),
            'mean_val_loss': float(np.mean(val_losses)),
            'std_val_loss': float(np.std(val_losses)),

            'wall_clock_training_seconds': float(training_duration),
        }


        last_model_path = Path(self.path_results, 'model_checkpoints', 'last_epoch.pth')
        torch.save(self.model.state_dict(), last_model_path)


        with open(Path(self.path_results, 'training_info.yaml'), 'w') as file:
            yaml.dump(training_info, file)


        self.logger.info("Final training completed. Best model saved.")
        self._log_time('Final Training End')
        del dataloaders #make sure GPU memory is freed
        gc.collect()
        return self.model

    def save_model(self, model, filename):
        model_path = Path(self.path_results, 'model_checkpoints', filename)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)


    def load_best_model(self, return_best_epoch=False, allow_missing_scaling_factor: bool = False):
        """
        Loads the saved model configuration and best checkpoint on the experiment device.

        Args:
            return_best_epoch (bool, optional): Also reads the saved best epoch.
            allow_missing_scaling_factor (bool, optional): Allows checkpoints without
                scaling_factor buffers. Pretraining scaling must then be recomputed
                on the original training split before evaluation.

        Returns:
            model or tuple: The loaded model, optionally with the saved best epoch.
        """
        best_model_path = Path(self.path_results, 'model_checkpoints', 'best_model.pth')
        if not best_model_path.exists():
            raise FileNotFoundError(f"Best model not found at {best_model_path}")
        if not Path(self.path_results, "model_config.yaml").exists():
            raise FileNotFoundError(f"Model config not found at {Path(self.path_results, 'model_config.yaml')}")
        with open(Path(self.path_results, "model_config.yaml"), 'r') as file:
            config = yaml.safe_load(file)
        config['chosen_device']=self.device

        self.model,_ = self.get_untrained_model(config=config)
        state_dict = torch.load(best_model_path, map_location=self.model.device, weights_only=True)
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as exc:
            if "scaling_factor" not in str(exc):
                raise

            if not allow_missing_scaling_factor:
                raise RuntimeError(
                    "Checkpoint is missing persisted NODE scaling_factor buffers. "
                    "Refusing to load silently because this can change NODE predictions. "
                    "Use allow_missing_scaling_factor=True only in protocol code that immediately "
                    "recomputes scaling with model.pretraining() on the original training split. "
                    f"Checkpoint: {best_model_path}"
                ) from exc

            incompatible = self.model.load_state_dict(state_dict, strict=False)
            missing = list(incompatible.missing_keys)
            unexpected = list(incompatible.unexpected_keys)
            non_scaling_missing = [key for key in missing if not key.endswith("scaling_factor")]
            non_scaling_unexpected = [key for key in unexpected if not key.endswith("scaling_factor")]

            if non_scaling_missing or non_scaling_unexpected:
                raise RuntimeError(
                    "Checkpoint is incompatible for reasons beyond missing NODE scaling_factor buffers. "
                    f"Missing: {missing}. Unexpected: {unexpected}."
                ) from exc

            self.logger.warning(
                "Loaded an older checkpoint without persisted NODE scaling_factor buffers. "
                "If this is a NODE model with pretraining_scaling=True, recompute pretraining scaling "
                "on the intended training loader before evaluation."
            )
        self.model.to(self.device)


        if return_best_epoch:
            with open(Path(self.path_results,'model_checkpoints', "best_model_info.txt"), 'r') as file:
                for line in file:
                    if line.startswith("Best model found at epoch:"):
                        best_epoch = int(line.split(":", 1)[1].strip())
                        break
                else:
                    raise ValueError("Best epoch is missing from best_model_info.txt.")
            return self.model,best_epoch

        return self.model

    def performance_measurement(self, model, dataset):
        """
        Returns the model's prediction results for the given dataset or data loader.
        """
        self.logger.info("Measuring performance")
        y_pred = model.predict(dataset)
        return y_pred

    def write_results(self, results: dict, filename: str):

        # Convert numpy types to native Python types for JSON compatibility
        """
        Writes results as filename.json in the experiment directory.
        Converts NumPy values to Python values and non-finite floats to null.
        """
        def convert_to_python(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert arrays to lists
            elif isinstance(obj, (np.generic, np.number)):
                return convert_to_python(obj.item())  # Convert numpy scalars to Python scalars
            elif isinstance(obj, float) and not math.isfinite(obj):
                return None
            elif isinstance(obj, dict):
                return {k: convert_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python(v) for v in obj]
            else:
                return obj

        readable_results = convert_to_python(results)

        with open(Path(self.path_results, filename+'.json'), 'w') as file:
            json.dump(readable_results, file, indent=4)

    def copy_results_to(self, destination_path,overwrite=False):
        """
        Copies the experiment results directory to destination_path.
        An existing destination is replaced only if overwrite is True.
        """
        destination = Path(destination_path)
        if destination.exists():
            if overwrite:
                shutil.rmtree(destination)
                self.logger.info(f"Overwriting existing results at {destination}")
            else:
                raise FileExistsError(f"Destination {destination} already exists. Set overwrite=True to overwrite.")
        shutil.copytree(self.path_results, destination)

    def load_data(self, return_dataloader=True,shuffle_train=True,batch_size=None):
        """
        Loads the training, validation, and test splits for the experiment.

        Args:
            return_dataloader (bool, optional): Returns data loaders if True, datasets
                otherwise. Defaults to True.
            shuffle_train (bool, optional): Shuffles only the training loader.
                Defaults to True.
            batch_size (int, optional): Batch size; otherwise uses the experiment config.

        Returns:
            dict: Data loaders or datasets under 'train', 'val', and 'test'.
        """
        train_dataset = self.dataset_class(self.config['dataset'], data_split='train',device=self.device)
        val_dataset = self.dataset_class(self.config['dataset'], data_split='val',device=self.device)
        test_dataset = self.dataset_class(self.config['dataset'], data_split='test',device=self.device)


        if return_dataloader:
            if batch_size is None:
                batch_size=self.config['experiment']['batch_size']
            train_loader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=shuffle_train,
                                      collate_fn=self.dataset_class.collate_fn)
            val_loader = DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    collate_fn=self.dataset_class.collate_fn)
            test_loader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     collate_fn=self.dataset_class.collate_fn)
            return {"train": train_loader, "val": val_loader, "test": test_loader}
        return {"train": train_dataset, "val": val_dataset, "test": test_dataset}



    def get_model_defaults(self):

        """
        Returns the full configuration with model search-space entries replaced by
        their 'default' values, preserving the nested model structure.
        """
        def recursive_defaults(d):
            if isinstance(d, dict):
                out = {}
                for key, value in d.items():
                    if isinstance(value, dict) and "default" in value:
                        out[key] = value["default"]
                    else:
                        out[key] = recursive_defaults(value)
                return out
            else:
                return d

        config = self.config.copy()
        if "model" in config:
            config["model"] = recursive_defaults(config["model"])
        return config






    def performance_evaluation(self, y_true, y_pred, metrics):
        """
        Evaluates predictions against ground truth using the experiment's scoring method.

        Args:
            y_true: Ground truth values.
            y_pred: Model predictions.
            metrics: Metrics passed to the scoring method.

        Returns:
            The result returned by scoring.
        """
        self.logger.info("Evaluating performance")
        return self.scoring(y_true, y_pred, metrics)

    @abstractmethod
    def run_experiment(self, default_parameters=False):
        """
        Defines the main experiment flow; implemented by derived classes.

        Args:
            default_parameters (bool, optional): Selects default model parameters when
                supported by the experiment.
        """
        pass

    def store_best_model(self, seed, model):
        """
        Appends the model to the list of stored models for the given seed.
        """
        if seed not in self.best_models:
            self.best_models[seed] = []
        self.best_models[seed].append(model)


    def load_train_losses(self):
        """
        Loads the training and validation loss histories from training_logs.

        Returns:
            tuple: Training losses and validation losses.
        """
        train_losses_path = Path(self.path_results, 'training_logs', 'train_losses.yaml')
        val_losses_path = Path(self.path_results, 'training_logs', 'val_losses.yaml')
        if not train_losses_path.exists():
            raise FileNotFoundError(f"Train losses not found at {train_losses_path}")

        if not val_losses_path.exists():
            raise FileNotFoundError(f"Val losses not found at {val_losses_path}")

        with open(train_losses_path, 'r') as file:
            train_losses = yaml.safe_load(file)

        with open(val_losses_path, 'r') as file:
            val_losses = yaml.safe_load(file)

        return train_losses, val_losses
    def get_name_channels_dataset(self):
        """
        Returns the dataset's channel names through its get_channels method.
        """
        if hasattr(self.dataset_class, 'get_channels'):
            return self.dataset_class.get_channels()
        else:
            raise NotImplementedError("get_channels not implemented for this dataset class")
