import os
from pathlib import Path
import sys


PROJECT_PATH = Path(os.path.abspath(__file__)).parent
sys.path.append(str(Path(PROJECT_PATH, 'src')))
DATA_PATH = PROJECT_PATH / 'data'
RAW_DATA_PATH = PROJECT_PATH / 'Datasets'
OUTPUT_PATH = PROJECT_PATH / 'output'

RESULTS_PATH = Path(PROJECT_PATH, 'results')
THESIS_RESULTS_PATH = RESULTS_PATH / 'thesis_results'
NEW_RESULTS_PATH = RESULTS_PATH / 'new_runs'
EVALUATION_SOURCE = 'thesis_results'  # 'thesis_results' or 'new_runs'
if EVALUATION_SOURCE not in ['thesis_results', 'new_runs']:
    raise ValueError("EVALUATION_SOURCE must be 'thesis_results' or 'new_runs'")
EVALUATION_RESULTS_PATH = RESULTS_PATH / EVALUATION_SOURCE
CONFIGS_PATH = Path(PROJECT_PATH, 'configs')
SOURCE_PATH = Path(PROJECT_PATH, 'src')

VERBOSE = False
VERBOSE_BATCHES=False
DEBUG=False
