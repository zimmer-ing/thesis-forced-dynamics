"""Download or generate missing datasets and run the existing preparation scripts."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Constants as const

datasets = ['indpensim', 'tanks', 'robot', 'nanodrone', 'smd', 'rlc', 'fractional']
external_datasets = ['indpensim', 'tanks', 'robot', 'nanodrone']
TANKS_URLS = [
    'https://data.4tu.nl/file/d4810b78-6cdd-48fe-8950-9bd601e5f47f/3b697e42-01a4-4979-a370-813a456c36f5',
    'https://drive.google.com/uc?export=download&id=1HnQf_gu0g_UlggoBqy2s34l9YJiFdN01',
]
ROBOT_URL = 'https://fdm-fallback.uni-kl.de/TUK/FB/MV/WSKL/0001/Robot_Identification_Benchmark_Without_Raw_Data.rar'
METADATA_FILES = ['scaler_params.json', 'split_files.json', 'metadata.json']


def output_groups(dataset):
    data = const.DATA_PATH
    if dataset == 'indpensim':
        batches = list(range(1, 15)) + list(range(30, 45)) + list(range(61, 75))
        return [(data / 'IndPenSim/minmax_scaled', [f'Batch_{i}.csv' for i in batches] + ['scaler_params.json'], None)]
    if dataset in ['tanks', 'robot', 'nanodrone']:
        from src.datasets import CascadedTanks, IndustrialRobotForward, NanoDroneOOD, NanoDroneRunSplit
        if dataset == 'tanks':
            dataset_classes = [CascadedTanks]
        elif dataset == 'robot':
            dataset_classes = [IndustrialRobotForward]
        else:
            dataset_classes = [NanoDroneOOD, NanoDroneRunSplit]
        groups = []
        for dataset_class in dataset_classes:
            names = dataset_class.TRAIN_FILES + dataset_class.VAL_FILES + dataset_class.TEST_FILES
            groups.append((dataset_class.DATA_DIR, names + METADATA_FILES, dataset_class))
        return groups
    if dataset == 'smd':
        file_order = json.loads((ROOT / 'src/datasets/SMD_file_order.json').read_text())
        names = []
        for split, count in [('train', 300), ('val', 150), ('test', 150)]:
            expected = {f'{split}/Sample{i}.csv' for i in range(count)}
            if len(file_order[split]) != count or set(file_order[split]) != expected:
                raise ValueError(f'Invalid SMD file order for {split}.')
            names.extend(file_order[split])
        return [(data / 'SMD_Dataset', names, None)]
    if dataset == 'rlc':
        return [(data / '4th_order_RLC', [f'{split}/{split}_data.csv.gz' for split in ['train', 'val', 'test']], None)]
    names = [f'{task}/{split}/traj_{i:04d}.csv' for task in ['reconstruction', 'extrapolation', 'forcing']
             for split, count in [('train', 40), ('val', 20), ('test', 20)] for i in range(1, count + 1)]
    return [(data / 'Fractional_Data/fractional_benchmark_data', names, None)]


def data_status(directory, names, dataset_class=None):
    missing = [name for name in names if not (directory / name).is_file() or (directory / name).stat().st_size == 0]
    if missing:
        if directory.exists() and any(p.is_file() for p in directory.rglob('*')):
            raise ValueError(f'Incomplete dataset in {directory}: {len(missing)} required files missing or empty. '
                             'Existing files were kept. Restore the missing files or move this directory aside before retrying.')
        return False
    for name in names:
        path = directory / name
        if path.suffix == '.json':
            content = json.loads(path.read_text())
            if not isinstance(content, dict) or not content:
                raise ValueError(f'Invalid metadata: {path}')
            if name == 'split_files.json':
                split_files = [filename for split in ['train', 'val', 'test'] for filename in content[split]]
                if set(split_files) != {item for item in names if item.endswith('.csv')} or len(split_files) != len(set(split_files)):
                    raise ValueError(f'Invalid split file list: {path}')
                if dataset_class is not None:
                    expected = {'train': dataset_class.TRAIN_FILES,
                                'val': dataset_class.VAL_FILES,
                                'test': dataset_class.TEST_FILES}
                    if content != expected:
                        raise ValueError(f'Prepared split does not match {dataset_class.__name__}: {path}. '
                                         'Move the prepared directory aside and run preparation again.')
            if name == 'scaler_params.json' and not content.get('scaling_details'):
                raise ValueError(f'Missing scaling parameters: {path}')
        else:
            opener = gzip.open if path.suffix == '.gz' else open
            with opener(path, 'rt', newline='') as handle:
                reader = csv.reader(handle)
                header = next(reader)
                first_row = next(reader)
                if len(header) < 3 or len(first_row) != len(header):
                    raise ValueError(f'Invalid CSV: {path}')
    return True


def file_digest(path, algorithm):
    digest = hashlib.new(algorithm)
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(url, path, expected_size, expected_md5):
    if path.is_file():
        if path.stat().st_size != expected_size or file_digest(path, 'md5') != expected_md5:
            raise ValueError(f'Invalid archive: {path}. Move it aside before retrying.')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.download')
    print(f'Download: {url}', flush=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'thesis-forced-dynamics-data-preparation'})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    if temporary.stat().st_size != expected_size or file_digest(temporary, 'md5') != expected_md5:
        raise ValueError(f'Download checksum mismatch: {temporary}')
    temporary.replace(path)


def download_tanks():
    cache = const.RAW_DATA_PATH / 'NonlinearBenchmarks/_cache/Cascaded_Tanks'
    mat = cache / 'CascadedTanksFiles/dataBenchmark.mat'
    if mat.is_file():
        if file_digest(mat, 'sha256') != 'cb2f88d4388be4d3f2a24c6402fba804976aac5f2e1f26cda59ea0a38d016eab':
            raise ValueError(f'Unexpected Cascaded Tanks source: {mat}')
        return
    archive = cache / 'CascadedTanksFiles.zip'
    errors = []
    for url in TANKS_URLS:
        try:
            download(url, archive, 7520592, 'f832776ed1efdca4572338f467b7baec')
            break
        except (OSError, ValueError) as error:
            errors.append(str(error))
    else:
        raise RuntimeError('Cascaded Tanks download failed. Place the original dataBenchmark.mat at '
                           f'{mat}. Sources: https://doi.org/10.4121/12960104.v1. ' + '; '.join(errors))
    with zipfile.ZipFile(archive) as handle:
        content = handle.read('CascadedTanksFiles/dataBenchmark.mat')
    if hashlib.sha256(content).hexdigest() != 'cb2f88d4388be4d3f2a24c6402fba804976aac5f2e1f26cda59ea0a38d016eab':
        raise ValueError('Cascaded Tanks MAT checksum mismatch')
    mat.parent.mkdir(parents=True, exist_ok=True)
    mat.write_bytes(content)


def download_robot():
    directory = const.RAW_DATA_PATH / 'IndustrialRobot/Rohdaten'
    mat = directory / 'forward_identification_without_raw_data.mat'
    if mat.is_file():
        return
    tool = next((name for name in ['bsdtar', 'unar', 'unrar', '7z', '7zz'] if shutil.which(name)), None)
    if tool is None:
        raise RuntimeError('Industrial Robot needs bsdtar, unar, unrar or 7z to unpack the RAR. '
                           f'Alternatively, place forward_identification_without_raw_data.mat at {mat}.')
    archive = directory / 'Robot_Identification_Benchmark_Without_Raw_Data.rar'
    download(ROBOT_URL, archive, 12717003, '6fffa993f60d65f85100bad053888df2')
    with tempfile.TemporaryDirectory(prefix='robot-', dir=directory) as temp:
        if tool == 'bsdtar':
            command = [tool, '-xf', str(archive), '-C', temp, mat.name]
        elif tool == 'unar':
            command = [tool, '-q', '-o', temp, str(archive), mat.name]
        elif tool == 'unrar':
            command = [tool, 'x', '-idq', str(archive), mat.name, temp + '/']
        else:
            command = [tool, 'x', '-y', '-o' + temp, str(archive), mat.name]
        subprocess.run(command, check=True)
        matches = list(Path(temp).rglob(mat.name))
        if len(matches) != 1:
            raise ValueError('The Robot archive does not contain exactly one forward MAT file')
        shutil.copy2(matches[0], mat)


def prepare(dataset, complete):
    if dataset == 'indpensim':
        from Scripts.DataPreparation.Load_Preprocess_IndPenSim import main
        main([])
    elif dataset == 'tanks':
        download_tanks()
        from Scripts.DataPreparation.prepare_nonlinearbenchmark_core_datasets import main
        main()
    elif dataset == 'robot':
        download_robot()
        from Scripts.DataPreparation.build_forward_dataset import main
        main([])
    elif dataset == 'nanodrone':
        from Scripts.DataPreparation.download_nanodrone_lfs_csvs import main
        main()
        if not complete[0]:
            from Scripts.DataPreparation.prepare_nanodrone_dataset import main
            main()
        if not complete[1]:
            from Scripts.DataPreparation.prepare_nanodrone_run_split_dataset import main
            main()
    elif dataset in ['smd', 'rlc']:
        import torch
        import yaml
        if dataset == 'smd':
            from src.datasets.SMD_Dataset import SMDSystem
            torch.manual_seed(42)
            config = yaml.safe_load((ROOT / 'configs/datasets/SMDSystem.yaml').read_text())
            SMDSystem(config, data_split='train', device='cpu')
        else:
            from src.datasets.RLC_Datasets import Order_4_Dataset
            directory = const.DATA_PATH / '4th_order_RLC'
            for split in ['train', 'val', 'test']:
                if (directory / split).is_dir():
                    (directory / split).rmdir()
            config = yaml.safe_load((ROOT / 'configs/datasets/RLCDataset.yaml').read_text())
            Order_4_Dataset(config, data_split='train', device='cpu')
    else:
        matlab = shutil.which('matlab')
        if matlab is None:
            raise RuntimeError('Fractional CSVs are missing. Restore data/Fractional_Data from the repository '
                               'or install MATLAB with FOMCON on its search path to generate them.')
        subprocess.run([matlab, '-batch', "assert(exist('fotf','file') ~= 0, 'FOMCON is not on the MATLAB path'); "
                                        "run('Scripts/DataPreparation/SimpleFDE_DSgen.m')"], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', nargs='?', default='all', choices=['all', 'external'] + datasets)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Show existing and missing datasets without downloading or preparing data.')
    mode.add_argument('--check', action='store_true', help='Check required prepared files without creating anything.')
    args = parser.parse_args()
    if args.dataset == 'all':
        selected = datasets
    elif args.dataset == 'external':
        selected = external_datasets
    else:
        selected = [args.dataset]
    failures = []
    for dataset in selected:
        try:
            groups = output_groups(dataset)
            complete = [data_status(directory, names, dataset_class) for directory, names, dataset_class in groups]
            if all(complete):
                print(f'{dataset}: prepared files checked; skipping download and preprocessing.', flush=True)
                continue
            if args.dry_run:
                print(f'{dataset}: missing; would download/generate and prepare data.', flush=True)
                continue
            if args.check:
                raise FileNotFoundError(f'{dataset}: required prepared data is missing.')
            print(f'{dataset}: preparing missing data.', flush=True)
            prepare(dataset, complete)
            if not all(data_status(directory, names, dataset_class) for directory, names, dataset_class in groups):
                raise RuntimeError(f'{dataset}: preparation did not produce all required files.')
            print(f'{dataset}: ready.', flush=True)
        except (OSError, ValueError, RuntimeError, KeyError, StopIteration, ImportError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
            failures.append(dataset)
            print(f'{dataset}: {error}', flush=True)
    if failures:
        print('Not ready: ' + ', '.join(failures), flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
