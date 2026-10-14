"""Prepare IndPenSim with the historical min-max fit over all 100 batches."""

import argparse
import hashlib
import http.client
import sys
import urllib.request
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DOWNLOAD_URL = "https://data.mendeley.com/public-files/datasets/pdnjz7zz5x/files/e7007dff-4061-494f-a9f5-16c20f457d1d/file_downloaded"
ARCHIVE_SIZE = 503792426
ARCHIVE_SHA256 = "bc434c6cf96444fee3c915d434ac84e2a9c6492be4daebcfa4aed9cdff181f00"
REQUIRED_ZIP_MEMBERS = (
    "Mendeley_data/100_Batches_IndPenSim_V3.csv",
    "Mendeley_data/100_Batches_IndPenSim_Statistics.csv",
)


def validate_zip(path: Path) -> None:
    if not zipfile.is_zipfile(path):
        raise ValueError("IndPenSim file is not a ZIP archive.")
    if path.stat().st_size != ARCHIVE_SIZE:
        raise ValueError("IndPenSim ZIP does not match the original archive size.")
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            checksum.update(chunk)
    if checksum.hexdigest() != ARCHIVE_SHA256:
        raise ValueError("IndPenSim ZIP does not match the original SHA-256 checksum.")
    with zipfile.ZipFile(path) as archive:
        for member in REQUIRED_ZIP_MEMBERS:
            if member not in archive.namelist() or archive.getinfo(member).file_size == 0:
                raise ValueError(f"Missing or empty IndPenSim archive member: {member}")
        invalid_member = archive.testzip()
        if invalid_member is not None:
            raise zipfile.BadZipFile(f"Corrupt IndPenSim archive member: {invalid_member}")


def download_archive(url: str, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix(path.suffix + ".download")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "thesis-forced-dynamics-data-preparation"})
        with urllib.request.urlopen(request, timeout=180) as response, tmp.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        validate_zip(tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the original IndPenSim dataset with min-max scaling fitted to all 100 batches.")
    parser.parse_args(argv)
    import pandas as pd
    import Constants as const
    from sklearn.preprocessing import MinMaxScaler
    from src.datasets.dataset_base import save_scaler_details

    # Define the directory for raw files under IndPenSim
    raw_dir = Path(const.RAW_DATA_PATH, "IndPenSim", "raw")
    raw_dir.mkdir(exist_ok=True, parents=True)

    # Define the path for the downloaded zip file
    raw_zip_path = raw_dir / "data.zip"

    try:
        if raw_zip_path.exists():
            print(f"Found IndPenSim ZIP: {raw_zip_path}. Checking the existing file...", flush=True)
            validate_zip(raw_zip_path)
        else:
            print("Downloading and checking the IndPenSim ZIP...", flush=True)
            download_archive(DOWNLOAD_URL, raw_zip_path)
    except (OSError, ValueError, zipfile.BadZipFile, http.client.HTTPException) as error:
        raise RuntimeError(
            f"IndPenSim ZIP could not be downloaded or verified: {error}\n"
            "Please open this link and download the original ZIP:\n"
            f"{DOWNLOAD_URL}\n"
            "Save it at the following path (replace data.zip if it is invalid):\n"
            f"{raw_zip_path}\n"
            "Then rerun from the repository root:\n"
            "bash Scripts/Prepare_Data.sh indpensim"
        ) from None

    print("IndPenSim ZIP verified. Preprocessing now...", flush=True)

    with zipfile.ZipFile(raw_zip_path) as archive:
        with archive.open(REQUIRED_ZIP_MEMBERS[0]) as handle:
            columns = pd.read_csv(handle, nrows=0).columns.tolist()
        process_columns = columns[:columns.index("2400")]
        with archive.open(REQUIRED_ZIP_MEMBERS[0]) as handle:
            # The original NumPy split converted all process columns to float.
            variable_data = pd.read_csv(handle, usecols=process_columns).astype(float)

        # Rename columns correctly
        variable_data = variable_data.rename(
            columns={
                '2-PAT control(PAT_ref:PAT ref)': 'Batch reference(Batch_ref:Batch ref)',
                'Batch reference(Batch_ref:Batch ref)': '2-PAT control(PAT_ref:PAT ref)'
            }
        )

        # Group data by batch reference and save each batch as a separate CSV file in "Unscaled"
        unscaled_dir = Path(const.DATA_PATH, "IndPenSim", "Unscaled")
        unscaled_dir.mkdir(exist_ok=True, parents=True)
        bp = variable_data.groupby('Batch reference(Batch_ref:Batch ref)')
        for batch, df in bp:
            df.to_csv(Path(unscaled_dir, f"Batch_{int(batch)}.csv"), index=False)

        # Create a description file with batch information in the main IndPenSim folder
        description = (
            "Batches 1-30: Controlled by recipe driven approach\n"
            "Batches 31-60: Controlled by operators\n"
            "Batches 61-90: Controlled by an Advanced Process Control (APC) solution using the Raman spectroscopy\n"
            "Batches 91-100: Contain faults resulting in process deviations"
        )
        with open(Path(Path(const.DATA_PATH, "IndPenSim"), "batch_description.txt"), "w") as file:
            file.write(description)

        # -----------------------------
        # Fit on all 100 original batches before the loader selects train/val/test.
        # This retains the historical TII preprocessing, including held-out batches.
        # -----------------------------
        # Define list of columns not to be scaled
        exclude_columns = ['Time (h)','Batch reference(Batch_ref:Batch ref)']
        # Determine which columns will be scaled
        columns_to_scale = [col for col in variable_data.columns if col not in exclude_columns]

        print("Applying MinMax scaling to the dataset (excluding columns:", exclude_columns, ")...")
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(variable_data[columns_to_scale])

        # Create a copy of the original dataframe and replace only the columns to be scaled
        scaled_df = variable_data.copy()
        scaled_df[columns_to_scale] = scaled_values

        # Define the directory for scaled data and save the results
        scaled_dir = Path(const.DATA_PATH, "IndPenSim", "minmax_scaled")
        scaled_dir.mkdir(exist_ok=True, parents=True)

        # Save each batch as a separate CSV file in the scaled directory
        bp_scaled = scaled_df.groupby('Batch reference(Batch_ref:Batch ref)')
        for batch, df in bp_scaled:
            df.to_csv(Path(scaled_dir, f"Batch_{int(batch)}.csv"), index=False)

        # Save the scaler object so it can be loaded later in the dataset class
        save_scaler_details(
            scaler=scaler,
            scaled_columns=columns_to_scale,
            exclude_columns=exclude_columns,
            output_dir=scaled_dir)

        print("Processing complete. The zip file is retained at:", raw_zip_path)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        sys.exit(1)
