"""
Data downloading utilities for time series datasets.

Provides functions to download energy demand and commodity price data via yfinance.
"""

import os
import tempfile
import time
import urllib.request
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

RANDOM_SEED = 42


def _get_processed_data_dir() -> Path:
    """Get the path to the processed data directory."""
    return Path(__file__).parent / "processed"


def _is_file_stale(filepath: Path, hours: int = 24) -> bool:
    """Check if a file is older than the specified number of hours.

    Args:
        filepath: Path to the file to check.
        hours: Number of hours to consider as stale threshold.

    Returns:
        True if file is stale or doesn't exist, False otherwise.
    """
    if not filepath.exists():
        return True

    file_age_seconds = time.time() - os.path.getmtime(filepath)
    stale_threshold_seconds = hours * 3600
    return file_age_seconds > stale_threshold_seconds


def download_energy() -> None:
    """Download energy demand data.

    Downloads the UCI household electric power consumption dataset and saves
    it as daily aggregated data to data/processed/energy.csv.

    Raises:
        RuntimeError: If the download fails.
    """
    output_path = _get_processed_data_dir() / "energy.csv"

    try:
        # Try to download from UCI ML Repository
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"

        # Create a temporary directory for the zip file
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "energy.zip"

            # Download the zip file
            urllib.request.urlretrieve(url, zip_path)

            # Extract and read the data
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            # Find the CSV file
            csv_files = list(Path(tmpdir).glob("*.txt"))
            if not csv_files:
                raise FileNotFoundError("No CSV/TXT file found in archive")

            csv_path = csv_files[0]

            # Read the data
            df = pd.read_csv(
                csv_path,
                sep=";",
                na_values="?",
                low_memory=False
            )

            # Combine Date and Time columns
            df["ds"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")

            # Drop rows with missing values in Global_active_power
            df = df.dropna(subset=["Global_active_power"])

            # Resample to daily and take the mean
            df = df.set_index("ds")
            df_daily = df[["Global_active_power"]].resample("D").mean()

            # Reset index and rename columns
            df_daily = df_daily.reset_index()
            df_daily.columns = ["ds", "y"]

            # Remove any remaining NaN values
            df_daily = df_daily.dropna()

            # Save to CSV
            df_daily.to_csv(output_path, index=False)
    except Exception as e:
        raise RuntimeError(f"Failed to download energy data from UCI: {e}")


def download_commodity() -> None:
    """Download commodity price data using yfinance.

    Downloads crude oil futures ("CL=F") with a 24h staleness check.
    Resamples to daily business days and forward-fills gaps (≤5 consecutive).
    """
    output_path = _get_processed_data_dir() / "commodity.csv"

    # Check staleness: only download if file is older than 24 hours
    if not _is_file_stale(output_path, hours=24):
        return

    try:
        # Download 10 years of data
        df = yf.download("CL=F", period="10y", auto_adjust=True, progress=False)

        # Extract the Close column and reset index
        df = df[["Close"]].reset_index()

        # Rename columns
        df.columns = ["ds", "y"]

        # Ensure datetime type
        df["ds"] = pd.to_datetime(df["ds"])

        # Resample to daily business days (exclude weekends)
        df = df.set_index("ds")
        df = df.asfreq("B")  # Business day frequency

        # Forward-fill gaps (up to 5 consecutive)
        df["y"] = df["y"].ffill(limit=5)

        # Remove rows with NaN values
        df = df.dropna()

        # Reset index
        df = df.reset_index()

        # Save to CSV
        df.to_csv(output_path, index=False)
    except Exception as e:
        raise RuntimeError(f"Failed to download commodity data: {e}")


def load_dataset(name: str) -> pd.DataFrame:
    """Load a dataset from the processed data directory.

    Args:
        name: Name of the dataset (without .csv extension).
              Valid options: "energy", "commodity"

    Returns:
        DataFrame with 'ds' (datetime) and 'y' (float) columns.

    Raises:
        FileNotFoundError: If the dataset file doesn't exist.
    """
    filepath = _get_processed_data_dir() / f"{name}.csv"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset '{name}' not found at {filepath}. "
            f"Run download_energy() and download_commodity() first."
        )

    df = pd.read_csv(filepath)

    # Ensure 'ds' is datetime
    if df["ds"].dtype != "datetime64[ns]":
        df["ds"] = pd.to_datetime(df["ds"])

    return df


if __name__ == "__main__":
    # Download both datasets
    print("Downloading energy dataset...")
    download_energy()

    print("Downloading commodity dataset...")
    download_commodity()

    # Load and verify
    print("\nLoading and verifying datasets...")
    energy = load_dataset("energy")
    commodity = load_dataset("commodity")

    # Run smoke tests
    assert len(energy) >= 365, f"Energy too short: {len(energy)} rows"
    assert len(commodity) >= 2500, f"Commodity too short: {len(commodity)} rows"
    assert list(energy.columns) == ["ds", "y"], f"Wrong columns: {energy.columns.tolist()}"
    assert list(commodity.columns) == ["ds", "y"], f"Wrong columns: {commodity.columns.tolist()}"
    assert energy["ds"].dtype == "datetime64[ns]", "energy ds not datetime"
    assert commodity["ds"].dtype == "datetime64[ns]", "commodity ds not datetime"

    # Verify no gaps in resampled index
    energy_dates = pd.to_datetime(energy["ds"]).sort_values().reset_index(drop=True)
    diffs = energy_dates.diff().dropna()
    assert (diffs <= pd.Timedelta(days=5)).all(), f"Energy has gaps > 5 days: max gap {diffs.max()}"

    commodity_dates = pd.to_datetime(commodity["ds"]).sort_values().reset_index(drop=True)
    diffs = commodity_dates.diff().dropna()
    assert (diffs <= pd.Timedelta(days=5)).all(), f"Commodity has gaps > 5 business days: max gap {diffs.max()}"

    print(f"Energy: {len(energy)} rows, {energy['ds'].min()} to {energy['ds'].max()}")
    print(f"Commodity: {len(commodity)} rows, {commodity['ds'].min()} to {commodity['ds'].max()}")
    print("All checks passed.")
