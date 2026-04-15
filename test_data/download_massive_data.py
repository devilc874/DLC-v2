#!/usr/bin/env python3
"""
Download the Jena Climate Dataset (420K+ continuous sensor readings).

This provides a mathematically pure, unlooped testbed of real-world
floating-point data to benchmark DLC v2 against GZIP/ZLIB.

Source: Google TensorFlow datasets (Jena Climate 2009-2016)
Columns extracted: Temperature (degC), Pressure (mbar)

Usage:
    python test_data/download_massive_data.py
"""

import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
ZIP_PATH = os.path.join(SCRIPT_DIR, "jena_climate.zip")
CSV_PATH = os.path.join(SCRIPT_DIR, "jena_climate_2009_2016.csv")


def download_and_extract():
    print(f"Downloading massive Jena Climate dataset from Google...")
    print(f"URL: {DATASET_URL}")
    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)

    print("Extracting ZIP file...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(SCRIPT_DIR)

    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)


def process_to_bin():
    print("Loading CSV into memory (this might take a few seconds)...")
    df = pd.read_csv(CSV_PATH)

    # Extract Temperature column
    temp_data = df['T (degC)'].values.astype('<f8')
    temp_bin_path = os.path.join(SCRIPT_DIR, "real_jena_temperature_420k.bin")
    temp_data.tofile(temp_bin_path)
    print(f"Saved {len(temp_data):,} samples to {temp_bin_path} "
          f"({os.path.getsize(temp_bin_path)/1e6:.2f} MB)")

    # Extract Pressure column
    pressure_data = df['p (mbar)'].values.astype('<f8')
    pressure_bin_path = os.path.join(SCRIPT_DIR, "real_jena_pressure_420k.bin")
    pressure_data.tofile(pressure_bin_path)
    print(f"Saved {len(pressure_data):,} samples to {pressure_bin_path} "
          f"({os.path.getsize(pressure_bin_path)/1e6:.2f} MB)")

    if os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)

    print("\nSuccess! Unlooped datasets ready for benchmarking.")


if __name__ == "__main__":
    download_and_extract()
    process_to_bin()
