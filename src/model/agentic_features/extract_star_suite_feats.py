#!/usr/bin/env python3
"""
extract_star_suite_feats.py

Parallel feature extraction script for the StarFeatureSuite (our 222-feature architecture).
Designed with the exact CLI argument structure and multiprocessing pattern as 
handcrafted_features/extract_feats.py for direct, seamless benchmark comparison.

NOTE: Does NOT include any FATS or light_curve (LC_extractor) handcrafted features.
Extracts strictly our autonomous AI designed StarFeatureSuite features.
"""

from datasets import load_from_disk
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import time
import os
import warnings

# Import our custom feature extractor
from star_feature_suite import StarFeatureExtractor

# Suppress runtime warnings during feature extraction
warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_suite_columns(bands_to_process):
    """
    Generates the complete, deterministic list of feature column names produced 
    by StarFeatureExtractor for the given passbands.
    """
    extractor = StarFeatureExtractor(bands=bands_to_process)
    
    # Create a synthetic dummy star with observations in all target bands
    dummy_star = {
        'sourceid': 'dummy',
        'period': 1.234,
        'ra': 180.0,
        'dec': 30.0,
        'bands_data': {}
    }
    for b in bands_to_process:
        dummy_star['bands_data'][b] = {
            'mjd': np.linspace(58000.0, 59000.0, 50),
            'target': np.random.normal(15.0, 0.1, 50),
            'past_feat_dynamic_real': np.full(50, 0.015)
        }
        
    sample_feats = extractor.extract_features(dummy_star)
    # Exclude non-numeric identifier 'sourceid'
    cols = [k for k in sorted(sample_feats.keys()) if k != 'sourceid']
    return cols


def process_batch(batch_data, batch_idx, batch_size, bands_to_process, suite_columns):
    """
    Processes a single batch of astronomical objects in parallel across worker processes.
    """
    extractor = StarFeatureExtractor(bands=bands_to_process)
    batch_features = pd.DataFrame(columns=suite_columns)
    
    start_time = time.time()
    
    # Determine the number of objects in this batch (handling both dict-of-lists from HF dataset slice and list-of-dicts)
    if isinstance(batch_data, dict):
        # HuggingFace slice dict where keys are column names mapping to lists
        first_key = list(batch_data.keys())[0]
        n_objects = len(batch_data[first_key])
        is_dict_of_lists = True
    else:
        n_objects = len(batch_data)
        is_dict_of_lists = False
        
    for i in range(n_objects):
        star_idx = batch_idx * batch_size + i
        
        if is_dict_of_lists:
            star = {k: batch_data[k][i] for k in batch_data.keys()}
        else:
            star = batch_data[i]
            
        try:
            feats = extractor.extract_features(star)
            # Filter out sourceid and align exactly to suite_columns, sanitizing inf/nan/overflows
            row_vals = [float(np.nan_to_num(feats.get(col, 0.0), nan=0.0, posinf=0.0, neginf=0.0)) for col in suite_columns]
            batch_features.loc[star_idx, :] = row_vals
        except Exception as e:
            # Fallback to zeros/NaNs if a malformed record is encountered
            batch_features.loc[star_idx, :] = np.full(len(suite_columns), 0.0)
            
    batch_time = time.time() - start_time
    print(f"Finished batch {batch_idx} ({n_objects} objects) in {batch_time:.2f}s")
    return batch_features, batch_time


def compile_star_suite_features(data, bands_to_process, suite_columns, num_workers):
    """
    Compiles StarFeatureSuite features across the entire dataset split using multiprocessing.
    """
    compiled_features = pd.DataFrame(columns=suite_columns)
    
    # Multiprocessing configuration
    num_workers = max(1, num_workers - 1)  # Leave one CPU free
    batch_size = max(1, len(data) // (num_workers * 2))
    
    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batches.append((
            batch, i // batch_size, batch_size,
            bands_to_process, suite_columns
        ))
        
    print(f"Processing {len(data)} objects with {num_workers} workers in {len(batches)} batches")
    print(f"Feature dimensionality per object: {len(suite_columns)}")
    
    # Run multiprocessing extraction pool
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.starmap(process_batch, batches),
            total=len(batches),
            desc="Computing StarFeatureSuite features"
        ))
        
    total_extraction_time = 0.0
    for batch_features, batch_time in results:
        compiled_features = pd.concat([compiled_features, batch_features])
        total_extraction_time += batch_time
        
    compiled_features = compiled_features.sort_index()
    
    print("\n-- Time spent calculating StarFeatureSuite features --")
    print(f"Total cumulative worker extraction time: {total_extraction_time/60:.2f} minutes")
    
    return compiled_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract StarFeatureSuite features from dataset")
    parser.add_argument(
        '--split', type=str, choices=['train', 'validation', 'test', 'anom'],
        required=True, help='Dataset split to process (train, validation, test, or anom)'
    )
    parser.add_argument(
        "--dataset_path", type=str,
        required=True, help='Path to dataset on disk'
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path to save the extracted features as a CSV file"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of worker processes to use for parallel feature extraction"
    )
    parser.add_argument(
        "--bands_to_process", type=str,
        required=False, help='Bands to process (comma-separated list, e.g. "g,r,i")'
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path
    split = args.split
    num_workers = args.num_workers
    bands_to_process = args.bands_to_process
    output_path = args.output_path

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)

    # Determine bands present
    if bands_to_process is not None:
        bands_to_process = [b.strip() for b in bands_to_process.split(',')]
    else:
        first_example = dataset[split][0]
        if "bands_data" not in first_example:
            raise ValueError("Expected 'bands_data' key in dataset example, but not found.")
        bands_to_process = list(first_example["bands_data"].keys())
        print(f"Bands discovered in dataset: {bands_to_process}")

    suite_columns = get_suite_columns(bands_to_process)
    print(f"Target Feature Suite columns ({len(suite_columns)} features across bands {bands_to_process})")

    # Process split
    print(f"\nExtracting features for split: {split}")
    start_wall_time = time.time()
    suite_feats = compile_star_suite_features(
        dataset[split], bands_to_process, suite_columns, num_workers
    )
    wall_time_min = (time.time() - start_wall_time) / 60.0
    print(f"Total wall-clock execution time: {wall_time_min:.2f} minutes")

    # Save results
    os.makedirs(output_path, exist_ok=True)
    output_file = f"{output_path}/{split}_{os.path.basename(dataset_path)}.csv"
    suite_feats.to_csv(output_file, index=None)
    print(f"\nSuccess! Extracted {suite_feats.shape[1]} features across {suite_feats.shape[0]} objects.")
    print(f"Features saved to: {output_file}")
