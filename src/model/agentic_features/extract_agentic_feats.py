#!/usr/bin/env python3
"""
Parallel Multiprocessing Extractor for Agent-Engineered Features.

Extracts the 36 agent-engineered features per band across an entire dataset split
(train, validation, test, or anom) using Python multiprocessing.

Saves the output as both:
  1. A CSV file (for quick inspection and legacy compatibility)
  2. A HuggingFace Dataset with standardized column names (`embeddings_g`, `embeddings_r`,
     `combined_embedding`, `class_str`, `sourceid`), ready for immediate evaluation in
     `src/benchmark/classification/linear_knn.py` and `rf_hpo.py`.

Usage:
  python extract_agentic_feats.py \
      --split validation \
      --dataset_path /path/to/dataset \
      --output_path /path/to/output_dir \
      --num_workers 4
"""

import os
import time
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk, Dataset, DatasetDict

from generated_features import calc_agentic_features, FEATURE_NAMES, NUM_FEATURES


def process_batch(batch_data, batch_idx, batch_size, bands_to_process, columns):
    """Process a batch of astronomical objects and return their agent-engineered features."""
    batch_features = pd.DataFrame(columns=columns)
    batch_time = 0.0

    for i, star in enumerate(batch_data['bands_data']):
        star_idx = batch_idx * batch_size + i
        start_time = time.time()
        
        band_feats = []
        for band in bands_to_process:
            try:
                feats = calc_agentic_features(star.get(band))
            except Exception:
                feats = np.full(NUM_FEATURES, -3.0, dtype=np.float32)
            band_feats.append(feats)
            
        batch_time += time.time() - start_time
        batch_features.loc[star_idx, :] = np.concatenate(band_feats)

    return batch_features, batch_time


def compile_agentic_features(data, bands_to_process, columns, num_workers):
    """Compile agent-engineered features across a dataset split using multiprocessing."""
    features_df = pd.DataFrame(columns=columns)
    total_time = 0.0

    num_workers = max(1, num_workers)
    batch_size = max(1, len(data) // (num_workers * 4))

    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batches.append((
            batch, i // batch_size, batch_size,
            bands_to_process, columns
        ))

    print(f"Processing {len(data)} objects with {num_workers} workers across {len(batches)} batches...")

    if num_workers > 1 and len(batches) > 1:
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.starmap(process_batch, batches),
                total=len(batches),
                desc="Extracting Agentic Features"
            ))
    else:
        results = []
        for b in tqdm(batches, desc="Extracting Agentic Features (Sequential)"):
            results.append(process_batch(*b))

    for batch_features, batch_time in results:
        features_df = pd.concat([features_df, batch_features])
        total_time += batch_time

    features_df = features_df.sort_index()
    print(f"\nTotal CPU time spent computing agentic features: {total_time/60:.2f} minutes")
    return features_df


def main():
    parser = argparse.ArgumentParser(description="Extract Agent-Engineered Features from Dataset")
    parser.add_argument(
        '--split', type=str, choices=['train', 'validation', 'test', 'anom', 'all'],
        default='validation', help='Dataset split to process (or "all" for all splits)'
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help='Path to input HuggingFace dataset on disk'
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Directory path to save the extracted features (CSV and HF Dataset)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of CPU worker processes for parallel extraction"
    )
    parser.add_argument(
        "--bands_to_process", type=str, required=False,
        help='Comma-separated list of bands (e.g. "g,r" or "I,V"). If omitted, auto-detected.'
    )
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    print(f"Loading dataset from: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)

    # Determine splits to process
    if isinstance(dataset, DatasetDict):
        splits_to_process = list(dataset.keys()) if args.split == 'all' else [args.split]
    else:
        # Wrap single dataset in dict
        dataset = DatasetDict({"validation": dataset})
        splits_to_process = ["validation"]

    for split in splits_to_process:
        print(f"\n============================================================")
        print(f"Processing Split: {split} ({len(dataset[split])} samples)")
        print(f"============================================================")
        
        split_data = dataset[split]
        if len(split_data) == 0:
            print(f"Split '{split}' is empty. Skipping.")
            continue

        # Determine bands
        if args.bands_to_process:
            bands = args.bands_to_process.split(',')
        else:
            first_ex = split_data[0]
            if "bands_data" not in first_ex:
                raise ValueError("Expected 'bands_data' key in dataset example.")
            bands = list(first_ex["bands_data"].keys())
        print(f"Target bands: {bands}")

        columns = [f"{band}_{feat}" for band in bands for feat in FEATURE_NAMES]

        # Extract features
        feats_df = compile_agentic_features(split_data, bands, columns, args.num_workers)

        # 1. Save as CSV
        csv_file = os.path.join(args.output_path, f"{split}_agentic_features.csv")
        feats_df.to_csv(csv_file, index=False)
        print(f"Saved CSV report to: {csv_file}")

        # 2. Build standardized HuggingFace Dataset for Benchmarks
        print("Constructing HuggingFace Dataset formatted for benchmark suite...")
        hf_dict = {}
        
        # Copy essential metadata columns
        for col in ["sourceid", "class_str", "period", "ra", "dec"]:
            if col in split_data.column_names:
                hf_dict[col] = split_data[col]

        # Create embeddings_{band} columns (each row is a list of floats of length NUM_FEATURES)
        combined_embeddings = []
        for i in range(len(feats_df)):
            row_vals = feats_df.iloc[i]
            comb = []
            for band in bands:
                band_cols = [f"{band}_{feat}" for feat in FEATURE_NAMES]
                band_vec = row_vals[band_cols].values.astype(np.float32).tolist()
                
                if f"embeddings_{band}" not in hf_dict:
                    hf_dict[f"embeddings_{band}"] = []
                hf_dict[f"embeddings_{band}"].append(band_vec)
                comb.extend(band_vec)
            combined_embeddings.append(comb)

        # Add ready-to-use combined_embedding (concatenated across bands)
        hf_dict["combined_embedding"] = combined_embeddings

        out_hf = Dataset.from_dict(hf_dict)
        hf_split_dir = os.path.join(args.output_path, split)
        out_hf.save_to_disk(hf_split_dir)
        print(f"Saved HuggingFace benchmark dataset to: {hf_split_dir}")
        print(f"  -> Features per band: {NUM_FEATURES}")
        print(f"  -> Combined embedding dimension: {len(combined_embeddings[0])}")
        print(f"  -> Ready for evaluation in linear_knn.py and rf_hpo.py!")

    print("\nAll splits processed successfully!")

    if len(splits_to_process) > 1:
        print("\nCombining all processed splits into a unified DatasetDict...")
        full_dict = DatasetDict({
            split: load_from_disk(os.path.join(args.output_path, split))
            for split in splits_to_process if os.path.exists(os.path.join(args.output_path, split))
        })
        dict_path = os.path.join(args.output_path, "hf_dataset_dict")
        full_dict.save_to_disk(dict_path)
        print(f"Saved complete benchmark DatasetDict to: {dict_path}")
        print(f"-> Ready for immediate evaluation with: --input_embs {dict_path}")


if __name__ == "__main__":
    main()
