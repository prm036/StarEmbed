#!/usr/bin/env python3
"""
AD_test_star_suite.py
Out-of-Distribution / Anomaly Detection Benchmark using Isolation Forest.
Compatible with StarFeatureSuite and handcrafted feature datasets.
"""
import os
import argparse
import numpy as np
import pandas as pd
from datasets import load_from_disk
from sklearn.ensemble import IsolationForest

def add_embedding_batch(examples):
    arr_g = np.array(examples["embeddings_g"], dtype=np.float32)
    arr_r = np.array(examples["embeddings_r"], dtype=np.float32)

    if arr_g.ndim == 4 and arr_g.shape[1] == 1:
        arr_g = arr_g.squeeze(axis=1)
        arr_r = arr_r.squeeze(axis=1)

    if arr_g.ndim == 3:
        g = arr_g.mean(axis=1)
        r = arr_r.mean(axis=1)
    else:
        g, r = arr_g, arr_r

    return {"g_embedding": g, "r_embedding": r}

def run_ood_evaluation(main_ds_path, anom_ds_path, output_dir, seeds=[0, 42, 123]):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading main dataset: {main_ds_path}")
    ds = load_from_disk(main_ds_path, keep_in_memory=False)
    print(f"Loading anomaly dataset: {anom_ds_path}")
    ds_anom = load_from_disk(anom_ds_path, keep_in_memory=False)

    if 'embeddings_g' in ds['validation'].features and 'g_embedding' not in ds['validation'].features:
        print("Mapping embeddings_g/embeddings_r -> g_embedding/r_embedding...")
        ds = ds.map(
            add_embedding_batch,
            batched=True,
            batch_size=512,
            num_proc=4,
            remove_columns=["embeddings_g", "embeddings_r"],
            keep_in_memory=False,
        )
        
    if 'embeddings_g' in ds_anom['train'].features and 'g_embedding' not in ds_anom['train'].features:
        ds_anom = ds_anom.map(
            add_embedding_batch,
            batched=True,
            batch_size=512,
            num_proc=4,
            remove_columns=["embeddings_g", "embeddings_r"],
            keep_in_memory=False,
        )

    ds.set_format(type="numpy", columns=["g_embedding", "r_embedding", "class_str"])
    ds_anom.set_format(type="numpy", columns=["g_embedding", "r_embedding", "class_str"])

    def batched_xy(split):
        if split == 'anoms':
            anom_split_name = 'anoms' if 'anoms' in ds_anom else 'train'
            X = np.concatenate([ds_anom[anom_split_name]["g_embedding"], ds_anom[anom_split_name]["r_embedding"]], axis=1)
            y = ds_anom[anom_split_name]["class_str"]
        else:
            X = np.concatenate([ds[split]["g_embedding"], ds[split]["r_embedding"]], axis=1)
            y = ds[split]["class_str"]
        return X, y

    X_train, y_train = batched_xy("train")
    X_val, y_val = batched_xy("validation")
    X_test, y_test = batched_xy("test")
    X_anom, y_anom = batched_xy("anoms")

    print(f"Test shape: {X_test.shape}, Anomaly shape: {X_anom.shape}")

    X_combined_test = np.concatenate([X_test, X_anom], axis=0)
    y_combined_test = np.concatenate([y_test, y_anom], axis=0)

    for seed in seeds:
        print(f"\n--- Running IsolationForest for seed {seed} ---")
        models = {}
        unique_classes = np.unique(np.asarray(y_val))
        
        for i in unique_classes:
            idx = np.where(np.asarray(y_val) == i)[0]
            m = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)
            m.fit(X_train[idx, :])
            models[i] = m

        df = pd.DataFrame()
        df['True'] = y_combined_test
        for key in models:
            df[key] = models[key].decision_function(X_combined_test)

        dataset_basename = os.path.basename(main_ds_path.rstrip('/'))
        out_csv = os.path.join(output_dir, f"{dataset_basename}_AD_seed{seed}.csv")
        df.to_csv(out_csv, index=False)
        print(f"✅ Saved anomaly decision scores to: {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Isolation Forest OOD Detection Benchmark")
    parser.add_argument("--main_ds", type=str, required=True, help="Path to main HF embeddings directory")
    parser.add_argument("--anom_ds", type=str, required=True, help="Path to anomaly HF embeddings directory")
    parser.add_argument("--output_dir", type=str, default="output/ood", help="Directory to save OOD decision scores")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123], help="Random seeds to evaluate")
    args = parser.parse_args()

    run_ood_evaluation(args.main_ds, args.anom_ds, args.output_dir, args.seeds)
