#!/usr/bin/env python3
"""
Universal Out-of-Distribution (OOD) / Anomaly Detection Benchmark.

Evaluates how well stellar time-series embeddings distinguish normal variable stars
(test split) from out-of-distribution or anomalous astronomical objects (anom split).
Uses Isolation Forest ensembles and evaluates exact AUROC and AUPRC metrics across random seeds.
"""

import os
import argparse
import numpy as np
import pandas as pd
from datasets import load_from_disk
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
import json


def parse_args():
    p = argparse.ArgumentParser(description="OOD / Anomaly Detection Benchmark")
    p.add_argument("--input_embs", type=str, required=True,
                   help="Path to HF DatasetDict containing train, test, and anom splits")
    p.add_argument("--output_dir", type=str, default="ood_results",
                   help="Directory to save anomaly scores and AUROC/AUPRC reports")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123],
                   help="Random seeds for Isolation Forest ensembles")
    p.add_argument("--n_estimators", type=int, default=200,
                   help="Number of trees in Isolation Forest")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"================ OOD Anomaly Detection: {args.input_embs} ================")
    ds = load_from_disk(args.input_embs)

    if "combined_embedding" not in ds["train"].features:
        raise ValueError("Dataset must contain 'combined_embedding' column!")

    print("Converting splits to NumPy arrays...")
    X_train = np.array(ds["train"]["combined_embedding"], dtype=np.float32)
    y_train = np.array(ds["train"]["class_str"]) if "class_str" in ds["train"].features else np.zeros(len(X_train))

    X_test = np.array(ds["test"]["combined_embedding"], dtype=np.float32)
    X_anom = np.array(ds["anom"]["combined_embedding"], dtype=np.float32)

    print(f"Train normal samples: {len(X_train)} | Test normal samples: {len(X_test)} | Anomaly samples: {len(X_anom)}")

    # Create binary ground truth for OOD evaluation: 0 = Normal (test), 1 = Anomaly (anom)
    X_eval = np.vstack([X_test, X_anom])
    y_eval_binary = np.concatenate([np.zeros(len(X_test), dtype=int), np.ones(len(X_anom), dtype=int)])

    results_by_seed = []

    for seed in args.seeds:
        print(f"\n--- Running Isolation Forest (Seed {seed}) ---")
        # Fit global Isolation Forest on normal training data
        iso_forest = IsolationForest(n_estimators=args.n_estimators, random_state=seed, n_jobs=-1)
        iso_forest.fit(X_train)

        # In sklearn IsolationForest, decision_function returns lower scores for anomalies and higher for normal.
        # So we negate decision_function to get anomaly score (higher = more anomalous).
        anom_scores = -iso_forest.decision_function(X_eval)

        auroc = roc_auc_score(y_eval_binary, anom_scores)
        auprc = average_precision_score(y_eval_binary, anom_scores)

        print(f"Seed {seed} -> AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")

        # Also compute class-conditional anomaly models (matching AD_test.py structure)
        classes = np.unique(y_train)
        class_scores = {}
        for cls_name in classes:
            idx = np.where(y_train == cls_name)[0]
            if len(idx) < 10:
                continue
            clf_cls = IsolationForest(n_estimators=args.n_estimators, random_state=seed, n_jobs=-1)
            clf_cls.fit(X_train[idx])
            class_scores[f"score_{cls_name}"] = -clf_cls.decision_function(X_eval)

        # Save scores to CSV
        df_scores = pd.DataFrame({"is_anomaly": y_eval_binary, "global_anomaly_score": anom_scores})
        for col, vals in class_scores.items():
            df_scores[col] = vals
        
        csv_path = os.path.join(args.output_dir, f"ood_scores_seed{seed}.csv")
        df_scores.to_csv(csv_path, index=False)
        print(f"Saved anomaly scores to: {csv_path}")

        results_by_seed.append({"seed": seed, "auroc": float(auroc), "auprc": float(auprc)})

    # Calculate mean and std
    aurocs = [r["auroc"] for r in results_by_seed]
    auprcs = [r["auprc"] for r in results_by_seed]

    mean_auroc, std_auroc = np.mean(aurocs), np.std(aurocs)
    mean_auprc, std_auprc = np.mean(auprcs), np.std(auprcs)

    print("\n================ OOD BENCHMARK SUMMARY ================")
    print(f"AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
    print(f"AUPRC: {mean_auprc:.4f} ± {std_auprc:.4f}")
    print("=======================================================")

    summary_data = {
        "dataset": args.input_embs,
        "mean_auroc": float(mean_auroc),
        "std_auroc": float(std_auroc),
        "mean_auprc": float(mean_auprc),
        "std_auprc": float(std_auprc),
        "runs": results_by_seed
    }

    summary_file = os.path.join(args.output_dir, "ood_summary_report.json")
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved summary report to: {summary_file}")


if __name__ == "__main__":
    main()
