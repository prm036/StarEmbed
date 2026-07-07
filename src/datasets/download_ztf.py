#!/usr/bin/env python3
"""
Download and Save the StarEmbed ZTF_40k Dataset from Hugging Face Hub.

Fetches the dataset from `StarEmbed/ZTF_40k` and saves it locally to disk
so we can run our agentic feature extraction and benchmarking pipeline.
"""

import os
import argparse
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Download StarEmbed ZTF_40k Dataset")
    parser.add_argument(
        "--repo_id", type=str, default="StarEmbed/ZTF_40k",
        help="Hugging Face dataset repository ID"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/ZTF_40k",
        help="Local directory path to save the dataset to disk"
    )
    parser.add_argument(
        "--split", type=str, default="all",
        help="Which split to download (e.g. 'validation', 'train', 'test', or 'all' for entire dataset)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Connecting to Hugging Face Hub: {args.repo_id}...")

    if args.split == "all":
        print("Downloading all dataset splits...")
        ds = load_dataset(args.repo_id)
    else:
        print(f"Downloading '{args.split}' split...")
        ds = load_dataset(args.repo_id, split=args.split)

    print(f"Download complete! Saving dataset to disk at: {args.output_dir} ...")
    ds.save_to_disk(args.output_dir)
    print(f"\nSuccess! Dataset successfully saved to: {args.output_dir}")
    print("You can now run feature extraction:")
    print(f"  python src/model/agentic_features/extract_agentic_feats.py --split validation --dataset_path {args.output_dir} --output_path ./output_agentic")


if __name__ == "__main__":
    main()
