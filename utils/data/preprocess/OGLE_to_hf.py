from datasets import Dataset, Features, Value, Sequence
import multiprocessing
from tqdm import tqdm
import argparse
import time

from read_OGLE import (
    load_catalog, merge_remarks, merge_ident, read_light_curve, get_period_feature_columns
)


# Standardized StarEmbed schema with some columns unique to Catalina
band_schema = Features({
    "mjd": Sequence(feature=Value("float64")),
    "target": Sequence(feature=Value("float64")),  # mag
    "past_feat_dynamic_real": Sequence(feature=Value("float64")),  # mag unc
    "feat_dynamic_real": Sequence(feature=Value("float64")),  # delta t between observations
    "length": Value("int64"),
})

schema = Features({
    # From catalog file
    "sourceid": Value("string"),
    "avg_mag_I": Value("float64"),
    "avg_mag_V": Value("float64"),

    "parent_type": Value("string"),
    "sub_type": Value("string"),
    "class_str": Value("string"),
    "region": Value("string"),

    # From light curve files
    "bands_data": {
        "I": band_schema,
        "V": band_schema,
    },

    # From remarks file
    "remarks": Value("string"),

    # From ident file
    "OGLE_IV_id": Value("string"),
    "OGLE_III_id": Value("string"),
    "OGLE_II_id": Value("string"),
    "other_id": Value("string"),
    "ra": Value("string"),
    "dec": Value("string"),
} | {feature: Value("float64") for feature in get_period_feature_columns(3)})


def process_star(cat_idx, catalog, catalog_desc):
    star_info = catalog.iloc[cat_idx].to_dict()
    multiband_lc = read_light_curve(*catalog_desc, star_info['sourceid'])

    if multiband_lc is None:
        return None, star_info['sourceid']

    entry = star_info | {"bands_data": multiband_lc}
    return entry, None


def create_dataset(num_workers):
    catalogs_to_process = [
        # region, parent_type, sub_type
        ("blg", "rrlyr", "RRab"),
        ("blg", "rrlyr", "RRc"),
        ("blg", "rrlyr", "RRd"),
    ]

    # Create empty lists to store dataset entries
    dataset_entries = []

    # List of IDs that don't have light curves
    no_lc_ids = []

    print(f"Processing {len(catalogs_to_process)} catalogs")
    for catalog_to_process in catalogs_to_process:
        start_time = time.time()
        print(f"  Starting catalog-level data for {catalog_to_process[0]} {catalog_to_process[2]}")

        cat = load_catalog(*catalog_to_process)
        cat = merge_remarks(*catalog_to_process, cat)
        cat = merge_ident(*catalog_to_process, cat)
        cat.reset_index(drop=True, inplace=True)

        cat_read_time = time.time()
        print(f"  Finished catalog-level data ({cat_read_time - start_time:.2f}s)")

        for star_ID in tqdm(
            cat['sourceid'],
            desc=f"Processing {catalog_to_process[0]} {catalog_to_process[2]}"
        ):
            star_info = cat[cat['sourceid'] == star_ID].to_dict(orient='records')[0]

            # Get light curve, create entry
            multiband_lc = read_light_curve(*catalog_to_process, star_ID)

            if multiband_lc is None:
                no_lc_ids.append(star_ID)
                continue

            # Create entry following schema
            entry = star_info | {"bands_data": multiband_lc}

            dataset_entries.append(entry)

    # Create HuggingFace dataset
    dataset = Dataset.from_list(dataset_entries, features=schema)

    print(f"Created dataset with {len(dataset_entries)} entries")
    print(f"No lightcurve data found for {len(no_lc_ids)} IDs")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process OGLE data to HuggingFace format')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for parallel processing (default: 4)')

    args = parser.parse_args()
    num_workers = args.num_workers

    dataset = create_dataset(num_workers)
    dataset.save_to_disk(
        "../../../data/ogle4_hf",
        num_proc=num_workers,
        max_shard_size="100MB"
    )
    print("Done writing OGLE data to HF format\n")
