from datasets import Dataset, Features, Value, Sequence
from tqdm import tqdm
import datasets

from utils.data.preprocess.read_OGLE import (
    load_catalog, merge_remarks, merge_ident, read_light_curve
)

# Standardized StarEmbed schema with some columns unique to Catalina
band_schema = Features({
    "mjd": Sequence(feature=Value("float64")),
    "target": Sequence(feature=Value("float64")),
    "past_feat_dynamic_real": Sequence(feature=Value("float64")),
    "feat_dynamic_real": Sequence(feature=Value("float64")),
    "length": Value("int64"),
})

schema = Features({
    "sourceid": Value("string"),
    "numerical_id": Value("string"),
    "bands_data": {
        "I": band_schema,
        "V": band_schema,
    },
    "avg_mag_V": Value("float64"),
    "period": Value("float64"),
    "class_str": Value("string"),
    "class_int": Value("int64"),
    "ra": Value("float64"),
    "dec": Value("float64")
})


if __name__ == "__main__":

    catalogs_to_process = [
        # region, parent_type, sub_type
        ("blg", "cep", "cep1O"),
    ]

    # Create empty lists to store dataset entries
    dataset_entries = []

    # List of IDs that don't have light curves
    no_lc_ids = []

    for catalog_to_process in catalogs_to_process:
        cat = load_catalog(*catalog_to_process)
        cat = merge_remarks(*catalog_to_process, cat)
        cat = merge_ident(*catalog_to_process, cat)

        for star_ID in cat['ID']:
            # Get light curve, create entry
            multiband_lc = read_light_curve(*catalog_to_process, star_ID)

            if multiband_lc is None:
                # print(f"No light curve found for {star_ID}")
                no_lc_ids.append(star_ID)
                continue

            # Create entry following schema
            entry = {
                "sourceid": star_ID,
                "bands_data": multiband_lc,
            }

    # Read in lightcurves
    # Set up schema
    # Assosciate light curves with catalogs (by ID) and create HF entries
    # Create HF dataset
    # Write HF dataset to disk

    # When to concatenate catalogs?
