import os
import glob
import pandas as pd
import pyarrow.parquet as pq
import re
from datasets import Dataset, DatasetDict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm import tqdm

def extract_field_id(filename):
    """Extract field_id from filename like field_<field_id>_vs.csv"""
    match = re.search(r'field_(\d+)_vs\.csv', os.path.basename(filename))
    if match:
        return match.group(1)
    return None

def group_csv_files_by_area(csv_dir):
    """Group CSV files by field_id"""
    csv_files = glob.glob(os.path.join(csv_dir, "field_*_vs.csv"))
    field_to_files = {}
    
    for csv_file in csv_files:
        field_id = extract_field_id(csv_file)
        if field_id:
            if field_id not in field_to_files:
                field_to_files[field_id] = []
            field_to_files[field_id].append(csv_file)
    
    return field_to_files

def get_parquet_path(base_dir, field_id):
    """Get corresponding parquet file path for an field_id"""
    # Create the pattern with wildcard
    pattern = os.path.join(base_dir, f"*/field000{field_id}/ztf_000{field_id}_*.parquet")
    
    # Use glob to find all matching files
    matching_files = glob.glob(pattern)
    
    if not matching_files:
        print(f"Warning: No parquet files found matching pattern: {pattern}")
        return None
    
    return matching_files[0]

def process_field_batch(args):
    """Process one field's data - designed to be used with multiprocessing"""
    field_id, csv_files, parquet_dir, batch_size, output_base_dir = args

    print(f"Processing field {field_id} with {len(csv_files)} CSV files")
    print(f"field_id: {field_id}, csv_files: {csv_files}, parquet_dir: {parquet_dir}, output_base_dir: {output_base_dir}")
    
    # Get corresponding parquet file
    parquet_path = get_parquet_path(parquet_dir, field_id)
    
    # Check if parquet file exists
    if not os.path.exists(parquet_path):
        print(f"Warning: Parquet file not found for field {field_id}: {parquet_path}")
        return None
    
    # Read all IDs from CSV files for this area and create a set for quick lookup
    all_ids = set()
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Using the 'id' column from the CSV file as specified
        id_col = [col for col in df.columns if col == 'id' or col.endswith('_id')]
        if id_col:
            all_ids.update(df[id_col[0]].astype(str).tolist())
    
    if not all_ids:
        print(f"No IDs found in CSV files for field {field_id}")
        return None

    print(f"For field {field_id}, found {len(all_ids)} unique IDs")
        
    # Convert to list and sort for deterministic behavior
    all_ids = sorted(list(all_ids))
    
    # Read parquet file metadata to get schema and number of rows
    parquet_file = pq.ParquetFile(parquet_path)
    
    # We know the ID column in parquet is 'objectid'
    id_column = 'objectid'
    
    # Verify the column exists in the parquet schema
    schema = parquet_file.schema
    if id_column not in schema.names:
        print(f"Warning: 'objectid' column not found in parquet file for field {field_id}")
        # Fall back to looking for alternative ID columns if needed
        for col in schema.names:
            if 'id' in col.lower():
                id_column = col
                print(f"Using alternative ID column: {id_column}")
                break
        
    if not id_column:
        print(f"No ID column found in parquet file for field {field_id}")
        return None
    
    # Process parquet in batches to avoid loading everything into memory
    results = []
    num_batches = parquet_file.num_row_groups
    
    # Create a dictionary for fast ID lookups
    id_lookup = {id_val: True for id_val in all_ids}
    
    # Only read the columns we need to minimize memory usage
    # First, read just the ID column to find matching rows
    id_indices = []
    id_to_csv_data = {}
    
    # Read additional metadata from CSV files to join with parquet data
    for csv_file in csv_files:
        csv_df = pd.read_csv(csv_file)
        id_col = [col for col in csv_df.columns if col == '_id'][0]
        # Store additional CSV data keyed by ID for later joining
        for _, row in csv_df.iterrows():
            item_id = str(row[id_col])
            id_to_csv_data[item_id] = {col: row[col] for col in csv_df.columns 
                                      if col != id_col}
    
    # Process each row group (batch) in the parquet file
    for batch_idx in range(num_batches):
        # First, only read the ID column to check for matches
        batch_col_indices = [parquet_file.schema.names.index(id_column)]
        id_batch = parquet_file.read_row_group(batch_idx, columns=[id_column]).to_pandas()
        
        # Convert IDs to string for consistent comparison
        id_batch[id_column] = id_batch[id_column].astype(str)
        
        # Find which rows match our IDs
        mask = id_batch[id_column].isin(all_ids)
        
        # If we have matches, get the full rows for those matches
        if mask.any():
            print(f"Found {mask.sum()} matches in batch {batch_idx}")
            match_indices = id_batch.index[mask].tolist()
            
            # Now read the full rows just for the matching IDs
            # We'll read all columns this time
            full_batch = parquet_file.read_row_group(batch_idx).to_pandas()
            filtered_batch = full_batch.iloc[match_indices]
            
            # Join with the corresponding CSV data
            for idx, row in filtered_batch.iterrows():
                item_id = str(row[id_column])
                if item_id in id_to_csv_data:
                    # Add CSV columns to the result
                    for col, val in id_to_csv_data[item_id].items():
                        filtered_batch.at[idx, f"csv_{col}"] = val
            
            if not filtered_batch.empty:
                results.append(filtered_batch)
                
    
    # Concatenate all results if we found any
    if results:
        print(f"Saving field {field_id} to {output_base_dir}")
        output_path = os.path.join(output_base_dir, f"field_{field_id}.csv")
        field_extracted_df = pd.concat(results, ignore_index=True)
        field_extracted_df.to_csv(output_path, index=False)
        return field_extracted_df
    
    return None

def create_hf_dataset(processed_data):
    """Create a HuggingFace dataset from processed dataframes"""
    # Filter out None results
    valid_data = [df for df in processed_data if df is not None]
    
    if not valid_data:
        print("No data to save")
        return None
    
    # Concatenate all dataframes
    combined_df = pd.concat(valid_data, ignore_index=True)
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_pandas(combined_df)
    
    return dataset

def optimized_query_algorithm(csv_dir, parquet_dir, output_path, num_workers=4, batch_size=10000):
    """Main function to coordinate the optimized query process"""
    print("Grouping CSV files by area...")
    field_to_files = group_csv_files_by_area(csv_dir)
    
    print(f"Found {len(field_to_files)} distinct fields")
    
    # Prepare arguments for parallel processing
    process_args = [
        (field_id, csv_files, parquet_dir, batch_size, output_path) 
        for field_id, csv_files in field_to_files.items()
    ]
    
    # Process areas in parallel
    processed_data = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(process_field_batch, process_args), 
                          total=len(process_args), 
                          desc="Processing fields"):
            processed_data.append(result)
    
    # Create HuggingFace dataset
    print("Creating HuggingFace dataset...")
    # dataset = create_hf_dataset(processed_data)
    dataset = None
    
    if dataset:
        # Save the dataset
        print(f"Saving dataset to {output_path}...")
        dataset.save_to_disk(output_path)
        print(f"Dataset saved successfully with {len(dataset)} records")
        
        # Print dataset statistics
        print("\nDataset Statistics:")
        print(f"Total records: {len(dataset)}")
        print(f"Columns: {dataset.column_names}")
        print(f"Sample record: {dataset[0] if len(dataset) > 0 else None}")
    
    return dataset

if __name__ == "__main__":
    # Example usage
    csv_directory = "/projects/p32795/weijian/tmps/03_02_xgb_095_vnv"
    parquet_directory = "/projects/b1094/stroh/software/catalogs/ztf/lc_dr23/"
    output_dataset_path = "/projects/p32795/weijian/queried_scope_from_ztf/"
    
    # Adjust these parameters based on your system capabilities
    num_workers = 8  # Number of parallel processes
    batch_size = 10000  # Adjust based on available memory
    
    optimized_query_algorithm(csv_directory, parquet_directory, output_dataset_path, 
                             num_workers=num_workers, batch_size=batch_size)