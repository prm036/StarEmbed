import os
import glob
import time
from datetime import datetime

def get_file_time_range(directory, pattern="*.csv"):
    """
    Get the earliest and latest modification times of files in a directory
    
    Parameters:
    -----------
    directory : str
        Directory to scan
    pattern : str
        Glob pattern to match files
        
    Returns:
    --------
    (earliest_time, latest_time, duration) : tuple
        Earliest and latest file modification times and the duration between them
    """
    # Find all matching files
    file_paths = glob.glob(os.path.join(directory, pattern))
    
    if not file_paths:
        print(f"No files matching {pattern} found in {directory}")
        return None, None, None
    
    # Get modification times
    mod_times = [os.path.getmtime(file_path) for file_path in file_paths]
    
    # Find earliest and latest
    earliest_time = min(mod_times)
    latest_time = max(mod_times)
    
    # Calculate duration
    duration_seconds = latest_time - earliest_time
    
    # Convert to human-readable format
    earliest_datetime = datetime.fromtimestamp(earliest_time)
    latest_datetime = datetime.fromtimestamp(latest_time)
    
    # Format duration as hours:minutes:seconds
    hours, remainder = divmod(duration_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_formatted = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    
    return earliest_datetime, latest_datetime, duration_formatted

# Example usage
output_dir = "/projects/p32795/weijian/queried_scope_from_ztf/"
earliest, latest, duration = get_file_time_range(output_dir)

if earliest and latest:
    print(f"Processing started at: {earliest}")
    print(f"Processing finished at: {latest}")
    print(f"Total duration: {duration}")
    
    # Print number of files generated
    file_count = len(glob.glob(os.path.join(output_dir, "*.csv")))
    print(f"Total files generated: {file_count}")