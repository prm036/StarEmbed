from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import os
import glob
import pandas as pd
import pyarrow.parquet as pq

# Your star coordinates
ra_deg = 30.49325
dec_deg = 6.925177

# Create a SkyCoord object
coords = SkyCoord(ra=ra_deg*u.degree, dec=dec_deg*u.degree)

# Display coordinates in HMS/DMS format
ra_hms = coords.ra.to_string(unit=u.hourangle, sep=('h', 'm', 's'), precision=2, pad=True)
dec_dms = coords.dec.to_string(unit=u.degree, sep=('d', 'm', 's'), precision=2, pad=True)

print(f"Searching for source at:")
print(f"RA (HMS): {ra_hms}")
print(f"Dec (DMS): {dec_dms}")

# Path to your local ZTF catalog files
ztf_local_path = "/projects/b1094/stroh/software/catalogs/ztf/lc_dr23/0/field000451"  # Update this to your actual path

def query_local_ztf_catalog(coords, radius_arcsec=2.0):
    """
    Query local ZTF catalog using coordinates from parquet files
    
    Parameters:
    -----------
    coords : SkyCoord
        Coordinates to search
    radius_arcsec : float
        Search radius in arcseconds
    
    Returns:
    --------
    matches : astropy.table.Table
        Table containing matching sources
    """
    # Find all parquet files
    catalog_files = glob.glob(os.path.join(ztf_local_path, "**/*.parquet"), recursive=True)
    
    if not catalog_files:
        print(f"No parquet files found in {ztf_local_path}")
        return None
    
    all_matches = []
    
    for catalog_file in catalog_files:
        print(f"Searching in {os.path.basename(catalog_file)}...")
        
        # Load parquet file
        try:
            # Read parquet file into pandas DataFrame
            df = pd.read_parquet(catalog_file)
            
            # Convert to astropy Table
            catalog = Table.from_pandas(df)

            # print length of catalog
            print(f"Length of catalog: {len(catalog)}")
            # print(f"Columns in catalog: {catalog.colnames}")
            # print(f"First row of catalog: {catalog[0]}")
        except Exception as e:
            print(f"Error reading {catalog_file}: {e}")
            continue
        
        # Create SkyCoord objects for the catalog entries
        try:
            # Column names might vary - adjust as needed
            catalog_coords = SkyCoord(ra=catalog['objra']*u.degree, dec=catalog['objdec']*u.degree)
        except KeyError:
            print(f"Could not find RA/Dec columns in {catalog_file}")
            continue
                
        
        # # Find matches within the specified radius
        # idx, sep2d, _ = coords.match_to_catalog_sky(catalog_coords)
        
        # # Filter matches by separation
        # mask = sep2d < (radius_arcsec * u.arcsec)

        # Calculate separations directly instead of using match_to_catalog_sky
        separations = coords.separation(catalog_coords)
        
        # Find entries within the search radius
        mask = separations < (radius_arcsec * u.arcsec)
        
        # if np.any(mask):
        #     # print shape of mask
        #     print(f"Shape of mask: {mask.shape}")
        #     # Add file information to matches
        #     matches = catalog[idx[mask]]
        #     matches['catalog_file'] = [os.path.basename(catalog_file)] * len(matches)
        #     matches['separation_arcsec'] = sep2d[mask].to(u.arcsec).value
        #     all_matches.append(matches)

        if np.any(mask):
            # Get matches using the mask
            print(f"Shape of mask: {mask.shape}")
            matches = catalog[mask]
            matches['catalog_file'] = [os.path.basename(catalog_file)] * len(matches)
            matches['separation_arcsec'] = separations[mask].to(u.arcsec).value
            all_matches.append(matches)
    
    if all_matches:
        # Combine all matches into a single table
        try:
            result = Table(np.concatenate(all_matches))
            return result
        except ValueError:
            # If tables have different columns, concatenate differently
            combined = Table()
            for i, table in enumerate(all_matches):
                for row in table:
                    combined.add_row(row)
            return combined
    else:
        print(f"No sources found within {radius_arcsec} arcseconds")
        return None

# Perform the search
matches = query_local_ztf_catalog(coords)



# Display results
if matches is not None and len(matches) > 0:
    print(f"\nFound {len(matches)} matching source(s):")

    # Sort by separation to find the closest match
    matches.sort('separation_arcsec')

    # Take only the closest match
    match = matches[0]

    # Format coordinates for display
    match_coords = SkyCoord(ra=match['objra']*u.degree, dec=match['objdec']*u.degree)
    match_ra_hms = match_coords.ra.to_string(unit=u.hourangle, sep=('h', 'm', 's'), precision=2, pad=True)
    match_dec_dms = match_coords.dec.to_string(unit=u.degree, sep=('d', 'm', 's'), precision=2, pad=True)
    
    # Print basic info - adjust column names as needed for your data
    print(f"Source ID: {match['objectid']}")
    print(f"Field ID: {match['fieldid']}")
    print(f"RA: {match['objra']:.6f}° ({match_ra_hms})")
    print(f"Dec: {match['objdec']:.6f}° ({match_dec_dms})")
    print(f"Separation: {match['separation_arcsec']:.3f} arcsec")
    
    print(f"Found in: {match['catalog_file']}")
else:
    print("No matching sources found")