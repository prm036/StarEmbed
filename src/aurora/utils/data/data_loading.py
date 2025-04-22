import pickle
import pandas as pd
CSDR1_RAW = f"/projects/p32015/git/moirai_supsup/data_download/cache/all_objects_47054_None.pkl"
CSDR1_META = f"/projects/p32015/git/moirai_supsup/data_download/CSDR1_varstars.txt"

def load_csdr1_raw(path):
    if not path:
        path = CSDR1_RAW
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_csdr1_meta(csv_path):
    """
    Load periods from the variable stars catalog
    """
    if not csv_path:
        csv_path = CSDR1_META
    column_names = ['ID', 'RAh', 'RAm', 'RAs', 'Decsign', 'DEm', 'DEs', 
                   'magV', 'P', "Amp", "class", 'flag']
    
    varstars = pd.read_csv(
        csv_path,
        header=33,
        sep=r'\s+',
        names=column_names
    )
    
    return varstars
    
def load_periods(csv_path):
    """
    Load periods from the variable stars catalog
    """
    if not csv_path:
        csv_path = CSDR1_META
    column_names = ['ID', 'RAh', 'RAm', 'RAs', 'Decsign', 'DEm', 'DEs', 
                   'magV', 'P', "Amp", "class", 'flag']
    
    varstars = pd.read_csv(
        csv_path,
        header=33,
        sep=r'\s+',
        names=column_names
    )
    
    # Extract periods as a list
    periods = varstars['P'].values
    return periods