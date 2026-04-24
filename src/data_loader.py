"""
data_loader.py - Handles reading input data (CSV/Excel)
"""
import pandas as pd
import numpy as np
from pathlib import Path

def load_input_data(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data based on the configuration.
    
    Returns:
        X (np.ndarray): 1D array of X coordinates
        Y (np.ndarray): 1D array of Y coordinates
        Z (np.ndarray): 1D array of values to interpolate
    """
    input_cfg = config.get('input', {})
    filepath = Path(input_cfg.get('filepath', 'input_data.csv'))
    fmt = input_cfg.get('format', 'csv').lower()
    
    cols = input_cfg.get('columns', {})
    col_x = cols.get('x', 'X')
    col_y = cols.get('y', 'Y')
    col_val = cols.get('value', 'Value')
    
    if not filepath.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
        
    if fmt == 'excel':
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)
        
    # Check columns
    missing = [c for c in [col_x, col_y, col_val] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in input file: {missing}")
        
    # Drop NaNs
    df = df.dropna(subset=[col_x, col_y, col_val])
    
    return df[col_x].values, df[col_y].values, df[col_val].values
