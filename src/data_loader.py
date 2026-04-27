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
    fmt = input_cfg.get('format', '').lower()
    if not fmt:
        fmt = filepath.suffix.lower().lstrip('.')
        if fmt == 'shp':
            fmt = 'shapefile'
    
    cols = input_cfg.get('columns', {})
    col_x = cols.get('x') or 'X'
    col_y = cols.get('y') or 'Y'
    col_val = cols.get('value') or 'Value'
    
    if not filepath.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
        
    if fmt == 'excel' or fmt in ['xls', 'xlsx']:
        df = pd.read_excel(filepath)
    elif fmt == 'shapefile' or fmt == 'shp':
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "Reading ESRI shapefiles requires the 'geopandas' library. "
                "Please install it (e.g., pip install geopandas) or convert your shapefile to a CSV."
            )
        df = gpd.read_file(filepath)
        if hasattr(df, 'geometry'):
            if col_x not in df.columns:
                df[col_x] = df.geometry.x
            if col_y not in df.columns:
                df[col_y] = df.geometry.y
    else:
        df = pd.read_csv(filepath)
        
    # Check columns
    missing = [c for c in [col_x, col_y, col_val] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in input file: {missing}")
        
    # Drop NaNs
    df = df.dropna(subset=[col_x, col_y, col_val])
    
    return df[col_x].values.astype(np.float64), df[col_y].values.astype(np.float64), df[col_val].values.astype(np.float64)


def load_custom_prediction_points(filepath: str | Path, col_x: str, col_y: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load custom output prediction points.
    
    Supports .csv, .xlsx, and optionally .shp (if geopandas is installed).
    Assumes points are in the same CRS as the input data.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Custom prediction points file not found: {filepath}")
        
    ext = filepath.suffix.lower()
    
    if ext == '.shp':
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "Reading ESRI shapefiles requires the 'geopandas' library. "
                "Please install it (e.g., pip install geopandas) or convert your shapefile to a CSV."
            )
        df = gpd.read_file(filepath)
        if hasattr(df, 'geometry'):
            if col_x not in df.columns:
                df[col_x] = df.geometry.x
            if col_y not in df.columns:
                df[col_y] = df.geometry.y
                
        missing = [c for c in [col_x, col_y] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected columns in shapefile: {missing}")
        df = df.dropna(subset=[col_x, col_y])
        return df[col_x].values.astype(np.float64), df[col_y].values.astype(np.float64), df
        
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)
        
    missing = [c for c in [col_x, col_y] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in custom prediction file: {missing}")
        
    df = df.dropna(subset=[col_x, col_y])
    return df[col_x].values.astype(np.float64), df[col_y].values.astype(np.float64), df

