"""
exporter.py - Handles exporting the predicted grid to NetCDF, GeoTIFF, or CSV
"""
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path


def export_to_netcdf(
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    output_dir: Path,
    engine_name: str,
    z_dim_name: str = "Depth",
):
    """Export interpolation results to a CF-compliant NetCDF file for Paraview."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"predicted_{engine_name}.nc"

    x_coords = X_grid[0, :]
    y_coords = Y_grid[:, 0]

    ds = xr.Dataset(
        {
            "predicted_mean": xr.DataArray(
                mean_pred.astype(np.float32),
                dims=["Y", "X"],
                attrs={"long_name": "Predicted Mean Value"},
            ),
            "predicted_std": xr.DataArray(
                std_pred.astype(np.float32),
                dims=["Y", "X"],
                attrs={"long_name": "Prediction Standard Deviation (Uncertainty)"},
            ),
        },
        coords={
            "X": xr.DataArray(x_coords, dims=["X"], attrs={"units": "m", "axis": "X"}),
            "Y": xr.DataArray(y_coords, dims=["Y"], attrs={"units": "m", "axis": "Y"}),
        },
        attrs={
            "description": f"Spatial Interpolation Results using {engine_name}",
            "created_by": "20260423_Interp_Engine",
            "Conventions": "CF-1.6",
        },
    )

    ds.to_netcdf(out_path)
    print(f"       ✓ predicted_{engine_name}.nc")


def export_to_geotiff(
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    output_dir: Path,
    engine_name: str,
):
    """
    Export predicted mean and std as two single-band GeoTIFF files.

    Requires rasterio. CRS is not set (data coordinates are assumed to be in the
    same projection as the input data). To embed a CRS, set `crs` in rasterio.open()
    to e.g. rasterio.crs.CRS.from_epsg(32650).
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        raise ImportError(
            "rasterio is required for GeoTIFF export. "
            "Install it with: conda install -c conda-forge rasterio"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_min, x_max = float(X_grid[0, 0]), float(X_grid[0, -1])
    y_min, y_max = float(Y_grid[-1, 0]), float(Y_grid[0, 0])
    nrows, ncols = mean_pred.shape

    # Pixel-edge transform (origin at top-left corner)
    transform = from_bounds(x_min, y_min, x_max, y_max, ncols, nrows)

    tif_profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": ncols,
        "height": nrows,
        "count": 1,
        "crs": None,          # set to e.g. rasterio.crs.CRS.from_epsg(32650) if known
        "transform": transform,
        "nodata": float("nan"),
    }

    for arr, suffix in [(mean_pred, "mean"), (std_pred, "std")]:
        out_path = output_dir / f"predicted_{engine_name}_{suffix}.tif"
        with rasterio.open(out_path, "w", **tif_profile) as dst:
            # rasterio expects row 0 at the top; flip Y axis to match
            dst.write(np.flipud(arr).astype(np.float32), 1)
        print(f"       ✓ predicted_{engine_name}_{suffix}.tif")


def export_grid_to_csv(
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    output_dir: Path,
    engine_name: str,
):
    """Export the prediction grid as a flat CSV (one row per valid grid cell)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"predicted_{engine_name}.csv"

    df = pd.DataFrame({
        "X": X_grid.flatten(),
        "Y": Y_grid.flatten(),
        "predicted_mean": mean_pred.flatten(),
        "predicted_std": std_pred.flatten(),
    })
    df = df.dropna(subset=["predicted_mean"])   # drop cells outside convex hull
    df.to_csv(out_path, index=False)
    print(f"       ✓ predicted_{engine_name}.csv")


_GRID_FORMAT_HANDLERS = {
    "nc":  export_to_netcdf,
    "tif": export_to_geotiff,
    "csv": export_grid_to_csv,
}

_VALID_GRID_FORMATS  = set(_GRID_FORMAT_HANDLERS.keys())
_VALID_POINT_FORMATS = {"csv", "xz"}


def export_grid(
    formats: list,
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    output_dir: Path,
    engine_name: str,
    z_dim_name: str = "Depth",
):
    """
    Export the prediction grid in all requested formats.

    Args:
        formats     : List of format strings, e.g. ["nc", "tif", "csv"]
        X_grid      : 2D meshgrid of X coordinates
        Y_grid      : 2D meshgrid of Y coordinates
        mean_pred   : 2D array of predicted means (NaNs outside hull)
        std_pred    : 2D array of predicted std devs (NaNs outside hull)
        output_dir  : Directory to write files into
        engine_name : Engine label used in filenames
        z_dim_name  : Z dimension label (NetCDF / Paraview only)
    """
    unknown = set(formats) - _VALID_GRID_FORMATS
    if unknown:
        raise ValueError(
            f"Unknown grid export format(s): {sorted(unknown)}. "
            f"Valid options are: {sorted(_VALID_GRID_FORMATS)}"
        )

    for fmt in formats:
        handler = _GRID_FORMAT_HANDLERS[fmt]
        if fmt == "nc":
            handler(X_grid, Y_grid, mean_pred, std_pred, output_dir, engine_name, z_dim_name)
        else:
            handler(X_grid, Y_grid, mean_pred, std_pred, output_dir, engine_name)
