"""
exporter.py - Handles exporting the predicted grid to NetCDF
"""
import xarray as xr
import numpy as np
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
    """
    Export the interpolation results to a CF-compliant NetCDF file for Paraview.

    Args:
        X_grid      : 2D array of X coordinates
        Y_grid      : 2D array of Y coordinates
        mean_pred   : 2D array of predicted means (NaNs outside hull)
        std_pred    : 2D array of predicted std devs (NaNs outside hull)
        output_dir  : Directory to write the NetCDF file into
        engine_name : Name of the engine used (e.g. "gp" or "kriging")
        z_dim_name  : Name for the Z dimension attribute (Paraview label)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"predicted_{engine_name}.nc"

    # Extract 1D coordinate vectors from the meshgrid
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
    print(f"  NetCDF saved to: {out_path}")
