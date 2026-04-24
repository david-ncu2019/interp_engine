"""
generate_test_data.py - Create synthetic spatial datasets for engine testing.
Uses geostatspy for stochastic simulation (SGSIM) to create challenging 
geostatistical fields instead of simple deterministic functions.

For each scenario, produces:
  - {name}.csv              : sparse sample points (noisy or clustered)
  - {name}_ground_truth.csv : dense regular grid of the noise-free stochastic field
"""
import numpy as np
import pandas as pd
from pathlib import Path
import geostatspy.geostats as geostats

def generate_sgsim_field(nx, ny, xsiz, ysiz, vario_in, seed=42):
    """
    Generate an unconditional Sequential Gaussian Simulation field.
    Returns a (ny, nx) array.
    """
    # geostatspy.geostats.sgsim requires ALL keys even if nst=1
    vario = {
        'nug': 0.0, 'nst': 1,
        'it1': 1, 'cc1': 1.0, 'azi1': 0, 'hmaj1': 100, 'hmin1': 100,
        'it2': 1, 'cc2': 0.0, 'azi2': 0, 'hmaj2': 0, 'hmin2': 0
    }
    vario.update(vario_in)

    # geostatspy.geostats.sgsim often fails with empty data (unconditional).
    # We provide a single dummy point far away to satisfy the search tree logic.
    df_dummy = pd.DataFrame({'X': [-9999.0], 'Y': [-9999.0], 'Value': [0.0]})
    
    # xmn, ymn are the centers of the first cell
    xmn = xsiz / 2.0
    ymn = ysiz / 2.0
    
    # Unconditional SGSIM
    # itrans=0: no normal score transform (direct simulation)
    # ktype=0: simple kriging
    sim = geostats.sgsim(
        df_dummy, 'X', 'Y', 'Value', wcol=-1, scol=-1, 
        tmin=-999, tmax=999, itrans=0, ismooth=0, dftrans=None, 
        tcol=-1, twtcol=-1, zmin=-999, zmax=999, ltail=1, ltpar=0, utail=1, utpar=0, 
        nsim=1, nx=nx, xmn=xmn, xsiz=xsiz, ny=ny, ymn=ymn, ysiz=ysiz, 
        seed=seed, ndmin=0, ndmax=1, nodmax=10, mults=0, nmult=2, noct=0, 
        ktype=0, colocorr=0, sec_map=None, vario=vario
    )
    # geostats.sgsim returns (nsim, ny, nx)
    return sim[0]

def sample_field(grid, xsiz, ysiz, n_points, method='random', noise=0.05, seed=42):
    """
    Sample a grid to create sparse data points.
    Methods: 'random', 'clustered'
    """
    np.random.seed(seed)
    ny, nx = grid.shape
    
    if method == 'random':
        # Randomly pick indices
        indices = np.random.choice(nx * ny, n_points, replace=False)
        iy, ix = np.unravel_index(indices, (ny, nx))
    elif method == 'clustered':
        # Concentrate samples in two clusters
        n_cluster = n_points // 2
        
        # Cluster 1 (Bottom Left)
        ix1 = np.random.normal(nx*0.2, nx*0.1, n_cluster).astype(int)
        iy1 = np.random.normal(ny*0.2, ny*0.1, n_cluster).astype(int)
        
        # Cluster 2 (Top Right)
        ix2 = np.random.normal(nx*0.8, nx*0.1, n_cluster).astype(int)
        iy2 = np.random.normal(ny*0.8, ny*0.1, n_cluster).astype(int)
        
        ix = np.concatenate([ix1, ix2])
        iy = np.concatenate([iy1, iy2])
        
        # Clip to grid bounds
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)
    
    # Convert grid indices to coordinates
    x_coords = ix * xsiz + (xsiz / 2.0)
    y_coords = iy * ysiz + (ysiz / 2.0)
    values = grid[iy, ix] + np.random.normal(0, noise, len(ix))
    
    return pd.DataFrame({"X": x_coords, "Y": y_coords, "Value": values})

def main():
    out_dir = Path("test_data")
    out_dir.mkdir(exist_ok=True)
    
    # Grid Setup (1000m x 1000m at 10m resolution)
    nx, ny = 100, 100
    xsiz, ysiz = 10.0, 10.0
    
    # it1: 1=spherical, 2=exponential, 3=gaussian
    scenarios = [
        {
            "name": "S5_SGS_Extreme_Aniso",
            "desc": "High anisotropy ratio (10:1) at 30 degrees",
            "vario": {'nug': 0.05, 'nst': 1, 'it1': 2, 'cc1': 0.95, 'azi1': 30, 'hmaj1': 600, 'hmin1': 60},
            "n_points": 300,
            "sample_method": "random",
            "noise": 0.05
        },
        {
            "name": "S6_SGS_Nested",
            "desc": "Short-range isotropic + Long-range anisotropic",
            "vario": {
                'nug': 0.0, 'nst': 2, 
                'it1': 1, 'cc1': 0.4, 'azi1': 0, 'hmaj1': 100, 'hmin1': 100, # Short isotropic
                'it2': 2, 'cc2': 0.6, 'azi2': 120, 'hmaj2': 800, 'hmin2': 200 # Long anisotropic
            },
            "n_points": 400,
            "sample_method": "random",
            "noise": 0.02
        },
        {
            "name": "S7_SGS_Clustered",
            "desc": "Samples concentrated in clusters, leaving gaps",
            "vario": {'nug': 0.1, 'nst': 1, 'it1': 3, 'cc1': 0.9, 'azi1': 45, 'hmaj1': 400, 'hmin1': 200},
            "n_points": 200,
            "sample_method": "clustered",
            "noise": 0.05
        },
        {
            "name": "S8_SGS_HighNugget",
            "desc": "Strong white noise component (high nugget)",
            "vario": {'nug': 0.6, 'nst': 1, 'it1': 2, 'cc1': 0.4, 'azi1': 0, 'hmaj1': 300, 'hmin1': 300},
            "n_points": 600,
            "sample_method": "random",
            "noise": 0.2
        }
    ]

    print("="*60)
    print("  Generating Geostatistical Test Datasets (SGSIM)")
    print("="*60)

    for s in scenarios:
        print(f"\nScenario: {s['name']}")
        print(f"  Description: {s['desc']}")
        
        # 1. Generate Ground Truth Grid
        grid = generate_sgsim_field(nx, ny, xsiz, ysiz, s['vario'], seed=123)
        
        # Optional: Add a large-scale trend to S5 or similar if desired
        if s['name'] == "S5_SGS_Extreme_Aniso":
            # Add a gentle slope
            X_grid, Y_grid = np.meshgrid(np.arange(nx)*xsiz, np.arange(ny)*ysiz)
            grid += (X_grid / 1000.0) * 2.0  # +2.0 increase across X
        
        # Save Ground Truth
        x_coords = np.arange(nx) * xsiz + (xsiz / 2.0)
        y_coords = np.arange(ny) * ysiz + (ysiz / 2.0)
        XX, YY = np.meshgrid(x_coords, y_coords)
        df_truth = pd.DataFrame({
            "X": XX.ravel(),
            "Y": YY.ravel(),
            "Value": grid.ravel()
        })
        f_truth = out_dir / f"{s['name']}_ground_truth.csv"
        df_truth.to_csv(f_truth, index=False)
        
        # 2. Generate Sparse Samples
        df_samples = sample_field(
            grid, xsiz, ysiz, 
            n_points=s['n_points'], 
            method=s['sample_method'], 
            noise=s['noise'],
            seed=456
        )
        f_samples = out_dir / f"{s['name']}.csv"
        df_samples.to_csv(f_samples, index=False)

        print(f"  -> Samples: {len(df_samples)} saved to {f_samples.name}")
        print(f"  -> Truth:   {len(df_truth)} saved to {f_truth.name}")

    print("\n" + "="*60)
    print("Done. All geostatistical scenarios generated.")
    print("="*60)

if __name__ == "__main__":
    main()
