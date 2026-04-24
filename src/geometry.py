"""
geometry.py - Handles convex hull computation and grid generation
"""
import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path


def generate_prediction_grid(X: np.ndarray, Y: np.ndarray, config: dict):
    """
    Generate a 2D grid bounded by the (buffered) convex hull of the input points.

    Returns
    -------
    X_grid : np.ndarray  – 2D meshgrid X coordinates
    Y_grid : np.ndarray  – 2D meshgrid Y coordinates
    mask   : np.ndarray  – 1D boolean mask (True = inside buffered hull)
    grid_shape : tuple   – (ny, nx)
    hull_vertices : np.ndarray – (M, 2) vertices of the *buffered* convex hull
                                 (closed polygon, first == last)
    """
    geom_cfg = config.get('geometry', {})
    resolution = geom_cfg.get('resolution_m', 50.0)
    buffer_pct = geom_cfg.get('convex_hull_buffer_percent', 5.0) / 100.0

    # 1. Bounding box with buffer
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()

    x_range = x_max - x_min
    y_range = y_max - y_min

    x_min -= x_range * buffer_pct
    x_max += x_range * buffer_pct
    y_min -= y_range * buffer_pct
    y_max += y_range * buffer_pct

    # 2. 1D grid vectors
    nx = int(np.ceil((x_max - x_min) / resolution)) + 1
    ny = int(np.ceil((y_max - y_min) / resolution)) + 1

    x_vec = x_min + np.arange(nx) * resolution
    y_vec = y_min + np.arange(ny) * resolution

    # 3. 2D meshgrid
    X_grid, Y_grid = np.meshgrid(x_vec, y_vec)
    grid_shape = X_grid.shape

    # 4. Convex Hull
    points = np.column_stack((X, Y))
    hull = ConvexHull(points)
    hull_verts = points[hull.vertices]

    # Expand hull outward from centroid by buffer_pct
    centroid = np.mean(hull_verts, axis=0)
    buffered_verts = centroid + (hull_verts - centroid) * (1.0 + buffer_pct)

    # Close the polygon (first vertex == last vertex)
    buffered_verts_closed = np.vstack([buffered_verts, buffered_verts[0]])

    # 5. Mask grid points inside buffered hull
    hull_path = Path(buffered_verts)
    grid_points = np.column_stack((X_grid.flatten(), Y_grid.flatten()))
    mask = hull_path.contains_points(grid_points)

    return X_grid, Y_grid, mask, grid_shape, buffered_verts_closed
