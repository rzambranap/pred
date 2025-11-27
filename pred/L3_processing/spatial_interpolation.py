import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from PREDICT.utils.geospatial import from_df_get_bounding_box

def inverse_distance_weighting(df_data, df_metadata, resolution):
    # Get bounding box from metadata
    bbox = from_df_get_bounding_box(df_metadata)
    min_lat, max_lat, min_lon, max_lon = bbox['min_lat'], bbox['max_lat'], bbox['min_lon'], bbox['max_lon']
    
    # Create grid
    lat_grid = np.arange(min_lat, max_lat, resolution)
    lon_grid = np.arange(min_lon, max_lon, resolution)
    lon, lat = np.meshgrid(lon_grid, lat_grid)
    
    # Prepare output xarray
    if isinstance(df_data, pd.DataFrame):
        times = df_data.index
        output = xr.DataArray(np.zeros((len(times), len(lat_grid), len(lon_grid))),
                              coords=[times, lat_grid, lon_grid],
                              dims=['time', 'lat', 'lon'])
    else:
        output = xr.DataArray(np.zeros((len(lat_grid), len(lon_grid))),
                              coords=[lat_grid, lon_grid],
                              dims=['lat', 'lon'])
    
    # Prepare coordinates and values
    coords = df_metadata[['lat', 'lon']].values
    if isinstance(df_data, pd.DataFrame):
        for time in df_data.index:
            values = df_data.loc[time].values
            output.loc[time] = idw_interpolation(coords, values, lat, lon)
    else:
        values = df_data.values
        output[:] = idw_interpolation(coords, values, lat, lon)
    
    return output

def idw_interpolation(coords, values, lat, lon, power=2):
    tree = cKDTree(coords)
    grid_shape = lat.shape
    lat_lon_pairs = np.vstack([lat.ravel(), lon.ravel()]).T
    distances, indices = tree.query(lat_lon_pairs, k=len(coords))
    weights = 1 / distances**power
    weights /= weights.sum(axis=1)[:, None]
    interpolated_values = np.sum(weights * values[indices], axis=1)
    return interpolated_values.reshape(grid_shape)