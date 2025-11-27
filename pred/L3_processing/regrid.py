import numpy as np
from ..utils.geospatial import get_bbox_from_xarray
# import decimal
import xarray as xr
from scipy.interpolate import griddata


def regrid_dataarray(da, resolution=None, lat_grid=None, lon_grid=None):
    if len(da.shape) > 2:
        raise TypeError("Only dataarrays of 2 dims for this function da's {len(da.shape)} is a no-go")
    if ((lat_grid is None) or (lon_grid is None)):
        if resolution is None:
            resolution = get_naive_resolution_from_radar(da)
        lat_grid, lon_grid = calc_lat_lon_grids_from_radar(da, resolution)
    lons, lats = da.lon.values.flatten(), da.lat.values.flatten()
    values = da.fillna(0).values.flatten()  # in order to not get a nan-filled da
    ppoints = np.array([lons, lats]).T
    '''
    decimals = -(decimal.Decimal(str(resolution))).as_tuple().exponent

    lon_min, lon_max = np.round((lons.min(), lons.max()), decimals=decimals)
    lon_grid = np.arange(lon_min, lon_max+res, res)

    lat_min, lat_max = np.round((lats.min(), lats.max()), decimals=decimals)
    lat_grid = np.arange(lat_min, lat_max+res, res)
    '''
    grid_lon, grid_lat = np.meshgrid(lon_grid, lat_grid)

    grid_z = griddata(ppoints, values, (grid_lon, grid_lat), method='linear')
    nda = xr.DataArray(grid_z, coords=[("lat", lat_grid), ("lon", lon_grid)])
    # nda = xr.DataArray(grid_z, dims=['lon', 'lat'], coords={'lon': grid_lon, 'lat': grid_lat})
    return nda


def get_naive_resolution_from_radar(ds):
    azi1 = ds.azimuth[0].data
    azi2 = ds.azimuth[1].data
    # rmax = ds.range.max()

    p1 = ds.sel({'azimuth': azi1})
    p2 = ds.sel({'azimuth': azi2})

    diff_coords = np.abs(p1.lon - p2.lon)
    res = diff_coords.median().data
    return res


def calc_lat_lon_grids_from_radar(ds, resolution):
    bbox = get_bbox_from_xarray(ds)
    lon_grid = np.arange(bbox.min_lon, bbox.max_lon, resolution)
    lat_grid = np.arange(bbox.min_lat, bbox.max_lat, resolution)
    return lat_grid, lon_grid
