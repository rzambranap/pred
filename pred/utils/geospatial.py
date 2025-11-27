import numpy as np
import xarray as xr
from global_land_mask import globe


class BBox:
    def __init__(self, min_lat, min_lon, max_lat, max_lon):
        self.min_lat = min_lat
        self.min_lon = min_lon
        self.max_lat = max_lat
        self.max_lon = max_lon

    def to_shape_compliant(self):
        shp = [self.min_lon, self.min_lat, self.max_lon, self.max_lat]
        return shp


def get_data_in_bbox(data, bbox):
    """
    Copied for our purposes from weatherforce spacetime module
    Theirs is a lot more complicated, we only need these lines

    Extracts data inside a BBox from xarray Dataset or DataArray

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Dataset or DataArray from which we want the data.
    bbox : BBox
        Bounding box of interest

    Returns
    -------
    data : xarray.Dataset or xarray.DataArray
        Extracted data with bbox

    See Also
    --------
    .common.BBox : class for bounding boxes.

    Examples
    --------
    test
    """
    data = data.sel({'lon': slice(bbox.min_lon, bbox.max_lon)})
    if data['lat'][0] < data['lat'][-1]:
        # Latitude Ordered Negative to Positive
        return data.sel({'lat': slice(bbox.min_lat, bbox.max_lat)})
    else:
        # Latitude Ordered Positive to Negative
        return data.sel({'lat': slice(bbox.max_lat, bbox.min_lat)})
    return None


def get_bbox_from_xarray(array, lon_name='lon', lat_name='lat'):
    """
    Creates a BBox bounding box from xarray Dataset or DataArray

    Parameters
    ----------
    array : xarray.Dataset or xarray.DataArray
        Dataset or DataArray for which we want the bounding box.
    lon_name : str
        Name of longitude dimension (by default 'lon')
    lat_name : str
        Name of latitude dimension (by default 'lat')

    Returns
    -------
    box : BBox
        Class to descript a bounding box.

    See Also
    --------
    BBox : class for bounding boxes.

    Examples
    --------
    test
    """
    lons = array[lon_name]
    lats = array[lat_name]
    # The function only needs to grab the maxs and mins
    min_lat = lats.min().data
    max_lat = lats.max().data
    min_lon = lons.min().data
    max_lon = lons.max().data

    # And create a bounding box as follows
    box = BBox(min_lat=min_lat, min_lon=min_lon,
               max_lat=max_lat, max_lon=max_lon)
    return box


def add_land_mask(data):
    """
    Adds a land mask to a given xarray dataset or data array.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        The data to add a land mask to.

    Returns
    -------
    xarray.Dataset
        The data with the land mask added as a new variable called 'mask'.

    Raises
    ------
    TypeError
        If `data` is not an xarray DataArray or Dataset.

    Notes
    -----
    This function uses the `globe.is_land` function to create a global land mask.

    """
    if isinstance(data, xr.DataArray):
        data = data.to_dataset()
    lat = data.lat.to_numpy()
    lon = data.lon.to_numpy()
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    globe_land_mask = globe.is_land(lat_grid, lon_grid)
    data = data.assign({'mask': (('lat', 'lon'), globe_land_mask)})
    return data


def divide_dataset_land_sea(dataset):
    """
    Splits a dataset into two datasets based on land and sea masks.

    Parameters:
    -----------
    dataset : xarray.Dataset or xarray.DataArray
        The input dataset to be divided into land and sea components.
        It must have "lat" and "lon" dimensions.

    Returns:
    --------
    ds_land : xarray.Dataset or xarray.DataArray
        The dataset containing only the values of the input dataset that are over land.
    ds_sea : xarray.Dataset or xarray.DataArray
        The dataset containing only the values of the input dataset that are over sea.
    """
    ds_mask = add_land_mask(dataset)
    ds_land = dataset.where(ds_mask.mask)
    ds_sea = dataset.where(~ds_mask.mask)
    return ds_land, ds_sea


def subdivide(ds, nrows, ncols):
    """
    Subdivides a given xarray Dataset `ds` into a grid of smaller divisions.

    Args:
        ds (xarray.Dataset): The input xarray Dataset to be subdivided.
        nrows (int): The number of rows in the grid.
        ncols (int): The number of columns in the grid.

    Returns:
        list: A list of xarray Datasets representing the subdivisions.
    """
    lons = ds.lon.to_numpy()
    lats = ds.lat.to_numpy()
    nlons = lons.shape[0]
    nlats = lats.shape[0]
    col_sze = nlons//ncols
    row_sze = nlats//nrows
    col_slices = [slice(i*col_sze, (i+1)*col_sze) for i in range(0, ncols)]
    row_slices = [slice(i*row_sze, (i+1)*row_sze) for i in range(0, nrows)]
    divisions = []
    for j in range(0, ncols):
        for i in range(0, nrows):
            slice_col = col_slices[j]
            slice_row = row_slices[i]
            div = ds[:, slice_row, slice_col]
            divisions.append(div)
    return divisions


def get_val_coord(lon, lat, scan):
    if len(scan.lat.shape) == 1:
        return scan.sel({'lat': lat, 'lon': lon}, method='nearest')
    if len(scan.lat.shape) > 2:
        abslat = np.abs(scan.lat.mean(dim='time') - lat)
        abslon = np.abs(scan.lon.mean(dim='time') - lon)
    else:
        abslat = np.abs(scan.lat - lat)
        abslon = np.abs(scan.lon - lon)
    c = np.maximum(abslon.values, abslat.values)
    xloc, yloc = np.where(c == np.min(c))
    point_value = scan[:, xloc[0], yloc[0]]
    return point_value


def ds_to_df_at_point(ds, point):
    lon, lat = point
    if isinstance(ds, xr.DataArray):
        vars = [ds.name]
    if isinstance(ds, xr.Dataset):
        vars = list(ds.data_vars)
    ds_at_point = get_val_coord(lon, lat, ds)
    df_at_point = ds_at_point.to_dataframe()
    df_at_point = df_at_point.loc[:, vars]
    return df_at_point


def calc_ds_center(sample):
    """
    Calculates the geographic midpoint (central longitude and latitude) of a dataset.

    Parameters:
    - sample: A dataset containing 'lon' and 'lat' coordinates.

    Returns:
    - (lon0, lat0): A tuple representing the central longitude and latitude of the dataset.

    Example usage:
    >>> sample_data = xr.Dataset({'lon': (['x'], [-10, 0, 10]), 'lat': (['y'], [50, 60])})
    >>> calc_radar_origin(sample_data)
    (0.0, 55.0)
    """
    lon_max, lon_min = sample.lon.values.max(), sample.lon.values.min()
    lat_max, lat_min = sample.lat.values.max(), sample.lat.values.min()

    # Calculate the midpoint for longitude and latitude
    lon0, lat0 = lon_min + (lon_max - lon_min) / 2, lat_min + (lat_max - lat_min) / 2

    return lon0, lat0


def get_bbox_from_gdf(gdf):
    minlon, minlat = gdf.boundary.get_coordinates().min()
    maxlon, maxlat = gdf.boundary.get_coordinates().max()
    return BBox(minlat, minlon, maxlat, maxlon)


def get_surrounding_bbox_from_bboxs(bboxs):
    minlat = np.min([i.min_lat for i in bboxs])
    minlon = np.min([i.min_lon for i in bboxs])
    maxlat = np.max([i.max_lat for i in bboxs])
    maxlon = np.max([i.max_lon for i in bboxs])
    return BBox(minlat, minlon, maxlat, maxlon)

def get_bbox_from_df(df):
    min_lat, max_lat = df.lat.min(), df.lat.max()
    min_lon, max_lon = df.lon.min(), df.lon.max()
    return BBox(min_lat, min_lon, max_lat, max_lon)


def get_extent_from_bbox(bbox, pad=0):
    return (bbox.min_lon - pad, bbox.max_lon + pad, bbox.min_lat - pad, bbox.max_lat + pad)

