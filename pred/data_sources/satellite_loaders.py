import xarray as xr
from datetime import datetime as dt
import pandas as pd
import numpy as np
import gzip


def convert_to_dt(imerg_ds):
    """
    Converts time index of imerg dataset to datetime object

    Parameters
    ----------
    imerg_ds : xarray.Dataset or xarray.DataArray
        Loaded dataset of IMERG data.

    Returns
    -------
    imerg_ds : xarray.Dataset or xarray.DataArray
        description.

    See Also
    --------
    load_imerg : TODO.

    Examples
    --------
    test
    """
    new_index = [dt.strptime(str(it), '%Y-%m-%d %H:%M:%S')
                 for it in imerg_ds.time.data]
    imerg_ds['time'] = new_index
    return imerg_ds


def load_gsmap_to_xr(fpath):
    """
    Loads raw GSMaP files to an xarray.DataArray

    Using gzip numpy and xarray

    Parameters
    ----------
    fpath : str
        Filepath

    Returns
    -------
    gsmap_xr : xr.DataArray
        DataArray with the loaded GSMaP data

    See Also
    --------
    load_daily_gsmap : loads GSMaP data for a whole day given the date.

    Examples
    --------
    test
    """
    gz = gzip.GzipFile(fpath, 'rb')
    dd = np.frombuffer(gz.read(), dtype=np.float32)
    # 1200 = nb of lat ; 3600 = np of lon; pre=mm/hr
    pre = dd.reshape((1200, 3600, 1))
    nm = fpath.split('/')[-1]
    time = np.array(pd.to_datetime(nm.split('.')[1] + (nm.split('.')[2]))).reshape((1))
    # Create lon and lat coordinates
    lon = np.linspace(0.05, 359.95, 3600)  # centroids of lon pixels
    lat = np.linspace(59.95, -59.95, 1200)  # centroids of lat pixels
    lon[lon > 180] = lon[lon > 180]-360
    gsmap_xr = xr.DataArray(data=pre,
                            dims=['lat', 'lon', 'time'],
                            coords={'lat': lat,
                                    'lon': lon,
                                    'time': time})
    gsmap_xr = gsmap_xr.sortby('lon')
    return gsmap_xr


def open_mf_gsmap(fpaths):
    if isinstance(fpaths, str):
        fpaths = [fpaths]
    xrs = [load_gsmap_to_xr(fpath) for fpath in fpaths]
    dataset = xr.concat(xrs, dim='time')
    return dataset


def open_mf_imerg(fpaths, field=None):
    if isinstance(fpaths, str):
        fpaths = [fpaths]

    if isinstance(field, str):
        field = [field]

    ds = xr.open_mfdataset(fpaths, group='Grid/')
    ds = convert_to_dt(ds)
    if isinstance(field, list):
        return ds[field]
    else:
        return ds


def load_daily_gsmap(date, filepaths_df):
    """
    Loads raw GSMaP files for a whole day to an xarray.DataArray

    Given the date and that we have the files

    Parameters
    ----------
    date : str
        'YYYY-MM-DD' date for which we want the data
    filepaths_df : pd.DataFrame
        Dataframe containing datetimes and filepaths created using the
        'get_df_dates_filepaths' function

    Returns
    -------
    dataset : xr.DataArray
        DataArray with the loaded GSMaP data for the whole day

    See Also
    --------
    load_daily_gsmap : loads GSMaP data for a whole day given the date.

    Examples
    --------
    test
    """
    daily_paths = filepaths_df.loc[date]
    daily_index = daily_paths.index.tolist()
    paths = daily_paths.paths.tolist()
    xrs = [load_gsmap_to_xr(fpath) for fpath in paths]
    dataset = xr.concat(xrs, daily_index)
    dataset = dataset.rename({'concat_dim': 'time'})
    return dataset


def load_daily_imerg(date, filepaths_df, field=None):
    """
    Loads raw IMERG files for a whole day to an xarray.DataArray

    Given the date and that we have the files

    Parameters
    ----------
    date : str
        'YYYY-MM-DD' date for which we want the data
    filepaths_df : pd.DataFrame
        Dataframe containing datetimes and filepaths created using the
        'get_df_dates_filepaths' function
    field : list of strs or str
        Field to load such as 'precipitationCal', or None in order to
        load everything

    Returns
    -------
    dataset : xr.DataArray
        DataArray with the loaded GSMaP data for the whole day

    See Also
    --------
    load_daily_gsmap : loads GSMaP data for a whole day given the date.

    Examples
    --------
    test
    """
    if isinstance(field, str):
        field = [field]
    fpaths = filepaths_df.loc[date].paths.tolist()
    ds = xr.open_mfdataset(fpaths, group='Grid/')
    ds = convert_to_dt(ds)
    if isinstance(field, list):
        return ds[field]
    else:
        return ds
