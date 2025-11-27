import xarray as xr
import numpy as np
import pandas as pd
import pyart
from datetime import timedelta as td
import warnings


def open_mf_pyart_dataset(fpaths, elevation):
    radars = [pyart.io.read(fpath) for fpath in fpaths]
    dss = [extract_sweep_from_pyart_radar(rad, elevation) for rad in radars]
    if determine_need_normalization(dss):
        dss = normalize_range_wise(dss)
    big_ds = xr.concat(dss, dim='time')
    return big_ds


def extract_sweep_from_pyart_radar(radar, sweep_number):
    sweep = radar.extract_sweeps([sweep_number])
    coords = {'azimuth': sweep.azimuth,
              'range': sweep.range}

    site = {'longitude': sweep.longitude,
            'latitude': sweep.latitude,
            'altitude': sweep.altitude}

    fields = sweep.fields

    ds = xr.Dataset()

    for key in coords.keys():
        ccopy = coords[key].copy()
        da = xr.DataArray(ccopy.pop('data'), dims=key, attrs=ccopy)
        da.name = key
        ds.coords[key] = da

    time = pd.to_datetime(sweep.time['units'].split(' ')[-1][:-1])
    rtime = [time + td(seconds=int(i)) for i in sweep.time['data']]
    ds = ds.assign_coords(
        {'elevation': (['azimuth'], sweep.elevation['data']),
         'rtime': (['azimuth'], rtime)})

    ds = ds.assign_coords({'time': ds.rtime.to_numpy().min()})

    for key in fields.keys():
        ccopy = fields[key].copy()
        da = xr.DataArray(ccopy.pop('data'),
                          dims=['azimuth', 'range'],
                          coords={'range': ds['range'], 'azimuth': ds['azimuth']},
                          attrs=ccopy)
        ds[key] = da

    for key in site.keys():
        ccopy = site[key].copy()
        da = xr.DataArray(ccopy.pop('data'), attrs=ccopy)
        da.name = key
        ds.coords[key] = da
    ds.attrs = {'fixed_angle': sweep.fixed_angle['data'],
                'other': sweep.metadata}
    ds = ds.interp({'azimuth': np.arange(0.5, 360, 1)}, method='nearest')
    return ds


def determine_need_normalization(ds_list):
    if len(set([ds.range.shape for ds in ds_list])) > 1:
        return True
    else:
        return False


def normalize_range_wise(ds_list):
    warnings.warn("Different range values, interpolating", category=Warning)
    ds_ranges = [ds.range.data for ds in ds_list]
    range_lengths = [len(r) for r in ds_ranges]
    ds_dict = {'ranges': ds_ranges,
               'rlength': range_lengths}
    ds_df = pd.DataFrame.from_dict(ds_dict)
    min_range = ds_df.sort_values('rlength').iloc[0, 0]
    interped_dss = [ds.interp({'range': min_range}, method='nearest') for ds in ds_list]
    return interped_dss
