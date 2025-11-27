import numpy as np


def get_val_coord(lon, lat, scan):
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


def get_coords_station(gauge_meta, station):
    st_df = gauge_meta.loc[station, :]
    lat = st_df.lat
    lon = st_df.lon
    return lat, lon
