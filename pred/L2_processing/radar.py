import numpy as np
import xarray as xr


def calc_rate_from_refl(ref, a=200, b=1.6, offset=None):
    if offset is not None:
        ref = ref + offset
    radar_final = ((10**(ref / 10)) / a) ** (1 / b)
    return radar_final


def calc_refl_from_rate(rate_df, a=200, b=1.6):
    non_zero = rate_df+0.0001
    final = 10*np.log10(a*(non_zero**(b)))
    return final


def calculate_radar_estimate_averaging_elevations(raw_DSs, ref_var, a=205, b=1.44, offset=5.7):
    """
    Calculate radar estimate by averaging reflectivities from three elevations.

    Args:
        raw_DSs (list): List of xarray Datasets containing reflectivity data for different elevations.
        ref_var (str): Name of the reflectivity variable in the Datasets.
        a (float, optional): Parameter 'a' used in rate calculation. Defaults to 205.
        b (float, optional): Parameter 'b' used in rate calculation. Defaults to 1.44.
        offset (float, optional): Offset value used in rate calculation. Defaults to 5.7.

    Returns:
        xarray.Dataset: Dataset containing the radar estimate of rain rate.

    """
    # Extract reflectivity data from raw_DSs
    reflectivities = [i[ref_var] for i in raw_DSs]

    # Extract necessary coordinate data
    azis = reflectivities[0].azimuth.data
    rans = reflectivities[0].range.data
    time = reflectivities[0].time.data
    several_lons = [refl.lon.data for refl in reflectivities]
    several_lats = [refl.lat.data for refl in reflectivities]

    # Convert reflectivity data to numpy arrays
    reflectivities = [i.to_numpy() for i in reflectivities]
    reflectivities = [np.ma.masked_invalid(i) for i in reflectivities]

    # Calculate rain rates from reflectivities
    rates = [calc_rate_from_refl(refl, a=a, b=b, offset=offset) for refl in reflectivities]

    # Average the rain rates from different elevations
    rate_final = np.ma.stack(rates).mean(axis=0)

    # Compute average latitudes and longitudes
    lat_final = np.stack(several_lats).mean(axis=0)
    lon_final = np.stack(several_lons).mean(axis=0)

    # Create a new xarray Dataset for the final radar estimate
    final_ds = xr.Dataset(
        data_vars={'rain_rate': (['time', 'azimuth', 'range'], rate_final)},
        coords={'time': time, 'azimuth': azis, 'range': rans}
    )

    # Assign latitude and longitude coordinates to the final Dataset
    final_ds = final_ds.assign_coords({
        'lon': (['azimuth', 'range'], lon_final),
        'lat': (['azimuth', 'range'], lat_final)
    })

    return final_ds
