import numpy as np
import xradar as xd
import warnings


def calculate_georeferencing(ds):
    """
    Calculate georeferencing coordinates for an xarray dataset.

    Args:
        ds (xr.Dataset): Input xarray dataset.

    Returns:
        xr.Dataset: Output xarray dataset with georeferencing coordinates added.
    """
    mfds = ds.copy()
    ranges = mfds.range.to_numpy()
    azimuths = mfds.azimuth.to_numpy()
    elevations = mfds.elevation.mean().data
    site_altitude = mfds.altitude.to_numpy()

    x_origin = mfds.longitude.data
    y_origin = mfds.latitude.data

    xs = np.empty((len(azimuths), len(ranges)))
    ys = np.empty_like(xs)
    zs = np.empty_like(xs)
    for az in range(0, len(azimuths)):
        for r in range(0, len(ranges)):
            x, y, z = xd.georeference.antenna_to_cartesian(
                ranges[r], azimuths[az], elevations, site_altitude=site_altitude
            )
            xs[az, r] = x
            ys[az, r] = y
            zs[az, r] = z

    lon, lat = cartesian_to_geographic_aeqd(xs, ys, x_origin, y_origin)

    alt = mfds.altitude.data
    alts = zs + alt

    out_ds = mfds.assign_coords(
        {'lon': (['azimuth', 'range'], lon),
         'lat': (['azimuth', 'range'], lat),
         'alt': (['azimuth', 'range'], alts)}
    )

    return out_ds


def cartesian_to_geographic_aeqd(x, y, lon_0, lat_0, R=6370997.0):
    """
    Azimuthal equidistant Cartesian to geographic coordinate transform.

    Transform a set of Cartesian/Cartographic coordinates (x, y) to
    geographic coordinate system (lat, lon) using a azimuthal equidistant
    map projection [1]_.

    .. math::

        lat = \\arcsin(\\cos(c) * \\sin(lat_0) +
                       (y * \\sin(c) * \\cos(lat_0) / \\rho))

        lon = lon_0 + \\arctan2(
            x * \\sin(c),
            \\rho * \\cos(lat_0) * \\cos(c) - y * \\sin(lat_0) * \\sin(c))

        \\rho = \\sqrt(x^2 + y^2)

        c = \\rho / R

    Where x, y are the Cartesian position from the center of projection;
    lat, lon the corresponding latitude and longitude; lat_0, lon_0 are the
    latitude and longitude of the center of the projection; R is the radius of
    the earth (defaults to ~6371 km). lon is adjusted to be between -180 and
    180.

    Parameters
    ----------
    x, y : array-like
        Cartesian coordinates in the same units as R, typically meters.
    lon_0, lat_0 : float
        Longitude and latitude, in degrees, of the center of the projection.
    R : float, optional
        Earth radius in the same units as x and y. The default value is in
        units of meters.

    Returns
    -------
    lon, lat : array
        Longitude and latitude of Cartesian coordinates in degrees.

    References
    ----------
    .. [1] Snyder, J. P. Map Projections--A Working Manual. U. S. Geological
        Survey Professional Paper 1395, 1987, pp. 191-202.

    """
    x = np.atleast_1d(np.asarray(x))
    y = np.atleast_1d(np.asarray(y))

    lat_0_rad = np.deg2rad(lat_0)
    lon_0_rad = np.deg2rad(lon_0)

    rho = np.sqrt(x * x + y * y)
    c = rho / R

    with warnings.catch_warnings():
        # division by zero may occur here but is properly addressed below so
        # the warnings can be ignored
        warnings.simplefilter("ignore", RuntimeWarning)
        lat_rad = np.arcsin(
            np.cos(c) * np.sin(lat_0_rad) + y * np.sin(c) * np.cos(lat_0_rad) / rho
        )
    lat_deg = np.rad2deg(lat_rad)
    # fix cases where the distance from the center of the projection is zero
    lat_deg[rho == 0] = lat_0

    x1 = x * np.sin(c)
    x2 = rho * np.cos(lat_0_rad) * np.cos(c) - y * np.sin(lat_0_rad) * np.sin(c)
    lon_rad = lon_0_rad + np.arctan2(x1, x2)
    lon_deg = np.rad2deg(lon_rad)
    # Longitudes should be from -180 to 180 degrees
    lon_deg[lon_deg > 180] -= 360.0
    lon_deg[lon_deg < -180] += 360.0

    return lon_deg, lat_deg
