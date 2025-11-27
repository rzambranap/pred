import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import xarray as xr
import numpy as np
import pyproj
from shapely.geometry import Point
from shapely.ops import transform
import geopandas as gpd
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib
import os


def save_fig(
        fig: matplotlib.figure.Figure, 
        fig_name: str, 
        fig_dir: str, 
        fig_fmt: str,
        fig_size: tuple[float, float] = None, 
        save: bool = True, 
        dpi: int = 300,
        transparent_png = True,
    ):
    """This procedure stores the generated matplotlib figure to the specified 
    directory with the specified name and format.

    Parameters
    ----------
    fig : [type]
        Matplotlib figure instance
    fig_name : str
        File name where the figure is saved
    fig_dir : str
        Path to the directory where the figure is saved
    fig_fmt : str
        Format of the figure, the format should be supported by matplotlib 
        (additional logic only for pdf and png formats)
    fig_size : Tuple[float, float]
        Size of the figure in inches, by default [6.4, 4] 
    save : bool, optional
        If the figure should be saved, by default True. Set it to False if you 
        do not want to override already produced figures.
    dpi : int, optional
        Dots per inch - the density for rasterized format (png), by default 300
    transparent_png : bool, optional
        If the background should be transparent for png, by default True
    """
    if not save:
        return
    if fig_size is not None:
        fig.set_size_inches(fig_size, forward=False)
    else:
        fig.set_size_inches(fig.get_size_inches(), forward=False)        
    fig_fmt = fig_fmt.lower()
    fig_dir = os.path.join(fig_dir, fig_fmt)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    pth = os.path.join(
        fig_dir,
        '{}.{}'.format(fig_name, fig_fmt.lower())
    )
    if fig_fmt == 'pdf':
        metadata={
            'Creator' : '',
            'Producer': '',
            'CreationDate': None
        }
        fig.savefig(pth, bbox_inches='tight', metadata=metadata)
    elif fig_fmt == 'png':
        alpha = 0 if transparent_png else 1
        axes = fig.get_axes()
        fig.patch.set_alpha(alpha)
        for ax in axes:
            ax.patch.set_alpha(alpha)
        fig.savefig(
            pth, 
            bbox_inches='tight',
            dpi=dpi,
        )
    else:
        try:
            fig.savefig(pth, bbox_inches='tight')
        except Exception as e:
            print("Cannot save figure: {}".format(e)) 




def create_circle(coord, radius_km):
    # Central point
    central_point = Point(coord)
    
    # Define the projection
    wgs84 = pyproj.CRS("EPSG:4326")
    aeqd_proj = pyproj.CRS(proj="aeqd", lat_0=coord[1], lon_0=coord[0])
    
    # Project to azimuthal equidistant projection
    project = pyproj.Transformer.from_crs(wgs84, aeqd_proj, always_xy=True).transform
    central_point_aeqd = transform(project, central_point)
    
    # Create the buffer (radius in meters)
    buffer_aeqd = central_point_aeqd.buffer(radius_km * 1000)
    
    # Re-project the buffer back to WGS84
    project_back = pyproj.Transformer.from_crs(aeqd_proj, wgs84, always_xy=True).transform
    buffer_wgs84 = transform(project_back, buffer_aeqd)
    
    return buffer_wgs84


def create_radar_circles_and_centers(radar_csv_path, radius_km=150):
    """
    Creates radar circles and their central points from a CSV file containing radar coordinates.

    This function reads radar coordinates from a CSV file, filters the coordinates based on a bounding box,
    and generates circular geometries around each radar coordinate with a specified radius. It returns a 
    GeoDataFrame containing the radar circles and a GeoSeries containing the central points of the circles.

    Parameters:
    radar_cvs_path (str): The file path to the CSV file containing radar coordinates.
    radius_km (float, optional): The radius of the circles in kilometers. Default is 150 km.

    Returns:
    tuple: A tuple containing:
        - circle_gdf (GeoDataFrame): A GeoDataFrame with radar circle geometries.
        - central_points (GeoSeries): A GeoSeries with the central points of the radar circles.
    """
    rad_df = pd.read_csv(radar_csv_path)
    # TO DO put this in bbox form
    rad_df = rad_df.loc[rad_df.loc[:,'lat']<9]
    rad_df = rad_df.loc[rad_df.loc[:,'lon']>-80]
    rad_df = rad_df.loc[rad_df.loc[:,'lon']<-50]
    rad_df = rad_df.loc[rad_df.loc[:,'lat']>-6]

    coordinates = rad_df[['lon', 'lat']].values

    circles = [create_circle(coord, radius_km) for coord in coordinates]

    circle_gdf = gpd.GeoDataFrame({'name': rad_df['name'], 'geometry': circles}, crs="EPSG:4326")
    central_points = gpd.GeoSeries([Point(coord) for coord in coordinates], crs="EPSG:4326")

    return circle_gdf, central_points


def plot_stamen(ax=None, fig=None, level=2):
    """
    Add a Stamen terrain map to the given matplotlib figure and axes.

    Parameters
    ----------
    ax : matplotlib Axes, optional
        The Axes to add the Stamen terrain map to. If not provided, the
        current Axes will be used.
    fig : matplotlib Figure, optional
        The Figure that the Axes belongs to. If not provided, the current
        Figure will be used.
    level : int, optional
        The zoom level of the Stamen terrain map. A level of 2 corresponds
        to the global view, while higher levels correspond to more detailed
        views of smaller areas. The default level is 2.

    Returns
    -------
    None

    Notes
    -----
    The Stamen terrain map is a global map that shows physical features such
    as land and ocean elevations, and is provided by the Stamen Design
    cartography company.
    """
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    stamen_terrain = cimgt.Stamen('terrain-background')
    ax.add_image(stamen_terrain, level)
    return


def plot_simple_map(ax=None, fig=None):
    """
    Plot a simple map with ocean, coastline, rivers, borders, and land.

    Parameters
    ----------
    ax : Axes, optional
        The Axes object to draw on. If not provided, the current Axes will be used.
    fig : Figure, optional
        The Figure object to draw on. If not provided, the current Figure will be used.

    Returns
    -------
    None
    """
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND)
    gl = ax.gridlines()
    gl.bottom_labels = True
    gl.left_labels = True
    return


def plot_cml_network(df_metadata,
                     lons1_col='lons1', lons2_col='lons2',
                     lats1_col='lats1', lats2_col='lats2',
                     ax=None, c='b'):
    """
    This function plots a network based on the data stored in a pandas dataframe.

    Parameters:
    df_metadata (pd.DataFrame): A dataframe with the metadata for the network
    lons1_col (str, optional): Column name for the longitudes of the first endpoint of each edge. Default is 'lons1'.
    lons2_col (str, optional): Column name for the longitudes of the second endpoint of each edge. Default is 'lons2'.
    lats1_col (str, optional): Column name for the latitudes of the first endpoint of each edge. Default is 'lats1'.
    lats2_col (str, optional): Column name for the latitudes of the second endpoint of each edge. Default is 'lats2'.
    ax (matplotlib.axes._subplots.AxesSubplot, optional): Matplotlib axes object to plot the network on.
    Default is None, in which case it will be created.
    c (str, optional): Color to plot. Default is 'b' (blue).

    Returns:
    None
    """
    if ax is None:
        ax = plt.gca()
    idxs = df_metadata.index.tolist()
    for idx in idxs:
        lons = df_metadata.loc[idx, lons1_col], df_metadata.loc[idx, lons2_col]
        lats = df_metadata.loc[idx, lats1_col], df_metadata.loc[idx, lats2_col]
        ax.plot(lons, lats, c=c)
    return


def plot_gauge_network(df, ax=None, display_names=False, c='b'):
    if ax is None:
        ax = plt.gca()
    # Scatter plot of rain gauge locations
    ax.scatter(df['lon'], df['lat'], c=c, marker='x', s=50)

    # Iterate over each row in the DataFrame
    name_list = ['name', 'station']
    if display_names:
        name_index = [i for i in name_list if i in df.columns][0]
        for _, row in df.iterrows():
            name = row[name_index]
            x, y = row['lon'], row['lat']

            # Adjust text size based on the marker size
            marker_size = 50  # Change this value according to your preference
            text_size = marker_size / 10

            # Calculate the text position offset based on the marker size
            text_offset = marker_size * 0.05

            # Add the text annotation
            ax.annotate(name, (x, y), xytext=(text_offset, -text_offset),
                        textcoords='offset points', fontsize=text_size,
                        ha='left', va='bottom')

    return


def plot_radar_coverage(sample, ax=None, fig=None):
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    if isinstance(sample, xr.Dataset):
        fields = list(sample.data_vars)
        da = sample[fields[0]]
    elif isinstance(sample, xr.DataArray):
        da = sample
    else:
        raise TypeError("can only plot if xr dataset or data array")
    if 'time' in da.dims:
        single_tstep = da.isel({'time': 0})
    else:
        single_tstep = da

    single_tstep = single_tstep * 0
    coords = list(single_tstep.coords)

    if ('lat' in coords) and ('lon' in coords):
        single_tstep.plot(x='lon', y='lat', add_colorbar=False, alpha=0.5, ax=ax)

    return


def decide_type_plot(ds):
    dims = ds.dims
    ndims = len(dims)
    if ndims == 1:
        return 'time_series_plot'
    elif ndims == 2:
        return 'spatial_plot'
    else:
        raise ValueError(f'{ndims} dims in dataset, not yet taken into account')

def plot_ds(ds, joint=False, plot_map=True, match_scale=True, pcp_var=None, discretize=False, max_global=None, **kwargs):
    vars = list(ds.data_vars)
    nvars = len(vars)

    global_min = np.min(np.array([ds[var].min().values for var in vars]))
    global_max = np.max(np.array([ds[var].max().values for var in vars]))
    if max_global is not None:
        global_max = max_global

    plot_type = decide_type_plot(ds)

    figsize = ((7+2)*nvars, 7)
    fig, axes = plt.subplots(1,
                             nvars,
                             subplot_kw=dict(projection=ccrs.PlateCarree()),
                             figsize=figsize)

    for i in range(nvars):
        ax = axes[i]
        var = vars[i]
        if plot_map:
            plot_simple_map(ax=ax)
        vmin, vmax = ds[var].min().values, ds[var].max().values

        if match_scale:
            vmin, vmax = global_min, global_max
        if discretize:
            lvs = calculate_plotting_levels([ds[v] for v in vars], max_global=max_global)
            ds[var].plot(x='lon', y='lat', ax=ax, levels=lvs, **kwargs)
            ax.set_title(var)
        else:    
            ds[var].plot(x='lon', y='lat', ax=ax, vmin=vmin, vmax=vmax, **kwargs)
            ax.set_title(var)
    return


def plot_bbox_rectangle(bbox, style=None):
    if style is None:
        style={'linewidth':1,
               'edgecolor':'black',
               'facecolor':'none',
               'alpha':0.5}
    rect = Rectangle(
        (bbox.min_lon, bbox.min_lat),  # Bottom-left corner of the rectangle (min_lon, min_lat)
        bbox.max_lon - bbox.min_lon,   # Width of the rectangle (max_lon - min_lon)
        bbox.max_lat - bbox.min_lat,   # Height of the rectangle (max_lat - min_lat)
        transform=ccrs.PlateCarree(),
        **style
    )
    ax = plt.gca()
    ax.add_patch(rect)
    return


def calculate_plotting_levels(das_to_plot, nlevs=None, max_global=None):
    if isinstance(das_to_plot, xr.Dataset):
        das_to_plot = [das_to_plot[i] for i in das_to_plot.data_vars]
    if not isinstance(das_to_plot, list):
        das_to_plot = [das_to_plot]
    maxs = [i.max().values for i in das_to_plot]
    mins = [i.min().values for i in das_to_plot]
    vmax = np.max(maxs)
    if max_global is not None:
        vmax = max_global
    vmin = np.min(mins)
    if vmax//10 >= 3:
        vmax = int(vmax//10 * 10 + 10)
        if nlevs is None:
            nlevs = vmax//10 + 1
        lvs = np.linspace(0, vmax, nlevs)
    else:
        vmax = int(vmax//1 * 1 + 1)
        if nlevs is None:
            nlevs = vmax//1 + 1
        lvs = np.linspace(int(vmin//1), vmax, nlevs)
    return lvs