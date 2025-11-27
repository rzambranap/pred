from .file_utils import get_df_dates_filepaths
import numpy as np
import xarray as xr
import pandas as pd
import haversine as hvs

class Gauges:
    def __init__(self, data_path, meta_path):
        meta = pd.read_csv(meta_path, index_col=0)
        data = pd.read_csv(data_path, index_col=0, parse_dates=[0])
        if isinstance(meta.index[0], (int, np.int64)):
            data.columns = data.columns.astype(int)
        self.data = data
        self.metadata = meta
        return

    def calc_distance_to_point(self, point: '(lat, lon)', return_df=False):
        if not isinstance(point, tuple):
            raise ValueError(f'point must be a tuple (lon, lat), instead is {type(point)}')
        gau_coords = self.metadata.loc[:, ['lat', 'lon']].to_numpy()
        rad_coords = np.array(point)
        distances = hvs.haversine_vector(rad_coords, gau_coords, comb=True)
        self.metadata.loc[:, 'distance_to_point'] = distances
        if return_df:
            return self.metadata
        return

    def get_coords_station(self, station):
        st_df = self.metadata.loc[station, :]
        lat = st_df.lat
        lon = st_df.lon
        return lon, lat


class Radar:
    def __init__(self, region, name):
        self.region = region
        self.name = name
        return

    def build_catalog(self, level, dirpath, **kwargs):
        catalog = self.__catalog_builders[level](dirpath, **kwargs)
        setattr(self, f'l{level}_catalog', catalog)
        return

    def load_sample(self, level=None, date=None):
        if level is None:
            retrieve_catalog  = 'l2_catalog'
        else:
            retrieve_catalog  = f'l{level}_catalog'
        if hasattr(self, retrieve_catalog):
            cat = getattr(self, retrieve_catalog)
            if date is None:
                return xr.open_dataset(cat.iloc[0, 0])
            else:
                fpath = cat.loc[date, 'paths']
                if not isinstance(fpath, str):
                    fpath = fpath.iloc[0]
                return xr.open_dataset(fpath)
        else:
            raise ValueError(f'no catalog yet built for {self.name}')

    __catalog_builders = {0: get_df_dates_filepaths,
                          1: get_df_dates_filepaths,
                          2: get_df_dates_filepaths}


class SPP:
    def __init__(self, product, name):
        self.product = product
        self.name = name
        return

    def build_catalog(self, level, dirpath, **kwargs):
        catalog = self.__catalog_builders[level](dirpath, **kwargs)
        setattr(self, f'l{level}_catalog', catalog)
        return

    def load_sample(self, level=None, date=None):
        if level is None:
            retrieve_catalog  = 'l2_catalog'
        else:
            retrieve_catalog  = f'l{level}_catalog'
        if hasattr(self, retrieve_catalog):
            cat = getattr(self, retrieve_catalog)
            if date is None:
                return xr.open_dataset(cat.iloc[0, 0])
            else:
                fpath = cat.loc[date, 'paths']
                if not isinstance(fpath, str):
                    fpath = fpath.iloc[0]
                return xr.open_dataset(fpath)
        else:
            raise ValueError(f'no catalog yet built for {self.name}')

    __catalog_builders = {0: get_df_dates_filepaths,
                          1: get_df_dates_filepaths,
                          2: get_df_dates_filepaths,
                          3: get_df_dates_filepaths}



def determine_analysis_scope_for_gauge_network(full_gauge_dataset, df_files_radar,
                                               daily_threshold=20,
                                               min_gauge_points=6, min_hours_radar=8,
                                               min_radar_files=32):
    # TODO Add functionnality of checking raw radar files for radar data availability
    # Use only gauge data when we have radar data available

    unique_dates_radar = list(set([wea.date() for wea in df_files_radar.index.tolist()]))
    dates_to_test = [str(wea) for wea in unique_dates_radar]
    gauge_data = pd.concat([full_gauge_dataset.loc[day] for day in dates_to_test])
    display(gauge_data)
    stations = gauge_data.columns.tolist()  # Define stations
    summary_of_ellegible_stations = {}
    for station in stations:
        ds = gauge_data.loc[:, station]
        ds_acc_daily = ds.groupby(pd.Grouper(freq='D')).sum()
        ds_over_threshold = ds_acc_daily.loc[ds_acc_daily > daily_threshold]
        dates_for_station = [str(item.date()) for item in ds_over_threshold.index.tolist()]
        if len(dates_for_station) < 1:
            continue
        gauge_data_for_station = pd.concat([gauge_data.loc[day, station] for day in dates_for_station])
        gauge_data_for_station = gauge_data_for_station.dropna()
        npoints = gauge_data_for_station.shape[0]
        if npoints > min_gauge_points:
            time_step = (gauge_data_for_station.index[1] - gauge_data_for_station.index[0])
            mins = int(((time_step.to_numpy().astype(float))/1e9)/60)
            summary_of_ellegible_stations[station] = {'npoints': npoints,
                                                      'dates': dates_for_station,
                                                      'time_step': mins,
                                                      'ndays': len(dates_for_station),
                                                      'gauge_data_for_station': gauge_data_for_station}
    return pd.DataFrame.from_dict(summary_of_ellegible_stations).T
