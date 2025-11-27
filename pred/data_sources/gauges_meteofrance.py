import pandas as pd
import os
import numpy as np
import re
from lat_lon_parser import parse

IGNORE_FILES = ['6MIN_R_S200705010000_E200904302354.data']

valid_acronyms = {'RR1': 'ACC_PRECIPITATION_1HOUR',
                  'DRR1': 'LENGTH_EPISODE',
                  'RR6': 'ACC_PRECIPITATION_6MIN',
                  'RR': 'ACC_PRECIPITATION_1DAY'}

date_format = {'RR1': '%Y%m%d%H',
               'DRR1': '%Y%m%d%H',
               'RR6': '%Y%m%d%H%M',
               'RR': '%Y%m%d'}


def process_data_files(dir_path):
    """
    Process data files from the specified directory.

    Args:
        dir_path (str): Path to the directory containing the data files.

    Returns:
        list: List of dictionaries containing the processed meteorological fields.
    """
    data_paths = get_data_paths(dir_path)
    met_fields = []
    for data_path in data_paths:
        if data_path.split('/')[-1] in IGNORE_FILES:
            continue
        meteorological_fields = process_data_file(data_path)
        met_fields.append(meteorological_fields)
    return met_fields


def get_data_paths(dir_path):
    """
    Get the paths of the data files from the specified directory.

    Args:
        dir_path (str): Path to the directory containing the data files.

    Returns:
        list: List of data file paths.
    """
    mf_list = os.listdir(dir_path)
    data_paths = [os.path.join(dir_path, i) for i in mf_list if '.data' in i]
    return data_paths


def get_html_paths(dir_path):
    """
    Get the paths of the data files from the specified directory.

    Args:
        dir_path (str): Path to the directory containing the data files.

    Returns:
        list: List of data file paths.
    """
    mf_list = os.listdir(dir_path)
    html_paths = [os.path.join(dir_path, i) for i in mf_list if '.html' in i]
    return html_paths


def process_data_file(data_path):
    """
    Process a single data file.

    Args:
        data_path (str): Path to the data file.

    Returns:
        dict: Dictionary containing the processed meteorological fields present in the data file.
    """
    print(data_path)
    test = pd.read_csv(data_path, sep=';', decimal=",")
    measured = find_measured_field(test)
    test['time'] = pd.to_datetime(test['DATE'], format=date_format[measured])
    test = test.drop('DATE', axis=1)
    group_poste = test.groupby('POSTE')
    meteorological_fields = {field: [] for field in valid_acronyms.values()}
    for name, group in group_poste:
        watered_down = group.set_index('time').drop('POSTE', axis=1)
        present_fields = [field for field in watered_down.columns if field in valid_acronyms.keys()]
        separated_fields = {
            valid_acronyms[field]: watered_down[field].to_frame().rename(columns={field: name})
            for field in present_fields
        }
        for field in separated_fields.keys():
            meteorological_fields[field].append(separated_fields[field])
    for key in list(meteorological_fields.keys()):
        if len(meteorological_fields[key]) < 1:
            _ = meteorological_fields.pop(key)
        else:
            meteorological_fields[key] = pd.concat(meteorological_fields[key], axis=1)
    return meteorological_fields


def find_measured_field(data):
    """
    Find the field indicating the measured variable from the data.

    Args:
        data (pd.DataFrame): Dataframe containing the data.

    Returns:
        str: Field name of the measured variable.
    """
    for field in valid_acronyms.keys():
        if field in data.columns:
            return field
    raise ValueError("No valid measured field found in the data.")


def parse_field(field, met_field):
    """
    Parse a specific meteorological field from the met_fields data.

    Args:
        field (str): Name of the meteorological field to parse.
        met_fields (list): List of dictionaries containing the processed meteorological fields.

    Returns:
        pd.DataFrame: Parsed meteorological field in UTC except for daily data.
    """
    field_list = [fil[field] for fil in met_field if field in fil.keys()]
    field_chill = pd.concat(field_list).sort_index()
    if field != 'ACC_PRECIPITATION_1DAY':
        field_chill.index = field_chill.index + pd.to_timedelta('3 hours')
    mask = ~field_chill.index.duplicated()
    field_chill = field_chill.loc[mask]
    return field_chill


def parse_met_fields(met_fields):
    """
    Parse meteorological fields from the met_fields data.

    Args:
        met_fields (list): List of dictionaries containing the processed meteorological fields.

    Returns:
        dict: Dictionary of parsed meteorological fields.
    """
    parsed_fields = {}
    for field in valid_acronyms.values():
        print(field)
        parsed_field = parse_field(field, met_fields)
        parsed_fields[field] = parsed_field
    return parsed_fields


def process_html_file(path):
    print(path)
    dfs_html = pd.read_html(path, )
    short_info = dfs_html[3]
    mask = ~np.array([i in valid_acronyms.keys() for i in short_info.iloc[:, 0].tolist()])
    station_ids = short_info.loc[mask].iloc[:, 0].tolist()
    long_info_idx = np.argmax([np.sum(df.shape) for df in dfs_html])
    long_info = dfs_html[long_info_idx]
    long_mask = [i in station_ids for i in long_info.iloc[:, 0]]
    meta_stations = long_info.loc[long_mask]
    meta_stations.columns = ['id', 'name', 'coordinates', 'lambert', 'altitude', 'owner']
    coordinates = [i.replace('O', 'W').split(' ') for i in meta_stations.coordinates.tolist()]
    lats = [parse(i[0]) for i in coordinates].copy()
    lons = [parse(i[1]) for i in coordinates].copy()
    meta_stations = meta_stations.assign(lat=lats)
    meta_stations = meta_stations.assign(lon=lons)
    meta_stations.loc[:, 'lon'] = lons
    meta_stations['altitude'] = [float(re.findall(r'\d+', i)[0]) for i in meta_stations.altitude]
    meta_stations['id'] = [eval(i) for i in meta_stations.loc[:, 'id']]
    meta_stations = meta_stations.drop(['coordinates', 'lambert', 'owner'], axis=1)
    return meta_stations


def process_html_files(dir_path):
    fpaths = get_html_paths(dir_path)
    dfs = []
    for path in fpaths:
        dfs.append(process_html_file(path))
    full_metadata = pd.concat(dfs)
    full_metadata = full_metadata.drop_duplicates(keep='first')
    full_metadata = full_metadata.set_index('id')
    full_metadata = full_metadata.loc[~full_metadata.index.duplicated()]
    return full_metadata


def process_meteofrance_data_html_files(dir_path):
    metfields = process_data_files(dir_path)

    parsed_met_fields = parse_met_fields(metfields)

    metadata = process_html_files(dir_path)

    return parsed_met_fields, metadata


def save_parsed_fields(parsed_fields, metadata, out_dir):
    """
    Save parsed fields and metadata to CSV files.

    Args:
        parsed_fields (dict): Dictionary of parsed meteorological fields.
        full_metadata (pd.DataFrame): Parsed metadata.
        dir_path (str): Directory path to save the files.
    """
    # Remove duplicates from metadata and save to CSV

    # Save parsed fields to CSV
    mdcopy = metadata.copy()
    for field in valid_acronyms.values():
        print(field)
        df = parsed_fields[field]
        mdrelevant = mdcopy.loc[df.columns, :]
        field_path = os.path.join(out_dir, f'{field}.csv')
        meta_name = f'{field}_metadata.csv'
        meta_path = os.path.join(out_dir, meta_name)
        mdrelevant.to_csv(meta_path)
        df.to_csv(field_path)
        print(field_path)
    return
