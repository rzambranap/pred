import os
from datetime import datetime as dt
import pandas as pd


def get_df_dates_filepaths(directory_path, date_char_st=None,
                           date_char_nd=None, date_format=None):
    """
    Creates a dataframe with all of the files in the directory
    with an index of datetimes for easy choosing of a specific time

    Using knowledge about the filenames of files in directory
    (files can be recurrent in directory) we determine the datetime
    for each file and have a path to it for easy opening

    Best used with 'directory_path' starting at home

    Parameters
    ----------
    directory_path : str
        Directory you wish to map out
    date_char_st : int
        Character of filename at wich the date time starts for file
    date_char_nd : int
        Character of filename at wich the date time ends for file
    date_format : str
        Date format in wich the filename stored datetime data, following
        datetime conventions

    Returns
    -------
    files_x : pd.DataFrame
        Dataframe with datetimes and filepaths for desired directory

    See Also
    --------
    function or class : descriptor.

    Examples
    --------
    test
    """
    paths_x = []
    dates_x = []
    for root, dirs, files in os.walk(directory_path):
        for name in files:
            fpath = os.path.join(root, name)
            if fpath[-4:] == 'tore':
                break
            if ((date_char_st is None) or (date_char_nd is None) or (date_format is None)):
                parser = get_dt_parser(name)
                date = parser(name)
            else:
                date_str = name[date_char_st:date_char_nd]
                date = dt.strptime(date_str, date_format)
            paths_x.append(fpath)
            dates_x.append(date)
    files_x = pd.DataFrame.from_dict({'paths': paths_x,
                                      'time': dates_x})
    files_x = files_x.sort_values('time')
    files_x = files_x.set_index('time')
    return files_x


def get_dt_parser(fname):
    for key in dtparsers.keys():
        str2match = strstomatch[key]
        if str2match in fname:
            return dtparsers[key]
    raise ValueError(f"fname {fname} didn't match any known files")


def get_imerg_datetime(fname):
    date_portion = fname.split('.')[4].split('-')[0:2]
    datestr = (date_portion[0] + date_portion[1]).replace('S', '')
    return pd.to_datetime(datestr)


def get_gsmap_datetime(fname):
    date_portion = fname.split('.')[1:3]
    datestr = (date_portion[0] + date_portion[1])
    return pd.to_datetime(datestr)


def get_romuald_datetime(fname):
    datestr = fname.split('_')[-2]
    return pd.to_datetime(datestr)


def get_funrad_datetime(fname):
    datestr = ''.join(filter(str.isdigit, fname.split('.')[0]))
    return pd.to_datetime(datestr, format='%y%m%d%H%M%S')


def get_recrad_datetime(fname):
    datestr = ''.join(filter(str.isdigit, fname.split('.')[0]))[:-2]
    return pd.to_datetime(datestr)


def get_generalized_datetime(fname):
    datestr = fname.split('_')[-2]
    return pd.to_datetime(datestr, format='S%Y%m%dT%H%M')


def get_fpaths_from_dir(dirpath):
    fpaths = []
    for root, dirs, files in os.walk(dirpath):
        for name in files:
            fpath = os.path.join(root, name)
            fpaths.append(fpath)
    return fpaths


dtparsers = {'imerg': get_imerg_datetime,
             'gsmap': get_gsmap_datetime,
             'romuald': get_romuald_datetime,
             'funrad': get_funrad_datetime,
             'recrad': get_recrad_datetime,
             'level1': get_generalized_datetime,
             'level2': get_generalized_datetime}

strstomatch = {'imerg': 'IMERG',
               'gsmap': 'gsmap',
               'romuald': 'Romuald',
               'funrad': '.RAW',
               'recrad': '.vol',
               'level1': 'L1_',
               'level2': 'L2_'}
