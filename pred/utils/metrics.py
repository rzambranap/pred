import numpy as np
import pandas as pd
from math import sqrt
from math import log10
#from HydroErr.HydroErr import kge_2012 as kge
#import xskillscore as xs


def kge(a, b):
    """
    Calculate Kling-Gupta Efficiency (KGE) between two numpy arrays.

    Parameters:
        a (numpy.ndarray): Observed values.
        b (numpy.ndarray): Simulated values.

    Returns:
        float: Kling-Gupta Efficiency (KGE).
    """
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    std_a = np.std(a)
    std_b = np.std(b)
    corr = np.corrcoef(a, b)[0, 1]

    kge = 1 - np.sqrt((corr - 1) ** 2 + (std_a / mean_a - std_b / mean_b) ** 2 + (mean_a / mean_b - 1) ** 2)
    return kge

def get_val_coord(lon, lat, scan):
    """
    Returns the value of a point on a 2D grid based on longitude, latitude coordinates.

    Args:
    - lon (float): The longitude coordinate of the point.
    - lat (float): The latitude coordinate of the point.
    - scan (xarray.Dataset): A 2D grid of data containing latitude and longitude coordinates.

    Returns:
    - point_value (xarray.Dataset): The value of the point at the given
    longitude and latitude coordinates or closest point.
    """
    if len(scan.lat.shape) > 2:
        abslat = np.abs(scan.lat.mean(dim='time')-lat)
        abslon = np.abs(scan.lon.mean(dim='time')-lon)
    else:
        abslat = np.abs(scan.lat-lat)
        abslon = np.abs(scan.lon-lon)
    c = np.maximum(abslon.values, abslat.values)
    xloc, yloc = np.where(c == np.min(c))
    point_value = scan[:, xloc[0], yloc[0]]
    return point_value


def get_coords_station(gauge_meta, station):
    """
    Returns the latitude and longitude coordinates of a given station.

    Args:
    - gauge_meta (pandas.DataFrame): A pandas DataFrame containing
    information about the stations, including latitude and longitude coordinates.
    - station (str): The identifier of the station

    Returns:
    - lat (float): The latitude coordinate of the station.
    - lon (float): The longitude coordinate of the station.
    """
    st_df = gauge_meta.loc[station, :]
    lat = st_df.lat
    lon = st_df.lon
    return lat, lon


def rad_sat_psnrs(rad, sat):
    time = rad.time
    psnrs = []
    for step in time:
        single_sat = sat.sel(time=step, method='nearest')
        single_sat = single_sat.data
        single_rad = rad.sel(time=step, method='nearest').data
        rad_mask = np.ma.masked_invalid(single_rad)
        gm_mask = np.ma.masked_array(single_sat, mask=rad_mask.mask)
        correlation_matrix = np.ma.corrcoef(
            rad_mask.flatten(), gm_mask.flatten())
        correlation_xy = correlation_matrix[0, 1]
        '''fig = plt.figure()
        plt.scatter(gm_mask, rad_mask)
        plt.show()'''
        # psnrs.append(psnr(rad_mask, gm_mask))
        psnrs.append(correlation_xy)
    score_df = pd.DataFrame.from_dict({'time': time, 'psnrs': psnrs})
    score_df = score_df.set_index('time')
    return score_df


def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))


def pearsonr_2D(x, y):
    """computes pearson correlation coefficient
       where x is a 1D and y a 2D array"""

    upper = np.sum((x - np.mean(x)) *
                   (y - np.mean(y, axis=1)[:, None]), axis=1)
    lower = np.sqrt(np.sum(np.power(x - np.mean(x), 2)) *
                    np.sum(np.power(y - np.mean(y,
                                                axis=1)[:, None],
                                    2),
                           axis=1))

    rho = upper / lower

    return rho


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = np.array([img1.max(), img2.max()])
    PIXEL_MAX = PIXEL_MAX.max()
    return 20 * log10(PIXEL_MAX / sqrt(mse))


def contingency_dataframe_timewise(observed,
                                   simulated,
                                   threshold=0,
                                   prefix=None):
    observed_over = (observed > threshold).sum(dim=['lat', 'lon'])
    # observed_under = (observed < threshold).sum(dim=['lat', 'lon'])

    # sim_over = (simulated > threshold).sum(dim=['lat', 'lon'])
    # sim_under = (simulated > threshold).sum(dim=['lat', 'lon'])

    true_positive = ((observed > threshold) * (simulated >
                     threshold)).sum(dim=['lat', 'lon'])
    false_positive = ((observed < threshold) * (simulated >
                      threshold)).sum(dim=['lat', 'lon'])
    false_negative = ((observed > threshold) * (simulated <
                      threshold)).sum(dim=['lat', 'lon'])
    true_negative = ((observed < threshold) * (simulated <
                     threshold)).sum(dim=['lat', 'lon'])

    POD = (true_positive / observed_over).to_dataframe()
    FAR = (false_positive / (false_positive + true_positive)).to_dataframe()
    F_SCORE = (false_positive / (false_positive +
               true_negative)).to_dataframe()

    cont_df = pd.concat([POD, FAR, F_SCORE,
                         true_positive.to_dataframe(),
                         false_negative.to_dataframe(),
                         false_positive.to_dataframe(),
                         true_negative.to_dataframe()],
                        axis=1)
    cont_df.columns = ['POD', 'FAR', 'F_SCORE', 'true_positive',
                       'false_negative', 'false_positive', 'true_negative']
    if prefix is not None:
        cont_df.columns = [prefix + '_' + i for i in cont_df.columns]

    return cont_df


def count_pixels_over_threshold(datasets, threshold, rename_cols=None):
    if type(datasets) != list:
        datasets = [datasets]
    counts = []
    for ds in datasets:
        count = (ds > threshold).sum(dim=['lat', 'lon']).to_dataframe()
        counts.append(count)
    if len(counts) > 1:
        count_df = pd.concat(counts, axis=1)
    else:
        count_df = count

    count_df = count_df.dropna()

    if rename_cols is None:
        return count_df
    else:
        try:
            count_df.columns = rename_cols
            return count_df
        except Exception as E:
            print(
                'rename_cols wrong format, pass'
                'list of names same length as list of datasets')
            print(E)
            print('returning dataframe with no renaming')
            return count_df


def average_over_domain(datasets, threshold=None, rename_cols=None):
    if type(datasets) != list:
        datasets = [datasets]
    averages = []
    for ds in datasets:
        if threshold is not None:
            average = (ds.where(ds > threshold)).mean(
                dim=['lat', 'lon']).to_dataframe()
        else:
            average = ds.mean(dim=['lat', 'lon']).to_dataframe()
        averages.append(average)

    if len(averages) > 1:
        average_df = pd.concat(averages, axis=1)
    else:
        average_df = average

    average_df = average_df.dropna()

    if rename_cols is None:
        return average_df
    else:
        try:
            average_df.columns = rename_cols
            return average_df
        except Exception as E:
            print(
                'rename_cols wrong format, pass list of'
                'names same length as list of datasets')
            print(E)
            print('returning dataframe with no renaming')
            return average_df


def calc_rel_bias(ds, char_pcp='radar'):
    """Calculate the relative bias between two variables in a given dataset.

    Args:
    - ds (xarray.Dataset): A dataset with two variables.
    - char_pcp (str, optional): A string that is a part of one of the variable names in the dataset that indicates the
    variable that represents precipitation. Default is 'radar'.

    Returns:
    - dict: A dictionary with the following key-value pair:
        - 'relative_bias': A float value that represents the relative bias between the two variables in the dataset.
    """
    fields = list(ds.keys())
    if len(fields) != 2:
        raise ValueError('for this function to work a dataset with two variables must be input, here there are ' +
                         str(len(fields)) + ' fields')

    idx_pcp = [i for i in range(0, len(fields)) if char_pcp in fields[i]][0]
    field1 = fields.pop(idx_pcp)
    field2 = fields[0]

    f1_full = ds[field1].sum()
    f2_full = ds[field2].sum()

    rel_bias = {'relative_bias': np.round(
        (((f2_full - f1_full) / f1_full)).values, decimals=4)}
    return rel_bias


def calculate_relative_bias(observed, simulated):
    sum_obs = np.sum(observed)
    sum_sim = np.sum(simulated)
    bias = np.round((sum_sim - sum_obs)/sum_obs, decimals=4)
    return bias


def calc_pixel_wise_contingency(ds, char_pcp='radar', threshold=2):
    """Calculate the probability of detection (POD) and false alarm rate (FAR) for a given threshold between two variables
    in a given dataset pixel-wise.

    Args:
    - ds (xarray.Dataset): A dataset with two variables.
    - char_pcp (str, optional): A string that is a part of one
    of the variable names in the dataset that indicates the
    variable that represents precipitation. Default is 'radar'.
    - threshold (int or float or list of ints or floats or numpy.ndarray
    of ints or floats, optional): A threshold value(s)
    to use for calculation of POD and FAR. Default is 2.

    Returns:
    - pandas.DataFrame: A dataframe with the following columns:
        - 'POD': A float value that represents the probability of detection for each threshold.
        - 'FAR': A float value that represents the false alarm rate for each threshold.
        The dataframe is indexed by the threshold value.

    Raises:
    - ValueError: If the input dataset does not have two variables.
    """
    fields = list(ds.keys())
    if len(fields) != 2:
        raise ValueError('for this function to work a dataset with two variables must be input, here there are ' +
                         str(len(fields)) + ' fields')

    idx_pcp = [i for i in range(0, len(fields)) if char_pcp in fields[i]][0]
    field1 = fields.pop(idx_pcp)
    field2 = fields[0]
    ok = False
    if isinstance(threshold, list):
        ok = True
    if isinstance(threshold, np.ndarray):
        ok = True
    if not ok:
        threshold = [threshold]
    pods = []
    fars = []
    for t in threshold:
        mask_pod = ds[field1] > t
        mask_far = ds[field1] < t

        detect_sat = (ds[field2].where(mask_pod) > t).sum()
        pod = detect_sat / mask_pod.sum()

        false_sat = (ds[field2].where(mask_far) > t).sum()
        far = 1 - false_sat / mask_far.sum()

        fars.append(far.values)
        pods.append(pod.values)
    out_dict = {'threshold': threshold,
                'POD': np.array(pods),
                'FAR': np.array(fars)}

    df = pd.DataFrame.from_dict(out_dict)
    df = df.set_index('threshold')
    return df


def calc_area_wise_contingency(ds, char_pcp='radar', threshold=2):
    """Calculate the probability of detection (POD) and false alarm rate (FAR) for a given threshold between two variables
    in a given dataset area-wise.

    Args:
    - ds (xarray.Dataset): A dataset with two variables.
    - char_pcp (str, optional): A string that is a part of one of
    the variable names in the dataset that indicates the
    variable that represents precipitation. Default is 'radar'.
    - threshold (int or float or list of ints or floats or numpy.ndarray of
    ints or floats, optional): A threshold value(s)
    to use for calculation of POD and FAR. Default is 2.

    Returns:
    - pandas.DataFrame: A dataframe with the following columns:
        - 'POD': A float value that represents the probability of detection for each threshold.
        - 'FAR': A float value that represents the false alarm rate for each threshold.
        The dataframe is indexed by the threshold value.

    Raises:
    - ValueError: If the input dataset does not have two variables.
    """
    ds = ds.mean(dim=['lat', 'lon'])
    fields = list(ds.keys())
    if len(fields) != 2:
        raise ValueError('for this function to work a dataset with two variables must be input, here there are ' +
                         str(len(fields)) + ' fields')

    idx_pcp = [i for i in range(0, len(fields)) if char_pcp in fields[i]][0]
    field1 = fields.pop(idx_pcp)
    field2 = fields[0]
    ok = False
    if isinstance(threshold, list):
        ok = True
    if isinstance(threshold, np.ndarray):
        ok = True
    if not ok:
        threshold = [threshold]
    pods = []
    fars = []
    for t in threshold:
        mask_pod = ds[field1] > t
        mask_far = ds[field1] < t

        detect_sat = (ds[field2].where(mask_pod) > t).sum()
        pod = detect_sat / mask_pod.sum()

        false_sat = (ds[field2].where(mask_far) > t).sum()
        far = false_sat / mask_far.sum()

        fars.append(far.values)
        pods.append(pod.values)
    out_dict = {'threshold': threshold,
                'POD': np.array(pods),
                'FAR': np.array(fars)}

    df = pd.DataFrame.from_dict(out_dict)
    df = df.set_index('threshold')
    return df


def score_ds(ds, char_pcp='radar', threshold=2, name=None, score_threshold=False):
    field = [f for f in ds.keys() if char_pcp in f][0]
    field2 = [f for f in ds.keys() if char_pcp not in f][0]
    if score_threshold:
        ds = ds.where(ds[field] > threshold)
    dict_out = {'ds': ds}
    # lin_reg = lin_reg_stats(ds, char_pcp, threshold)
    # dict_out.update(lin_reg)
    if isinstance(threshold, (int, float)):
        dict_out.update(calc_contingency(ds, mode='pixel', char_pcp=char_pcp,
                        threshold=threshold).loc[threshold, :].to_dict())
    else:
        dict_out.update({'contingency': calc_pixel_wise_contingency(
            ds, char_pcp=char_pcp, threshold=threshold)})
    dict_out.update({'valid_px': ds[field][0].count().values})
    dict_out.update({'area': ds[field][0].count().values * 100})
    dict_out.update(calc_rel_bias(ds, char_pcp=char_pcp))
    dict_out.update(kge_ds(ds, char_pcp=char_pcp))
    # new
    dict_out.update({'pearson_r_px': np.round(xs.pearson_r(
        ds[field], ds[field2], skipna=True).data, decimals=3)})
    dsm = ds.mean(dim=['lat', 'lon'])
    dict_out.update({'pearson_r_ar': np.round(xs.pearson_r(
        dsm[field], dsm[field2], skipna=True).data, decimals=3)})
    dict_out.update(
        {'corr_count_above_threshol': calc_corr_count_above_threshold(ds, threshold)})
    if name is None:
        name = field2
    dicto = {name: dict_out}

    df_out = pd.DataFrame.from_dict(dicto)
    return df_out.T


def score_ds_multiple_arrays(ds, char_pcp='radar', threshold=2, name=None, score_threshold=False):
    dfs = []
    fields = list(ds.keys())
    idx_pcp = [i for i in range(0, len(fields)) if char_pcp in fields[i]][0]
    field1 = fields.pop(idx_pcp)
    # field2 = fields[0]
    for field in fields:
        ds_single = ds[[field1, field]]
        df_score = score_ds(ds_single, char_pcp=char_pcp, threshold=threshold,
                            name=name, score_threshold=score_threshold)
        dfs.append(df_score)
    if len(dfs) == 1:
        return df_score
    elif len(dfs) >= 2:
        return pd.concat(dfs)
    else:
        raise ValueError('Something wrong')


def calc_corr_count_above_threshold(ds, threshold=0):
    """
    Calculates the Pearson correlation coefficient of the support
    above a threshold for between two variables in a dataset.

    Args:
        ds (xarray.Dataset): A two-dimensional Dataset object.
        threshold (int, float or array-like): threshold.

    Returns:
        float or array-like: The Pearson correlation coefficient between the
        two datasets after masking out values below the specified threshold(s).
        If a list or array-like object is passed as threshold, returns an
        array of correlation coefficients corresponding to each threshold value.

    """
    if isinstance(threshold, (list, np.ndarray)):
        corrs = []
        for t in threshold:
            corr = calc_corr_count_above_threshold(ds, t)
            corrs.append(corr)
        return corrs
    corr = ds.where(ds > threshold).sum(
        dim=['lat', 'lon']).to_dataframe().corr().iloc[0, 1]
    corr = np.round(corr, decimals=3)
    return corr


def kge_ds(ds, char_pcp='radar', **kwargs):
    """
    Calculates the Kling-Gupta Efficiency (KGE) for a Dataset
    object containing two variables.

    Args:
        ds (xarray.Dataset): A two-dimensional Dataset object.
        char_pcp (str): A string that is contained in the name of the variable
            that is considered the reference variable. Default is 'radar'.
        **kwargs: Additional keyword arguments passed to the kge() function.

    Returns:
        dict: A dictionary containing the KGE value.

    Raises:
        ValueError: If the input dataset does not contain exactly two variables.

    """
    fields = list(ds.keys())
    if len(fields) != 2:
        raise ValueError('for this function to work a dataset with two variables must be input, here there are ' +
                         str(len(fields)) + ' fields')

    idx_pcp = [i for i in range(0, len(fields)) if char_pcp in fields[i]][0]
    field1 = fields.pop(idx_pcp)
    field2 = fields[0]
    y = ds.to_dataframe()[field2].to_numpy()
    x = ds.to_dataframe()[field1].to_numpy()
    d_out = {'kge': np.round(kge(y, x), decimals=3)}
    return d_out


def calc_contingency(ds, mode='pixel', char_pcp='radar', threshold=2):
    if mode == 'area':
        ds = ds.mean(dim=['lat', 'lon'])

    fields = list(ds.keys())
    if len(fields) != 2:
        raise ValueError('for this function to work a dataset with two variables must be input, here there are ' +
                         str(len(fields)) + ' fields')

    idx_pcp = [i for i in range(0, len(fields)) if char_pcp in fields[i]][0]
    field1 = fields.pop(idx_pcp)
    field2 = fields[0]
    ok = False
    if isinstance(threshold, list):
        ok = True
    if isinstance(threshold, np.ndarray):
        ok = True
    if not ok:
        threshold = [threshold]
    pods = []
    fars = []
    for t in threshold:
        mask_pod = ds[field1] > t
        mask_far = ds[field1] < t

        detect_sat = (ds[field2].where(mask_pod) > t).sum()
        pod = detect_sat / mask_pod.sum()

        false_sat = (ds[field2].where(mask_far) > t).sum()
        far = false_sat / mask_far.sum()

        fars.append(far.values)
        pods.append(pod.values)
    out_dict = {'threshold': threshold,
                'POD_' + mode: np.array(pods),
                'FAR_' + mode: np.array(fars)}

    df = pd.DataFrame.from_dict(out_dict)
    df = df.set_index('threshold')
    return df
