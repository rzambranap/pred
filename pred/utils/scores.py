import numpy as np
import xarray as xr
import pandas as pd


def calculate_correlation(a: np.ndarray, b: np.ndarray, skipna: bool = False) -> float:
    if skipna:
        if a.shape != b.shape:
            raise ValueError(f"a and b of different shapes: {a.shape}, {b.shape}")
        df = pd.DataFrame.from_dict({'a': a, 'b': b}).dropna()
        return calculate_correlation(df.a.to_numpy(), df.b.to_numpy(), skipna=False)
    return {'correlation': np.corrcoef(a, b)[0, 1]}


def calculate_rmse(observed: np.ndarray, simulated: np.ndarray, skipna: bool = False) -> float:
    if skipna:
        if observed.shape != simulated.shape:
            raise ValueError(f"a and b of different shapes: {observed.shape}, {simulated.shape}")
        df = pd.DataFrame.from_dict({'observed': observed, 'simulated': simulated}).dropna()
        return calculate_rmse(df.observed.to_numpy(), df.simulated.to_numpy())
    rmse = np.sqrt(np.mean((observed-simulated)**2))
    return {'rmse': rmse}


def calculate_relative_bias(observed: np.ndarray, simulated: np.ndarray, skipna: bool = False) -> float:
    if skipna:
        if observed.shape != simulated.shape:
            raise ValueError(f"a and b of different shapes: {observed.shape}, {simulated.shape}")
        df = pd.DataFrame.from_dict({'observed': observed, 'simulated': simulated}).dropna()
        return calculate_relative_bias(df.observed.to_numpy(), df.simulated.to_numpy())
    sum_obs = np.sum(observed)
    sum_sim = np.sum(simulated)
    bias = (sum_sim - sum_obs) / sum_obs
    return {'relative_bias': bias}


def calculate_kge(observed: np.ndarray, simulated: np.ndarray, skipna: bool = False) -> float:
    """
    Calculate Kling-Gupta Efficiency (KGE) between two numpy arrays.

    Parameters:
        a (numpy.ndarray): Observed values.
        b (numpy.ndarray): Simulated values.

    Returns:
        float: Kling-Gupta Efficiency (KGE).
    """
    if skipna:
        if observed.shape != simulated.shape:
            raise ValueError(f"a and b of different shapes: {observed.shape}, {simulated.shape}")
        df = pd.DataFrame.from_dict({'observed': observed, 'simulated': simulated}).dropna()
        return calculate_kge(df.observed.to_numpy(), df.simulated.to_numpy())
    a = observed
    b = simulated
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    std_a = np.std(a)
    std_b = np.std(b)
    corr = np.corrcoef(a, b)[0, 1]

    kge = 1 - np.sqrt((corr - 1) ** 2 + (std_a / mean_a - std_b / mean_b) ** 2 + (mean_a / mean_b - 1) ** 2)
    return {'kge': kge}


def calculate_contingency_scores(obs, sim, thresholds,
                                 detection_threshold=0.2,
                                 export_confusion_matrix=False,
                                 threshold_is_global=False,
                                 skipna=False):
    # Ensure thresholds is a list or array (it could be a single scalar)
    args = [thresholds]
    kwargs = {'detection_threshold': detection_threshold,
              'export_confusion_matrix': export_confusion_matrix,
              'threshold_is_global': threshold_is_global,
              'skipna': skipna}
    if skipna:
        if obs.shape != sim.shape:
            raise ValueError(f"a and b of different shapes: {obs.shape}, {sim.shape}")
        df = pd.DataFrame.from_dict({'observed': obs, 'simulated': sim}).dropna()
        kwargs['skipna'] = False
        return calculate_contingency_scores(df.observed.to_numpy(), df.simulated.to_numpy(), *args, **kwargs)

    if not isinstance(thresholds, (list, np.ndarray)):
        thresholds = [thresholds]

    results = []  # To store results for each threshold
    confusion_matrices = {}  # To store confusion matrices if export_confusion_matrix is True

    # Loop over each threshold
    for threshold in thresholds:
        if threshold_is_global:
            detection_threshold = threshold
        obs_true = obs > threshold
        sim_true = sim > detection_threshold

        obs_neg = obs < threshold
        sim_neg = sim < detection_threshold

        true_positives = np.sum(obs_true * sim_true)
        true_negatives = np.sum(obs_neg * sim_neg)

        false_positives = np.sum(sim_true * obs_neg)
        false_negatives = np.sum(sim_neg * obs_true)

        observed_true_col = [true_positives, false_negatives]
        observed_neg_col = [true_negatives, false_positives]

        cmatrix_dict = {'observed_true': observed_true_col,
                        'observed_false': observed_neg_col}

        index_col = ['predicted_true', 'predicted_false']

        confusion_matrix = pd.DataFrame.from_dict(cmatrix_dict)
        confusion_matrix.index = index_col

        # Calculate the metrics
        pod = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else np.nan
        pond = true_negatives / (false_negatives + true_negatives) if (false_negatives + true_negatives) > 0 else np.nan
        far = false_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else np.nan

        scores_dict = {'threshold': threshold,
                       'pod': pod,
                       'pond': pond,
                       'far': far}

        results.append(scores_dict)

        if export_confusion_matrix:
            confusion_matrices[threshold] = confusion_matrix

    # Return results for all thresholds
    if export_confusion_matrix:
        return results, confusion_matrices
    if len(thresholds) == 1:
        return results[0]

    return results


def check_only_n_variables(a: xr.Dataset | pd.DataFrame, n: int = 2) -> bool:
    if isinstance(a, xr.Dataset):
        nvars = len(list(a.data_vars))
    elif isinstance(a, pd.DataFrame):
        nvars = len(a.columns)
    else:
        raise TypeError(f"a is {type(a)}, expecting xr.Dataset or pd.DataFrame")
    if nvars != n:
        raise ValueError(f"a has {nvars} variables, we were expecting {n}")
    return True


def apply_function_to_multi_input_dataset(input, function, reference_var, *args, **kwargs):
    if isinstance(input, xr.Dataset):
        dvars = list(input.data_vars)
    elif isinstance(input, pd.DataFrame):
        dvars = input.columns
    else:
        raise TypeError(f'Type {type(input)} is still unsupported')

    if reference_var not in dvars:
        raise ValueError(f"the reference variable {reference_var} is not in the dataset,\nthe variables are {dvars}")

    dvars_to_score = [var for var in dvars if var != reference_var]
    pairings = [[reference_var, dvar_to_score] for dvar_to_score in dvars_to_score]

    dvars_scores = {scored_var: None for scored_var in dvars_to_score}
    for pair, dvar_to_score in zip(pairings, dvars_to_score):
        score_result = apply_function_to_dual_input(input[pair], function, reference_var, *args, **kwargs)
        dvars_scores[dvar_to_score] = score_result
    return dvars_scores


def get_remaining_field(dataset, known_field):
    correct_variables = check_only_n_variables(dataset, 2)
    if isinstance(dataset, pd.DataFrame):
        # For pandas DataFrame
        all_columns = dataset.columns
    elif isinstance(dataset, xr.Dataset):
        # For xarray Dataset
        all_columns = dataset.data_vars  # This gets all variable names in xarray
    else:
        raise ValueError("Unsupported data type. Only pandas DataFrame or xarray Dataset are supported.")

    # Return remaining field(s)
    return [col for col in all_columns if col != known_field][0]


def apply_function_to_dual_input(input, function, reference_variable, *args, **kwargs):
    correct_variables = check_only_n_variables(input, 2)

    if isinstance(input, xr.Dataset):
        dvars = list(input.data_vars)
        input = input.to_dataframe().loc[:, dvars]

    observed = input[reference_variable]
    simulated = input[get_remaining_field(input, reference_variable)]

    result = function(observed, simulated, *args, **kwargs)
    return result


def score_rbias(ds, reference_var, **kwargs):
    score = apply_function_to_multi_input_dataset(ds, calculate_relative_bias, reference_var, **kwargs)
    return score


def score_rmse(ds, reference_var, **kwargs):
    score = apply_function_to_multi_input_dataset(ds, calculate_rmse, reference_var, **kwargs)
    return score


def score_kge(ds, reference_var, **kwargs):
    score = apply_function_to_multi_input_dataset(ds, calculate_kge, reference_var, **kwargs)
    return score


def score_contingency(ds, reference_var, threshold, **kwargs):
    score = apply_function_to_multi_input_dataset(ds, calculate_contingency_scores, reference_var, threshold, **kwargs)
    return score


def score_coeff_var_corr(ds, reference_var, **kwargs):
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"Only xr.Dataset is supported for this score, yours is {type(ds)}")
    cv_ds = calculate_CV_for_ds(ds)
    correls = apply_function_to_multi_input_dataset(cv_ds, calculate_correlation, 'rain_rate', skipna=True)
    result = change_deepest_keys(correls, {'correlation': 'coeff_var_correlation'})
    return result


def score_support_size_corr(ds, reference_var, threshold, **kwargs):
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"Only xr.Dataset is supported for this score, yours is {type(ds)}")
    ds_support_size = (ds > threshold).sum(dim=['lat', 'lon'])
    correls = apply_function_to_multi_input_dataset(ds_support_size, calculate_correlation, 'rain_rate', skipna=True)
    result = change_deepest_keys(correls, {'correlation': 'support_size_correlation'})
    return result


def score_spatial_avg_corr(ds, reference_var, **kwargs):
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"Only xr.Dataset is supported for this score, yours is {type(ds)}")
    ds_avg = ds.mean(dim=['lat', 'lon'])
    correls = apply_function_to_multi_input_dataset(ds_avg, calculate_correlation, 'rain_rate', skipna=True)
    result = change_deepest_keys(correls, {'correlation': 'spatial_avg_correlation'})
    return result


def calculate_CV_for_ds(ds, dim=['lat', 'lon']):
    ds_mean = ds.mean(dim=dim)
    ds_std = ds.std(dim=dim)
    ds_cv = ds_std / ds_mean
    return ds_cv


def change_deepest_keys(d, key_mapping):
    new_dict = {}
    for outer_key, inner_dict in d.items():
        new_inner_dict = {}
        for inner_key, value in inner_dict.items():
            # Replace old keys with new keys using the key_mapping dictionary
            new_key = key_mapping.get(inner_key, inner_key)  # Keep the same key if not in key_mapping
            new_inner_dict[new_key] = value
        new_dict[outer_key] = new_inner_dict
    return new_dict


def calculate_r2score(a: np.ndarray, b: np.ndarray, skipna: bool = False) -> dict:
    # a is obs
    # b is sim
    if skipna:
        if a.shape != b.shape:
            raise ValueError(f"a and b of different shapes: {a.shape}, {b.shape}")
        df = pd.DataFrame.from_dict({'a': a, 'b': b}).dropna()
        return calculate_r2score(df.a.to_numpy(), df.b.to_numpy(), skipna=False)
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return {'r2': r2}


def score_r2(ds, reference_var, **kwargs):
    score = apply_function_to_multi_input_dataset(ds, calculate_r2score, reference_var, **kwargs)
    return score


def score_correlation(ds, reference_var, **kwargs):
    score = apply_function_to_multi_input_dataset(ds, calculate_correlation, reference_var, **kwargs)
    return score
