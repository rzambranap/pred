import numpy as np
import xarray as xr


def split_dataset(ds,
                  training=0.8,
                  testing=0.2,
                  dim='time',
                  method='random',
                  **kwargs):
    """
    Split a dataset into training and testing sets.

    Parameters
    ----------
    ds: xarray.Dataset
        The dataset to be split.
    training: float
        The ratio of the dataset to use for training.
    testing: float
        The ratio of the dataset to use for testing.
    dim: str
        The dimension along which the dataset will be split.
    method: {'random', 'sequential'}
        The method used to split the dataset.
    kwargs:
        Additional arguments to be passed to the method.
    Returns
    -------
    train_ds: xarray.Dataset
        The training set of the dataset.
    test_ds: xarray.Dataset
        The testing set of the dataset.
    """

    valid_methods = ['random', 'sequential']
    if method not in valid_methods:
        raise ValueError(f'Unknown method {method}, valid methods are {valid_methods}')
    sum_ratios = training + testing
    if np.abs(sum_ratios - 1) > 0.001:
        raise ValueError(f'sum of set_ratios != 1, = {sum_ratios}')
    else:
        ds = ds.copy()
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f'The object ds must be an xarray.Dataset, it is a {type(ds)}')
    nsamples = ds['time'].size
    split_ratios = [training, testing]
    train_elts, test_elts = [np.floor(i * nsamples).astype(int) for i in split_ratios]
    print(train_elts, test_elts)
    if not all(x > 0 for x in [train_elts, test_elts]):
        raise ValueError(f'Not enough elements, splits of {train_elts, test_elts}')
    if method == 'sequential':
        train_idx = ds['time'][:train_elts]
        test_idx = ds['time'][train_elts:train_elts + test_elts]
    if method == 'random':
        tidcs = ds['time'].copy().to_numpy()
        if 'seed' in kwargs:
            nseed = kwargs['seed']
            np.random.seed(nseed)
        np.random.shuffle(tidcs)
        train_idx = tidcs[:train_elts]
        test_idx = tidcs[train_elts:train_elts + test_elts]
    train_ds = ds.sel(time=train_idx)
    test_ds = ds.sel(time=test_idx)

    return train_ds, test_ds
