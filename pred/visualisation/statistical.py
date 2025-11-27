import scipy
import matplotlib.pyplot as plt
import numpy as np


def plot_fit(x, y, ax=None, print_vals=False):
    """
    Plots a linear fit and the R-squared value on the given axes.

    Parameters
    ----------
    x : array-like
        The x-axis data with no nans or infs.
    y : array-like
        The y-axis data with no nans or infs.
    ax : matplotlib.axes.Axes, optional
        The axes to plot the fit on. If not provided, the current axes will be used.
    print_vals : bool, optional
        If True, the slope, intercept, R-value, p-value, and standard error will be printed.

    Returns
    -------
    slope : float
        The slope of the fitted line.
    """
    try:
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
            x, y)
        xmin = np.min(x)
        xmax = np.max(x)
        xvec = np.linspace(xmin, xmax, 50)
        yvec = slope * xvec + intercept
        if print_vals:
            print('rvalue', r_value)
            print('slope', slope)
            print('intercept', intercept)
            print('pvalue', p_value)
            print('std_errr', std_err)
        if ax is None:
            ax = plt.gca()
        label = 'y = ' + f'{slope:0.2f}' + 'x + ' + f'{intercept:0.2f}' + '\nr² = ' + f'{r_value:0.3f}'
        ax.plot(xvec, yvec, c='r', linestyle='--', label=label)
        ax.plot(np.linspace(0, xmax, 10), np.linspace(0, xmax, 10), c='b')
        return slope
    except Exception as E:
        print('wea hermano')
        print(E)
        return 0
