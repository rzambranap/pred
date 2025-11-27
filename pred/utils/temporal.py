import pandas as pd
import numpy as np


def align_dataframes_by_time_fixed(df1, df2, time_diff_minutes=6):
    """
    Aligns two pandas DataFrames based on their datetime indices, within a specified time difference.

    Parameters:
    - df1, df2: pandas DataFrames with datetime indices.
    - time_diff_minutes: The maximum time difference in minutes for which rows should be considered aligned.

    Returns:
    - Two pandas DataFrames with aligned indices based on the average (midpoint) between the matching indices
    of the input DataFrames.

    Example usage:
    >>> dates1 = pd.date_range('2024-01-01 00:00:00', periods=5, freq='T')  # Every minute
    >>> dates2 = pd.date_range('2024-01-01 00:02:30', periods=5, freq='T')  # Every minute, starting 2.5 minutes later
    >>> df1 = pd.DataFrame({"A": range(5)}, index=dates1)
    >>> df2 = pd.DataFrame({"B": range(5, 10)}, index=dates2)
    >>> aligned_df1, aligned_df2 = align_dataframes_by_time_fixed(df1, df2)
    """
    # Find insertion points for df2's indices into df1's indices array
    idx = np.searchsorted(df1.index.to_numpy(), df2.index.to_numpy())
    # Ensure idx does not go out of bounds for df1
    idx = np.minimum(idx, len(df1.index) - 1)

    # Creating DataFrame with corresponding indices from both DataFrames
    _dict = {"t1": df1.index.to_numpy()[idx], "t2": df2.index.to_numpy()}
    df = pd.DataFrame(_dict)

    # Calculate time differences in minutes between paired indices
    diff = [(item.total_seconds() / 60) for item in (df.t1 - df.t2).tolist()]
    df.loc[:, 'diff'] = diff

    # Filter pairs of indices by the specified time difference
    mask = df.loc[:, 'diff'].abs() <= time_diff_minutes
    selection = df.loc[mask]

    # Select rows from the original DataFrames based on the filtered indices
    sel_df1 = df1.loc[selection.t1.tolist()]
    sel_df2 = df2.loc[selection.t2.tolist()]

    # Align indices of the selected rows to their average (midpoint)
    idx = sel_df1.index + (sel_df1.index - sel_df2.index) / 2
    sel_df1.index, sel_df2.index = idx, idx

    return sel_df1, sel_df2


def match_closest_time_indexes(df1, df2):
    matched_df = pd.DataFrame(columns=df1.columns)

    for idx1, row1 in df1.iterrows():
        closest_idx2 = (df2.index - idx1).argmin()  # Find index with minimum time difference
        matched_row = pd.concat([row1, df2.iloc[closest_idx2]])
        matched_df = matched_df.append(matched_row, ignore_index=True)

    matched_df.index = df1.index
    return matched_df
