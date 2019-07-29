import os
import pandas as pd
import numpy as np
import random
from constants import *
from tslearn.utils import to_time_series_dataset
from scipy.signal import argrelextrema

def print_whole_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

def str_to_seconds(timestamp_string):
    if type(timestamp_string) != str and type(timestamp_string) != float and type(timestamp_string) != int:
        timestamp_string = timestamp_string.values[0]  # in case of using numpy apply
    res = -1
    if type(timestamp_string) == float or type(timestamp_string) == int or len(timestamp_string) <= 2:  # case '-1'
        pass
    elif len(timestamp_string) == 5:  # case 'MM:SS'
        m, s = timestamp_string.split(":")
        res = int(m) * 60 + int(s)
    elif len(timestamp_string) == 8:  # case 'MM:SS:MSMS'
        m, s, _ = timestamp_string.split(":")
        res = int(m) * 60 + int(s)
    else:
        res = np.nan
    return res

def get_file_paths(path_to_main_folder):
    contents = sorted(os.listdir(path_to_main_folder))
    file_paths = [path_to_main_folder + "/" + i for i in contents if ".txt" in i]
    return file_paths

def parse_df_splits_from_raw(raw_time_series):
    skip_index = raw_time_series.index[raw_time_series[0] == DIVIDE_MARKER].tolist()
    dfs = []
    if len(skip_index) > 0:
        for i in range(len(skip_index)):
            if i == 0:
                dfs.append(raw_time_series.iloc[0: skip_index[i], :])
            if i == len(skip_index) - 1:
                dfs.append(raw_time_series.iloc[skip_index[i] + SKIP_ROWS:, :])
            if i != 0 and i != len(skip_index) - 1 and len(skip_index) >= 3:
                dfs.append(raw_time_series.iloc[skip_index[i - 1] + SKIP_ROWS: skip_index[i], :])
    else:
        dfs = [raw_time_series]
    return dfs

def coerce_invalid_col_values_from_df(col_i, df):
    col = df[df.columns[col_i]]
    col = pd.to_numeric(col, errors='coerce')
    col.fillna(method="pad")  # use the last available value to fill NaNs
    return col

def standardize_col(col, col_i):
    if col_i != 0:  # the first column is the timestamp, which doesn't need to be normalized
        col = (col - col.mean()) / col.std()  # standardize column
    return col

def rectify_time_diff(timestamps_df):
    timestamps_df.iloc[:, TIME_DIFF_INDEX] = timestamps_df.iloc[:, TIME_DIFF_INDEX].apply(str_to_seconds)
    start_time_col = timestamps_df.iloc[:, ROW_MAPPER_CUTOFF_INDEX].apply(str_to_seconds)
    for column in timestamps_df.iloc[:, ROW_MAPPER_CUTOFF_INDEX:].columns:
        timestamps_df[column] = timestamps_df[column].apply(str_to_seconds)
        # timestamps_df[column] = timestamps_df[column] - time_diff_col
        timestamps_df[column] = timestamps_df[column] - start_time_col
    return timestamps_df

def extract_mus_data_in_time_range(df, start_time, end_time):
    start_index = df.iloc[:, TIMESTAMP_INDEX].searchsorted(start_time)[0]
    end_index = df.iloc[:, TIMESTAMP_INDEX].searchsorted(end_time)[0]
    df_slice = df.iloc[start_index : end_index]
    mus1_data = df_slice.iloc[:, MUS1_INDEX].values
    mus2_data = df_slice.iloc[:, MUS2_INDEX].values
    mus3_data = df_slice.iloc[:, MUS3_INDEX].values
    mus4_data = df_slice.iloc[:, MUS4_INDEX].values
    mus5_data = df_slice.iloc[:, MUS5_INDEX].values
    mus6_data = df_slice.iloc[:, MUS6_INDEX].values
    return mus1_data, mus2_data, mus3_data, mus4_data, mus5_data, mus6_data

def downsample_ts_dict(ts_dict, MUS_key, downsampling_divisor=DOWNSAMPLING_DIVISOR):
    # mus_data = get_minmax_pts_only(ts_dict, MUS_key)
    mus_data = ts_dict[MUS_key]
    mus_data_downsampled = _downsample_to_min_len(mus_data, downsampling_divisor)
    return mus_data_downsampled

# def get_minmax_pts_only(ts_dict, MUS_key):
    # mus_data = []
    # for knot in ts_dict[MUS_key]:
    #     # TAKE MINMAX
    #     # for local maxima
    #     max_inds = argrelextrema(knot.astype(float), np.greater)[0]
    #     # for local minima
    #     min_inds = argrelextrema(knot.astype(float), np.less)[0]
    #     minmax_inds = np.union1d(max_inds, min_inds)
    #     if len(minmax_inds) > 0:
    #         mus_data.append(knot[minmax_inds])
    #     else:
    #         mus_data.append(knot)
    # return mus_data
    # return ts_dict[MUS_key]

def _downsample_to_min_len(mus_data, downsampling_divisor=DOWNSAMPLING_DIVISOR):
    min_len = min([len(knot) for knot in mus_data])
    mus_data_downsampled = []
    for knot in mus_data:
        x_to_subsample = np.array(range(0, len(knot), int(len(knot)/min_len*downsampling_divisor)), dtype='float64')
        x = np.array(range(len(knot)), dtype='float64')
        subsample = np.interp(x_to_subsample, x, np.array(knot, dtype='float64'))
        subsample = fillna_numpy_array(subsample)
        mus_data_downsampled.append(subsample)
    return mus_data_downsampled

def fillna_numpy_array(subsample):
    subsample_df = pd.DataFrame(subsample).fillna(method="ffill")
    subsample = subsample_df.iloc[:, 0].values
    return subsample

def append_mus_data_to_dict(ts_dict,
                            mus1_data, mus2_data, mus3_data,
                            mus4_data, mus5_data, mus6_data):
    ts_dict[MUS1].append(mus1_data)
    ts_dict[MUS2].append(mus2_data)
    ts_dict[MUS3].append(mus3_data)
    ts_dict[MUS4].append(mus4_data)
    ts_dict[MUS5].append(mus5_data)
    ts_dict[MUS6].append(mus6_data)

def convert_mus_data_to_time_series(ts_dict):
    ts_dict[MUS1] = np.nan_to_num(to_time_series_dataset(ts_dict[MUS1]))
    ts_dict[MUS2] = np.nan_to_num(to_time_series_dataset(ts_dict[MUS2]))
    ts_dict[MUS3] = np.nan_to_num(to_time_series_dataset(ts_dict[MUS3]))
    ts_dict[MUS4] = np.nan_to_num(to_time_series_dataset(ts_dict[MUS4]))
    ts_dict[MUS5] = np.nan_to_num(to_time_series_dataset(ts_dict[MUS5]))
    ts_dict[MUS6] = np.nan_to_num(to_time_series_dataset(ts_dict[MUS6]))

def shuffle_data_general(timestamps, data, ratings=[]):
    shuffled_order = list(range(len(timestamps)))
    random.seed(30)
    random.shuffle(shuffled_order)
    shuff_ts = [timestamps[i] for i in shuffled_order]

    shuff_ratings = ratings
    if len(ratings) > 0:
        shuff_ratings = [ratings[i] for i in shuffled_order]

    shuff_data = {}
    shuff_data[MUS1] = [data[MUS1][i] for i in shuffled_order]
    shuff_data[MUS2] = [data[MUS2][i] for i in shuffled_order]
    shuff_data[MUS3] = [data[MUS3][i] for i in shuffled_order]
    shuff_data[MUS4] = [data[MUS4][i] for i in shuffled_order]
    shuff_data[MUS5] = [data[MUS5][i] for i in shuffled_order]
    shuff_data[MUS6] = [data[MUS6][i] for i in shuffled_order]

    return shuff_ts, shuff_data, shuff_ratings