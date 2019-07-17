import os
import pandas as pd
from constants import *
from tslearn.utils import to_time_series_dataset
from scipy.signal import resample

def _print_whole_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

def str_to_seconds(timestamp_string):
    if type(timestamp_string) != str:
        timestamp_string = timestamp_string.values[0]  # in case of using numpy apply
    res = -1
    if len(timestamp_string) <= 2:  # case '-1'
        pass
    elif len(timestamp_string) == 5:  # case 'MM:SS'
        m, s = timestamp_string.split(":")
        res = int(m) * 60 + int(s)
    elif len(timestamp_string) == 7:  # case 'MM:SS:MSMS'
        m, s, _ = timestamp_string.split(":")
        res = int(m) * 60 + int(s)
    else:
        res = float(timestamp_string)
    return res

def get_file_paths(path_to_main_folder):
    contents = os.listdir(path_to_main_folder)
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
    for column in timestamps_df.iloc[:, ROW_MAPPER_CUTOFF_INDEX:].columns:
        timestamps_df[column] = timestamps_df[column].apply(str_to_seconds)
        timestamps_df[column] = timestamps_df[column] - timestamps_df.iloc[:, TIME_DIFF_INDEX]
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

def downsample_ts_dict(ts_dict, MUS_key):
    min_len = min([len(knot) for knot in ts_dict[MUS_key]])
    mus_data_downsampled = []
    for knot in ts_dict[MUS_key]:
        subsample = resample(knot, min_len)
        mus_data_downsampled.append(subsample)
    return mus_data_downsampled

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
    ts_dict[MUS1] = to_time_series_dataset(ts_dict[MUS1])
    ts_dict[MUS2] = to_time_series_dataset(ts_dict[MUS2])
    ts_dict[MUS3] = to_time_series_dataset(ts_dict[MUS3])
    ts_dict[MUS4] = to_time_series_dataset(ts_dict[MUS4])
    ts_dict[MUS5] = to_time_series_dataset(ts_dict[MUS5])
    ts_dict[MUS6] = to_time_series_dataset(ts_dict[MUS6])