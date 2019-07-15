"""
The preprocessing required from the data is as follows:
1. Read each time series in each text file as a dataframe
2. Filter out NaN rows  !!! NOTE THIS MAKES THE TIME SERIES OF VARIABLE LENGTH !!!
3. Normalize each of data columns based on values in that column, ignoring the first column
4. (more operations may be needed)
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing  # used for normalization
import os
from constants import *
from utilities import *

class EMGDataManager:
    def __init__(self, path_to_main_folder, path_to_timestamps, path_to_ratings=None):
        self.path_to_main_folder = path_to_main_folder
        self.path_to_timestamps = path_to_timestamps
        self.paths_to_texts = self._get_file_paths(self.path_to_main_folder)
        # TODO: path to ratings not implemented rn

        # each of the keys corresponds to a 2D numpy array where
        #   each array is a time series
        self.ROB_datasets = {MUS1: None, MUS2: None, MUS3: None, MUS4: None, MUS5: None, MUS6:None}
        self.LAP_datasets = {MUS1: None, MUS2: None, MUS3: None, MUS4: None, MUS5: None, MUS6:None}
        self.ROB_datasets_times = {MUS1: None, MUS2: None, MUS3: None, MUS4: None, MUS5: None, MUS6:None}
        self.LAP_datasets_times = {MUS1: None, MUS2: None, MUS3: None, MUS4: None, MUS5: None, MUS6: None}
        self.ROB_datasets_ds = {MUS1: None, MUS2: None, MUS3: None, MUS4: None, MUS5: None, MUS6: None} # downsampled
        self.LAP_datasets_ds = {MUS1: None, MUS2: None, MUS3: None, MUS4: None, MUS5: None, MUS6: None}
        self.ROB_ratings = []
        self.LAP_ratings = []
        # each of the keys correpsonds to an array of dicts where
        #   each dict is like {ACT1: None, ACT2: None, etc} and corresponds to one knot
        self.ROB_action_labels = []
        self.LAP_action_labels = []

        self.row_map = None  # Dataframe of the first four columns of the ratings
        self.actions = []  # array of strings

        self.preprocess()

    def get_ROB_data(self):
        return self.ROB_datasets, self.ROB_ratings, self.ROB_action_labels

    def get_LAP_ratings(self):
        return self.LAP_datasets, self.LAP_ratings, self.LAP_action_labels

    def get_row_map(self):
        return self.row_map

    def preprocess(self):
        self._preprocess_timestamps()  # this has to run first because this inits the row map
        data_file_paths = self._get_file_paths(self.path_to_main_folder)
        data_index = 0
        for data_file in data_file_paths:
            raw_time_series = pd.read_table(data_file, skiprows=SKIP_ROWS, header=None,
                                            usecols=COLUMNS_USED, encoding="mac-roman")
            dfs = self._parse_df_splits_from_raw(raw_time_series)
            for df in dfs:
                # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
                #     print(df)
                df = self._trim_time_diff(data_index, df)
                df = self._preprocess_column_values(df)
                # TODO: further split df on each knot that is recorded thru row_map
                data_index += 1
                for col_i in range(1, len(df.columns)):
                    if self.row_map.iloc[data_index][ROB_OR_LAP_INDEX] == ROB:
                        pass
                    elif self.row_map.iloc[data_index][ROB_OR_LAP_INDEX] == LAP:
                        pass
        #     TODO: preprocess ratings

    def _trim_time_diff(self, data_index, df):
        time_diff = self.row_map.iloc[data_index][TIME_DIFF_INDEX]
        pass_index = 0
        for index, row in df.iterrows():
            if float(row[0]) > time_diff:
                pass_index = index
                break
        df = df.iloc[pass_index:]
        return df

    def _preprocess_column_values(self, df):
        for col_i in range(len(df.columns)):
            col = self._coerce_invalid_col_values_from_df(col_i, df)
            col = self._standardize_col(col, col_i)
            df[df.columns[col_i]] = col
            return df

    def _standardize_col(self, col, col_i):
        if col_i != 0:  # the first column is the timestamp, which doesn't need to be normalized
            col = (col - col.mean()) / col.std()  # standardize column
        return col

    def _coerce_invalid_col_values_from_df(self, col_i, df):
        col = df[df.columns[col_i]]
        col = pd.to_numeric(col, errors='coerce')
        col.fillna(method="pad")  # use the last available value to fill NaNs
        return col

    def _parse_df_splits_from_raw(self, raw_time_series):
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

    def _preprocess_timestamps(self):
        timestamps_df = pd.read_csv(self.path_to_timestamps)
        self._preprocess_row_map(timestamps_df)
        self._preprocess_actions(timestamps_df)
        timestamps_sets = timestamps_df.iloc[:, ROW_MAPPER_CUTOFF_INDEX:].to_dict('index')
        for timestamp_index in range(len(timestamps_sets)):
            timestamps = timestamps_sets[timestamp_index]
            for action, time_str in timestamps.items():
                time = str_to_seconds(time_str)  # convert to seconds
                time = time - int(self.row_map.iloc[timestamp_index][INITIAL_OFFSET_INDEX]) # sub initial offset
                timestamps[action] = time
            if self.row_map.iloc[timestamp_index][ROB_OR_LAP_INDEX] == ROB:
                self.ROB_action_labels.append(timestamps)
            elif self.row_map.iloc[timestamp_index][ROB_OR_LAP_INDEX] == LAP:
                self.LAP_action_labels.append(timestamps)

    def _preprocess_row_map(self, timestamps_df):
        self.row_map = timestamps_df.iloc[:, :ROW_MAPPER_CUTOFF_INDEX]  # contains the row mapping data for later preprocessing
        self.row_map.iloc[:, INITIAL_OFFSET_INDEX:] = self.row_map.iloc[:, INITIAL_OFFSET_INDEX:].apply(str_to_seconds, axis=1)  # change the initial offset to seconds

    def _preprocess_actions(self, timestamps_df):
        self.actions = timestamps_df.columns[ROW_MAPPER_CUTOFF_INDEX:]

    def _get_file_paths(self, path_to_main_folder):
        contents = os.listdir(path_to_main_folder)
        file_paths = [path_to_main_folder + "/" + i for i in contents if ".txt" in i]
        return file_paths