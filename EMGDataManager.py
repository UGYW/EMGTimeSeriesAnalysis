"""
The preprocessing required from the data is as follows:
1. Read each time series in each text file as a dataframe
2. Filter out NaN rows  !!! NOTE THIS MAKES THE TIME SERIES OF VARIABLE LENGTH !!!
3. Normalize each of data columns based on values in that column, ignoring the first column
4. (more operations may be needed)
"""

import pandas as pd
import numpy as np
from tslearn.utils import to_time_series_dataset
import os
from constants import *
from utilities import *
import logging

class EMGDataManager:
    def __init__(self, path_to_main_folder, path_to_timestamps, path_to_ratings=None, downsampler=True):
        self.path_to_main_folder = path_to_main_folder
        self.path_to_timestamps = path_to_timestamps
        self.paths_to_texts = self._get_file_paths(self.path_to_main_folder)
        # TODO: path to ratings not implemented rn

        self.downsampler_active = downsampler  # determines whether or not to init downsampling

        # each of the keys corresponds to a 2D numpy array where
        #   each array is a time series
        self.ROB_datasets = {MUS1: [], MUS2: [], MUS3: [], MUS4: [], MUS5: [], MUS6: []}
        self.LAP_datasets = {MUS1: [], MUS2: [], MUS3: [], MUS4: [], MUS5: [], MUS6: []}
        self.ROB_datasets_times = {MUS1: [], MUS2: [], MUS3: [], MUS4: [], MUS5: [], MUS6: []}
        self.LAP_datasets_times = {MUS1: [], MUS2: [], MUS3: [], MUS4: [], MUS5: [], MUS6: []}
        self.ROB_datasets_downsampled = {MUS1: [], MUS2: [], MUS3: [], MUS4: [], MUS5: [], MUS6: []}
        self.LAP_datasets_downsampled = {MUS1: [], MUS2: [], MUS3: [], MUS4: [], MUS5: [], MUS6: []}
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
        for data_file_path in data_file_paths:
            logging.info("LOADING " + data_file_path)

            raw_time_series = pd.read_table(data_file_path, skiprows=SKIP_ROWS, header=None,
                                            usecols=COLUMNS_USED, encoding="mac-roman")
            data_file_key = os.path.splitext(os.path.basename(data_file_path))[0]
            dfs = self._parse_df_splits_from_raw(raw_time_series)

            for df in dfs:
                df = self._preprocess_column_values(df)
                df = self._trim_time_diff(data_index, df)

                rob_or_lap = self.row_map.iloc[data_index][KNOT_TYPE_INDEX]
                while data_index < len(self.row_map) and \
                    data_file_key == self.row_map.iloc[data_index][CODE_INDEX] and \
                    rob_or_lap == self.row_map.iloc[data_index][KNOT_TYPE_INDEX]:
                    print(data_file_key)
                    print(data_index)
                    print(rob_or_lap)
                    start_time = self.row_map.iloc[data_index][START_TIME_INDEX]
                    end_time = self.row_map.iloc[data_index][END_TIME_INDEX]

                    mus1_data, mus2_data, mus3_data, mus4_data, mus5_data, mus6_data = \
                        self._extract_mus_data_in_time_range(df, start_time, end_time)

                    if self.row_map.iloc[data_index][KNOT_TYPE_INDEX] == ROB:
                        self._load_time_series_to_dict(self.ROB_datasets, mus1_data, mus2_data, mus3_data,
                                                                          mus4_data, mus5_data, mus6_data)
                    elif self.row_map.iloc[data_index][KNOT_TYPE_INDEX] == LAP:
                        self._load_time_series_to_dict(self.LAP_datasets, mus1_data, mus2_data, mus3_data,
                                                                          mus4_data, mus5_data, mus6_data)

                    # TODO: turn mus data into 2d numpy arrays of time series

                    if self.downsampler_active:
                        # TODO: downsample from loaded ROB and LAP datasets
                        mus1_data_downsampled = []
                        mus2_data_downsampled = []
                        mus3_data_downsampled = []
                        mus4_data_downsampled = []
                        mus5_data_downsampled = []
                        mus6_data_downsampled = []

                        if self.row_map.iloc[data_index][KNOT_TYPE_INDEX] == ROB:
                            self._load_time_series_to_dict(self.ROB_datasets_downsampled,
                                                           mus1_data_downsampled, mus2_data_downsampled,
                                                           mus3_data_downsampled, mus4_data_downsampled,
                                                           mus5_data_downsampled, mus6_data_downsampled)

                        elif self.row_map.iloc[data_index][KNOT_TYPE_INDEX] == LAP:
                            self._load_time_series_to_dict(self.LAP_datasets_downsampled,
                                                           mus1_data_downsampled, mus2_data_downsampled,
                                                           mus3_data_downsampled, mus4_data_downsampled,
                                                           mus5_data_downsampled, mus6_data_downsampled)

                    rob_or_lap = self.row_map.iloc[data_index][KNOT_TYPE_INDEX]
                    data_index += 1

                if data_index > len(self.row_map) - 1:
                    break
                elif self.row_map.iloc[data_index][CODE_INDEX] != data_file_key:
                    break  # takes care of situation where a recording only has LAP or ROB
        self._convert_mus_data_to_time_series(self.ROB_datasets)
        self._convert_mus_data_to_time_series(self.LAP_datasets)
        #     TODO: preprocess ratings

    def _convert_mus_data_to_time_series(self, ts_dict):
        ts_dict[MUS1] = to_time_series_dataset(ts_dict[MUS1])
        ts_dict[MUS2] = to_time_series_dataset(ts_dict[MUS2])
        ts_dict[MUS3] = to_time_series_dataset(ts_dict[MUS3])
        ts_dict[MUS4] = to_time_series_dataset(ts_dict[MUS4])
        ts_dict[MUS5] = to_time_series_dataset(ts_dict[MUS5])
        ts_dict[MUS6] = to_time_series_dataset(ts_dict[MUS6])

    def _extract_mus_data_in_time_range(self, df, start_time, end_time):
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

    def _load_time_series_to_dict(self, ts_dict,
                                  mus1_data, mus2_data, mus3_data,
                                  mus4_data, mus5_data, mus6_data):
        ts_dict[MUS1].append(mus1_data)
        ts_dict[MUS2].append(mus2_data)
        ts_dict[MUS3].append(mus3_data)
        ts_dict[MUS4].append(mus4_data)
        ts_dict[MUS5].append(mus5_data)
        ts_dict[MUS6].append(mus6_data)

    def _trim_time_diff(self, data_index, df):
        time_diff = self.row_map.iloc[data_index][TIME_DIFF_INDEX]
        pass_index = df.iloc[:, TIMESTAMP_INDEX].searchsorted(time_diff)[0]
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
                time = time - int(self.row_map.iloc[timestamp_index][TIME_DIFF_INDEX]) # sub initial offset
                timestamps[action] = time
            if self.row_map.iloc[timestamp_index][KNOT_TYPE_INDEX] == ROB:
                self.ROB_action_labels.append(timestamps)
            elif self.row_map.iloc[timestamp_index][KNOT_TYPE_INDEX] == LAP:
                self.LAP_action_labels.append(timestamps)

    def _preprocess_row_map(self, timestamps_df):
        self.row_map = timestamps_df.iloc[:, ROW_MAPPER_INDICES]  # contains the row mapping data for later preprocessing
        self.row_map.iloc[:, TIME_DIFF_INDEX] = self.row_map.iloc[:, TIME_DIFF_INDEX].apply(str_to_seconds)  # change the initial offset to seconds
        # TODO: apply method - if value is -1 then use average or prev value to pad
        self.row_map.iloc[:, START_TIME_INDEX] = self.row_map.iloc[:, START_TIME_INDEX].apply(str_to_seconds)
        self.row_map.iloc[:, END_TIME_INDEX] = self.row_map.iloc[:, END_TIME_INDEX].apply(str_to_seconds)

    def _preprocess_actions(self, timestamps_df):
        self.actions = timestamps_df.columns[ROW_MAPPER_CUTOFF_INDEX:]

    def _get_file_paths(self, path_to_main_folder):
        contents = os.listdir(path_to_main_folder)
        file_paths = [path_to_main_folder + "/" + i for i in contents if ".txt" in i]
        return file_paths