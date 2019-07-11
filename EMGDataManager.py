"""
The preprocessing required from the data is as follows:
1. Read each time series in each text file as a dataframe
2. Filter out NaN rows  !!! NOTE THIS MAKES THE TIME SERIES OF VARIABLE LENGTH !!!
3. Normalize each of data columns based on values in that column, ignoring the first column
4. (more operations may be needed)
"""

import pandas as pd
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
        #     TODO

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
        print(self.ROB_action_labels)
        print(self.LAP_action_labels)

    def _preprocess_actions(self, timestamps_df):
        self.actions = timestamps_df.columns[ROW_MAPPER_CUTOFF_INDEX:]

    def _preprocess_row_map(self, timestamps_df):
        self.row_map = timestamps_df.iloc[:, :ROW_MAPPER_CUTOFF_INDEX]  # contains the row mapping data for later preprocessing
        self.row_map.iloc[:, INITIAL_OFFSET_INDEX:] = self.row_map.iloc[:, INITIAL_OFFSET_INDEX:].apply(str_to_seconds, axis=1)  # change the initial offset to seconds

    def _get_file_paths(self, path_to_main_folder):
        contents = os.listdir(path_to_main_folder)
        file_paths = [path_to_main_folder + "/" + i for i in contents if ".txt" in i]
        return file_paths