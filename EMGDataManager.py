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

class EMGDataManager:
    def __init__(self, path_to_main_folder, path_to_ratings, path_to_action_labels):
        self.path_to_main_folder = path_to_main_folder
        self.path_to_ratings = path_to_ratings
        self.path_to_action_labels = path_to_action_labels

        self.preprocessor = EMGDataPreprocessor()

        # These are synced using the file names
        # time_series_collection format:
        # {MUS1: [time series set 1], MUS2: [time series set 2], ... MUS6: [time series set 6]}
        # where each set is a list of all the available time series for that muscle
        # ts_map format: (multiple time series per file!)
        # [filename1.0, filename1.1, filename2... ]
        # used to map time series to ratings and action labels
        # timestamps format:
        # [[timestamps of file 1], [timestamps of file 2], ...]
        self.time_series_collection, self.ts_map, self.timestamps = \
            self.preprocessor.preprocess_time_series_input(self.path_to_main_folder)
        self.time_series_ratings = self.preprocessor.preprocess_ratings(self.path_to_ratings)
        self.time_series_action_labels = self.preprocessor.preprocess_action_labels(self.path_to_action_labels)

    def get_time_series_collection(self):
        return self.time_series_collection

    def get_time_series_ratings(self):
        return self.time_series_ratings

    def get_time_series_action_labels(self):
        return self.time_series_action_labels

class EMGDataPreprocessor:
    def __init__(self):
        pass

    def preprocess_time_series_input(self, path_to_main_folder):
        time_series_collection = {MUS1:[], MUS2:[], MUS3:[], MUS4:[], MUS5:[], MUS6:[]}
        ts_map = []
        timestamps = []
        file_paths = self._get_file_paths(path_to_main_folder)
        for file_path in file_paths:
            time_series_from_file = self._get_time_series(file_path)
            time_series_index = 0
            for time_series in time_series_from_file:
                # add the filename without .txt
                # appends for each time series in that file
                # todo: ask how diff time series within same file are distinguished    vvvvvvvvv is it like this?
                ts_map.append(os.path.basename(os.path.normpath(file_path))[:-4] + str(time_series_index))
                timestamps.append(self._get_column(time_series, T2COL[TIME]))
                for mus, col in MUS2COL.items():
                    time_series_collection[mus].append(self._get_column(time_series, col))
        return time_series_collection, ts_map, timestamps

    def preprocess_ratings(self, path_to_ratings):
        # TODO: Get ratings in the order of the time series
        return []

    def preprocess_action_labels(self, path_to_action_labels):
        # TODO: Get action labels for each of the time series, in order
        return [[]]

    def _get_file_paths(self, path_to_main_folder):
        contents = os.listdir(path_to_main_folder)
        file_paths = [path_to_main_folder + "/" + i for i in contents if ".txt" in i]
        return file_paths

    def _get_time_series(self, path_to_text):
        # TODO
        return []

    def _get_column(self, time_series, col):
        # TODO
        return []  # make sure to return this as an array

    def _get_time_series_from_file(self, path_to_text):
        # TODO: Each file contains at least one time series, and each needs to be made into a dataframe
        # this code reads the whole file as one dataframe. it's for reference only
        # instead of this below, you need to be able to identify and parse the multiple time series stored in the file
        return []

    def _filter_NaN_rows(self, time_series):
        # TODO: filter NaN rows from a single dataframe and make the value the average of the previous and after
        # Also justify values if needed
        return None

    def normalize_time_series(self, raw_time_series):
        # TODO: normalize a single array
        # NOTE: DON'T RUN THIS UNTIL **AFTER** PARTITIONING TO TEST AND TRAIN - OTHERWISE CAUSES DATA LEAK
        return None