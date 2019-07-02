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

SKIP_ROWS = 7
COLUMNS_USED = [0, 2, 3, 4, 5, 6, 7]  # omit date

class EMGDataManager:
    def __init__(self, path_to_main_folder, path_to_ratings, path_to_action_labels):
        self.path_to_main_folder = path_to_main_folder
        self.path_to_ratings = path_to_ratings
        self.path_to_action_labels = path_to_action_labels

        self.preprocessor = EMGDataPreprocessor()

        # These are synced using the file names
        self.time_series_collection = self.preprocessor.preprocess_time_series_collection(self.path_to_main_folder)  # array of dataframes
        self.time_series_ratings = self.preprocessor.preprocess_ratings(self.path_to_ratings)
        self.time_series_action_labels = self.preprocessor.preprocess_action_labels(self.path_to_action_labels)

    def get_time_series_collection(self):
        return self.time_series_collection

class EMGDataPreprocessor:
    def __init__(self):
        pass

    def preprocess_time_series_collection(self, path_to_main_folder):
        file_paths = self._get_file_paths(path_to_main_folder)
        time_series_collection = []
        for file_path in file_paths:
            time_series_collection += self._get_time_series(file_path)  # concatenates the returned values
        return time_series_collection

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
        raw_time_series = self._get_time_series_from_file(path_to_text)
        normal_time_series_collection = []
        for raw_ts in raw_time_series:
            ts = self._filter_NaN_rows(raw_ts)  # filters out invalid rows, including NaNs
            normal_ts = self.normalize_time_series(ts)  # normalizes the dataframes based on EMG columns
            normal_time_series_collection.append(normal_ts)
        return normal_time_series_collection

    def _get_time_series_from_file(self, path_to_text):
        # TODO: Each file contains at least one time series, and each needs to be made into a dataframe
        # this code reads the whole file as one dataframe. it's for reference only
        # instead of this, you need to be able to identify and parse the multiple time series stored in the file
        # raw_time_series = pd.read_table(path_to_text, skiprows=SKIP_ROWS, header=None,
        #                                 usecols=COLUMNS_USED, encoding="mac-roman")
        return []

    def _filter_NaN_rows(self, time_series):
        # TODO: filter NaN rows from a single dataframe and make the value the average of the previous and after
        # Also justify values if needed
        return None

    def normalize_time_series(self, raw_time_series):
        # TODO: normalize a single dataframe's last 6 columns by column value
        return None