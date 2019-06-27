"""
The preprocessing required from the data is as follows:
1. Read each time series in each text file as a dataframe
2. Filter out NaN rows  !!! NOTE THIS MAKES THE TIME SERIES OF VARIABLE LENGTH !!!
3. Normalize each of data columns based on values in that column, ignoring the first column
4. (more operations may be needed)
"""

import pandas as pd
from sklearn import preprocessing
import os

SKIP_ROWS = 7
COLUMNS_USED = [0, 2, 3, 4, 5, 6, 7]  # omit date

class EMGPreprocessor():
    def __init__(self, path_to_main_folder):
        self.time_series_collection = self._get_time_series_collection(path_to_main_folder)  # array of dataframes

    def get_time_series_collection(self):
        return self.time_series_collection

    def _get_time_series_collection(self, path_to_main_folder):
        file_paths = self._get_file_paths(path_to_main_folder)
        time_series_collection = []
        for file_path in file_paths:
            time_series_collection += self._get_time_series(file_path)  # concatenates the returned values
        return time_series_collection

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
        # raw_time_series = pd.read_table(path_to_text, skiprows=SKIP_ROWS, header=None,
        #                                 usecols=COLUMNS_USED, encoding="mac-roman")
        return []

    def _filter_NaN_rows(self, time_series):
        # TODO: filter NaN rows from a single dataframe
        return None

    def normalize_time_series(self, raw_time_series):
        # TODO: normalize a single dataframe's last 6 columns by column value
        return None