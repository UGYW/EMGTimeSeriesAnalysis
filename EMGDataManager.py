from data_utils import *
import logging

class EMGDataManager:
    def __init__(self, path_to_main_folder, path_to_timestamps, path_to_ratings=None, downsampler=False):
        self.path_to_main_folder = path_to_main_folder
        self.path_to_timestamps = path_to_timestamps
        # TODO: path to ratings not implemented rn

        self.downsampler_active = downsampler  # determines whether or not to init downsampling

        # each of the keys corresponds to a 2D numpy array where
        #   each array is a time series
        self.ROB_datasets = {MUS1: [], MUS2: [], MUS3: [], MUS4: [], MUS5: [], MUS6: []}
        self.LAP_datasets = {MUS1: [], MUS2: [], MUS3: [], MUS4: [], MUS5: [], MUS6: []}
        self.ROB_datasets_times = []
        self.LAP_datasets_times = []
        self.ROB_datasets_downsampled = {MUS1: [], MUS2: [], MUS3: [], MUS4: [], MUS5: [], MUS6: []}
        self.LAP_datasets_downsampled = {MUS1: [], MUS2: [], MUS3: [], MUS4: [], MUS5: [], MUS6: []}
        self.ROB_ratings = []
        self.LAP_ratings = []
        # each of the keys correpsonds to an array of dicts where
        #   each dict is like {ACT1: None, ACT2: None, etc} and corresponds to one knot
        self.ROB_action_timestamps = []
        self.LAP_action_timestamps = []

        self.row_map = None  # Dataframe of the first four columns of the ratings
        self.actions = []  # array of strings

        self.preprocess()

    def get_ROB_data(self):
        return self.ROB_datasets

    def get_ROB_metadata(self):
        return self.ROB_datasets_times, self.ROB_ratings, self.ROB_action_timestamps

    def get_LAP_data(self):
        return self.LAP_datasets

    def get_LAP_metadata(self):
        return self.LAP_datasets_times, self.LAP_ratings, self.LAP_action_timestamps

    def get_ROB_data_downsampled(self):
        return self.ROB_datasets_downsampled

    def get_LAP_data_downsampled(self):
        return self.LAP_datasets_downsampled

    def get_row_map(self):
        return self.row_map

    def get_actions(self):
        return self.actions

    def preprocess(self):
        # ACTION LABELS
        self._preprocess_timestamps()  # this has to run first because this inits the row map

        # INPUT DATA
        self._preprocess_data_files(get_file_paths(self.path_to_main_folder))
        if self.downsampler_active:
            self._make_downsampled_datasets()
        self._convert_datasets_to_time_series()

        # RATINGS
        #     TODO

    def _preprocess_data_files(self, data_file_paths):
        data_index = 0
        for data_file_path in data_file_paths:
            logging.info("LOADING " + data_file_path)
            data_file_key, video_sesh_dfs = self._extract_video_sessions_data(data_file_path)

            for df in video_sesh_dfs:
                df = self._preprocess_column_values(df)
                df = self._trim_time_diff(data_index, df)
                rob_or_lap = self.row_map.iloc[data_index][KNOT_TYPE_INDEX]

                while self.check_is_same_video(data_index, data_file_key, rob_or_lap):
                    self._preprocess_knot(data_file_key, data_index, df, rob_or_lap)
                    rob_or_lap = self.row_map.iloc[data_index][KNOT_TYPE_INDEX]
                    data_index += 1

                if data_index > len(self.row_map) - 1 or \
                                self.row_map.iloc[data_index][CODE_INDEX] != data_file_key:
                    break

    def _make_downsampled_datasets(self):
        # !!! ASSUMES ROB AND LAP DATASETS HAVE BEEN LOADED BUT NOT YET CONVERTED TO TIME SERIES !!!
        self.ROB_datasets_downsampled[MUS1] = downsample_ts_dict(self.ROB_datasets, MUS1)
        self.ROB_datasets_downsampled[MUS2] = downsample_ts_dict(self.ROB_datasets, MUS2)
        self.ROB_datasets_downsampled[MUS3] = downsample_ts_dict(self.ROB_datasets, MUS3)
        self.ROB_datasets_downsampled[MUS4] = downsample_ts_dict(self.ROB_datasets, MUS4)
        self.ROB_datasets_downsampled[MUS5] = downsample_ts_dict(self.ROB_datasets, MUS5)
        self.ROB_datasets_downsampled[MUS6] = downsample_ts_dict(self.ROB_datasets, MUS6)

        self.LAP_datasets_downsampled[MUS1] = downsample_ts_dict(self.LAP_datasets, MUS1)
        self.LAP_datasets_downsampled[MUS2] = downsample_ts_dict(self.LAP_datasets, MUS2)
        self.LAP_datasets_downsampled[MUS3] = downsample_ts_dict(self.LAP_datasets, MUS3)
        self.LAP_datasets_downsampled[MUS4] = downsample_ts_dict(self.LAP_datasets, MUS4)
        self.LAP_datasets_downsampled[MUS5] = downsample_ts_dict(self.LAP_datasets, MUS5)
        self.LAP_datasets_downsampled[MUS6] = downsample_ts_dict(self.LAP_datasets, MUS6)

    def _convert_datasets_to_time_series(self):
        convert_mus_data_to_time_series(self.ROB_datasets)
        convert_mus_data_to_time_series(self.LAP_datasets)
        if self.downsampler_active:
            convert_mus_data_to_time_series(self.ROB_datasets_downsampled)
            convert_mus_data_to_time_series(self.LAP_datasets_downsampled)

    def _preprocess_knot(self, data_file_key, data_index, df, rob_or_lap):
        logging.info("PROCESSING KNOT FROM " + data_file_key + " OF TYPE " + rob_or_lap)
        start_time = self.row_map.iloc[data_index][START_TIME_INDEX]
        end_time = self.row_map.iloc[data_index][END_TIME_INDEX]
        mus1_data, mus2_data, mus3_data, mus4_data, mus5_data, mus6_data = \
            extract_mus_data_in_time_range(df, start_time, end_time)
        self.load_datasets(self.ROB_datasets, self.LAP_datasets, data_index,
                           mus1_data, mus2_data, mus3_data, mus4_data, mus5_data, mus6_data,
                           start_time, end_time)

    def _extract_video_sessions_data(self, data_file_path):
        raw_time_series = pd.read_table(data_file_path, skiprows=SKIP_ROWS, header=None,
                                        usecols=COLUMNS_USED, encoding="mac-roman")
        data_file_key = os.path.splitext(os.path.basename(data_file_path))[0]
        dfs = parse_df_splits_from_raw(raw_time_series)
        return data_file_key, dfs

    def check_is_same_video(self, data_index, data_file_key, rob_or_lap):
        is_same_video = data_index < len(self.row_map) and \
            data_file_key == self.row_map.iloc[data_index][CODE_INDEX] and \
            rob_or_lap == self.row_map.iloc[data_index][KNOT_TYPE_INDEX]
        return is_same_video

    def _preprocess_column_values(self, df):
        for col_i in range(len(df.columns)):
            col = coerce_invalid_col_values_from_df(col_i, df)
            col = standardize_col(col, col_i)
            df[df.columns[col_i]] = col
            return df

    def _preprocess_timestamps(self):
        timestamps_df = rectify_time_diff(pd.read_csv(self.path_to_timestamps))
        self._preprocess_actions(timestamps_df)
        self._preprocess_row_map(timestamps_df)
        # timestamps_sets = timestamps_df.iloc[:, ROW_MAPPER_CUTOFF_INDEX:].to_dict('index')
        timestamps_sets = timestamps_df.iloc[:, ROW_MAPPER_CUTOFF_INDEX:].as_matrix()
        for timestamp_index in range(len(timestamps_sets)):
            timestamps = timestamps_sets[timestamp_index]
            if self.row_map.iloc[timestamp_index][KNOT_TYPE_INDEX] == ROB:
                self.ROB_action_timestamps.append(timestamps)
            elif self.row_map.iloc[timestamp_index][KNOT_TYPE_INDEX] == LAP:
                self.LAP_action_timestamps.append(timestamps)
        print(self.ROB_action_timestamps)
        print(self.LAP_action_timestamps)

    def _preprocess_actions(self, timestamps_df):
        self.actions = timestamps_df.columns[ROW_MAPPER_CUTOFF_INDEX:]

    def _preprocess_row_map(self, timestamps_df):
        self.row_map = timestamps_df.iloc[:, ROW_MAPPER_INDICES]  # gets the first four columns
        self.row_map.iloc[:, START_TIME_INDEX] = timestamps_df.iloc[:, START_TIME_INDEX:].apply(np.min, axis=1)
        self.row_map.iloc[:, END_TIME_INDEX] = timestamps_df.iloc[:, START_TIME_INDEX:].apply(np.max, axis=1)

    def load_datasets(self, ROB_dataset, LAP_dataset, data_index,
                            mus1_data, mus2_data, mus3_data, mus4_data, mus5_data, mus6_data,
                            start_time=None, end_time=None):
        if self.row_map.iloc[data_index][KNOT_TYPE_INDEX] == ROB:
            append_mus_data_to_dict(ROB_dataset, mus1_data, mus2_data, mus3_data,
                                    mus4_data, mus5_data, mus6_data)
            if start_time is not None and end_time is not None:
                self.ROB_datasets_times.append(end_time - start_time)
        elif self.row_map.iloc[data_index][KNOT_TYPE_INDEX] == LAP:
            append_mus_data_to_dict(LAP_dataset, mus1_data, mus2_data, mus3_data,
                                    mus4_data, mus5_data, mus6_data)
            if start_time is not None and end_time is not None:
                self.LAP_datasets_times.append(end_time - start_time)
        else:
            raise Exception("Knot Type " + self.row_map.iloc[data_index][KNOT_TYPE_INDEX] +
                            " is not " + ROB + " or " + LAP)

    def _trim_time_diff(self, data_index, df):
        time_diff = self.row_map.iloc[data_index][TIME_DIFF_INDEX]
        pass_index = df.iloc[:, TIMESTAMP_INDEX].searchsorted(time_diff)[0]
        df = df.iloc[pass_index:]
        return df