from copy import deepcopy
import numpy as np

class NeighbourClassifier:
    def __init__(self, neighbour_classifier):
        self.neighbour_classifier = deepcopy(neighbour_classifier)

        self.SPLIT_RATIO = 0.7  # % of data used to train the model
        self.split_index = None

        self.mus1_input_data_train = []
        self.mus1_input_data_test = []
        self.mus2_input_data_train = []
        self.mus2_input_data_test = []
        self.mus3_input_data_train = []
        self.mus3_input_data_test = []
        self.mus4_input_data_train = []
        self.mus4_input_data_test = []
        self.mus5_input_data_train = []
        self.mus5_input_data_test = []
        self.mus6_input_data_train = []
        self.mus6_input_data_test = []

        self.timestamps = None

    def load_data(self, input_data_mus1, input_data_mus2, input_data_mus3,
                        input_data_mus4, input_data_mus5, input_data_mus6,
                        timestamps):
        self.split_index = int(round(self.SPLIT_RATIO * len(input_data_mus1)))
        self.mus1_input_data_train = input_data_mus1[:self.split_index]
        self.mus1_input_data_test = input_data_mus1[self.split_index:]
        self.mus2_input_data_train = input_data_mus2[:self.split_index]
        self.mus2_input_data_test = input_data_mus2[self.split_index:]
        self.mus3_input_data_train = input_data_mus3[:self.split_index]
        self.mus3_input_data_test = input_data_mus3[self.split_index:]
        self.mus4_input_data_train = input_data_mus4[:self.split_index]
        self.mus4_input_data_test = input_data_mus4[self.split_index:]
        self.mus5_input_data_train = input_data_mus5[:self.split_index]
        self.mus5_input_data_test = input_data_mus5[self.split_index:]
        self.mus6_input_data_train = input_data_mus6[:self.split_index]
        self.mus6_input_data_test = input_data_mus6[self.split_index:]

        self.timestamps = timestamps

    def predict(self):
        predicted_timestamps = []
        num_test_points = len(self.mus1_input_data_test)
        for test_pt_i in range(num_test_points):
            average_timestamp = self.find_average_timestamp(
                self.mus1_input_data_test[test_pt_i],
                self.mus2_input_data_test[test_pt_i],
                self.mus3_input_data_test[test_pt_i],
                self.mus4_input_data_test[test_pt_i],
                self.mus5_input_data_test[test_pt_i],
                self.mus6_input_data_test[test_pt_i]
            )
            predicted_timestamps.append(average_timestamp)
        return predicted_timestamps

    def get_test_timestamps(self):
        return self.timestamps[self.split_index:]
        
    def find_average_timestamp(self, input_point_mus1, input_point_mus2, input_point_mus3,
                                     input_point_mus4, input_point_mus5, input_point_mus6):
        nbr_idx_mus1, nbr_idx_mus2, nbr_idx_mus3, \
        nbr_idx_mus4, nbr_idx_mus5, nbr_idx_mus6 = self.find_nbrs(input_point_mus1, input_point_mus2, input_point_mus3,
                                                                  input_point_mus4, input_point_mus5, input_point_mus6)
        avg_timestamp_mus1 = self._calc_average_timestamp(nbr_idx_mus1)
        avg_timestamp_mus2 = self._calc_average_timestamp(nbr_idx_mus2)
        avg_timestamp_mus3 = self._calc_average_timestamp(nbr_idx_mus3)
        avg_timestamp_mus4 = self._calc_average_timestamp(nbr_idx_mus4)
        avg_timestamp_mus5 = self._calc_average_timestamp(nbr_idx_mus5)
        avg_timestamp_mus6 = self._calc_average_timestamp(nbr_idx_mus6)
        avg_timestamp = self._average_timestamps(np.array([
            avg_timestamp_mus1,
            avg_timestamp_mus2,
            avg_timestamp_mus3,
            avg_timestamp_mus4,
            avg_timestamp_mus5,
            avg_timestamp_mus6
        ]))
        return avg_timestamp
    
    def find_nbrs(self, input_point_mus1, input_point_mus2, input_point_mus3,
                        input_point_mus4, input_point_mus5, input_point_mus6):
        nbr_idx_mus1 = self._find_nbr_idx(input_point_mus1, self.mus1_input_data_train)
        nbr_idx_mus2 = self._find_nbr_idx(input_point_mus2, self.mus2_input_data_train)
        nbr_idx_mus3 = self._find_nbr_idx(input_point_mus3, self.mus3_input_data_train)
        nbr_idx_mus4 = self._find_nbr_idx(input_point_mus4, self.mus4_input_data_train)
        nbr_idx_mus5 = self._find_nbr_idx(input_point_mus5, self.mus5_input_data_train)
        nbr_idx_mus6 = self._find_nbr_idx(input_point_mus6, self.mus6_input_data_train)
        return nbr_idx_mus1, nbr_idx_mus2, nbr_idx_mus3, \
               nbr_idx_mus4, nbr_idx_mus5, nbr_idx_mus6

    def _calc_average_timestamp(self, nbr_idx):
        # avg_timestamp = dict.fromkeys(self.actions, -1)
        selected_timestamps = np.array(self.timestamps)[nbr_idx]
        average_timestamp = self._average_timestamps(selected_timestamps)
        return average_timestamp

    def _average_timestamps(self, timestamp_sets):  # takes in an array of dictionaries
        return np.mean(timestamp_sets, axis=0)

    def _find_nbr_idx(self, input_point, mus_data):
        # TODO: other methods for finding neighbours can be added here
        # To make it work on 3.6
        # self.neighbour_classifier.fit(np.concatenate((mus_data, np.array([input_point]))))
        nbr_idx = self.neighbour_classifier.kneighbors(np.concatenate((mus_data, np.array([input_point]))),
                                                       return_distance=False)
        nbr_idx = nbr_idx[-1][1:] # get the neighbours of the test point, not including itself as the closest
        return nbr_idx