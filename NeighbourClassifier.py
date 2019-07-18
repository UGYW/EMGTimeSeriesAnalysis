from copy import deepcopy
import numpy as np

class NeighbourClassifier:
    def __init__(self, neighbour_classifier, actions):
        self.neighbour_classifier = neighbour_classifier
        self.actions = actions
        self.mus1_classifier = deepcopy(self.neighbour_classifier)
        self.mus2_classifier = deepcopy(self.neighbour_classifier)
        self.mus3_classifier = deepcopy(self.neighbour_classifier)
        self.mus4_classifier = deepcopy(self.neighbour_classifier)
        self.mus5_classifier = deepcopy(self.neighbour_classifier)
        self.mus6_classifier = deepcopy(self.neighbour_classifier)

        self.SPLIT_RATIO = 0.8  # % of data used to train the model
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
        self.split_index = self.SPLIT_RATIO * len(input_data_mus1)
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

    def predict(self):
        # TODO: find average timestamp for each test point
        predicted_timestamps = []
        return predicted_timestamps

    def get_test_timestamps(self):
        return self.timestamps[self.split_index:]
        
    def find_average_timestamp(self, input_point_mus1, input_point_mus2, input_point_mus3,
                                     input_point_mus4, input_point_mus5, input_point_mus6):
        nbr_idx_mus1, nbr_idx_mus2, nbr_idx_mus3, \
        nbr_idx_mus4, nbr_idx_mus5, nbr_idx_mus6 = self.find_nbrs(input_point_mus1, input_point_mus2, input_point_mus3,
                                                                  input_point_mus4, input_point_mus5, input_point_mus6)
        avg_timestamp_mus1 = self.find_average_timestamp(nbr_idx_mus1)
        avg_timestamp_mus2 = self.find_average_timestamp(nbr_idx_mus2)
        avg_timestamp_mus3 = self.find_average_timestamp(nbr_idx_mus3)
        avg_timestamp_mus4 = self.find_average_timestamp(nbr_idx_mus4)
        avg_timestamp_mus5 = self.find_average_timestamp(nbr_idx_mus5)
        avg_timestamp_mus6 = self.find_average_timestamp(nbr_idx_mus6)
        avg_timestamp = self._average_timestamps([
            avg_timestamp_mus1,
            avg_timestamp_mus2,
            avg_timestamp_mus3,
            avg_timestamp_mus4,
            avg_timestamp_mus5,
            avg_timestamp_mus6
        ])
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

    def _find_average_timestamp(self, nbr_idx):
        # TODO
        return {}

    def _average_timestamps(self, timestamp_sets):  # takes in an array of dictionaries
        return []

    def _find_nbr_idx(self, input_point, mus_data):
        # TODO
        return []