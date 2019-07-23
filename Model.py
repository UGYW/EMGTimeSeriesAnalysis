""" Abstraction of models used for analysis"""
from copy import deepcopy
import numpy as np
from tslearn.svm import TimeSeriesSVR

class Model:
    def __init__(self, individual_model, ensemble_model):
        self.individual_model = deepcopy(individual_model)
        self.ensemble_model = deepcopy(ensemble_model)
        self.mus1_model = deepcopy(self.individual_model)
        self.mus2_model = deepcopy(self.individual_model)
        self.mus3_model = deepcopy(self.individual_model)
        self.mus4_model = deepcopy(self.individual_model)
        self.mus5_model = deepcopy(self.individual_model)
        self.mus6_model = deepcopy(self.individual_model)

        # we shouldn't need to shuffle the input_data if we're using clustering methods
        self.SPLIT_RATIO = [0.5, 0.3] # train and interm respectively. the remaining is test
        self.split_train_index = -1
        self.split_interm_index = -1

        self.mus1_input_data_train = None
        self.mus1_input_data_interm = None
        self.mus1_input_data_test = None
        self.mus2_input_data_train = None
        self.mus2_input_data_interm = None
        self.mus2_input_data_test = None
        self.mus3_input_data_train = None
        self.mus3_input_data_interm = None
        self.mus3_input_data_test = None
        self.mus4_input_data_train = None
        self.mus4_input_data_interm = None
        self.mus4_input_data_test = None
        self.mus5_input_data_train = None
        self.mus5_input_data_interm = None
        self.mus5_input_data_test = None
        self.mus6_input_data_train = None
        self.mus6_input_data_interm = None
        self.mus6_input_data_test = None

        self.label_train = None
        self.label_interm = None
        self.label_test = None

        # times are not used in the initial training phase due to dtw
        self.times_interm = None
        self.times_test = None

        self.data_loaded = False

    def get_label_train(self):
        return self.label_train

    def get_label_interm(self):
        return self.label_interm

    def get_label_test(self):
        return self.label_test

    def load_data(self, input_data_mus1, input_data_mus2, input_data_mus3,
                  input_data_mus4, input_data_mus5, input_data_mus6,
                  labels, times):
        self.data_loaded = True
        self._load_input_data(input_data_mus1, input_data_mus2, input_data_mus3, input_data_mus4, input_data_mus5, input_data_mus6)
        self._load_labels(labels)
        self._load_times(times)

    def load_svr_models(self):
        # THIS OVERWRITES WHAT WAS PREVIOUSLY IN THESE MODELS!
        if not self.data_loaded:
            raise Exception("Must have loaded all the data first.")
        self.mus1_model = TimeSeriesSVR(sz=len(self.mus1_input_data_train[0]), d=1)
        self.mus2_model = TimeSeriesSVR(sz=len(self.mus2_input_data_train[0]), d=1)
        self.mus3_model = TimeSeriesSVR(sz=len(self.mus3_input_data_train[0]), d=1)
        self.mus4_model = TimeSeriesSVR(sz=len(self.mus4_input_data_train[0]), d=1)
        self.mus5_model = TimeSeriesSVR(sz=len(self.mus5_input_data_train[0]), d=1)
        self.mus6_model = TimeSeriesSVR(sz=len(self.mus6_input_data_train[0]), d=1)

    def _load_input_data(self, input_data_mus1, input_data_mus2, input_data_mus3, input_data_mus4, input_data_mus5, input_data_mus6):
        if self.split_train_index == -1 or self.split_interm_index == -1:
            input_data_len = len(input_data_mus1)  # assume all 6 are of same length
            self._calc_input_data_split(input_data_len)
        self._load_input_data_mus1(input_data_mus1)
        self._load_input_data_mus2(input_data_mus2)
        self._load_input_data_mus3(input_data_mus3)
        self._load_input_data_mus4(input_data_mus4)
        self._load_input_data_mus5(input_data_mus5)
        self._load_input_data_mus6(input_data_mus6)

    def _load_labels(self, labels):
        if self.split_train_index == -1 or self.split_interm_index == -1:
            input_data_len = len(labels)
            self._calc_input_data_split(input_data_len)
        self.label_train = labels[: self.split_train_index]
        self.label_interm = labels[self.split_train_index: self.split_interm_index]
        self.label_test = labels[self.split_interm_index:]

    def _load_times(self, times):
        if self.split_train_index == -1 or self.split_interm_index == -1:
            input_data_len = len(times)
            self._calc_input_data_split(input_data_len)
        self.times_interm = times[self.split_train_index: self.split_interm_index]
        self.times_test = times[self.split_interm_index:]

    def fit(self):
        self._fit_individual_models()
        interm_pred = self._calc_interm_pred(self.mus1_input_data_interm, self.mus2_input_data_interm,
                                             self.mus3_input_data_interm, self.mus4_input_data_interm,
                                             self.mus5_input_data_interm, self.mus6_input_data_interm,
                                             self.times_interm)
        interm_labels = self.label_interm
        self.ensemble_model.fit(interm_pred, interm_labels)

    # Predict for ensemble using test data
    def predict(self):
        interm_pred = self._calc_interm_pred(self.mus1_input_data_test, self.mus2_input_data_test,
                                             self.mus3_input_data_test, self.mus4_input_data_test,
                                             self.mus5_input_data_test, self.mus6_input_data_test,
                                             self.times_test)
        return self.ensemble_model.predict(interm_pred)

    def _calc_input_data_split(self, input_data_len):
        self.split_train_index = int(self.SPLIT_RATIO[0] * input_data_len)
        self.split_interm_index = self.split_train_index + int(self.SPLIT_RATIO[1] * input_data_len)

    def _load_input_data_mus1(self, input_data):
        self.mus1_input_data_train = input_data[: self.split_train_index]
        self.mus1_input_data_interm = input_data[self.split_train_index : self.split_interm_index]
        self.mus1_input_data_test = input_data[self.split_interm_index : ]

    def _load_input_data_mus2(self, input_data):
        self.mus2_input_data_train = input_data[: self.split_train_index]
        self.mus2_input_data_interm = input_data[self.split_train_index : self.split_interm_index]
        self.mus2_input_data_test = input_data[self.split_interm_index : ]

    def _load_input_data_mus3(self, input_data):
        self.mus3_input_data_train = input_data[: self.split_train_index]
        self.mus3_input_data_interm = input_data[self.split_train_index : self.split_interm_index]
        self.mus3_input_data_test = input_data[self.split_interm_index : ]

    def _load_input_data_mus4(self, input_data):
        self.mus4_input_data_train = input_data[: self.split_train_index]
        self.mus4_input_data_interm = input_data[self.split_train_index : self.split_interm_index]
        self.mus4_input_data_test = input_data[self.split_interm_index : ]

    def _load_input_data_mus5(self, input_data):
        self.mus5_input_data_train = input_data[: self.split_train_index]
        self.mus5_input_data_interm = input_data[self.split_train_index : self.split_interm_index]
        self.mus5_input_data_test = input_data[self.split_interm_index : ]

    def _load_input_data_mus6(self, input_data):
        self.mus6_input_data_train = input_data[: self.split_train_index]
        self.mus6_input_data_interm = input_data[self.split_train_index : self.split_interm_index]
        self.mus6_input_data_test = input_data[self.split_interm_index : ]

    def _fit_individual_models(self):
        self.mus1_model.fit(self.mus1_input_data_train, self.label_train)
        self.mus2_model.fit(self.mus2_input_data_train, self.label_train)
        self.mus3_model.fit(self.mus3_input_data_train, self.label_train)
        self.mus4_model.fit(self.mus4_input_data_train, self.label_train)
        self.mus5_model.fit(self.mus5_input_data_train, self.label_train)
        self.mus6_model.fit(self.mus6_input_data_train, self.label_train)

    def _calc_interm_pred(self, mus1_data_interm, mus2_data_interm, mus3_data_interm,
                          mus4_data_interm, mus5_data_interm, mus6_data_interm, times):
        interm_pred1 = self.mus1_model.predict(mus1_data_interm)
        interm_pred2 = self.mus2_model.predict(mus2_data_interm)
        interm_pred3 = self.mus3_model.predict(mus3_data_interm)
        interm_pred4 = self.mus4_model.predict(mus4_data_interm)
        interm_pred5 = self.mus5_model.predict(mus5_data_interm)
        interm_pred6 = self.mus6_model.predict(mus6_data_interm)
        assert self.times_interm is not None
        interm_pred = np.stack((interm_pred1,
                                interm_pred2,
                                interm_pred3,
                                interm_pred4,
                                interm_pred5,
                                interm_pred6,
                                times)).transpose()
        return interm_pred