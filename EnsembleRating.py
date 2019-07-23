"""
The EnsembleRating task is as follows:
1. Preprocess the EMG data using EMGPreprocessor, put some aside as test data
2. Makes a model for each of the EMG record points (6 total)
    i.e One for all the bicep non-dom time series, one for all the bicep doms, etc.
3. Split training data into individual and interim
4. Train the individual models using tslearn package and individual data
5. Train the ensemble method using the interim data
6. Apply the model to test data. Take the mode rating from the models, breaking ties with uncertainty.
"""

from Model import Model
from EMGDataManager import EMGDataManager
from tslearn.clustering import TimeSeriesKMeans  # currently used
from sklearn.linear_model import SGDRegressor, LinearRegression

import logging
logging.basicConfig(level=logging.DEBUG)

PATH_TO_DATA = "./data_sample/"
PATH_TO_ACTION_LABELS = "./data_sample/sample_action_labels.csv"
PATH_TO_RATINGS = "./data_sample/sample_ratings.csv"

def main():
    dm = EMGDataManager(PATH_TO_DATA,
                        path_to_timestamps=PATH_TO_ACTION_LABELS, path_to_ratings=PATH_TO_RATINGS,
                        downsampler=True)

    indv_model = TimeSeriesKMeans(metric="dtw")
    # indv_model = KNeighborsTimeSeriesClassifier(metric="dtw")
    ensm_model = LinearRegression()

    # LOADING DATA
    # uses lap data as an example
    lap_model = Model(indv_model, ensm_model)
    lap_data_mus1, lap_data_mus2, lap_data_mus3, lap_data_mus4, lap_data_mus5, lap_data_mus6 = \
        dm.get_LAP_data_downsampled()
    times, ratings, _ = dm.get_LAP_metadata()
    lap_model.load_data(lap_data_mus1, lap_data_mus2, lap_data_mus3,
                        lap_data_mus4, lap_data_mus5, lap_data_mus6,
                        ratings, times)
    lap_model.load_svr_models()
    lap_model.fit()

    pred = lap_model.predict()
    print("Prediction")
    print(pred)
    print("Actual")
    print(lap_model.get_label_test())

if __name__ == '__main__':
    main()