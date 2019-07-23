"""
The EnsembleSegmenting task is as follows:
1. Preprocess the EMG data using EMGPreprocessor, put some aside as test data
2. Makes a model for each of the EMG record points (6 total)
    i.e One for all the bicep non-dom time series, one for all the bicep doms, etc.
3. For each model, split data into training and validation
4. Build similarity model and use DTW with most similar K neighbours to determine action time stamps.
5. Average time stamps to produce action time stamp predictions.
"""

from EMGDataManager import EMGDataManager
from NeighbourClassifier import NeighbourClassifier
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
import logging
logging.basicConfig(level=logging.DEBUG)

PATH_TO_DATA = "./data_sample/"
PATH_TO_ACTION_LABELS = "./data_sample/sample_action_labels.csv"

N_NEIGHBOURS = 3

def main():
    dm = EMGDataManager(PATH_TO_DATA, path_to_timestamps=PATH_TO_ACTION_LABELS, downsampler=True)
    clf = KNeighborsTimeSeriesClassifier(n_neighbors=N_NEIGHBOURS, metric="dtw")
    nbr_clf_rob = NeighbourClassifier(clf)

    # LOADING DATA
    # we can only use downsampled data here
    rob_data_mus1, rob_data_mus2, rob_data_mus3, rob_data_mus4, rob_data_mus5, rob_data_mus6 = \
        dm.get_ROB_data_downsampled()
    _, _, timestamps = dm.get_ROB_metadata()
    nbr_clf_rob.load_data(rob_data_mus1, rob_data_mus2, rob_data_mus3,
                          rob_data_mus4, rob_data_mus5, rob_data_mus6,
                          timestamps)

    # PREDICTIONS
    predictions = nbr_clf_rob.predict()
    actual = nbr_clf_rob.get_test_timestamps()
    print(predictions)
    print(actual)

if __name__ == '__main__':
    main()