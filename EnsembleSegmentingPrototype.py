"""
The EnsembleSegmenting task is as follows:
1. Preprocess the EMG data using EMGPreprocessor, put some aside as test data
2. Makes a model for each of the EMG record points (6 total)
    i.e One for all the bicep non-dom time series, one for all the bicep doms, etc.
3. For each model, split data into training and validation
4. Build similarity model and use DTW with most similar K neighbours to determine action time stamps.
5. Average time stamps to produce action time stamp predictions.
"""

from tslearn.utils import to_time_series_dataset
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
import numpy as np

ACT1 = "ACT1"
ACT2 = "ACT2"
#                                 1                   2                   3                   4
mock_dataset_muscle1 = np.array([[1, 2, 3, 3],       [3, 2, 3, 5],       [1, 2, 6, 7],       [3, 2, 3, 6]])
mock_dataset_muscle2 = np.array([[2, 5, 6, 7, 8, 9], [3, 2, 5, 6, 7, 3], [6, 5, 3, 4, 7, 8], [2, 5, 6, 7, 8, 9]])
mock_timestamps = np.array([
    {ACT1: 0, ACT2: 0.8},
    {ACT1: 0.1, ACT2: 0.6},
    {ACT1: 0, ACT2: 0.8},
    {ACT1: 0.1, ACT2: 0.7}
])

def main():
    number_of_actions = len(mock_timestamps[0].values())

    print("MUSCLE 1")

    # CONSTRUCTS THE MODEL
    X1 = to_time_series_dataset(mock_dataset_muscle1)
    X_train1 = np.array(X1[:-1])
    X_test1 = np.array([X1[-1]])
    clf1 = KNeighborsTimeSeriesClassifier(n_neighbors=3, metric="dtw")

    # IDENTIFIES THE NEIGHBOURS
    # Makes the row in question always the last row
    nbrs_indices1 = clf1.kneighbors(np.concatenate((X_train1, X_test1)), return_distance=False)
    nbrs_indices1 = nbrs_indices1[-1][1:]  # gets the closest neighbours of the test row, not including self in [0]
    print("NEIGHBOURS INDICES")
    print(nbrs_indices1)

    # CALCULATES THE AVERAGE TIMESTAMPS
    avg_timestamps_1 = np.zeros(number_of_actions)
    for nbrs_ind in nbrs_indices1:
        avg_timestamps_1 += np.array(list(mock_timestamps[nbrs_ind].values()))
    avg_timestamps_1 /= len(nbrs_indices1)
    print("AVERAGE TIMESTAMPS")
    print(avg_timestamps_1)

    print("\n")

    print("MUSCLE 2")

    # CONSTRUCTS THE MODEL
    X2 = to_time_series_dataset(mock_dataset_muscle2)
    X_train2 = np.array(X2[:-1])
    X_test2 = np.array([X2[-1]])
    clf2 = KNeighborsTimeSeriesClassifier(n_neighbors=5, metric="dtw")

    # IDENTIFIES THE NEIGHBOURS
    # Makes the row in question always the last row
    nbrs_indices2 = clf2.kneighbors(np.concatenate((X_train2, X_test2)), return_distance=False)
    nbrs_indices2 = nbrs_indices2[-1][1:]  # gets the closest neighbours of the test row, not including self in [0]
    print("NEIGHBOURS INDICES")
    print(nbrs_indices1)

    # CALCULATES THE AVERAGE TIMESTAMPS
    avg_timestamps_2 = np.zeros(number_of_actions)
    for nbrs_ind in nbrs_indices2:
        avg_timestamps_2 += np.array(list(mock_timestamps[nbrs_ind].values()))
    avg_timestamps_2 /= len(nbrs_indices2)
    print("AVERAGE TIMESTAMPS")
    print(avg_timestamps_2)

    print("\n")

    # CALCULATE THE PREDICTION
    pred_avg_timestamp = ( avg_timestamps_1 + avg_timestamps_2 ) / 2
    print("PREDICTED AVERAGE TIMESTAMP")
    print(pred_avg_timestamp)

    # CALCULATE THE DIFFERENCE FROM PREDICTION
    goal_timestamp = np.array(list(mock_timestamps[-1].values()))
    diff = sum(np.abs(goal_timestamp) - np.abs(pred_avg_timestamp))
    print("ACTUAL")
    print(goal_timestamp)
    print("DIFFERENCE FROM ACTUAL")
    print(diff)

if __name__ == '__main__':
    main()