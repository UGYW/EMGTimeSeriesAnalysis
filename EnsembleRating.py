"""
The EnsembleRating task is as follows:
1. Preprocess the EMG data using EMGPreprocessor, put some aside as test data
2. Makes a model for each of the EMG record points (6 total)
    i.e One for all the bicep non-dom time series, one for all the bicep doms, etc.
3. For each model, split data into training and validation
4. Train the data using tslearn package.
5. Validate to determine best hyperparams
6. Apply the model to test data. Take the mode rating from the models, breaking ties with uncertainty.
"""

from tslearn.utils import to_time_series_dataset
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
import numpy as np

# this should actually be normalized as per their column value in data manager
#                                 1                   2                   3                   4
mock_dataset_muscle1 = np.array([[1, 2, 3, 3],       [3, 2, 3, 6],       [1, 2, 6, 7],       [3, 2, 3, 6]])
mock_dataset_muscle2 = np.array([[2, 5, 6, 7, 8, 9], [3, 2, 5, 6, 7, 3], [6, 5, 3, 4, 7, 8], [2, 5, 6, 7, 8, 9]])
mock_labels = np.array([1, 0, 2, 0])  # the rating scores

def main():
    X1 = to_time_series_dataset(mock_dataset_muscle1)
    y1 = mock_labels
    X_train1 = X1[:-1]
    y_train1 = y1[:-1]
    X_test1 = [X1[-1]]
    y_test1 = [y1[-1]]
    # X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.25, random_state=1)
    clf1 = KNeighborsTimeSeriesClassifier(n_neighbors=5, metric="dtw")
    clf1.fit(X_train1, y_train1)
    print("Prediction: " + str(clf1.predict(X_test1)))
    print("Actual: " + str(y_test1))

    X2 = to_time_series_dataset(mock_dataset_muscle2)
    y2 = mock_labels
    X_train2 = X2[:-1]
    y_train2 = y2[:-1]
    X_test2 = [X2[-1]]
    y_test2 = [y2[-1]]
    # X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.25, random_state=2)
    clf2 = KNeighborsTimeSeriesClassifier(n_neighbors=5, metric="dtw")
    clf2.fit(X_train2, y_train2)
    print("Prediction: " + str(clf2.predict(X_test2)))
    print("Actual: " + str(y_test2))

    # todo: make model for ratings and total time as a feature


if __name__ == '__main__':
    main()