from tslearn.utils import to_time_series_dataset
from sklearn.linear_model import SGDRegressor, SGDClassifier
import numpy as np

# Use one of these two models and the un-down-sampled uneven time series data
from tslearn.clustering import GlobalAlignmentKernelKMeans
from tslearn.clustering import TimeSeriesKMeans  # currently used

# this should actually be normalized as per their column value in data manager
mock_dataset_muscle1 = np.array([[1, 2, 3, 3],
                                 [3, 2, 3, 6],
                                 [1, 2, 6, 7],
                                 [2, 2, 3, 7],
                                 [3, 2, 3, 6]])
mock_dataset_muscle2 = np.array([[2, 5, 6, 7, 8, 9],
                                 [3, 2, 5, 6, 7, 3],
                                 [6, 5, 3, 4, 7, 8],
                                 [6, 5, 3, 4, 7, 8],
                                 [2, 5, 6, 7, 8, 9]])
mock_labels = np.array([1, 0, 3, 2, 1])  # the rating scores
mock_times = np.array([100, 200, 400, 300, 200])

def main():
    X1 = to_time_series_dataset(mock_dataset_muscle1)
    y1 = mock_labels
    X_train1 = X1[:-2]
    y_train1 = y1[:-2]
    X_test1 = X1[-2:]
    y_test1 = y1[-2:]
    # clf1 = KNeighborsTimeSeriesClassifier(n_neighbors=5, metric="dtw")
    clf1 = TimeSeriesKMeans(metric="dtw")
    clf1.fit(X_train1, y_train1)
    pred_train1 = clf1.predict(X_train1)
    pred_test1 = clf1.predict(X_test1)
    print("TRAINING SET 1")
    print("Prediction: " + str(pred_test1))
    print("Actual: " + str(y_test1))

    print("\n")

    X2 = to_time_series_dataset(mock_dataset_muscle2)
    y2 = mock_labels
    X_train2 = X2[:-2]
    y_train2 = y2[:-2]
    X_test2 = X2[-2:]
    y_test2 = y2[-2:]
    clf2 = TimeSeriesKMeans(metric="dtw")
    # clf2 = KNeighborsTimeSeriesClassifier(n_neighbors=5, metric="dtw")
    clf2.fit(X_train2, y_train2)
    pred_train2 = clf2.predict(X_train2)
    pred_test2 = clf2.predict(X_test2)
    print("TRAINING SET 2")
    print("Prediction: " + str(pred_test2))
    print("Actual: " + str(y_test2))

    print("\n")

    times_train = mock_times[:-2]
    times_test = mock_times[-2:]
    X_train = np.stack((pred_train1, pred_train2, times_train)).transpose()
    X_test = np.stack((pred_test1, pred_test2, times_test)).transpose()
    y_train = np.array(mock_labels[:-2]).reshape((len(X_train),))
    y_test = mock_labels[-2:]
    sgd = SGDClassifier()
    sgd.fit(X_train, y_train)
    pred = sgd.predict(X_test)
    print("ENSEMBLE")
    print("Prediction: " + str(pred))
    print("Actual: " + str(y_test))
    print("Score: " + str(sgd.score(X_test, y_test)))

if __name__ == '__main__':
    main()