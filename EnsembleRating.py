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

PATH_TO_DATA = "./data_sample/"
PATH_TO_RATINGS = "./data_sample/sample_ratings.csv"

def main():
    pass

if __name__ == '__main__':
    main()