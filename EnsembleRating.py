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

def main():
    pass

if __name__ == '__main__':
    main()