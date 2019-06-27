"""
The EnsembleSegmenting task is as follows:
1. Preprocess the EMG data using EMGPreprocessor, put some aside as test data
2. Makes a model for each of the EMG record points (6 total)
    i.e One for all the bicep non-dom time series, one for all the bicep doms, etc.
3. For each model, split data into training and validation
4. Build similarity model and use DTW with most similar K neighbours to determine action time stamps.
5. Average time stamps to produce action time stamp predictions.
"""

def main():
    pass

if __name__ == '__main__':
    main()