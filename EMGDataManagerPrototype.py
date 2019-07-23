"""Used to test new implementations"""
from EMGDataManager import EMGDataManager
import logging
logging.basicConfig(level=logging.DEBUG)

PATH_TO_DATA = "./data_sample/"
PATH_TO_RATINGS = "./data_sample/sample_ratings.csv"
PATH_TO_ACTION_LABELS = "./data_sample/sample_action_labels.csv"

def main():
    dm = EMGDataManager(PATH_TO_DATA, path_to_timestamps=PATH_TO_ACTION_LABELS,
                        path_to_ratings=PATH_TO_RATINGS, downsampler=True)

if __name__ == '__main__':
    main()