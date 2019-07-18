"""Used to test new implementations"""
from EMGDataManager import EMGDataManager
import logging
logging.basicConfig(level=logging.DEBUG)

PATH_TO_DATA = "./data_sample/"
PATH_TO_ACTION_LABELS = "./data_sample/sample_action_labels.csv"

def main():
    dm = EMGDataManager(PATH_TO_DATA, PATH_TO_ACTION_LABELS, downsampler=True)

if __name__ == '__main__':
    main()