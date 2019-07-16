"""Used to test new implementations"""
from EMGDataManager import EMGDataManager
import logging
logging.basicConfig(level=logging.DEBUG)

PATH_TO_DATA = "./data_sample/"
PATH_TO_RATING_AND_TIMESTAMPS = "./data_sample/sample.csv"

def main():
    dm = EMGDataManager(PATH_TO_DATA, PATH_TO_RATING_AND_TIMESTAMPS)

if __name__ == '__main__':
    main()