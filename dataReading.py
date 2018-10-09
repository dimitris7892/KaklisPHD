from pandas import Series
import numpy as np

class BaseSeriesReader:
    # Imports data from time series file. Expected format: CSV with 2 fields (velocity, RPM)
    def readSeriesDataFromFile(self, path):
        # Load file
        series = Series.from_csv(path, sep=',', header=0)
        # Separate values
        Y = series.values
        X = series._index.values
        return X,Y


class UnseenSeriesReader (BaseSeriesReader):
    # Imports data from time series file. Expected format: CSV with 2 fields (velocity, RPM)
    def readSeriesDataFromFile(self, path):
        pass # TODO: Complete