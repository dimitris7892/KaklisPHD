from pandas import Series
import numpy as np
import pandas as pd

class BaseSeriesReader:
    # Imports data from time series file. Expected format: CSV with 2 fields (velocity, RPM)
    def readSeriesDataFromFile(self, path,start,end,nrows):
        # Load file
        data = pd.read_csv(
            path, nrows=nrows)
        dt = data.values[ 0:, 3:23 ]
        X = dt[ start:end, 1 ]
        Y = dt[ start:end, 4 ]

        return X, Y


class UnseenSeriesReader (BaseSeriesReader):
    # Imports data from time series file. Expected format: CSV with 2 fields (velocity, RPM)
   pass