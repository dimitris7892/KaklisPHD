from pandas import Series
import numpy as np
import pandas as pd

class BaseSeriesReader:
    # Imports data from time series file. Expected format: CSV with 2 fields (velocity, RPM)
    def readSeriesDataFromFile(self, path,start,end,nrows,case):
        # Load file
        data = pd.read_csv(
            path)#,nrows=nrows)
        dt = data.values[ 0:, 3:23 ]
        if self.__class__.__name__=='UnseenSeriesReader':
            X = dt[ start:end, 1 ]
            Y = dt[ start:end, 4 ]
            W = dt[start:end,0]
        else:
            #X = dt[ :, 1 ]
            #Y = dt[ :, 4 ]
            X = dt[ 100000:103000, 1 ]
            Y = dt[ 100000:103000, 4 ]
            W = dt[ 100000:103000, 0]

        return np.nan_to_num(X.astype(float)),np.nan_to_num(Y.astype(float)) , np.nan_to_num(W.astype(float))


class UnseenSeriesReader (BaseSeriesReader):
    # Imports data from time series file. Expected format: CSV with 2 fields (velocity, RPM)
   pass