from pandas import Series
import numpy as np

# Imports data from time series file
def importSeriesDataFromFile(path):
    series = Series.from_csv(path, sep=',', header=0)
    Y = series.values
    X = series._index.values
    Xnew=[]
    prev=[[X[i],np.mean(X[i-30:i])] for i in range(30,len(X))]
    Xnew.append(prev)
    Xnew = np.array(Xnew).reshape(-1,2)
    return Xnew,Y