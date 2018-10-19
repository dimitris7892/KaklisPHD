import numpy as np

class BaseFeatureExtractor:
    # Get a series of tuples, where the i-th tuple contains the current sample value X_i and the average X values over the previous 30 samples before i
    def extractFeatures(self, X, Y,history):
        HISTORY_SIZE = history

        Xnew = []
        prev = [[X[i], np.mean(X[i - HISTORY_SIZE:i])] for i in range(HISTORY_SIZE, len(X))]
        Xnew.append(prev)
        Xnew = np.array(Xnew).reshape(-1, 2)

        # Return data (ignoring lines that have no target value)
        return Xnew, Y[HISTORY_SIZE:]


    def extractFeatureswithVariance(self,X,Y,futureN):
        FUTUREv = futureN
        Xnew = [ ]
        next = [ [ X[ i ], np.var(X[ i:i+FUTUREv ]) ] for i in range(0, len(X)) ]
        Xnew.append(next)
        Xnew = np.array(Xnew).reshape(-1, 2)

        # Return data (ignoring lines that have no target value)
        return Xnew,Y