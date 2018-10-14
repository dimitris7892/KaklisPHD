import sklearn.linear_model as sk
import numpy as np
import numpy.linalg
from scipy.spatial import Delaunay
import random

class BasePartitionModeler:
    def createModelsFor(self,partitionsX, partitionsY, partition_labels):
        pass

    def getBestModelForPoint(self, point):
        mBest = None
        dBestFit = 0
        # For each model
        for m in self._models:
            # If it is a better for the point
            dCurFit = self.getFitnessOfModelForPoint(m, point)
            if dCurFit > dBestFit:
                # Update the selected best model and corresponding fit
                dBestFit = dCurFit
                mBest = m

        return mBest

    def  getFitnessOfModelForPoint(self, model, point):
        return 0.0


class LinearRegressionModeler(BasePartitionModeler):
    def createModelsFor(self, partitionsX, partitionsY, partition_labels):
        # Init result model list
        models = []
        # Init model to partition map
        self._partitionsPerModel = {}
        # For each label/partition
        for idx, pCurLbl in enumerate(partition_labels):
            # Create a linear model
            curModel = sk.LinearRegression()
            # Fit to data
            curModel.fit(partitionsX[idx], partitionsY[idx])
            # Add to returned list
            models.append(curModel)
            self._partitionsPerModel[curModel] = partitionsX[idx]

        # Update private models
        self._models = models

        # Return list of models
        return models



    def getFitnessOfModelForPoint(self, model, point):
        return numpy.linalg.norm(np.mean(self._partitionsPerModel[model])-point)

    def getTriangle(self,X,Y,usedV,usedRpm,errorBound):
        xy=np.vstack([X,Y])
        if usedV==[]:
            pointV1 = random.sample(X)
            pointV2RPM =random.sample(list(xy).index(n>pointV1 for n in xy[0,:]))

        else:
            pointV1= random.sample(X[filter(n not in usedV for n in range(0,len(X)))])
            p = [xy[filter(n not in usedV for n in range(0,len(xy)))]]
            pointV2RPM = random.sample(list((p).index(n > pointV1 for n in (p[0,:]))))

        pointRpm = [np.mean(n) for n in Y[list(X.index(pointV1))]]

        tri = Delaunay([pointV1,pointRpm],[pointV2RPM[0],pointV2RPM[1]+errorBound],[pointV2RPM[0],pointV2RPM[1]-errorBound])
        return tri
