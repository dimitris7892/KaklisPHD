import sklearn.linear_model as sk
import numpy as np
import numpy.linalg

class BasePartitionModeler:
    def createModelsFor(self, partitions):
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


class LinearRegressionModeler:
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
        return numpy.linalg.norm(mean(self._partitionsPerModel[model])-point)