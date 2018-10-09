import sklearn.linear_model as sk

class BasePartitionModeler:
    def createModelsFor(self, partitions):
        pass

class LinearRegressionModeler:
    def createModelsFor(self, partitionsX, partitionsY, partition_labels):
        # Init result model list
        models = []
        # For each label/partition
        for idx, pCurLbl in enumerate(partition_labels):
            # Create a linear model
            curModel = sk.LinearRegression()
            # Fit to data
            curModel.fit(partitionsX[idx], partitionsY[idx])
            # Add to returned list
            models.append(curModel)

        # Return list of models
        return models
