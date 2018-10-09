from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
import numpy as np

class DefaultClusterer:
    def clusteringV(self, dataX,nClusters, showPlot=False):
        return ([], [], [])

class KMeansClusterer:
    # Clusters velocity (V) data, for a given number of clusters
    def clusteringV(self, dataX, dataY = None, nClusters = None, showPlot=False, random_state=1000):

        # Init clustering model
        self._nClusters = nClusters
        self.random_state = random_state
        clusteringModel = self.getClusterer()

        # Fit the input data
        dataModel = clusteringModel.fit(dataX)
        self._dataModel = dataModel
        # Get the cluster labels
        labels = dataModel.predict(dataX)
        # Extract centroid values
        centroids = self.getCentroids()

        # Show plot, if requested
        if showPlot:
            plt.scatter(dataX[:,0],dataX[:,1], c=labels)
            plt.title("K-means with "+str(nClusters)+" clusters")
            plt.show()

        # Create partitions, based on common label
        partitionsX = []
        partitionsY = []
        partitionLabels = []
        # For each label
        for curLbl in np.unique(labels):
            # Create a partition for X using records with corresponding label equal to the current
            partitionsX.append(np.asarray(dataX[labels == curLbl]))
            # Create a partition for Y using records with corresponding label equal to the current
            partitionsY.append(np.asarray(dataY[labels == curLbl]))
            # Keep partition label to ascertain same order of results
            partitionLabels.append(curLbl)

        return partitionsX, partitionsY, partitionLabels , centroids , clusteringModel

    def getClusterer(self):
        # Assign a default number of clusters if not provided
        if self._nClusters == None:
            self._nClusters = int(math.log2(len(dataX) + 1))

        # Return clusterer
        return KMeans(n_clusters=self._nClusters , random_state=self.random_state)

    def getCentroids(self):
        return self._dataModel.cluster_centers_