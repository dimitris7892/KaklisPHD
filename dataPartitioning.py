from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import math
import numpy as np
from scipy import spatial

from utils import *

class DefaultClusterer:
    '''
    Performs a clustering, returning:
    partitionsX: An array of instance sets from X, where each instance set belongs to the same cluster.
    partitionsY: An array of instance sets from Y, where each instance set belongs to the same cluster.
    partitionLabels: The array of labels mapped to each partition.
    centroids: A representative instance from each partition.
    clusteringModel: The sklearn clustering model (or None if not applicable)
    '''
    def clustering(self, dataX, nClusters, showPlot=False):
        return ([], [], [], [], None)

    def showClusterPlot(self, dataX, labels, nClusters):
        xData = np.asarray(dataX[:, 0]).T[0]
        yData = np.asarray(dataX[:, 1]).T[0]
        colors=tuple([numberToRGBAColor(l) for l in labels])

        plt.scatter(xData, yData, c=colors, cmap=plt.cm.gnuplot)
        plt.title("K-means with " + str(nClusters) + " clusters")
        plt.show()


class KMeansClusterer(DefaultClusterer):
    # Clusters data, for a given number of clusters
    def clustering(self, dataX, dataY = None, nClusters = None, showPlot=False, random_state=1000):

        # Check if we need to use dataY
        dataUpdatedX = dataX
        if dataY is not None:
            dataUpdatedX = np.append(dataX, np.asmatrix([dataY]).T, axis=1)

            # Init clustering model
        self._nClusters = nClusters
        self.random_state = random_state
        clusteringModel = self.getClusterer()

        # Fit the input data
        dataModel = clusteringModel.fit(dataUpdatedX)
        self._dataModel = dataModel
        # Get the cluster labels
        labels = dataModel.predict(dataUpdatedX)
        # Extract centroid values
        centroids = self.getCentroids()

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

        # Show plot, if requested
        if (showPlot):
            self.showClusterPlot(dataUpdatedX, labels, nClusters)

        return partitionsX, partitionsY, partitionLabels , centroids , clusteringModel


    def getClusterer(self, dataX=None, dataY = None):
        # Assign a default number of clusters if not provided
        if self._nClusters == None:
            if dataX != None:
                self._nClusters = int(math.log2(len(dataX) + 1))
            else:
                self._nClusters = 2

                # Return clusterer
        return KMeans(n_clusters=self._nClusters , random_state=self.random_state)

    def getCentroids(self):
        return self._dataModel.cluster_centers_


class BoundedProximityClusterer (DefaultClusterer):
    def clustering(self, dataX, dataY = None, nClusters = None, showPlot=False, random_state=1000):
        return self._clustering(dataX, dataY, nClusters, showPlot, random_state)

    # Clusters data, for a given number of clusters
    def _clustering(self, dataX, dataY = None, nClusters = None, showPlot=False, random_state=1000, xBound = None, yBound = None):
        # Init return arrays
        partitionsX = []
        partitionsY = []
        partitionLabels = np.zeros((dataX.shape[0], 1)) # Init to input data size
        representatives = []

        # V is the set of all V values in the training set.
        V = np.unique(dataX, axis=0)

        # R is the set of RPM values in the training set.
        R = np.unique(dataY, axis=0)

        # yBound is the bound of error in y axis
        if yBound is None:
            yBound = np.mean(np.max(dataY) - np.min(dataY)) / math.log(2.0 + dataY.shape[0], 2.0)

        # Initialize the set U of used V values to the empty set.
        U = np.empty((1, dataX.shape[1]))

        canContinue = True
        iCurClusterLbl = -1 # Init label counter
        while canContinue:
            # Get next cluster label
            iCurClusterLbl += 1

            # (Randomly) Select a value v1 in the V space not already in U.
            notInU = self.setDiff(U, V)
            v1 = notInU[np.random.randint(0, notInU.shape[0], 1)]
            # Add v1 to the U set
            U  = np.append(U, v1, axis=0)
            # Select for the given v1 a value r which is the average/median of R values for points with coordinate V=v1
            r = np.mean(self.getAllRowsYsWhereRowsXEqV(dataX, dataY, v1))
            # Collapse all V to one dimension, based on PCA
            pcaMapping = PCA(1)
            pcaMapping.fit(dataX)
            Vmapped = pcaMapping.transform(dataX)
            # xBound is the bound of error in x PCA axis
            if xBound is None:
                xBound = np.mean(np.max(Vmapped) - np.min(Vmapped)) / math.log(2.0 + dataY.shape[0], 2.0)

            # Select another point (v2, r) with r_v2 - r_v1 < bound in the V space not in U.
            notInU = self.setDiff(U, V)

            # Init selected v2 indices
            selectionIndices = []
            ## For each point in notInU
            for iIdx in range(notInU.shape[0]):
                # Examine whether the R of its corresponding group
                # is within xBound distance from the v1 R
                # and its distance in the PCA space is also equally bounded by yBound
                v2 = np.asarray([notInU[iIdx]])
                pcaDist = spatial.distance.euclidean(pcaMapping.transform(v2), pcaMapping.transform(v1))
                # Find all points with v2 as X value
                allCandidateV2Indices = self.getAllRowsYsWhereRowsXEqV(dataX, np.arange(dataY.shape[0]), v2[0])
                rDist = np.abs(np.mean(R[allCandidateV2Indices]) - r)
                if (pcaDist <= xBound) and (rDist <= yBound):
                    # then add to selection indices
                    selectionIndices.extend(allCandidateV2Indices)
                    # Update U with used v2
                    U  = np.append(U, v2, axis=0)

            # If nothing found
            if len(selectionIndices) == 0:
                # Go to next v1
                canContinue = notInU.size > 2
                continue
            # Add all points in T to new Cluster
            clusterX = np.append(v1, dataX[selectionIndices], axis=0)
            clusterY = np.append(np.asarray([r]), dataY[selectionIndices], axis=0)
            # Add to returned partitions
            partitionsX.append(clusterX)
            partitionsY.append(clusterY)
            # Create representative (centroid)
            representatives.append(np.append(np.mean(np.asarray(dataX), axis=0), np.mean(clusterY))) # TODO: Check if we need Y column

            # Store cluster
            partitionLabels[selectionIndices] = iCurClusterLbl

            # Add all points in T to U
            U = np.append(U, dataX[selectionIndices], axis=0)
            notInU = self.setDiff(U, V)

            # Repeat until no more points can be added.
            canContinue = notInU.size > 2

        # Show plot, if requested
        if (showPlot):
            self.showClusterPlot(dataX, partitionLabels, nClusters)

        # Return
        return partitionsX, partitionsY, partitionLabels, representatives, None

    def getAllRowsYsWhereRowsXEqV(self, dataX, dataY, v):
        return dataY[(dataX == v).all(axis=1)]

    def setDiff(self, U, V):
        notInU = set(map(tuple, V)).difference(map(tuple, U))
        notInU = np.asarray(list(map(list, notInU)))
        return notInU
