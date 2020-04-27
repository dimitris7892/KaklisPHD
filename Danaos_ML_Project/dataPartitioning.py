from sklearn.cluster import KMeans , DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import math
import numpy as np
from scipy.spatial import Delaunay,ConvexHull
from scipy import spatial,random
import colorsys
#from sklearn.model_selection import KFold
import sys
sys.setrecursionlimit(1000000)
#import SSplines as simplexSplines
#from utils import *
#from dtw import dtw
#from matplotlib import rc
#plt.rcParams.update({'font.size':45})
#rc('text', usetex=True)

class DefaultPartitioner:
    '''
    Performs a partitioning of the data, returning:
    partitionsX: An array of instance sets from X, where each instance set belongs to the same cluster.
    partitionsY: An array of instance sets from Y, where each instance set belongs to the same cluster.
    partitionLabels: The array of labels mapped to each partition.
    centroids: A representative instance from each partition.
    clusteringModel: The sklearn clustering model (or None if not applicable)

    The Default action is one partition containing everything.
    '''

    def clustering(self, dataX, dataY=None, nClusters=None, showPlot=False, random_state=1000):
        if nClusters is not None:
            print("WARNING: DefaultPartitioner ignores number of clusters.")
        return ([dataX], [dataY], [0], [dataX[0]], None)

    def showClusterPlot(self, dataX, labels, nClusters):
        xData = np.asarray(dataX[:, 0]).T[0]
        yData = np.asarray(dataX[:, 1]).T[0]
        #colors=tuple([numberToRGBAColor(l) for l in labels])

        plt.scatter(xData, yData, c=labels)
        plt.title("K-means with " + str(nClusters) + " clusters")
        plt.show()

class KMeansPartitioner(DefaultPartitioner):
    # Clusters data, for a given number of clusters
    def clustering(self, dataX, dataY = None, dataW=None,nClusters = None, showPlot=False, random_state=1000):

        # Check if we need to use dataY
        dataUpdatedX = dataX
        #dataY=None
        dataX = dataX.reshape(-1,1) if len(dataX.shape)==1 else dataX
        if dataY is not None:
            #dataXcl = dataX[:,0:3]
            #dataYcl=np.append(dataX[:,3].reshape(-1,1), np.asmatrix([dataY]).T, axis=1)
            trnslatedDatax = np.array(np.append(dataX[:,0].reshape(-1,1),np.asmatrix([dataX[:,1],dataX[:,3],dataX[:,4]]).T,axis=1))
            dataUpdatedX = np.append(trnslatedDatax, np.asmatrix([dataY]).T, axis=1)

            # Init clustering model
        self._nClusters = nClusters
        self.random_state = random_state
        clusteringModel = self.getClusterer()

        # Fit the input data
        #dataUpdatedX=np.
        #dataUpdatedX = [dataX, dataY]
        try:
            dataModel = clusteringModel.fit(np.nan_to_num(dataUpdatedX))
        except:
            print ("Error in clustering")
        self._dataModel = dataModel
        # Get the cluster labels
        #labels = dataModel.labels_
        labels = dataModel.predict(dataUpdatedX)
        # Extract centroid values
        centroids = self.getCentroids()
        from scipy.interpolate import interp1d
        import scipy as sp
        import scipy.interpolate
        #spline = sp.interpolate.Rbf(centroids[0:,0], centroids[0:,1], centroids[0:,2], function='thin_plate', smooth=5, episilon=5)
        #f2 = interp1d(centroids[0:,0], centroids[0:,2], kind='cubic')
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

        #partitionsX = []
        #partitionsY = []
        #partitionLabels = []
        ##ballast

        # Show plot, if requested
        #if (showPlot):
            #self.showClusterPlot(dataUpdatedX, labels, nClusters)
        #for k in partitionLabels:
            #print("RPM variance of cluster "+str(k) +": " + str(np.var(partitionsY[k]))+
            #"\n"+ "Velocity variance of cluster "+str(k)+": "+str(np.var(partitionsX[k])))
        return partitionsX, partitionsY, partitionLabels , dataX , dataY,centroids


    def getClusterer(self, dataX=None, dataY = None):
        # Assign a default number of clusters if not provided
        if self._nClusters == None:
            if dataX != None:
                self._nClusters = int(math.log2(len(dataX) + 1))
            else:
                self._nClusters = 2

        print("Number of clusters: %d"%(self._nClusters))
                # Return clusterer
        #return DBSCAN(min_samples=2,eps=1)
        initCentroids = np.array([[7,1],[7,7],[7,21],[7,30],
                                 [10,1],[10,7],[10,21],[10,30],
                                 [12,1],[12,7],[12,21],[12,30],
                                 [14,1],[14,7],[14,21],[14,30]]
                                )
        #n_custers=self._nClusters
        return KMeans(n_clusters=self._nClusters,random_state=self.random_state)#init=initCentroids)

    def getCentroids(self):
        return self._dataModel.cluster_centers_










