from sklearn.cluster import KMeans , DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import math
import numpy as np
from scipy.interpolate import interp1d
import scipy as sp
import scipy.interpolate
from scipy.spatial import Delaunay,ConvexHull
from scipy import spatial,random
import colorsys
#from sklearn.model_selection import KFold
import sys
from matplotlib import pyplot
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import SpectralClustering
sys.setrecursionlimit(1000000)
#import plotly as py
#import plotly.graph_objs as go
#import SSplines as simplexSplines
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
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

    def showClusterPlot(self, dataX, labels, nClusters,centers,partitionsX,partitionsY):
        xData = np.asarray(dataX[:, 0]).T#[0]
        yData = np.asarray(dataX[:, 1]).T#[0]
        zData  = np.asarray(dataX[:, 2]).T#[0]
        #colors=tuple([numberToRGBAColor(l) for l in labels])
        ax = plt.axes(projection='3d')

        ax.scatter3D(xData,yData,zData,c= labels,cmap ='viridis')
        ax.scatter3D(centers[:, 0], centers[:, 1],centers[:,2], c='black', s=230, alpha=0.5)

        #plt.scatter(xData, yData, c=labels,s=60, cmap='viridis')
        plt.xlabel("STW")
        plt.ylabel("DRAFT")
        #plt.zlabel("DRAFT")
        #plt.scatter(centers[:, 0], centers[:, 1], c='black', s=230, alpha=0.5);
        plt.title("K-means with " + str(nClusters) + " clusters")
        #plt.show()
        #for i in range(0,len(partitionsX)):
        trace1 = go.Scatter3d(
            x=partitionsX[0][:,0],
            y=partitionsX[0][:,1],
            z=partitionsX[0][:,3],
            mode="markers",
            name="Cluster 0",
            marker=dict(color='rgba(255, 128, 255, 0.8)'),
            text=None)

        # trace2 is for 'Cluster 1'
        trace2 = go.Scatter3d(
            x=partitionsX[1][:,0],
            y=partitionsX[1][:,1],
            z=partitionsX[1][:,3],
            mode="markers",
            name="Cluster 1",
            marker=dict(color='rgba(255, 128, 2, 0.8)'),
            text=None)

        # trace3 is for 'Cluster 2'
        trace3 = go.Scatter3d(
            x=partitionsX[2][:,0],
            y=partitionsX[2][:,1],
            z=partitionsX[2][:,3],
            mode="markers",
            name="Cluster 2",
            marker=dict(color='rgba(0, 255, 200, 0.8)'),
            text=None)

        trace4 = go.Scatter3d(
            x=partitionsX[3][:, 0],
            y=partitionsX[3][:, 1],
            z=partitionsX[3][:, 3],
            mode="markers",
            name="Cluster 3",
            marker=dict(color='rgba(0, 255, 90, 0.8)'),
            text=None)

        trace5 = go.Scatter3d(
            x=partitionsX[4][:, 0],
            y=partitionsX[4][:, 1],
            z=partitionsX[4][:, 3],
            mode="markers",
            name="Cluster 4",
            marker=dict(color='rgba(0, 80, 200, 0.8)'),
            text=None)

        data = [trace1, trace2, trace3,trace4,trace5]

        title = "Visualizing Clusters in Three Dimensions "

        layout = dict(title=title,
                      xaxis=dict(title='STW', ticklen=5, zeroline=False),
                      yaxis=dict(title='WS', ticklen=5, zeroline=False)
                      )

        fig = dict(data=data, layout=layout)

        plot(fig)

        x=0

class KMeansPartitioner(DefaultPartitioner):
    # Clusters data, for a given number of clusters
    def clustering(self, dataX, dataY = None, dataW=None,nClusters = None, showPlot=False, random_state=1000):

        # Check if we need to use dataY
        dataUpdatedX = dataX
        initialDataX = dataX
        #dataY=None
        dataX = dataX.reshape(-1,1) if len(dataX.shape)==1 else dataX


        # define the model
        #model = RandomForestRegressor()
        # fit the model
        #model.fit(dataX, dataY)
        # get importance
        #importance = model.feature_importances_
        # summarize feature importance
        #for i, v in enumerate(importance):
            #print('Feature: %0d, Score: %.5f' % (i, v))
        # plot feature importance
        #plt.xlabel("Features")
        #plt.ylabel("Score")
        #plt.bar([x for x in range(len(importance))], importance)
        #plt.show()
        x=0

        if dataY is not None:
            #dataXcl = dataX[:,0:3]
            #dataYcl=np.append(dataX[:,3].reshape(-1,1), np.asmatrix([dataY]).T, axis=1)
            trnslatedDatax = np.array(np.append(dataX[:,0].reshape(-1,1),np.asmatrix([dataX[:,1],dataX[:,3]]).T,axis=1))
            dataUpdatedX = np.append(trnslatedDatax, np.asmatrix([dataY]).T, axis=1)

            # Init clustering model
        self._nClusters = nClusters
        self.random_state = random_state
        clusteringModel = self.getClusterer()

        # Fit the input data
        #dataUpdatedX=np.
        #dataUpdatedX = [dataX, dataY]
        try:
            dataModel = clusteringModel.fit(np.nan_to_num(trnslatedDatax))
        except:
            print ("Error in clustering")
        self._dataModel = dataModel
        # Get the cluster labels
        #labels = dataModel.labels_
        labels = dataModel.predict(trnslatedDatax)
        #labels = dataModel.labels_
        # Extract centroid values
        centroids = self.getCentroids()
        #centroids = None
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

        k = len(partitionsX)
        i=0
        if k > 1:
            indxForMerging=[]
            while i <k:
                if len(partitionsX[i]) <=1000:
                    partitionsX.remove(partitionsX[i])
                    partitionsY.remove(partitionsY[i])
                    indxForMerging.append(i)
                    k = len(partitionsX)
                i=i+1

            minDist =20000000000

            for k in indxForMerging:
                minI = 0
                for i in range(0,len(partitionsX)):
                    if i !=k:
                        dist = np.linalg.norm(np.mean(partitionsX[i],axis=0)-np.mean(partitionsX[k],axis=0))
                        if dist < minDist:
                            minDist = dist
                            minI=i
                partitionsX[minI] = np.concatenate([partitionsX[minI],partitionsX[k]])
                partitionsY[minI] = np.concatenate([partitionsY[minI], partitionsY[k]])

            partitionLabels = np.linspace(0, len(partitionsX), len(partitionsX))
        #for i in range(0,len(partitionsX)):
            #if len(partitionsX[i]) <=1000:
                #partitionsX, partitionsY, partitionLabels , dataX , dataY,centroids = self.clustering(dataX,dataY,None,nClusters-1 , False)
                #return partitionsX, partitionsY, partitionLabels , dataX , dataY,centroids



        #pca_2d = PCA(n_components=2)
        #trnslatedDatax = pca_2d.fit_transform(trnslatedDatax)

        #partitionsX = []
        #partitionsY = []
        #partitionLabels = []
        ##ballast
        print("NEW Number of clusters AFTER DELETING clusters with insufficient size: %d" % (len(partitionsX)))
        # Show plot, if requested
        #if (showPlot):
            #self.showClusterPlot(dataUpdatedX, labels, nClusters)
        #for k in partitionLabels:
            #print("RPM variance of cluster "+str(k) +": " + str(np.var(partitionsY[k]))+
            #"\n"+ "Velocity variance of cluster "+str(k)+": "+str(np.var(partitionsX[k])))
        #self.showClusterPlot(trnslatedDatax,labels,nClusters,centroids,partitionsX,partitionsY)
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
        #return SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                           #assign_labels='kmeans')
        #return DBSCAN(eps=0.6,min_samples=30)
        return KMeans(n_clusters=self._nClusters,random_state=self.random_state,)#init=initCentroids)

    def getCentroids(self):
        return self._dataModel.cluster_centers_


class KMeansPartitionerWS_WA(DefaultPartitioner):
    # Clusters data, for a given number of clusters
    def clustering(self, dataX, dataY = None, dataW=None,nClusters = None, showPlot=False, random_state=1000):

        # Check if we need to use dataY
        dataUpdatedX = dataX
        initialDataX = dataX
        #dataY=None
        dataX = dataX.reshape(-1,1) if len(dataX.shape)==1 else dataX


        # define the model
        #model = RandomForestRegressor()
        # fit the model
        #model.fit(dataX, dataY)
        # get importance
        #importance = model.feature_importances_
        # summarize feature importance
        #for i, v in enumerate(importance):
            #print('Feature: %0d, Score: %.5f' % (i, v))
        # plot feature importance
        #plt.xlabel("Features")
        #plt.ylabel("Score")
        #plt.bar([x for x in range(len(importance))], importance)
        #plt.show()
        x=0

        if dataY is not None:
            #dataXcl = dataX[:,0:3]
            #dataYcl=np.append(dataX[:,3].reshape(-1,1), np.asmatrix([dataY]).T, axis=1)
            trnslatedDatax = np.array(np.append(dataX[:,0].reshape(-1,1),np.asmatrix([dataX[:,1],dataX[:,2],dataX[:,3],dataX[:,4]]).T,axis=1))
            dataUpdatedX = np.append(trnslatedDatax, np.asmatrix([dataY]).T, axis=1)

            # Init clustering model
        self._nClusters = nClusters
        self.random_state = random_state
        clusteringModel = self.getClusterer()

        # Fit the input data
        #dataUpdatedX=np.
        #dataUpdatedX = [dataX, dataY]
        try:
            dataModel = clusteringModel.fit(np.nan_to_num(trnslatedDatax))
        except:
            print ("Error in clustering")
        self._dataModel = dataModel
        # Get the cluster labels
        #labels = dataModel.labels_
        labels = dataModel.predict(trnslatedDatax)
        #labels = dataModel.labels_
        # Extract centroid values
        centroids = self.getCentroids()
        #centroids = None
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

        #pca_2d = PCA(n_components=2)
        #trnslatedDatax = pca_2d.fit_transform(trnslatedDatax)

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
        #self.showClusterPlot(trnslatedDatax,labels,nClusters,centroids)
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
        #return SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                           #assign_labels='kmeans')
        #return DBSCAN(eps=0.6,min_samples=30)
        return KMeans(n_clusters=self._nClusters,random_state=self.random_state,)#init=initCentroids)

    def getCentroids(self):
        return self._dataModel.cluster_centers_


class KMeansPartitionerWH_WD(DefaultPartitioner):
    # Clusters data, for a given number of clusters
    def clustering(self, dataX, dataY = None, dataW=None,nClusters = None, showPlot=False, random_state=1000):

        # Check if we need to use dataY
        dataUpdatedX = dataX
        initialDataX = dataX
        #dataY=None
        dataX = dataX.reshape(-1,1) if len(dataX.shape)==1 else dataX


        # define the model
        #model = RandomForestRegressor()
        # fit the model
        #model.fit(dataX, dataY)
        # get importance
        #importance = model.feature_importances_
        # summarize feature importance
        #for i, v in enumerate(importance):
            #print('Feature: %0d, Score: %.5f' % (i, v))
        # plot feature importance
        #plt.xlabel("Features")
        #plt.ylabel("Score")
        #plt.bar([x for x in range(len(importance))], importance)
        #plt.show()
        x=0

        if dataY is not None:
            #dataXcl = dataX[:,0:3]
            #dataYcl=np.append(dataX[:,3].reshape(-1,1), np.asmatrix([dataY]).T, axis=1)
            trnslatedDatax = np.array(np.append(dataX[:,0].reshape(-1,1),np.asmatrix([dataX[:,1],dataX[:,2],dataX[:,5],dataX[:,6]]).T,axis=1))
            dataUpdatedX = np.append(trnslatedDatax, np.asmatrix([dataY]).T, axis=1)

            # Init clustering model
        self._nClusters = nClusters
        self.random_state = random_state
        clusteringModel = self.getClusterer()

        # Fit the input data
        #dataUpdatedX=np.
        #dataUpdatedX = [dataX, dataY]
        try:
            dataModel = clusteringModel.fit(np.nan_to_num(trnslatedDatax))
        except:
            print ("Error in clustering")
        self._dataModel = dataModel
        # Get the cluster labels
        #labels = dataModel.labels_
        labels = dataModel.predict(trnslatedDatax)
        #labels = dataModel.labels_
        # Extract centroid values
        centroids = self.getCentroids()
        #centroids = None
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

        #pca_2d = PCA(n_components=2)
        #trnslatedDatax = pca_2d.fit_transform(trnslatedDatax)

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
        #self.showClusterPlot(trnslatedDatax,labels,nClusters,centroids)
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
        #return SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                           #assign_labels='kmeans')
        #return DBSCAN(eps=0.6,min_samples=30)
        return KMeans(n_clusters=self._nClusters,random_state=self.random_state,)#init=initCentroids)

    def getCentroids(self):
        return self._dataModel.cluster_centers_


