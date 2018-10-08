from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class DefaultClusterer:
    def clusteringV(self, dataX,nClusters, showPlot=False):
        return ([], [], [])

class KMeansClusterer:
    # Clusters velocity (V) data, for a given number of clusters
    def clusteringV(self, dataX,nClusters, showPlot=False, random_state=1000):
        # Init clustering model
        clusteringModel = KMeans(n_clusters=nClusters, random_state=random_state)
        # Fitting the input data
        clusteringModel = clusteringModel.fit(dataX)
        # Getting the cluster labels
        labels = clusteringModel.predict(dataX)
        # Centroid values
        centroids = clusteringModel.cluster_centers_

        if (showPlot):
            plt.scatter(dataX[:,0],dataX[:,1], c=labels)
            plt.title("K-means with "+str(nClusters)+" clusters")
            plt.show()

        return labels , centroids , clusteringModel
