from sklearn.cluster import KMeans , DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import math
import numpy as np
import pyearth as sp
from tensorflow.keras import *
from scipy.spatial import Delaunay,ConvexHull
import tensorflow as tf
from tensorflow import keras
#from scipy import spatial,random
import colorsys
#from keras import losses
#from sklearn.model_selection import KFold
import sys
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

class TensorFlowCl(DefaultPartitioner):

    def getBestPartitionForPoint(self, point, partitions):
        # mBest = None
        mBest = None
        dBestFit = 0
        # For each model
        for m in range(0, len(partitions)):
            # If it is a better for the point
            dCurFit = self.getFitnessOfPoint(partitions, m, point)
            if dCurFit > dBestFit:
                # Update the selected best model and corresponding fit
                dBestFit = dCurFit
                mBest = m

        if mBest == None:
            return 0, 0
        else:
            return mBest, dBestFit

    def clustering(self, dataX, dataY=None, dataW=None, nClusters=None, showPlot=False, random_state=1000):

        models = []
        X=dataX
        Y=dataY
        # partitionsX=np.array(partitionsX[0])
        # partitionsY = np.array(partitionsY[0])

        #self.ClustersNum = len(partitionsX)
        self.modelId = -1
        # partition_labels = len(partitionsX)
        # Init model to partition map
        self._partitionsPerModel = {}



        def getFitnessOfPoint(partitions, cluster, point):
            return 1.0 / (1.0 + np.linalg.norm(np.mean(partitions[cluster]) - point))


        def baseline_modelDeepCl():
            # create model
            neurons=6
            model = keras.models.Sequential()

            model.add(keras.layers.Dense(20, input_shape=(7,)))
            #model.add(keras.layers.Dense(15, ))
            #model.add(keras.layers.Dense(10, ))
            #model.add(keras.layers.Dense(5, ))
            #while neurons >=2 :
                #model.add(keras.layers.Dense(neurons , ))
                #neurons = neurons-1

            #neurons = 3
            #while neurons <genModelKnots - 1 :
                #model.add(keras.layers.Dense(neurons , ))
                #neurons = neurons+1
            model.add(keras.layers.Dense(genModelKnots -1 , ))

            # Compile model
            model.compile(loss=custom_loss , optimizer=keras.optimizers.Adam())
            ###print(model.summary())
            return model

        def custom_loss(y_true,y_pred):

            #return tf.keras.losses.mean_squared_error(y_true,y_pred) + \
                   return tf.keras.losses.mean_squared_error(y_true,y_pred) #+ tf.keras.losses.categorical_crossentropy(y_true,y_pred) +\
                    #tf.keras.losses.kullback_leibler_divergence(y_true, y_pred)



        seed = 7
        np.random.seed(seed)

        self.genF = None

        sModel = []
        sr = sp.Earth(max_degree=1)
        sr.fit(X, Y)
        sModel.append(sr)
        import csv
        csvModels = []
        genModelKnots = []
        neurons=3


        self.modelId = 'Gen'
        self.countTimes = 0
        self.models = {"data": []}
        self.intercepts = []
        self.interceptsGen = 0
        self.SelectedFuncs = 0
        for models in sModel:
            modelSummary = str(models.summary()).split("\n")[4:]

            with open('./model_Gen_.csv', mode='w') as data:
                csvModels.append('./model_Gen_.csv')
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    ['Basis', 'Coeff'])
                for row in modelSummary:
                    row = np.delete(np.array(row.split(" ")), [i for i, x in enumerate(row.split(" ")) if x == ""])
                    try:
                        basis = row[0]
                        pruned = row[1]
                        coeff = row[2]
                        # if basis=='x0' :continue
                        if pruned == "No":
                            data_writer.writerow([basis, coeff])
                            genModelKnots.append(basis)
                    except:
                        x = 0

            genModelKnots = len(genModelKnots)
            # modelCount += 1
            # models.append(autoencoder)


        self.count = 0

        def extractFunctionsFromSplines(x0, x1, x2, x3, x4, x5, x6):
            piecewiseFunc = []
            self.count = self.count + 1
            for csvM in csvModels:
                if csvM != './model_' + str(self.modelId) + '_.csv':
                    continue
                # id = csvM.split("_")[ 1 ]
                # piecewiseFunc = [ ]

                with open(csvM) as csv_file:
                    data = csv.reader(csv_file, delimiter=',')
                    for row in data:
                        # for d in row:
                        if [w for w in row if w == "Basis"].__len__() > 0:
                            continue
                        if [w for w in row if w == "(Intercept)"].__len__() > 0:
                            self.interceptsGen = float(row[1])
                            continue

                        if row.__len__() == 0:
                            continue
                        d = row[0]
                        if self.count == 1:
                            self.intercepts.append(float(row[1]))

                        if d.split("*").__len__() == 1:
                            split = ""
                            try:
                                split = d.split('-')[0][2:3]
                                if split != "x":
                                    split = d.split('-')[1]
                                    num = float(d.split('-')[0].split('h(')[1])
                                    # if id == id:
                                    # if float(row[ 1 ]) < 10000:
                                    try:
                                        # piecewiseFunc.append(
                                        # tf.math.multiply(tf.cast(tf.math.less(x, num), tf.float32),
                                        # (num - inputs)))
                                        if split.__contains__("x0"):
                                            piecewiseFunc.append((num - x0))  # * float(row[ 1 ]))
                                        if split.__contains__("x1"):
                                            piecewiseFunc.append((num - x1))  # * float(row[ 1 ]))
                                        if split.__contains__("x2"):
                                            piecewiseFunc.append((num - x2))  # * float(row[ 1 ]))
                                        if split.__contains__("x3"):
                                            piecewiseFunc.append((num - x3))  # * float(row[ 1 ]))
                                        if split.__contains__("x4"):
                                            piecewiseFunc.append((num - x4))  # * float(row[ 1 ]))
                                        if split.__contains__("x5"):
                                            piecewiseFunc.append((num - x5))  # * float(row[ 1 ]))
                                        if split.__contains__("x6"):
                                            piecewiseFunc.append((num - x6))  # * float(row[ 1 ]))
                                    # if id ==  self.modelId:
                                    # inputs = tf.where(x >= num, float(row[ 1 ]) * (inputs - num), inputs)
                                    except:
                                        dc = 0
                                else:
                                    ##x0 or x1
                                    split = d.split('-')[0]
                                    num = float(d.split('-')[1].split(')')[0])
                                    # if id == id:
                                    # if float(row[ 1 ]) < 10000:
                                    try:
                                        if split.__contains__("x0"):
                                            piecewiseFunc.append((x0 - num))  # * float(row[ 1 ]))
                                        if split.__contains__("x1"):
                                            piecewiseFunc.append((x1 - num))  # * float(row[ 1 ]))
                                        if split.__contains__("x2"):
                                            piecewiseFunc.append((x2 - num))  # * float(row[ 1 ]))
                                        if split.__contains__("x3"):
                                            piecewiseFunc.append((x3 - num))  # * float(row[ 1 ]))
                                        if split.__contains__("x4"):
                                            piecewiseFunc.append((x4 - num))  # * float(row[ 1 ]))
                                        if split.__contains__("x5"):
                                            piecewiseFunc.append((x5 - num))  # * float(row[ 1 ]))
                                        if split.__contains__("x6"):
                                            piecewiseFunc.append((x6 - num))  # * float(row[ 1 ]))

                                        # piecewiseFunc.append(
                                        # tf.math.multiply(tf.cast(tf.math.greater(x, num), tf.float32),
                                        # (inputs - num)))
                                    # if id == self.modelId:
                                    # inputs = tf.where(x <= num, float(row[ 1 ]) * (num - inputs), inputs)
                                    except:
                                        dc = 0
                            except:
                                # if id == id:
                                # if float(row[ 1 ]) < 10000:
                                try:
                                    piecewiseFunc.append(x0)

                                    # piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                    # (inputs)))

                                    # inputs = tf.where(x >= 0, float(row[ 1 ]) * inputs, inputs)
                                # continue
                                except:
                                    dc = 0

                        else:
                            funcs = d.split("*")
                            nums = []
                            flgFirstx = False
                            flgs = []
                            for r in funcs:
                                try:
                                    if r.split('-')[0][2] != "x":
                                        flgFirstx = True
                                        nums.append(float(r.split('-')[0].split('h(')[1]))

                                    else:
                                        nums.append(float(r.split('-')[1].split(')')[0]))

                                    flgs.append(flgFirstx)
                                except:
                                    flgFirstx = False
                                    flgs = []
                                    split = d.split('-')[0][2]
                                    try:
                                        if d.split('-')[0][2] == "x":
                                            # if id == id:
                                            # if float(row[ 1 ]) < 10000:
                                            try:

                                                if d.split("-")[0].__contains__("x1") and d.split("-")[1].__contains__(
                                                        "x1"):
                                                    split = "x1"
                                                if d.split("-")[0].__contains__("x0") and d.split("-")[1].__contains__(
                                                        "x0"):
                                                    split = "x0"
                                                if d.split("-")[0].__contains__("x0") and d.split("-")[1].__contains__(
                                                        "x1"):
                                                    split = "x01"
                                                if d.split("-")[0].__contains__("x1") and d.split("-")[1].__contains__(
                                                        "x0"):
                                                    split = "x10"

                                                if split == "x0":
                                                    piecewiseFunc.append(x0 * (x0 - nums[0]) * float(row[1]))
                                                elif split == "x1":
                                                    piecewiseFunc.append(x1 * (x1 - nums[0]) * float(row[1]))
                                                elif split == "x01":
                                                    piecewiseFunc.append(x0 * (x1 - nums[0]) * float(row[1]))
                                                elif split == "x10":
                                                    piecewiseFunc.append(x1 * (x0 - nums[0]) * float(row[1]))
                                                # piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                                # (inputs) * (
                                                # inputs - nums[ 0 ])))

                                                # inputs = tf.where(x >= 0,
                                                # float(row[ 1 ]) * (inputs) * (inputs - nums[ 0 ]), inputs)
                                            except:
                                                dc = 0

                                        else:
                                            flgFirstx = True
                                            # if id == id:
                                            # if float(row[ 1 ]) < 10000:
                                            try:

                                                if d.split("-")[0].__contains__("x1") and d.split("-")[
                                                    1].__contains__("x1"):
                                                    split = "x1"
                                                if d.split("-")[0].__contains__("x0") and d.split("-")[
                                                    1].__contains__("x0"):
                                                    split = "x0"
                                                if d.split("-")[0].__contains__("x0") and d.split("-")[
                                                    1].__contains__("x1"):
                                                    split = "x01"
                                                if d.split("-")[0].__contains__("x1") and d.split("-")[
                                                    1].__contains__("x0"):
                                                    split = "x10"

                                                if split == "x0":
                                                    piecewiseFunc.append(x0 * (nums[0] - x0) * float(row[1]))
                                                elif split == "x1":
                                                    piecewiseFunc.append(x1 * (nums[0] - x1) * float(row[1]))
                                                elif split == "x01":
                                                    piecewiseFunc.append(x0 * (nums[0] - x1) * float(row[1]))
                                                elif split == "x10":
                                                    piecewiseFunc.append(x1 * (nums[0] - x0) * float(row[1]))

                                                # piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                                # (inputs) * (
                                                # nums[ 0 ] - inputs)))

                                                # inputs = tf.where(x > 0 ,
                                                # float(row[ 1 ]) * (inputs) * (nums[ 0 ] - inputs),inputs)
                                                flgs.append(flgFirstx)
                                            except:
                                                dc = 0

                                    except:
                                        # if id == id:
                                        # if float(row[ 1 ]) < 10000:
                                        try:
                                            piecewiseFunc.append(x0)

                                            # piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                            # (inputs)))

                                            # inputs = tf.where(x >= 0, float(row[ 1 ]) * (inputs), inputs)
                                        except:
                                            dc = 0
                            try:
                                # if id == id:
                                if flgs.count(True) == 2:
                                    # if float(row[ 1 ])<10000:
                                    try:

                                        if d.split("-")[0].__contains__("x1") and d.split("-")[1].__contains__(
                                                "x1"):
                                            split = "x1"
                                        if d.split("-")[0].__contains__("x0") and d.split("-")[1].__contains__(
                                                "x0"):
                                            split = "x0"
                                        if d.split("-")[0].__contains__("x0") and d.split("-")[1].__contains__(
                                                "x1"):
                                            split = "x01"
                                        if d.split("-")[0].__contains__("x1") and d.split("-")[1].__contains__(
                                                "x0"):
                                            split = "x10"

                                        if split == "x0":
                                            piecewiseFunc.append((nums[0] - x0) * (nums[1] - x0) * float(row[1]))
                                        elif split == "x1":
                                            piecewiseFunc.append((nums[0] - x1) * (nums[1] - x1) * float(row[1]))
                                        elif split == "x01":
                                            piecewiseFunc.append((nums[0] - x0) * (nums[1] - x1) * float(row[1]))
                                        elif split == "x10":
                                            piecewiseFunc.append((nums[0] - x1) * (nums[1] - x0) * float(row[1]))

                                        # piecewiseFunc.append(tf.math.multiply(tf.cast(
                                        # tf.math.logical_and(tf.math.less(x, nums[ 0 ]),
                                        # tf.math.less(x, nums[ 1 ])), tf.float32),
                                        # (nums[ 0 ] - inputs) * (
                                        # nums[ 1 ] - inputs)))

                                        # inputs = tf.where(x < nums[0] and x < nums[1],
                                        # float(row[ 1 ]) * (nums[ 0 ] - inputs) * (
                                        # nums[ 1 ] - inputs), inputs)
                                    except:
                                        dc = 0

                                elif flgs.count(False) == 2:
                                    # if float(row[ 1 ]) < 10000:
                                    try:
                                        if d.split("-")[0].__contains__("x1") and d.split("-")[1].__contains__(
                                                "x1"):
                                            split = "x1"
                                        if d.split("-")[0].__contains__("x0") and d.split("-")[1].__contains__(
                                                "x0"):
                                            split = "x0"
                                        if d.split("-")[0].__contains__("x0") and d.split("-")[1].__contains__(
                                                "x1"):
                                            split = "x01"
                                        if d.split("-")[0].__contains__("x1") and d.split("-")[1].__contains__(
                                                "x0"):
                                            split = "x10"

                                        if split == "x0":
                                            piecewiseFunc.append((x0 - nums[0]) * (x0 - nums[1]) * float(row[1]))
                                        elif split == "x1":
                                            piecewiseFunc.append((x1 - nums[0]) * (x1 - nums[1]) * float(row[1]))
                                        elif split == "x01":
                                            piecewiseFunc.append((x0 - nums[0]) * (x1 - nums[1]) * float(row[1]))
                                        elif split == "x10":
                                            piecewiseFunc.append((x1 - nums[0]) * (x0 - nums[1]) * float(row[1]))
                                        # inputs = tf.where(x > nums[ 0 ] and x > nums[ 1 ],
                                        # float(row[ 1 ]) * (inputs - nums[ 0 ]) * (
                                        # inputs - nums[ 1 ]), inputs)
                                    except:
                                        dc = 0
                                else:
                                    try:
                                        if flgs[0] == False:
                                            if nums.__len__() > 1:
                                                # if float(row[ 1 ]) < 10000:
                                                try:
                                                    if d.split("-")[0].__contains__("x1") and d.split("-")[
                                                        2].__contains__("x1"):
                                                        split = "x1"
                                                    if d.split("-")[0].__contains__("x0") and d.split("-")[
                                                        2].__contains__("x0"):
                                                        split = "x0"
                                                    if d.split("-")[0].__contains__("x0") and d.split("-")[
                                                        2].__contains__("x1"):
                                                        split = "x01"
                                                    if d.split("-")[0].__contains__("x1") and d.split("-")[
                                                        2].__contains__("x0"):
                                                        split = "x10"

                                                    if split == "x0":
                                                        piecewiseFunc.append(
                                                            (x0 - nums[0]) * (nums[1] - x0) * float(row[1]))
                                                    elif split == "x1":
                                                        piecewiseFunc.append(
                                                            (x1 - nums[0]) * (nums[1] - x1) * float(row[1]))
                                                    elif split == "x01":
                                                        piecewiseFunc.append(
                                                            (x0 - nums[0]) * (nums[1] - x1) * float(row[1]))
                                                    elif split == "x10":
                                                        piecewiseFunc.append(
                                                            (x1 - nums[0]) * (nums[1] - x0) * float(row[1]))




                                                # inputs = tf.where(x > nums[ 0 ] and x < nums[ 1 ],
                                                # float(row[ 1 ]) * (inputs - nums[ 0 ]) * (
                                                # nums[ 1 ] - inputs), inputs)
                                                except:
                                                    dc = 0
                                            else:
                                                # if float(row[ 1 ]) < 10000:
                                                try:
                                                    if d.split("-")[0].__contains__("x1") and d.split("-")[
                                                        1].__contains__("x1"):
                                                        split = "x1"
                                                    if d.split("-")[0].__contains__("x0") and d.split("-")[
                                                        1].__contains__("x0"):
                                                        split = "x0"
                                                    if d.split("-")[0].__contains__("x0") and d.split("-")[
                                                        1].__contains__("x1"):
                                                        split = "x01"
                                                    if d.split("-")[0].__contains__("x1") and d.split("-")[
                                                        1].__contains__("x0"):
                                                        split = "x10"

                                                    piecewiseFunc.append((x0 - nums[0]) * float(row[1]))

                                                    # inputs = tf.where(x > nums[0],
                                                    # float(row[ 1 ]) * (inputs - nums[ 0 ]), inputs)
                                                except:
                                                    dc = 0
                                        else:
                                            if nums.__len__() > 1:
                                                # if float(row[ 1 ]) < 10000:
                                                try:
                                                    if d.split("-")[0].__contains__("x1") and d.split("-")[
                                                        1].__contains__("x1"):
                                                        split = "x1"
                                                    if d.split("-")[0].__contains__("x0") and d.split("-")[
                                                        1].__contains__("x0"):
                                                        split = "x0"
                                                    if d.split("-")[0].__contains__("x0") and d.split("-")[
                                                        1].__contains__("x1"):
                                                        split = "x01"
                                                    if d.split("-")[0].__contains__("x1") and d.split("-")[
                                                        1].__contains__("x0"):
                                                        split = "x10"

                                                    if split == "x0":
                                                        piecewiseFunc.append(
                                                            (nums[0] - x0) * (x0 - nums[1]) * float(row[1]))
                                                    elif split == "x1":
                                                        piecewiseFunc.append(
                                                            (nums[0] - x1) * (x1 - nums[1]) * float(row[1]))
                                                    elif split == "x01":
                                                        piecewiseFunc.append(
                                                            (nums[0] - x0) * (x1 - nums[1]) * float(row[1]))
                                                    elif split == "x10":
                                                        piecewiseFunc.append(
                                                            (nums[0] - x1) * (x0 - nums[1]) * float(row[1]))
                                                    # inputs = tf.where(x < nums[ 0 ] and x > nums[1],
                                                    # float(row[ 1 ]) * (nums[ 0 ] - inputs) * (
                                                    # inputs - nums[ 1 ]), inputs)
                                                except:
                                                    dc = 0
                                            else:
                                                # if float(row[ 1 ]) < 10000:
                                                try:

                                                    if d.split("-")[0].__contains__("x1") and d.split("-")[
                                                        1].__contains__("x1"):
                                                        split = "x1"
                                                    if d.split("-")[0].__contains__("x0") and d.split("-")[
                                                        1].__contains__("x0"):
                                                        split = "x0"
                                                    if d.split("-")[0].__contains__("x0") and d.split("-")[
                                                        1].__contains__("x1"):
                                                        split = "x01"
                                                    if d.split("-")[0].__contains__("x1") and d.split("-")[
                                                        1].__contains__("x0"):
                                                        split = "x10"

                                                    if split == "x0":
                                                        piecewiseFunc.append((x0 - nums[0]) * float(row[1]))
                                                    elif split == "x1":
                                                        piecewiseFunc.append((x1 - nums[0]) * float(row[1]))
                                                    # piecewiseFunc.append(tf.math.multiply(tf.cast(
                                                    # tf.math.less(x, nums[ 0 ]), tf.float32),
                                                    # (
                                                    # inputs - nums[ 0 ])))

                                                    # inputs = tf.where(x < nums[ 0 ],
                                                    # float(row[ 1 ]) * (
                                                    # inputs - nums[ 0 ]), inputs)
                                                except:
                                                    dc = 0
                                    except:
                                        dc = 0
                            except:
                                dc = 0

            return piecewiseFunc

        self.flagGen = False

        estimator = baseline_modelDeepCl()
        XSplineVector = []
        velocities = []
        vectorWeights = []

        for i in range(0, len(X)):
            vector = extractFunctionsFromSplines(X[i][0], X[i][1],X[i][2],X[i][3],X[i][4],X[i][5],X[i][6])
            #vector = extractFunctionsFromSplines(X[i][0], X[i][1], X[i][3])
            #XSplineVector.append(np.append(X[i], vector))
            XSplineVector.append(vector)

        XSplineVector = np.array(XSplineVector)


        # estimator.layers[0].set_weights([weights, np.array([0] * (genModelKnots-1))])
        #dataUpdatedX = np.array(np.append(X[:, 0].reshape(-1, 1), np.asmatrix([X[:, 1],X[:, 2], X[:, 3]]).T, axis=1))

        estimator.fit(X, XSplineVector, epochs=20,verbose=0)

        self.flagGen = True
        from scipy.special import softmax
        # estimatorD = baseline_modelDeepCl()


        # input_img = keras.layers.Input(shape=(2,), name='input')
        # x = input_img
        # internal layers in encoder
        # for i in range(n_stacks - 1):
        # x =estimator.layers[ 2 ](x)

        # estimatorD.fit(X, Y, epochs=100)
        # keras.models.Model(inputs= keras.layers.Dense(input(estimator.layers[2].input)), outputs=estimator.layers[-1].output)
        # model2 = keras.models.Model(inputs=input_img, outputs=x)
        # model2 = keras.models.Model(inputs=estimator.layers[2].input, outputs=estimator.layers[-1].output)
        # model2 = estimator.layers[-2]
        # model2 =keras.models.Model(inputs=estimator.input, outputs=estimator.layers[-2].output)
        # model2.compile(optimizer=keras.optimizers.Adam(), loss= keras.losses.KLD)
        # model2.fit(X,Y,epochs=10)
        labels = np.unique(np.argmax(estimator.predict(X), axis=1))
        print(labels)
        labels = np.argmax(estimator.predict(X), axis=1)

        NNmodels = []
        scores = []

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


        return partitionsX, partitionsY, partitionLabels , dataX , dataY,None

    def getFitnessOfModelForPoint(self, model, point):
        return 1.0 / (1.0 + np.linalg.norm(np.mean(self._partitionsPerModel[model]) - point))

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

        '''k = len(partitionsX)
        i=0
        if k > 1:
            indxForMerging=[]
            while i <k:
                if len(partitionsX[i]) <=1000:
                    #partitionsX.remove(partitionsX[i])
                    #partitionsY.remove(partitionsY[i])
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
        #print("NEW Number of clusters AFTER DELETING clusters with insufficient size: %d" % (len(partitionsX)))'''
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


