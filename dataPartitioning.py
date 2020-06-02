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
from scipy import spatial,random
import colorsys
#from keras import losses
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

#class CrossValidationPartioner(DefaultPartitioner):

    #def NfoldValidation(self,dataX,dataY,dataW=None,folds=None,showPLot=False):
        #kf = KFold(n_splits=folds)
        #trainXList=[]
        #trainYList=[]
        #testXList = []
        #testYList = []
        #for train, test in kf.split(dataX):
            #train_X, test_X = dataX[ train ], dataX[ test ]
            #train_y, test_y = dataY[ train ], dataY[ train ]
            #trainXList.append(train_X)
            #trainYList.append(train_y)
            #testXList.append(test_X)
            #testYList.append(test_y)

        #return trainXList , trainYList  ,list(range(0,folds)), None,None,testXList , testYList

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
            neurons=3
            model = keras.models.Sequential()

            model.add(keras.layers.Dense(2, input_shape=(2,)))

            while neurons < genModelKnots -1:
                model.add(keras.layers.Dense(neurons, ))
                neurons = neurons+1
            model.add(keras.layers.Dense(genModelKnots - 1, ))

            # Compile model
            model.compile(loss=losses.mean_squared_error, optimizer=keras.optimizers.Adam())
            ###print(model.summary())
            return model



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

        def extractFunctionsFromSplines(x0, x1):
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
                                split = d.split('-')[0][2:4]
                                if split != "x0" and split != "x1":
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
            vector = extractFunctionsFromSplines(X[i][0], X[i][1])
            XSplineVector.append(vector)

        XSplineVector = np.array(XSplineVector)
        weights0 = np.mean(XSplineVector, axis=0)
        # weights1 = self.intercepts
        weights1 = estimator.layers[0].get_weights()[0][1]
        weights = np.array(
            np.append(weights0.reshape(-1, 1), np.asmatrix(weights0).reshape(-1, 1), axis=1).reshape(2, -1))

        # estimator.layers[0].set_weights([weights, np.array([0] * (genModelKnots-1))])

        estimator.fit(X, XSplineVector, epochs=20,verbose=0)

        self.flagGen = True
        from scipy.special import softmax
        # estimatorD = baseline_modelDeepCl()
        # dataUpdatedX = np.append(partitionsX, np.asmatrix([partitionsY]).T, axis=1)

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
        #dataY=None
        dataX = dataX.reshape(-1,1) if len(dataX.shape)==1 else dataX
        if dataY is not None:
            #dataXcl = dataX[:,0:3]
            #dataYcl=np.append(dataX[:,3].reshape(-1,1), np.asmatrix([dataY]).T, axis=1)
            #trnslatedDatax = np.array(np.append(dataX[:,0].reshape(-1,1),np.asmatrix([dataX[:,4]]).T,axis=1))
            dataUpdatedX = np.append(dataX, np.asmatrix([dataY]).T, axis=1)

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
        return KMeans(n_clusters=self._nClusters , random_state=self.random_state)

    def getCentroids(self):
        return self._dataModel.cluster_centers_


class BoundedProximityPartitioner (DefaultPartitioner):
    def clustering(self, dataX, dataY = None, nClusters = None, showPlot=False, random_state=1000):
        return self._clustering(dataX, dataY, nClusters, showPlot, random_state)

    # Clusters data, for a given number of clusters
    def _clustering(self, dataX, dataY = None,dataW=None ,nClusters = None, showPlot=False, random_state=1000, xBound = None, yBound = None):
        # Init return arrays
        partitionsX = []
        partitionsY = []
        partitionLabels = np.zeros((dataX.shape[0], 1)) # Init to input data size
        representatives = []

        # V is the set of all V values in the training set.
        #V, inverseIndex = np.unique(dataX, axis=0, return_inverse=True)
        V, inverseIndex = np.unique(dataX, return_inverse=True)

        # R is the set of corresponding RPM values in the training set.
        R = dataY[inverseIndex]

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
                xBound = (np.max(Vmapped) - np.min(Vmapped)) / math.log(2.0 + dataY.shape[0], 2.0)

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

            # DEBUG LINES
            print("Created a total of %d clusters. Remaining %d instances unclustered."%(len(representatives),
                                                                                         notInU.shape[0]))
            #############

            # Repeat until no more points can be added.
            canContinue = notInU.size > 2

        # Show plot, if requested
        #if (showPlot):
            #self.showClusterPlot(dataX, partitionLabels, nClusters)

        # Return
        return partitionsX, partitionsY, partitionLabels, representatives, None

    def getAllRowsYsWhereRowsXEqV(self, dataX, dataY, v):
        return dataY[(dataX == v).all(axis=1)]

    def setDiff(self, U, V):
        notInU = set(map(tuple, V)).difference(map(tuple, U))
        notInU = np.asarray(list(map(list, notInU)))
        return notInU

class DelaunayTriPartitioner:

    def showTriangle(self,x, y, z, error):
        # vertices = plotly_trisurf(x, y, z, tri.simplices)
        # polytops = list()
        k = np.vstack(np.array([ [ x ], [ y ], [ z ] ]))
        t = plt.Polygon(k, fill=False)
        plt.gca().add_patch(t)
        plt.xlim(np.min(k[ 0:, 0 ] - 10), np.max(k[ 0:, 0 ]) + 10)
        plt.ylim(np.min(k[ 0:, 1 ] - 10), np.max(k[ 0:, 1 ]) + 10)
        plt.title("Triangle with error bound " + str(round(error, 2)))
        plt.xlabel("Velocity")
        plt.ylabel("RPM +- " + str(round(error, 2)))
        plt.show()

    def sign(self,p1, p2, p3):
        return (p1[ 0 ] - p3[ 0 ]) * (p2[ 1 ] - p3[ 1 ]) - (p2[ 0 ] - p3[ 0 ]) * (p1[ 1 ] - p3[ 1 ])

    def distance(self,a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def is_between(self,a, c, b):
        return self.distance(a, c) + self.distance(c, b) == self.distance(a, b)

    def PointInTriangle(self,pt, v1, v2, v3):

        b1 = self.sign(pt, v1, v2) <= 0.0 or self.is_between(v1,pt,v2)
        b2 = self.sign(pt, v2, v3) <= 0.0 or self.is_between(v2,pt,v3)
        b3 = self.sign(pt, v3, v1) <= 0.0 or self.is_between(v3,pt,v1)

        return ((b1 == b2) and (b2 == b3))

    def getTriangle(self,X, Y, usedV, errorBound):
        X = X[ :, 0 ]
        indinXnotinU = filter(lambda x: x not in usedV, range(0, len(X)))
        if usedV == [ ]:
            pointV1 = random.sample(X, 1)
            index = random.sample(list([ num for num in range(0, len(X)) if X[ num ] > pointV1 ]), 1)
            pointV2 = X[ index ]
            pointRpm2 = Y[ index ]

        else:
            pointV1 = random.sample(X[ indinXnotinU ], 1)
            # p = [filter(lambda x: x not in usedV,range(0,len(X)))]
            x = X[ indinXnotinU ]
            y = Y[ indinXnotinU ]
            index = random.sample(list([ num for num in x if num > pointV1 ]), 1)
            pointV2 = index
            pointRpm2 = index

        pointRpm1 = np.mean([ Y[ n ] for n in range(0, len(Y)) if X[ n ] == pointV1 ])

        # points =np.vstack([[pointV1[0],pointRpm1],[pointV2[0],(pointRpm2+errorBound)[0]],[pointV2[0],(pointRpm2-errorBound)[0]]])
        # tri = Delaunay(points)
        self.showTriangle([ pointV1[ 0 ], pointRpm1 ], [ pointV2[ 0 ], (pointRpm1 + errorBound) ],
                     [ pointV2[ 0 ], (pointRpm1 - errorBound) ], errorBound)
        d = [
            self.PointInTriangle([ X[ i ], Y[ i ] ], [ pointV1[ 0 ], pointRpm1 ], [ pointV2[ 0 ], (pointRpm1 + errorBound) ],
                            [ pointV2[ 0 ], (pointRpm1 - errorBound) ])
            for i in indinXnotinU ]

        indexInTr = [ i for i in range(0, len(indinXnotinU)) if d[ i ] == True ]
        return indexInTr

    def tri_indices(self,simplices):
        # simplices is a numpy array defining the simplices of the triangularization
        # returns the lists of indices i, j, k

        return ([ triplet[ c ] for triplet in simplices ] for c in range(3))

    def plotly_trisurf(self,x, y, z, simplices):
        # x, y, z are lists of coordinates of the triangle vertices
        # simplices are the simplices that define the triangularization;
        # simplices  is a numpy array of shape (no_triangles, 3)
        # insert here the  type check for input data

        points3D = np.vstack((x, y, z)).T
        tri_vertices = map(lambda index: points3D[ index ], simplices)  # vertices of the surface triangles
        zmean = [ np.mean(tri[ :, 2 ]) for tri in tri_vertices ]  # mean values of z-coordinates of

        # triangle vertices
        min_zmean = np.min(zmean)
        max_zmean = np.max(zmean)

        I, J, K = self.tri_indices(simplices)
        return tri_vertices


    def HSVToRGB(self,h, s, v):

        (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
        return (int(255 * r), int(255 * g), int(255 * b))

    def getDistinctPixelss(self,n):
        huePartition = 1.0 / (n + 1)
        return np.array(self.HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n))

    def clustering(self, dataX, dataY = None,dataW=None ,cutOffvalues = None, showPlot=False,numofCl=None):
          vData = dataX
          points2D = np.array(vData)
          zx = range(0, len(vData[ :, 0 ]))
          tri = Delaunay(points2D,)
          vertices = self.plotly_trisurf(points2D[ :, 0 ], zx, points2D[ :, 1 ], tri.simplices)
          hull = ConvexHull(vData)

          #self.plotly_trisurf(dataX[ tri.vertices[ 0 ] ][ :, 0 ], zx[ 0:3 ], dataX[ tri.vertices[ 0 ] ][ :, 1 ],
                              #tri.simplices)

          #plt.plot(vData[ 0:, 0 ], vData[ 0:, 1 ], 'o', markersize=8)
          #for simplex in hull.simplices:
              #plt.plot(vData[ simplex, 0 ], vData[ simplex, 1 ], 'k-',linewidth=3)
          ###hull plot and data plot
          def in_hull(p, hull):
              """
              Test if points in `p` are in `hull`

              `p` should be a `NxK` coordinates of `N` points in `K` dimensions
              `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
              coordinates of `M` points in `K`dimensions for which Delaunay triangulation
              will be computed
              """

              if not isinstance(hull, Delaunay):
                  hull = Delaunay(hull)

              return hull.find_simplex(p) >= 0

          #for v in vertices:
              #k = np.array([ [ v[ 0 ][ 0 ], v[ 0 ][ 2 ] ], [ v[ 1 ][ 0 ], v[ 1 ][ 2 ] ], [ v[ 2 ][ 0 ], v[ 2 ][ 2 ] ] ])
              #t = plt.Polygon(k, fill=False,color='red',linewidth=3)
              #plt.gca().add_patch(t)

          #plt.xlabel('$V(t)$')
          #plt.ylabel(r'$\bar{\bf V}_N(t_i)$')
          #plt.show()
          x=1
          ### DT plot
          ##################################
          #dataX = np.append(dataX, np.asmatrix([ dataY ]).T, axis=1)
          #points2D = np.array(dataX)
          #zx = range(0, len(dataX[ :, 0 ]))
          #tri = Delaunay(points2D)
          #vertices = self.plotly_trisurf(points2D[ :, 0 ], zx, points2D[ :, 1 ], tri.simplices)
          cutoffPoint =cutOffvalues
          ##Convex HUll
          #XY=np.vstack([dataX[:,0],dataY]).T

          #################################
          #pcaMapping = PCA(1)
          #pcaMapping.fit(dataX)
          #Vmapped = pcaMapping.transform(dataX)
          #dataX = Vmapped
          ##################################

          #hull = ConvexHull(vData)
          history=20
          pointsy=[]
          pointsx0x1=[]

          #plt.plot(dataX[0:,0], dataX[0:,1], 'o',markersize=4 )
          #for simplex in hull.simplices:
              #plt.plot(vData[ simplex, 0 ], vData[simplex,1], 'k-')
          #plt.show()
          cl=0
          graph={}
          neighboringV=[]
          dataX=np.asarray(dataX)
          for i in range(0, len(dataX)):
              pointX = dataX[ i, 0 ]
              pointY = dataX[ i, 1 ]
              neighbors = [ ]
              sets = set()
              setsy=set()
              # for v in vertices:
              cluster = {}
              flag1 = True
              #c = tri.find_simplex(dataX[ i ])
              neighboringVertices=tri.vertex_neighbor_vertices[1][tri.vertex_neighbor_vertices[0][i]:tri.vertex_neighbor_vertices[0][i+1]]
              for n in list(neighboringVertices):
                  #if np.linalg.norm(np.array((dataX[i][0],dataX[i][1],dataY[i])) -np.array((dataX[n][0],dataX[n][1],dataY[n]))) <=cutoffPoint:
                  if spatial.distance.euclidean(dataX[ i ], dataX[ n ]) <= cutoffPoint :
                      #spatial.distance.euclidean([dataX[ i ][0],dataX[i][1],dataY[i]], [dataX[n][0],dataX[n][1],dataY[n]])<0.5:

                       # and  spatial.distance.euclidean( dataY[ i ],dataY[ n ] )<=cutoffPoint:
                            #and spatial.distance.euclidean( [dataX[i][1],dataY[ i ]], [dataX[n][1],dataY[ n ]] )<=cutoffPoint:
                                                     #[ dataX[ n ][ 0 ], dataY[ n ] ]) <= cutoffPoint:

                      #v=[dataX[n][0],dataX[n][1]]
                      v=n
                      vy=[dataY[n]]
                      #data=dataX[n].T
                      #if neighbors.__contains__(v) == False:
                      neighbors.append([dataX[n][0],dataX[n][1]])
                      sets.add(v)
                      setsy.add(tuple(vy))
                  else:
                      flag1 = False

              if flag1 :
                  cluster[ cl ] = neighbors

              else:
                  cl += 1
              #if list(sets) !=[]:
              graph[i]=[sets,False,dataY[i]]
              #graph[""].append(n)
              if neighbors!=[]:
                neighboringV.append(neighbors)

          #####################################

          ###################################
              # A function used by DFS
          def DFSUtil( v,i ,graph,cl,cly):
                  # Mark the current node as visited and print it
                  graph[ i ][ 1 ] = True
                  #print v,
                  cl.append(dataX[i])
                  cly.append(graph[ i ][2])
                  # Recur for all the vertices adjacent to
                  # this vertex
                  for k in list(graph[ i ][0]):
                      if graph[k][1] == False:
                          DFSUtil(dataX[k],k, graph,cl,cly)

                          # The function to do DFS traversal. It uses
              # recursive DFSUtil()
                  #return cl
          def DFS():
                  V = len(graph)  # total vertices
                  # Mark all the vertices as not visited
                  #visited = [ False ] * (V)
                  # Call the recursive helper function to print
                  # DFS traversal starting from all vertices one
                  # by one
                  clusters=[]
                  clustersy=[]
                  uncl=[]
                  uncly=[]
                  for i in range(0,len(graph)):
                      cl=[]
                      cly=[]
                      if graph[i][1] == False:
                          #try:
                          #  splitStr=i.replace('[', '').replace(']', '').split(' ')
                            tuplex=[]
                          #  for n in splitStr:
                          #      if n!='':
                            tuplex.append(dataX[i])
                            DFSUtil(tuple(tuplex),i ,graph,cl,cly)
                            if len(cl)>1:
                                clusters.append(cl)
                                clustersy.append(cly)
                            else:
                                uncl.append(cl)
                                uncly.append(cly)

                          #except:
                          #    print i
                  return clusters,clustersy,uncl,uncly
                          ######################################
          clusters,clustersy,uncl,uncly=DFS()
          train_x_list = [ ]
          train_y_list = [ ]
          trains_x_list = [ ]
          trains_y_list = [ ]
          trlabels=np.empty(sum(map(len,clusters)))
          trData=[]
          counter=0
          outliers=[]
          for i in range(0,len(clusters)):

              #clusters[i]=[clusters[i][ k ][ 0:2 ] for k in range(0,len(clusters[i]))]
              if len(np.array(clusters[ i ]))>4:
                trains_x_list.append(np.array(clusters[i]))
              #else:
                  #for k in range(0,len(np.array(clusters[ i ]))):
                    #outliers.append(clusters[i][k])
              if len(np.array(clusters[ i ]))>4:
                trains_y_list.append(np.array(clustersy[i]))
              #for v in clusters[i]:
                  #trData.append(v)
                  #trlabels[counter]=i
                  #counter+=1
         # return trains_x_list, trains_y_list,list(range(0,len(clusters))), None, None
              #train_y_list.append(np.array(cluster)[:,1])
              #if len(train_x_list) > 1:
              #trains_x_list.append(np.array(train_x_list).reshape(-1,1))
              #trains_y_list.append(np.array(train_y_list).reshape(-1,1))
              #train_x_list = [ ]
              #train_y_list = [ ]
          #colors = tuple([ numberToRGBAColor(l) for l in np.unique(trlabels) ])
          #for i in range(0, len(trains_x_list)):
             #t = plt.Polygon(trains_x_list[i], fill=False, color='red', linewidth=3)
             #plt.gca().add_patch(t)

          outliers=[]
          for l in range(0,len(vData)):
            point = vData[l]
            flagForPoint = False
            for i in range(0, len(trains_x_list)):
                #for k in range(0,len(trains_x_list[i])):
                    if(point in trains_x_list[i]) :
                        flagForPoint=True
                        break
            if(flagForPoint==False):
                outliers.append(point)

          outliers=np.array(outliers)
          vDataNew=[]

          for k in range(0,len(outliers)):
             for i in range(0, len(trains_x_list)):
                if outliers[k] not in trains_x_list[i]:
                        vDataNew.append(outliers[k])
          #plt.plot(vData[ 0:, 0 ], vData[ 0:, 1 ], 'o', markersize=3)
          vDataNew=np.array(vDataNew)
          #plt.plot(vDataNew[ 0:, 0 ], vDataNew[ 0:, 1 ], 'o', markersize=12,c='y')

          vDataSimple=[]
          for i in range(0,len(trains_x_list)):
              for k in range(0,len(trains_x_list[i])):
                vDataSimple.append(trains_x_list[i][k])

          vDataSimple=np.array(vDataSimple)
          #plt.plot(vDataSimple[ 0:, 0 ], vDataSimple[ 0:, 1 ], 'o', markersize=8)

          #plt.show()
          print ("\n"+ str(cutoffPoint))
          print (len(trains_x_list))

          return trains_x_list,trains_y_list, list(range(0, len(trains_x_list))),dataX,dataY , tri







