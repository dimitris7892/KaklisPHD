import sklearn.linear_model as sk
from sklearn.linear_model import  LinearRegression
import pyearth as sp
import numpy as np
import numpy.linalg
import sklearn.ensemble as skl
from scipy.spatial import Delaunay
import random
#from sklearn.cross_validation import train_test_split
import csv
import itertools
#from sklearn.model_selection import KFold as kf
from scipy import spatial
from sklearn.pipeline import Pipeline
#from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from gekko import GEKKO
import  matplotlib.pyplot as plt
from scipy.interpolate import BivariateSpline
import tensorflow as tf
from tensorflow import keras
from scipy.stats import pearsonr
#from sklearn.model_selection import KFold
#import pydot
#import graphviz
import scipy.stats as st
#from tensorflow.python.tools import inspect_checkpoint as chkp
from time import time
from sklearn.cluster import KMeans
tf.compat.v1.disable_eager_execution()
#tf.executing_eagerly()
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback
from sklearn import preprocessing
from scipy.special import softmax

class BasePartitionModeler:
    def createModelsFor(self,partitionsX, partitionsY, partition_labels):
        pass

    def getBestModelForPoint(self, point):
        #mBest = None
        mBest=None
        dBestFit = 0
        # For each model
        for m in self._models:
            # If it is a better for the point
            dCurFit = self.getFitnessOfModelForPoint(m, point)
            if dCurFit > dBestFit:
                # Update the selected best model and corresponding fit
                dBestFit = dCurFit
                mBest = m

        if mBest==None:
            return self._models[0]
        else: return mBest

    def getFitForEachPartitionForPoint(self, point, partitions):
        # For each model
        fits=[]
        for m in range(0, len(partitions)):
            # If it is a better for the point
            dCurFit = self.getFitnessOfPoint(partitions, m, point)
            fits.append(dCurFit)


        return fits

    def getBestPartitionForPoint(self, point,partitions):
            # mBest = None
            mBest = None
            dBestFit = 0
            # For each model
            for m in range(0,len(partitions)):
                # If it is a better for the point
                dCurFit = self.getFitnessOfPoint(partitions,m, point)
                if dCurFit > dBestFit:
                    # Update the selected best model and corresponding fit
                    dBestFit = dCurFit
                    mBest = m

            if mBest == None:
                return 0,0
            else:
                return mBest , dBestFit
    def getFitofPointIncluster(self,point,centroid):
        return 1.0 / (1.0 + numpy.linalg.norm(point - centroid))

    def  getFitnessOfModelForPoint(self, model, point):
        return 0.0

    def getFitsOfPoint(self, partitions, point):
        fits = [ ]
        for i in range(0, len(partitions)):
            fits.append(numpy.linalg.norm(np.mean(partitions[ i ]) - point))
        return fits

    def getFitnessOfPoint(self,partitions ,cluster, point):

        #return 1 / (1 + np.linalg.norm(np.mean(np.array(
            #np.append(partitions[cluster][:, 0].reshape(-1, 1), np.asmatrix(partitions[cluster][:, 4]).T, axis=1)),
                                               #axis=0) - np.array([point[0][0], point[0][4]])))
        #return 1 / (1 + np.linalg.norm(np.mean(np.array(
        #np.append(partitions[ cluster ][ :, 0 ].reshape(-1, 1),
                  #np.asmatrix([partitions[ cluster ][ :, 2 ], partitions[ cluster ][ :, 4 ]]).T,
                  #axis=1)),axis=0) - np.array([ point[ 0 ][ 0 ],point[ 0 ][ 2 ], point[ 0 ][ 4 ] ])))
        #partCl = np.array(np.append(partitions[ cluster ][:, 0].reshape(-1, 1),
                           #np.asmatrix([partitions[ cluster ][:, 3]]).T, axis=1))
        #point = np.array([point[0][0],point[0][3]])
        #return 1.0 / (1.0 + numpy.linalg.norm(np.mean(partCl,axis=0) - point))
        #normalizedPoint = [k for k in point (k - min(point))/(min(point) - max(point)) ]
        #pearsonr(np.mean(partitions[cluster], axis=0), point[0])
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(partitions[cluster], axis=0) - point))


class TensorFlowW2(BasePartitionModeler):


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


    def createModelsFor(self, partitionsX, partitionsY, partition_labels, tri, X, Y):
        models = []
        # partitionsX=np.array(partitionsX[0])
        # partitionsY = np.array(partitionsY[0])

        self.ClustersNum = len(partitionsX)
        self.modelId = -1
        # partition_labels = len(partitionsX)
        # Init model to partition map
        self._partitionsPerModel = {}

        def SplinesCoef(partitionsX, partitionsY):

            model = sp.Earth(use_fast=True)
            model.fit(partitionsX, partitionsY)

            return model.coef_

        def getFitnessOfPoint(partitions, cluster, point):
            return 1.0 / (1.0 + numpy.linalg.norm(np.mean(partitions[cluster]) - point))

        def baseline_modelDeepCl():
            # create model
            model = keras.models.Sequential()

            model.add(keras.layers.Dense(len(partition_labels) + 20, input_shape=(2,)))
            model.add(keras.layers.Dense(len(partition_labels) + 10, input_shape=(2,)))
            model.add(keras.layers.Dense(len(partition_labels), input_shape=(2,)))
            # model.add(keras.layers.Dense(len(partition_labels) * 2, input_shape=(2,)))
            # model.add(keras.layers.Dense(len(partition_labels), input_shape=(2,)))
            # model.add(keras.layers.Dense(10, input_shape=(2,)))
            # model.add(keras.layers.Dense(5, input_shape=(2,)))

            # model.add(keras.layers.Activation(custom_activation23))

            model.add(keras.layers.Dense(1, ))  # activation=custom_activation
            # model.add(keras.layers.Activation(custom_activation2))

            # model.add(keras.layers.Activation(custom_activation2))
            #
            # model.add(keras.layers.Activation(custom_activation))
            # model.add(keras.layers.Activation('linear'))  # activation=custom_activation
            # Compile model
            model.compile(loss='mse', optimizer=keras.optimizers.Adam())
            return model

        def baseline_model():
            # create model
            model = keras.models.Sequential()

            # model.add(keras.layers.Dense(20, input_shape=(7,)))
            # model.add(keras.layers.Dense(10, input_shape=(7,)))

            model.add(keras.layers.Dense(genModelKnots - 1, input_shape=(7 ,)))

            model.add(keras.layers.Dense(2))

            model.add(keras.layers.Dense(1))

            model.compile(loss=keras.losses.mean_squared_error,
                          optimizer=keras.optimizers.Adam(), )  # experimental_run_tf_function=False )
            # print(model.summary())
            return model

        seed = 7
        numpy.random.seed(seed)

        self.genF = None

        sModel = []
        sr = sp.Earth(max_degree=1)
        sr.fit(X, Y)
        sModel.append(sr)
        import csv
        csvModels = []
        genModelKnots = []

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

        srModels = []
        for idx, pCurLbl in enumerate(partition_labels):
            # maxTerms = if len(DeepCLpartitionsX) > 5000
            srM = sp.Earth(max_degree=1)
            srM.fit(np.array(partitionsX[idx]), np.array(partitionsY[idx]))
            srModels.append(srM)
        modelCount = 0
        import csv
        # csvModels = []
        ClModels = {"data": []}

        for models in srModels:
            modelSummary = str(models.summary()).split("\n")[4:]
            basisM = []
            with open('./model_' + str(modelCount) + '_.csv', mode='w') as data:
                csvModels.append('./model_' + str(modelCount) + '_.csv')
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    ['Basis', 'Coeff'])
                for row in modelSummary:
                    row = np.delete(np.array(row.split(" ")), [i for i, x in enumerate(row.split(" ")) if x == ""])
                    try:
                        basis = row[0]
                        pruned = row[1]
                        coeff = row[2]
                        # if basis == 'x0' : continue
                        if pruned == "No":
                            data_writer.writerow([basis, coeff])
                            basisM.append(basis)
                    except:
                        x = 0
                model = {}
                model["id"] = modelCount
                model["funcs"] = len(basisM)
                ClModels["data"].append(model)
            modelCount += 1

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

        estimator = baseline_model()
        XSplineVector = []
        velocities = []
        vectorWeights = []

        #for i in range(0, len(X)):
            #vector = extractFunctionsFromSplines(X[i][0], X[i][1], X[i][2], X[i][3], X[i][4], X[i][5], X[i][6])
            #XSplineVector.append(np.append(X[i], vector))

        #XSplineVectorGen = np.array(XSplineVector)
        # weights0 = np.mean(XSplineVector, axis=0)
        # weights1 = self.intercepts
        # weights1 = estimator.layers[0].get_weights()[0][1]
        # weights = np.array(
        # np.append(weights0.reshape(-1, 1), np.asmatrix(weights0).reshape(-1, 1), axis=1).reshape(9, -1))

        # estimator.layers[0].set_weights([weights, np.array([0] * (genModelKnots - 1))])

        self.flagGen = True

        # Plot training & validation accuracy values

        NNmodels = []
        scores = []

        if len(partition_labels) > 1:
            for idx, pCurLbl in enumerate(partition_labels):

                self.modelId = idx
                self.countTimes += 1

                #XSplineClusterVector = []
                #for i in range(0, len(partitionsX[idx])):
                    #vector = extractFunctionsFromSplines(partitionsX[idx][i][0], partitionsX[idx][i][1],
                                                         #partitionsX[idx][i][2], partitionsX[idx][i][3],
                                                         #partitionsX[idx][i][4], partitionsX[idx][i][5],
                                                         #partitionsX[idx][i][6])
                    #XSplineClusterVector.append(np.append(partitionsX[idx][i], vector))

                # X =  np.append(X, np.asmatrix([dataY]).T, axis=1)
                #XSplineClusterVector = np.array(XSplineClusterVector)

                # estimator = baseline_model()
                numOfNeurons = [x for x in ClModels['data'] if x['id'] == idx][0]['funcs']
                estimatorCl = keras.models.Sequential()
                # estimatorCl = baseline_model()

                estimatorCl.add(keras.layers.Dense(numOfNeurons - 1, input_shape=(7 ,)))
                # estimatorCl.add(keras.layers.Dense(genModelKnots -1, input_shape=(6,)))
                estimatorCl.add(keras.layers.Dense(5))
                estimatorCl.add(keras.layers.Dense(2))
                estimatorCl.add(keras.layers.Dense(1))
                estimatorCl.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(), )  # try:

                # weights0 = np.mean(XSplineClusterVector, axis=0)
                # weights1 = self.intercepts
                # weights1 = estimatorCl.layers[ 0 ].get_weights()[ 0 ][ 1 ]
                # weights = np.array(
                # np.append(weights0.reshape(-1, 1), np.asmatrix(weights0).reshape(-1, 1), axis=1).reshape(2, -1))

                # estimatorCl.layers[0].set_weights([weights, np.array([0] * (numOfNeurons - 1))])
                # modelId=idx

                estimatorCl.fit(np.array(partitionsX[idx]), np.array(partitionsY[idx]), epochs=30, )

                # x = input_img

                # x = estimatorCl.layers[2](x)

                # modelCl = keras.models.Model(inputs=input_img, outputs=x)
                # model2 = keras.models.Model(inputs=estimator.layers[2].input, outputs=estimator.layers[-1].output)

                # modelCl.compile(optimizer=keras.optimizers.Adam(), loss='mse')
                # modelCl.fit(partitionsX[idx], np.array(partitionsY[idx]), epochs=100)

                # Clscore = estimator.evaluate(np.array(partitionsX[idx]), np.array(partitionsY[idx]), verbose=1)
                # scores.append(Clscore)
                NNmodels.append(estimatorCl)

                # self._partitionsPerModel[ estimator ] = partitionsX[idx]
        # Update private models
        # models=[]

        # Plot training & validation loss values
        history = estimator.fit(X, Y, epochs=50, validation_split=0.33)
        NNmodels.append(estimator)

        self._models = NNmodels

        # Return list of models
        return estimator, history, scores, numpy.empty, vectorWeights  # , estimator , DeepCLpartitionsX


    def getFitnessOfModelForPoint(self, model, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(self._partitionsPerModel[model]) - point))

class TensorFlowW1(BasePartitionModeler):

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

    def createModelsFor(self, partitionsX, partitionsY, partition_labels, tri, X, Y):

        models = []
        # partitionsX=np.array(partitionsX[0])
        # partitionsY = np.array(partitionsY[0])

        self.ClustersNum = len(partitionsX)
        self.modelId = -1
        # partition_labels = len(partitionsX)
        # Init model to partition map
        self._partitionsPerModel = {}

        def SplinesCoef(partitionsX, partitionsY):

            model = sp.Earth(use_fast=True)
            model.fit(partitionsX, partitionsY)

            return model.coef_

        def getFitnessOfPoint(partitions, cluster, point):
            return 1.0 / (1.0 + numpy.linalg.norm(np.mean(partitions[cluster]) - point))

        def baseline_modelDeepCl():
            # create model
            model = keras.models.Sequential()

            model.add(keras.layers.Dense(len(partition_labels) + 20, input_shape=(2,)))
            model.add(keras.layers.Dense(len(partition_labels) + 10, input_shape=(2,)))
            model.add(keras.layers.Dense(len(partition_labels), input_shape=(2,)))
            # model.add(keras.layers.Dense(len(partition_labels) * 2, input_shape=(2,)))
            # model.add(keras.layers.Dense(len(partition_labels), input_shape=(2,)))
            # model.add(keras.layers.Dense(10, input_shape=(2,)))
            # model.add(keras.layers.Dense(5, input_shape=(2,)))

            # model.add(keras.layers.Activation(custom_activation23))

            model.add(keras.layers.Dense(1, ))  # activation=custom_activation
            # model.add(keras.layers.Activation(custom_activation2))

            # model.add(keras.layers.Activation(custom_activation2))
            #
            # model.add(keras.layers.Activation(custom_activation))
            # model.add(keras.layers.Activation('linear'))  # activation=custom_activation
            # Compile model
            model.compile(loss='mse', optimizer=keras.optimizers.Adam())
            return model

        def baseline_model():
            # create model
            model = keras.models.Sequential()

            #model.add(keras.layers.Dense(20, input_shape=(7,)))
            #model.add(keras.layers.Dense(10, input_shape=(7,)))

            model.add(keras.layers.Dense(genModelKnots -1,input_shape=(7+genModelKnots-1,)))
            #model.add(keras.layers.Dense(30))
            #model.add(keras.layers.Dense(20))
            #model.add(keras.layers.Dense(genModelKnots - 3,activation='relu'))
            model.add(keras.layers.Dense(2,))

            model.add(keras.layers.Dense(1))

            model.compile(loss=keras.losses.mean_squared_error,
                          optimizer=keras.optimizers.Adam(), )  # experimental_run_tf_function=False )
            # print(model.summary())
            return model

        seed = 7
        numpy.random.seed(seed)

        self.genF = None

        sModel = []
        sr = sp.Earth(max_degree=1)
        sr.fit(X, Y)
        sModel.append(sr)
        import csv
        csvModels = []
        genModelKnots = []

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

        srModels = []
        for idx, pCurLbl in enumerate(partition_labels):
            # maxTerms = if len(DeepCLpartitionsX) > 5000
            srM = sp.Earth(max_degree=1)
            srM.fit(np.array(partitionsX[idx]), np.array(partitionsY[idx]))
            srModels.append(srM)
        modelCount = 0
        import csv
        # csvModels = []
        ClModels = {"data": []}

        for models in srModels:
            modelSummary = str(models.summary()).split("\n")[4:]
            basisM = []
            with open('./model_' + str(modelCount) + '_.csv', mode='w') as data:
                csvModels.append('./model_' + str(modelCount) + '_.csv')
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    ['Basis', 'Coeff'])
                for row in modelSummary:
                    row = np.delete(np.array(row.split(" ")), [i for i, x in enumerate(row.split(" ")) if x == ""])
                    try:
                        basis = row[0]
                        pruned = row[1]
                        coeff = row[2]
                        # if basis == 'x0' : continue
                        if pruned == "No":
                            data_writer.writerow([basis, coeff])
                            basisM.append(basis)
                    except:
                        x = 0
                model = {}
                model["id"] = modelCount
                model["funcs"] = len(basisM)
                ClModels["data"].append(model)
            modelCount += 1

        self.count = 0

        def extractFunctionsFromSplines(x0, x1 , x2 , x3 ,x4 , x5 ,x6):
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

        estimator = baseline_model()
        XSplineVector = []
        velocities = []
        vectorWeights = []

        for i in range(0, len(X)):
            vector = extractFunctionsFromSplines(X[i][0], X[i][1],X[i][2],X[i][3],X[i][4],X[i][5],X[i][6])
            XSplineVector.append(np.append(X[i],vector))

        XSplineVectorGen = np.array(XSplineVector)
        #weights0 = np.mean(XSplineVector, axis=0)
        # weights1 = self.intercepts
        #weights1 = estimator.layers[0].get_weights()[0][1]
        #weights = np.array(
            #np.append(weights0.reshape(-1, 1), np.asmatrix(weights0).reshape(-1, 1), axis=1).reshape(9, -1))

        #estimator.layers[0].set_weights([weights, np.array([0] * (genModelKnots - 1))])



        self.flagGen = True

        # Plot training & validation accuracy values
        class TestCallback(Callback):
            def __init__(self, test_data):
                self.test_data = test_data

            def on_epoch_end(self, epoch, logs={}):
                x, y = self.test_data
                loss, acc = self.model.evaluate(x, y, verbose=0)
                print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


        NNmodels = []
        scores = []
        clScores=[]
        clustersTrScores=[]
        minXVars=[]
        minYVars = []
        stdWS=[]
        stdSTW = []
        stdsFOC=[]
        minEpochs=[]
        clusters=[]

        if len(partition_labels) > 1:
            for idx, pCurLbl in enumerate(partition_labels):

                self.modelId = idx
                self.countTimes += 1

                XSplineClusterVector = []
                for i in range(0, len(partitionsX[idx])):
                    vector = extractFunctionsFromSplines(partitionsX[idx][i][0], partitionsX[idx][i][1],partitionsX[idx][i][2],partitionsX[idx][i][3],partitionsX[idx][i][4],partitionsX[idx][i][5],partitionsX[idx][i][6])
                    XSplineClusterVector.append(np.append(partitionsX[idx][i],vector))

                # X =  np.append(X, np.asmatrix([dataY]).T, axis=1)
                XSplineClusterVector = np.array(XSplineClusterVector)

                # estimator = baseline_model()
                numOfNeurons = [x for x in ClModels['data'] if x['id'] == idx][0]['funcs']
                estimatorCl = keras.models.Sequential()
                #estimatorCl = baseline_model()

                estimatorCl.add(keras.layers.Dense(numOfNeurons - 1, input_shape=(7 + numOfNeurons - 1,)))
                #estimatorCl.add(keras.layers.Dense(numOfNeurons - 3, input_shape=(7 + numOfNeurons - 1,)))
                #estimatorCl.add(keras.layers.Dropout(0.0001))
                estimatorCl.add(keras.layers.Dense(2))
                #estimatorCl.add(keras.layers.Dropout(0.01))
                estimatorCl.add(keras.layers.Dense(1))
                estimatorCl.compile(loss=keras.losses.mean_squared_error,
                                    optimizer=keras.optimizers.Adam( ),)  # try:

                #weights0 = np.mean(XSplineClusterVector, axis=0)
                # weights1 = self.intercepts
                # weights1 = estimatorCl.layers[ 0 ].get_weights()[ 0 ][ 1 ]
                #weights = np.array(
                    #np.append(weights0.reshape(-1, 1), np.asmatrix(weights0).reshape(-1, 1), axis=1).reshape(2, -1))

                #estimatorCl.layers[0].set_weights([weights, np.array([0] * (numOfNeurons - 1))])
                # modelId=idx
                #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
                #X_train, X_test, y_train, y_test =  train_test_split(XSplineClusterVector, partitionsY[idx] ,test_size=0.33, random_state=42)
                modelsCl = []
                clustersScore = []
                clustersTrScore=[]
                varXCl=[]
                varYCl = []
                print("CLUSTER:  " +str(idx))
                for i in range(1,40):


                    #estimatorCl.fit(X_train,y_train, epochs=i ,verbose = 0)
                    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0)
                    #val
                    history = estimatorCl.fit(XSplineClusterVector, partitionsY[idx], validation_split=0.17,epochs=i,verbose = 0,callbacks=[es])
                    #historyTR = estimatorCl.fit(X_train, y_train, verbose=0,epochs=i)
                    #Clscore = estimatorCl.evaluate(X_test, y_test, verbose=0)
                    modelsCl.append(estimatorCl)
                    lastInd = len(history.history['val_loss'])-1

                    try:
                        #clustersScore.append(Clscore)
                        clustersScore.append(history.history['val_loss'][lastInd])
                        clustersTrScore.append(history.history['loss'][lastInd])
                        #clustersScore.append(history.history['val_loss'][i - 1])
                    except:
                        x=0

                #varXCl.append(np.var(partitionsX[idx]))
                #varYCl.append(np.var(partitionsY[idx]))


                minClscore = min(clustersScore)
                minClTrscore = min(clustersTrScore)
                minIndCl = clustersScore.index(min(clustersScore))
                minEpoch = minIndCl + 1
                minEpochs.append(minEpoch)
                #minXvar = varXCl[idx]
                #minYvar = varYCl[idx]

                minXVars.append(np.std(partitionsX[idx]))
                minYVars.append(np.std(partitionsY[idx]))
                stdWS.append(np.std(partitionsX[idx][:,3]))
                stdSTW.append(np.std(partitionsX[idx][:, 0]))
                stdsFOC.append(np.std(partitionsY[idx]))

                clScores.append(minClscore)
                clustersTrScores.append(minClTrscore)
                clusters.append(idx+1)
                #x = input_img

                #x = estimatorCl.layers[2](x)

                #modelCl = keras.models.Model(inputs=input_img, outputs=x)
                # model2 = keras.models.Model(inputs=estimator.layers[2].input, outputs=estimator.layers[-1].output)

                #modelCl.compile(optimizer=keras.optimizers.Adam(), loss='mse')
                #modelCl.fit(partitionsX[idx], np.array(partitionsY[idx]), epochs=100)

                #Clscore = estimator.evaluate(np.array(partitionsX[idx]), np.array(partitionsY[idx]), verbose=1)
                #scores.append(Clscore)
                NNmodels.append(modelsCl[minIndCl] )

                # self._partitionsPerModel[ estimator ] = partitionsX[idx]
        # Update private models
        # models=[]

        # Plot training & validation loss values
        print("GENERAL MODEL  ")
        #es = EarlyStopping(monitor='val_loss', mode='min', verbose=0)
        history = estimator.fit(XSplineVectorGen, Y, epochs=50, validation_split=0.17,verbose=0,)#callbacks=[es])


        #print("CORRELATION COEFF ERR AND STW: " +str(pearsonr(stdSTW,clustersTrScores)))
        #print("CORRELATION COEFF WS AND STW: " + str(pearsonr(stdSTW, stdWS)))
        #print("CORRELATION COEFF ERR AND WS: " + str(pearsonr(stdWS, clustersTrScores)))
        #print("CORRELATION COEFF ERR AND FOC: " + str(pearsonr(minYVars, clustersTrScores)))
        NNmodels.append(estimator)

        #normalizedSTDws = (stdWS - min(stdWS)) / (max(stdWS) - min(stdWS))
        #normalizedSTDstw = (stdSTW - min(stdSTW)) / (max(stdSTW) - min(stdSTW))
        #normalizedSTDFOC = (stdsFOC - min(stdsFOC)) / (max(stdsFOC) - min(stdsFOC))
        #normalizedErr = (clustersTrScores - min(clustersTrScores)) / (max(clustersTrScores) - min(clustersTrScores))
        #normalizedValErr = (clScores - min(clScores)) / (max(clScores) - min(clScores))
        #normalizedStw_ws = abs(normalizedSTDstw - normalizedSTDws)

        #print("CORRELATION COEFF Normalized WSstd-STWstd and ERROR: " + str(pearsonr(normalizedErr, normalizedStw_ws)))

        self._models = NNmodels

        # Return list of models
        #with open('./errorEpochCLusters.csv', mode='w') as data:
            #data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #data_writer.writerow(['cluster','trError','acc', 'epoch','stdX','stdY','stdSTW','stdWS','nErr','nStdWs','nStdSTW','nValErr','nSTW_WS','NFoc'])
            #for i in range(0,len(clScores)):

                #data_writer.writerow([clusters[i], clustersTrScores[i] ,clScores[i],minEpochs[i],minXVars[i],minYVars[i],stdWS[i],stdSTW[i],normalizedErr[i],normalizedSTDws[i],normalizedSTDstw[i],normalizedValErr[i],normalizedStw_ws,normalizedSTDFOC[i]])

        return estimator, history, scores, numpy.empty, vectorWeights  # , estimator , DeepCLpartitionsX

    def getFitnessOfModelForPoint(self, model, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(self._partitionsPerModel[model]) - point))

class TensorFlowW3(BasePartitionModeler):

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

    def createModelsFor(self, partitionsX, partitionsY, partition_labels, tri, X, Y):

        models = []
        # partitionsX=np.array(partitionsX[0])
        # partitionsY = np.array(partitionsY[0])

        self.ClustersNum = len(partitionsX)
        self.modelId = -1
        # partition_labels = len(partitionsX)
        # Init model to partition map
        self._partitionsPerModel = {}

        def SplinesCoef(partitionsX, partitionsY):

            model = sp.Earth(use_fast=True)
            model.fit(partitionsX, partitionsY)

            return model.coef_

        def getFitnessOfPoint(partitions, cluster, point):
            return 1.0 / (1.0 + numpy.linalg.norm(np.mean(partitions[cluster]) - point))

        def baseline_modelDeepCl():
            # create model
            model = keras.models.Sequential()

            model.add(keras.layers.Dense(len(partition_labels) + 20, input_shape=(2,)))
            model.add(keras.layers.Dense(len(partition_labels) + 10, input_shape=(2,)))
            model.add(keras.layers.Dense(len(partition_labels), input_shape=(2,)))
            # model.add(keras.layers.Dense(len(partition_labels) * 2, input_shape=(2,)))
            # model.add(keras.layers.Dense(len(partition_labels), input_shape=(2,)))
            # model.add(keras.layers.Dense(10, input_shape=(2,)))
            # model.add(keras.layers.Dense(5, input_shape=(2,)))

            # model.add(keras.layers.Activation(custom_activation23))

            model.add(keras.layers.Dense(1, ))  # activation=custom_activation
            # model.add(keras.layers.Activation(custom_activation2))

            # model.add(keras.layers.Activation(custom_activation2))
            #
            # model.add(keras.layers.Activation(custom_activation))
            # model.add(keras.layers.Activation('linear'))  # activation=custom_activation
            # Compile model
            model.compile(loss='mse', optimizer=keras.optimizers.Adam())
            return model

        def baseline_model():
            # create model
            model = keras.models.Sequential()

            #model.add(keras.layers.Dense(20, input_shape=(7,)))
            #model.add(keras.layers.Dense(10, input_shape=(7,)))

            model.add(keras.layers.Dense(genModelKnots - 1,input_shape=(7,)))
            #model.add(keras.layers.Dense(5))
            #model.add(keras.layers.Dense(2))

            model.add(keras.layers.Dense(1))

            model.compile(loss=keras.losses.mean_squared_error,
                          optimizer=keras.optimizers.Adam(), )  # experimental_run_tf_function=False )
            # print(model.summary())
            return model

        seed = 7
        numpy.random.seed(seed)

        self.genF = None

        sModel = []
        sr = sp.Earth(max_degree=1)
        sr.fit(X, Y)
        sModel.append(sr)
        import csv
        csvModels = []
        genModelKnots = []

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

        srModels = []
        for idx, pCurLbl in enumerate(partition_labels):
            # maxTerms = if len(DeepCLpartitionsX) > 5000
            srM = sp.Earth(max_degree=1)
            srM.fit(np.array(partitionsX[idx]), np.array(partitionsY[idx]))
            srModels.append(srM)
        modelCount = 0
        import csv
        # csvModels = []
        ClModels = {"data": []}

        for models in srModels:
            modelSummary = str(models.summary()).split("\n")[4:]
            basisM = []
            with open('./model_' + str(modelCount) + '_.csv', mode='w') as data:
                csvModels.append('./model_' + str(modelCount) + '_.csv')
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    ['Basis', 'Coeff'])
                for row in modelSummary:
                    row = np.delete(np.array(row.split(" ")), [i for i, x in enumerate(row.split(" ")) if x == ""])
                    try:
                        basis = row[0]
                        pruned = row[1]
                        coeff = row[2]
                        # if basis == 'x0' : continue
                        if pruned == "No":
                            data_writer.writerow([basis, coeff])
                            basisM.append(basis)
                    except:
                        x = 0
                model = {}
                model["id"] = modelCount
                model["funcs"] = len(basisM)
                ClModels["data"].append(model)
            modelCount += 1

        self.count = 0

        def extractFunctionsFromSplines(x0, x1 , x2 , x3 ,x4 , x5 ,x6):
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

        estimator = baseline_model()
        XSplineVector = []
        velocities = []
        vectorWeights = []

        for i in range(0, len(X)):
            vector = extractFunctionsFromSplines(X[i][0], X[i][1],X[i][2],X[i][3],X[i][4],X[i][5],X[i][6])
            XSplineVector.append(vector)
        #meanInputVector = np.mean(X,axis=0)
        #vector = extractFunctionsFromSplines(meanInputVector[0], meanInputVector[1],meanInputVector[2],meanInputVector[3],meanInputVector[4],
                                             #meanInputVector[5],meanInputVector[6])
        #XSplineVector.append(vector)

        XSplineVectorGen = np.array(XSplineVector)
        weights0 = np.mean(XSplineVectorGen, axis=0)
        # weights1 = self.intercepts
        #weights1 = estimator.layers[0].get_weights()[0][1]
        weights = np.array(np.append(weights0.reshape(-1, 1), np.asmatrix([weights0,weights0,weights0,weights0,weights0,weights0]).T, axis=1),).reshape(7,-1)

        estimator.layers[0].set_weights([weights,np.array([0] * (genModelKnots - 1)) ])
        #np.array([0] * (genModelKnots - 1)



        self.flagGen = True

        # Plot training & validation accuracy values



        NNmodels = []
        scores = []

        if len(partition_labels) > 1:
            for idx, pCurLbl in enumerate(partition_labels):

                self.modelId = idx
                self.countTimes += 1

                XSplineClusterVector = []
                for i in range(0, len(partitionsX[idx])):
                    vector = extractFunctionsFromSplines(partitionsX[idx][i][0], partitionsX[idx][i][1],partitionsX[idx][i][2],partitionsX[idx][i][3],partitionsX[idx][i][4],partitionsX[idx][i][5],partitionsX[idx][i][6])
                    XSplineClusterVector.append(vector)

                # X =  np.append(X, np.asmatrix([dataY]).T, axis=1)
                XSplineClusterVector = np.array(XSplineClusterVector)

                # estimator = baseline_model()
                numOfNeurons = [x for x in ClModels['data'] if x['id'] == idx][0]['funcs']
                estimatorCl = keras.models.Sequential()
                #estimatorCl = baseline_model()


                estimatorCl.add(keras.layers.Dense(numOfNeurons -1 ,input_shape=(7,)))
                #estimatorCl.add(keras.layers.Dense(genModelKnots -1, input_shape=(6,)))
                #estimatorCl.add(keras.layers.Dense(5))
                #estimatorCl.add(keras.layers.Dense(2))
                estimatorCl.add(keras.layers.Dense(1 ))
                estimatorCl.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(), )  # try:

                weights0 = np.mean(XSplineClusterVector, axis=0)
                # weights1 = self.intercepts
                # weights1 = estimatorCl.layers[ 0 ].get_weights()[ 0 ][ 1 ]
                weights = np.array(
                    np.append(weights0.reshape(-1, 1), np.asmatrix([weights0,weights0,weights0,weights0,weights0,weights0]).T, axis=1).reshape(7, -1))

                estimatorCl.layers[0].set_weights([weights, np.array([0] * (numOfNeurons - 1))])
                # modelId=idx

                estimatorCl.fit(np.array(partitionsX[idx]), np.array(partitionsY[idx]), epochs=30 ,validation_split=0.33)

                #x = input_img

                #x = estimatorCl.layers[2](x)

                #modelCl = keras.models.Model(inputs=input_img, outputs=x)
                # model2 = keras.models.Model(inputs=estimator.layers[2].input, outputs=estimator.layers[-1].output)

                #modelCl.compile(optimizer=keras.optimizers.Adam(), loss='mse')
                #modelCl.fit(partitionsX[idx], np.array(partitionsY[idx]), epochs=100)

                #Clscore = estimator.evaluate(np.array(partitionsX[idx]), np.array(partitionsY[idx]), verbose=1)
                #scores.append(Clscore)
                NNmodels.append(estimatorCl)

                # self._partitionsPerModel[ estimator ] = partitionsX[idx]
        # Update private models
        # models=[]

        # Plot training & validation loss values
        history = estimator.fit(X, Y, epochs=50, validation_split=0.33)
        NNmodels.append(estimator)


        self._models = NNmodels

        # Return list of models
        return estimator, history, scores, numpy.empty, vectorWeights  # , estimator , DeepCLpartitionsX

    def getFitnessOfModelForPoint(self, model, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(self._partitionsPerModel[model]) - point))
