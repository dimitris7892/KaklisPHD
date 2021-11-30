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
from tensorflow.keras import backend
from tensorflow.keras.callbacks import ReduceLROnPlateau
import itertools
import pandas as pd
from matplotlib import pyplot as plt
import math
#from sklearn.model_selection import KFold as kf
from scipy import spatial
from sklearn.pipeline import Pipeline
#from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from gekko import GEKKO
import  matplotlib.pyplot as plt
from scipy.interpolate import BivariateSpline
import tensorflow as tf
from tensorflow import keras
#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} )
#sess = tf.Session(config=config)
#keras.backend.set_session(sess)
from scipy.stats import pearsonr
#from sklearn.model_selection import KFold
#import pydot
#import graphviz
import scipy.stats as st
from scipy import *
from scipy.interpolate import RegularGridInterpolator
import scipy as scipy
#from tensorflow.python.tools import inspect_checkpoint as chkp
from time import time
from sklearn.cluster import KMeans
tf.compat.v1.disable_eager_execution()
#tf.executing_eagerly()
from tensorflow.keras.callbacks import EarlyStopping
import glob

class BasePartitionModeler:
    def createModelsFor(self,partitionsX, partitionsY, partition_labels):
        pass

    def extractFunctionsFromSplines(self, modelId ,x0, x1, x2, x3, x4, x5=None, x6=None):
        piecewiseFunc = []
        # csvModels = ['../../../../../Danaos_ML_Project/trainedModels/model_' + str(modelId) + '_.csv']
        # for csvM in csvModels:
        # if csvM != '../../../../../Danaos_ML_Project/trainedModels/model_' + str(modelId) + '_.csv':
        # continue
        # id = csvM.split("_")[ 1 ]
        # piecewiseFunc = [ ]

        # with open(csvM) as csv_file:
        # data = csv.reader(csv_file, delimiter=',')
        csvM = './trainedModels/model_Gen_.csv'
        dataCsvModels=[]
        dataCsvModel = []
        with open(csvM) as csv_file:
            data = csv.reader(csv_file, delimiter=',')
            for row in data:
                dataCsvModel.append(row)
        dataCsvModels.append(dataCsvModel)
        data = dataCsvModels[modelId] if modelId != 'Gen' else dataCsvModels[len(dataCsvModels) - 1]
        # id = modelId
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
            # if self.count == 1:
            # self.intercepts.append(float(row[1]))

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

        return piecewiseFunc , self.interceptsGen

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

    def createModelsFor(self, partitionsX, partitionsY, partition_labels,  X, Y, steps):#,numOfLayers,numOfNeurons):

        models = []
        # partitionsX=np.array(partitionsX[0])
        # partitionsY = np.array(partitionsY[0])

        self.ClustersNum = len(partitionsX)
        self.modelId = -1
        # partition_labels = len(partitionsX)
        # Init model to partition map
        self._partitionsPerModel = {}


        n_steps = steps

        def baseline_model():
            # create model
            model = keras.models.Sequential()

            model.add(keras.layers.LSTM(5 ,input_shape=(n_steps, 5 ,),))#return_sequences=True

            #model.add(keras.layers.Dense(5,))
            model.add(keras.layers.Dense(2, kernel_regularizer = tf.keras.regularizers.l1(0.01),
            activity_regularizer = tf.keras.regularizers.l2(0.01),
            bias_regularizer = tf.keras.regularizers.l2(0.01),))


            model.add(keras.layers.Dense(1,))

            model.compile(loss=keras.losses.mean_squared_error,
                          optimizer=keras.optimizers.Adam() )  # experimental_run_tf_function=False )
            # print(model.summary())

            return model

        seed = 7
        numpy.random.seed(seed)

        self.genF = None



        def split_sequence(sequence, n_steps):
            X, y = list(), list()
            for i in range(len(sequence)):
                # find the end of this pattern
                end_ix = i + n_steps
                # check if we are beyond the sequence
                if end_ix > len(sequence) - 1:
                    break
                # gather input and output parts of the pattern
                seq_x, seq_y = sequence[i:end_ix][:, 0:sequence.shape[1] - 1], sequence[end_ix-1][sequence.shape[1] - 1]
                X.append(seq_x)
                y.append(seq_y)
            return array(X), array(y)

            # define input sequence

        raw_seq = np.array(np.append(X, np.asmatrix(Y).T,axis=1))
        #raw_seq = X

        # split into samples
        Xlstm, Ylstm = split_sequence(raw_seq, n_steps)

        estimator = baseline_model()
        dot_img_file = '/home/dimitris/Desktop/neural.png'
        tf.keras.utils.plot_model(estimator, to_file=dot_img_file, show_shapes=True)
        #weights0 = np.mean(XSplineVector, axis=0)
        # weights1 = self.intercepts
        #weights1 = estimator.layers[0].get_weights()[0][1]
        #weights = np.array(
            #np.append(weights0.reshape(-1, 1), np.asmatrix(weights0).reshape(-1, 1), axis=1).reshape(9, -1))

        #estimator.layers[0].set_weights([weights, np.array([0] * (genModelKnots - 1))])


        print("Shape of training data: " + str(Xlstm.shape))
        print("GENERAL MODEL  ")
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0,restore_best_weights=True)
        rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1,)#restore_best_weights=True
        #XSplineVectorGen = np.reshape(XSplineVectorGen, (XSplineVectorGen.shape[0],1, XSplineVectorGen.shape[1]))
        #for i in range(0,10):
        size = 3000
        #X_test, Y_test = Xlstm[len(Xlstm)-size:len(Xlstm)] , Ylstm[len(Ylstm)-size:len(Ylstm)]
        #Xlstm, Ylstm = Xlstm[:len(Xlstm) - size], Ylstm[:len(Ylstm) - size]
        history = estimator.fit(Xlstm, Ylstm, epochs=100, verbose=0, callbacks=[rlrop] ,validation_split=0.05,)#shuffle=False,batch_size=120)callbacks=[rlrop]
        #XSplineVector =np.array(XSplineVector).reshape(-1,1)
        #history = estimator.fit(XSplineVectorGen, Y, epochs=50, verbose=0, callbacks=[rlrop], validation_split=0.1,batch_size=len(XSplineVectorGen),shuffle=False)
        #estimator.reset_states()
        mse =  estimator.evaluate(Xlstm,Ylstm)
        estimator.save('./DeployedModels/estimatorExpandedInput_.h5')
        #mse = estimator.evaluate(XSplineVectorGen, Y)
        patiences = [10, 20, 30, 40]
        lr_list, loss_list, acc_list, = list(), list(), list()
        '''for i in range(len(patiences)):
            rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patiences[i], min_delta=1E-7)
            lrm = LearningRateMonitor()
            history = estimator.fit(Xlstm, Ylstm, epochs=20, verbose=0,callbacks=[rlrop , lrm],validation_split=0.1,batch_size=12 ,shuffle=False)

            lr, loss = lrm.lrates, history.history['loss']

            lr_list.append(lr)
            loss_list.append(loss)'''
        print("LOSS: "+str(np.round(math.sqrt(mse),2)))
        print("VAL LOSS:  "+str(np.round(math.sqrt(np.mean(history.history['val_loss'])))))
        def line_plots(patiences, series):
            for i in range(len(patiences)):
                plt.subplot(220 + (i + 1))
                plt.plot(series[i])
                plt.title('patience=' + str(patiences[i]), pad=-80)
            plt.show()

        # plot learning rates
        #line_plots(patiences, lr_list)
        # plot loss
        #line_plots(patiences, loss_list)


        # plot train and validation loss
        print(estimator.metrics_names)

        #mse=history.history['loss'][0]
        #estimator.save('./DeployedModels/estimatorCl_Gen.h5')

        #print(estimator.summary())


        #print('Accuracy: %.2f' % (accuracy * 100))

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model train vs validation loss MAE: %.2f' % (np.round(math.sqrt(mse),2)) +"  "+str(np.round(math.sqrt(np.mean(history.history['val_loss'])))) )
        plt.ylabel('loss')
        plt.xlabel('epoch')
        #plt.ylim(0,10)
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()


        self.flagGen = True

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
        sizeTrDt=[]

        if len(partition_labels) > 1:
            for idx, pCurLbl in enumerate(partition_labels):

                self.modelId = idx
                self.countTimes += 1

                XSplineClusterVector = []
                for i in range(0, len(partitionsX[idx])):
                    #vector = extractFunctionsFromSplines(partitionsX[idx][i][0], partitionsX[idx][i][1],partitionsX[idx][i][2],partitionsX[idx][i][3],partitionsX[idx][i][4],partitionsX[idx][i][5],partitionsX[idx][i][6])
                    vector = extractFunctionsFromSplines(partitionsX[idx][i][0], partitionsX[idx][i][1],
                                                         partitionsX[idx][i][2], partitionsX[idx][i][3],
                                                         partitionsX[idx][i][4])
                    SplineVector = np.append(partitionsX[idx][i], vector)
                    # SplineVector = X[i]
                    XSplineVector.append(np.append(SplineVector, Y[i]))
                    #XSplineClusterVector.append(np.append(partitionsX[idx][i],vector))
                    XSplineClusterVector.append(np.append(SplineVector, partitionsY[idx][i]))

                # X =  np.append(X, np.asmatrix([dataY]).T, axis=1)
                XSplineClusterVector = np.array(XSplineClusterVector)
                raw_seq = XSplineClusterVector
                # split into samples
                Xlstm, Ylstm = split_sequence(raw_seq, n_steps)

                # estimator = baseline_model()
                numOfNeurons = [x for x in ClModels['data'] if x['id'] == idx][0]['funcs']
                estimatorCl = keras.models.Sequential()
                #estimatorCl = baseline_model()

                estimatorCl.add(keras.layers.LSTM(5 + numOfNeurons - 1, input_shape=(n_steps,5 + numOfNeurons - 1,)))
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
                #print("CLUSTER:  " +str(idx))
                #for i in range(1,40):


                #estimatorCl.fit(X_train,y_train, epochs=i ,verbose = 0)
                es = EarlyStopping(monitor='val_loss', mode='min', verbose=0)
                #val
                #XSplineClusterVector = np.reshape(XSplineClusterVector, (XSplineClusterVector.shape[0], XSplineClusterVector.shape[1], 1))
                history = estimatorCl.fit(Xlstm, Ylstm, validation_split=0.17,epochs=i,verbose = 0,callbacks=[es])#
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
                estimatorCl.save('./DeployedModels/estimatorCl_'+str(idx)+'.h5')
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
                sizeTrDt.append(len(partitionsX[idx]))
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



        NNmodels.append(estimator)

        if stdWS !=[]:
            normalizedSTDws = (stdWS - min(stdWS)) / (max(stdWS) - min(stdWS))
            normalizedSTDstw = (stdSTW - min(stdSTW)) / (max(stdSTW) - min(stdSTW))
            normalizedSTDFOC = (stdsFOC - min(stdsFOC)) / (max(stdsFOC) - min(stdsFOC))
            normalizedErr = (clustersTrScores - min(clustersTrScores)) / (max(clustersTrScores) - min(clustersTrScores))
            normalizedValErr = (clScores - min(clScores)) / (max(clScores) - min(clScores))
            normalizedStw_ws = abs(normalizedSTDstw - normalizedSTDws)

        #print("CORRELATION COEFF Normalized WSstd-STWstd and ERROR: " + str(pearsonr(normalizedErr, normalizedStw_ws)))

        self._models = NNmodels

        #Return list of models
        with open('./errorEpochCLusters.csv', mode='w') as data:
            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow(['cluster','clusterSize','trError','acc', 'epoch','stdX','stdY','stdSTW','stdWS','nErr','nStdWs','nStdSTW','nValErr','nSTW_WS'])
            for i in range(0,len(clScores)):

                data_writer.writerow([clusters[i],sizeTrDt[i], clustersTrScores[i] ,clScores[i],minEpochs[i],minXVars[i],minYVars[i],stdWS[i],stdSTW[i],normalizedErr[i],normalizedSTDws[i],normalizedSTDstw[i],normalizedValErr[i]])

        return estimator, history, scores

    def getFitnessOfModelForPoint(self, model, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(self._partitionsPerModel[model]) - point))
    def getFitnessOfPoint(self,partitions ,cluster, point):

        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(partitions[cluster], axis=0) - point))



def main():

    company = 'DANAOS'
    vessel = 'EXPRESS ATHENS'
    dataSet = []
    path = './legs/'+vessel+'/'
    modeler = TensorFlowW1()
    for infile in sorted(glob.glob(path + '*.csv')):
        data = pd.read_csv(infile, sep=',', decimal='.', skiprows=0, ).values
        print(str(infile))


        trData = np.array(
            np.append(data[:, 1].reshape(-1, 1), np.asmatrix([data[:, 4], data[:, 3], data[:, 0], data[:, 5],
                                                              data[:, 2]]).T,axis=1)).astype(float)

        for i in range(0, len(trData)):
            trData[i] = np.mean(trData[i:i + 15], axis=0)

        Xvector = trData[:,0:5]
        Yvector = trData[:,5]

        model, history,scores = modeler.createModelsFor([], [], [], Xvector, Yvector, 15)

    model.save('./DeployedModels/estimatorCl_Gen.h5')

if __name__ == "__main__":
    main()