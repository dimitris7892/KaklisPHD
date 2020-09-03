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
import parameters
import itertools
#from sklearn.model_selection import KFold as kf
from scipy import spatial
from sklearn.pipeline import Pipeline
#from sklearn.model_selection import RandomizedSearchCV
from basis_expansions1 import NaturalCubicSpline
from sklearn.preprocessing import StandardScaler
import  matplotlib.pyplot as plt
from scipy.interpolate import BivariateSpline
import tensorflow as tf
from tensorflow import keras
#from sklearn.model_selection import KFold
#import pydot
#import graphviz
import scipy.stats as st
#from tensorflow.python.tools import inspect_checkpoint as chkp
from time import time
import metrics
from sklearn.cluster import KMeans
tf.compat.v1.disable_eager_execution()
#tf.executing_eagerly()
from sklearn import preprocessing
##NEW ONE

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
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(partitions[ cluster ]) - point))

class TriInterpolantModeler(BasePartitionModeler):

    def getFitnessOfPoint(self,partitions ,cluster, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(partitions[ cluster ]) - point))

class LinearRegressionModeler(BasePartitionModeler):
    def createModelsFor(self, partitionsX, partitionsY, partition_labels,tri,X,Y):
        # Init result model list
        models = []
        # Init model to partition map
        self._partitionsPerModel = {}
        # For each label/partition
        for idx, pCurLbl in enumerate(partition_labels):
            # Create a linear model
            curModel = sk.LinearRegression()
            # Fit to data
            curModel.fit(partitionsX[idx], partitionsY[idx])
            # Add to returned list
            models.append(curModel)
            self._partitionsPerModel[curModel] = partitionsX[idx]

        # Update private models
        self._models = models

        # Return list of models
        return models , numpy.empty ,numpy.empty , None,None

    def getFitnessOfModelForPoint(self, model, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(self._partitionsPerModel[ model ]) - point))


class TensorFlowCA(BasePartitionModeler):

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

        def custom_activation23(inputs):

            x = inputs
            self.modelId = str(self.modelId)
            # models = {"data": [ ]}
            intercepts = []
            interceptsGen = 0
            for csvM in csvModels:
                id = csvM.split("_")[1]
                id = 0 if id == 'Gen' else int(id)
                piecewiseFunc = []

                with open(csvM) as csv_file:
                    data = csv.reader(csv_file, delimiter=',')
                    for row in data:
                        # for d in row:
                        if [w for w in row if w == "Basis"].__len__() > 0:
                            continue
                        if [w for w in row if w == "(Intercept)"].__len__() > 0:
                            intercepts.append(float(row[1]))
                            interceptsGen = float(row[1])
                            continue
                        if row.__len__() == 0:
                            continue
                        d = row[0]
                        coeffS = 1
                        # float(row[1])
                        if d.split("*").__len__() == 1:
                            split = ""
                            try:
                                split = d.split('-')[0][2]
                                if split != "x":
                                    num = float(d.split('-')[0].split('h(')[1])

                                    if id == int(self.modelId):
                                        # if float(row[ 1 ]) < 10000:
                                        # piecewiseFunc.append(
                                        # tf.math.multiply(tf.cast(tf.math.greater(x, num), tf.float32),
                                        # float(row[ 1 ]) * (inputs - num)))
                                        # if id ==  self.modelId:
                                        inputs = tf.where(x >= num, float(row[1]) * (inputs - num), inputs)
                                else:
                                    num = float(d.split('-')[1].split(')')[0])
                                    if id == int(self.modelId):
                                        # if float(row[ 1 ]) < 10000:
                                        # piecewiseFunc.append(
                                        # tf.math.multiply(tf.cast(tf.math.less(x, num), tf.float32),
                                        # float(row[ 1 ]) * (num - inputs)))
                                        # if id == self.modelId:
                                        inputs = tf.where(x <= num, float(row[1]) * (num - inputs), inputs)
                            except:
                                if id == int(self.modelId):
                                    # if float(row[ 1 ]) < 10000:
                                    # piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                    # float(row[ 1 ]) * (inputs)))

                                    inputs = tf.where(x >= 0, float(row[1]) * inputs, inputs)


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
                                        # if id == self.modelId:
                                        # piecewiseFunc.append(tf.math.multiply(tf.cast(
                                        # tf.math.logical_and(tf.math.less(x, nums[ 0 ]),
                                        # tf.math.greater(nums[ 1 ], x)), tf.float32),
                                        # float(row[ 1 ]) * (nums[ 0 ] - inputs) * (
                                        # inputs - nums[ 1 ])))
                                    # if id ==  self.modelId:
                                    # inputs = tf.where(x < nums[ 0 ] and x >= nums[ 0 ],
                                    # float(row[ 1 ]) * (nums[ 0 ] - inputs) * (
                                    flgs.append(flgFirstx)  # inputs - nums[ 1 ]), inputs)
                                except:
                                    flgFirstx = False
                                    flgs = []
                                    try:
                                        if d.split('-')[0][2] == "x":
                                            if id == int(self.modelId):
                                                # if float(row[ 1 ]) < 10000:
                                                # piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                                # float(row[ 1 ]) * (inputs) * (
                                                # inputs - nums[ 0 ])))

                                                inputs = tf.where(x >= 0,
                                                                  float(row[1]) * (inputs) * (inputs - nums[0]), inputs)

                                        else:
                                            flgFirstx = True
                                            if id == int(self.modelId):
                                                # if float(row[ 1 ]) < 10000:
                                                # piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                                # float(row[ 1 ]) * (inputs) * (
                                                # nums[ 0 ] - inputs)))

                                                inputs = tf.where(x > 0,
                                                                  float(row[1]) * (inputs) * (nums[0] - inputs), inputs)
                                        flgs.append(flgFirstx)
                                    except:
                                        if id == int(self.modelId):
                                            if float(row[1]) < 10000:
                                                # piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                                # float(row[ 1 ]) * (inputs)))

                                                inputs = tf.where(x >= 0, float(row[1]) * (inputs), inputs)
                            if id == int(self.modelId):
                                if flgs.count(True) == 2:
                                    # if float(row[ 1 ])<10000:
                                    # piecewiseFunc.append(tf.math.multiply(tf.cast(
                                    # tf.math.logical_and(tf.math.less(x, nums[ 0 ]),
                                    # tf.math.less(nums[ 1 ], x)), tf.float32),
                                    # float(row[ 1 ]) * (nums[ 0 ] - inputs) * (
                                    # nums[ 1 ] - inputs)))

                                    inputs = tf.where(tf.math.logical_and(tf.math.less(x, nums[0]),
                                                                          tf.math.less(nums[1], x)),
                                                      float(row[1]) * (nums[0] - inputs) * (
                                                              nums[1] - inputs), inputs)

                                elif flgs.count(False) == 2:
                                    # if float(row[ 1 ]) < 10000:
                                    # piecewiseFunc.append(tf.math.multiply(tf.cast(
                                    # tf.math.logical_and(tf.math.greater(x, nums[ 0 ]),
                                    # tf.math.greater(nums[ 1 ], x)), tf.float32),
                                    # float(row[ 1 ]) * (inputs - nums[ 0 ]) * (
                                    # inputs - nums[ 1 ])))

                                    inputs = tf.where(tf.math.logical_and(tf.math.greater(x, nums[0]),
                                                                          tf.math.greater(nums[1], x)),
                                                      float(row[1]) * (inputs - nums[0]) * (
                                                              inputs - nums[1]), inputs)
                                else:
                                    try:
                                        if flgs[0] == False:
                                            if nums.__len__() > 1:
                                                # if float(row[ 1 ]) < 10000:
                                                # piecewiseFunc.append(tf.math.multiply(tf.cast(
                                                # tf.math.logical_and(tf.math.greater(x, nums[ 0 ]),
                                                # tf.math.less(nums[ 1 ], x)), tf.float32),
                                                # float(row[ 1 ]) * (inputs - nums[ 0 ]) * (
                                                # nums[ 1 ] - inputs)))

                                                inputs = tf.where(tf.math.logical_and(tf.math.greater(x, nums[0]),
                                                                                      tf.math.less(nums[1], x)),
                                                                  float(row[1]) * (inputs - nums[0]) * (
                                                                          nums[1] - inputs), inputs)
                                            else:
                                                # if float(row[ 1 ]) < 10000:
                                                # pie#cewiseFunc.append(tf.math.multiply(tf.cast(
                                                # tf.math.greater(x, nums[ 0 ])
                                                # , tf.float32),
                                                # float(row[ 1 ]) * (inputs - nums[ 0 ])))

                                                inputs = tf.where(x > nums[0],
                                                                  float(row[1]) * (inputs - nums[0]), inputs)
                                        else:
                                            if nums.__len__() > 1:
                                                # if float(row[ 1 ]) < 10000:
                                                # piecewiseFunc.append(tf.math.multiply(tf.cast(
                                                # tf.math.logical_and(tf.math.less(x, nums[ 0 ]),
                                                # tf.math.greater(nums[ 1 ], x)), tf.float32),
                                                # float(row[ 1 ]) * (nums[ 0 ] - inputs) * (
                                                # inputs - nums[ 1 ])))

                                                inputs = tf.where(tf.math.logical_and(tf.math.less(x, nums[0]),
                                                                                      tf.math.greater(nums[1], x)),
                                                                  float(row[1]) * (nums[0] - inputs) * (
                                                                          inputs - nums[1]), inputs)
                                            else:
                                                # if float(row[ 1 ]) < 10000:
                                                # piecewiseFunc.append(tf.math.multiply(tf.cast(
                                                # tf.math.less(x, nums[ 0 ]), tf.float32),
                                                # float(row[ 1 ]) * (
                                                # inputs - nums[ 0 ])))

                                                inputs = tf.where(x < nums[0],
                                                                  float(row[1]) * (
                                                                          inputs - nums[0]), inputs)
                                    except:
                                        d = 0

                    model = {}
                    model["id"] = id
                    model["funcs"] = piecewiseFunc
                    self.models["data"].append(model)

            # modelId = 0 if modelId[ 'args' ] == -1 else modelId[ 'args' ]

            # interc = tf.cast(x, tf.float32) + intercepts[self.modelId]
            # funcs = [ x for x in models[ 'data' ] if x[ 'id' ] == str(modelId) ][ 0 ][ 'funcs' ]
            # for f in funcs:
            # inputs = f
            # funcs = [ x for x in models[ 'data' ] if x[ 'id' ] == str(self.modelId) ][ 0 ][ 'funcs' ]

            # intercept = interceptsGen if self.modelId == 'Gen' else intercepts[ int(self.modelId) ]
            # (intercept if intercept < 10000 else 0 )
            # SelectedFuncs = np.sum(
            # funcs) if len(funcs) > 0 else \
            # interceptsGen + np.sum(
            # [ x for x in models[ 'data' ] if x[ 'id' ] == 'Gen' ][ 0 ][ 'funcs' ])
            # intercept = tf.constant()
            # constants = intercepts[ 0 if self.modelId=='Gen' else self.modelId ]
            # k_constants = keras.backend.variable(constants)
            return inputs

        def custom_activation2(inputs):

            x = inputs
            self.modelId = str(self.modelId)

            if self.countTimes == 0:

                for csvM in csvModels:
                    id = csvM.split("_")[1]
                    piecewiseFunc = []

                    with open(csvM) as csv_file:
                        data = csv.reader(csv_file, delimiter=',')
                        for row in data:
                            # for d in row:
                            if [w for w in row if w == "Basis"].__len__() > 0:
                                continue
                            if [w for w in row if w == "(Intercept)"].__len__() > 0:
                                self.intercepts.append(float(row[1]))
                                self.interceptsGen = float(row[1])
                                continue
                            if row.__len__() == 0:
                                continue
                            d = row[0]
                            if d.split("*").__len__() == 1:
                                split = ""
                                try:
                                    split = d.split('-')[0][2]
                                    if split != "x":
                                        num = float(d.split('-')[0].split('h(')[1])
                                        # if id == id:
                                        # if float(row[ 1 ]) < 10000:
                                        try:
                                            piecewiseFunc.append(
                                                tf.math.multiply(tf.cast(tf.math.less(x, num), tf.float32),
                                                                 (num - inputs)))
                                        # if id ==  self.modelId:
                                        # inputs = tf.where(x >= num, float(row[ 1 ]) * (inputs - num), inputs)
                                        except:
                                            dc = 0
                                    else:
                                        num = float(d.split('-')[1].split(')')[0])
                                        # if id == id:
                                        # if float(row[ 1 ]) < 10000:
                                        try:

                                            piecewiseFunc.append(
                                                tf.math.multiply(tf.cast(tf.math.greater(x, num), tf.float32),
                                                                 (inputs - num)))
                                        # if id == self.modelId:
                                        # inputs = tf.where(x <= num, float(row[ 1 ]) * (num - inputs), inputs)
                                        except:
                                            dc = 0
                                except:
                                    # if id == id:
                                    # if float(row[ 1 ]) < 10000:
                                    try:
                                        piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                                                              (inputs)))

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
                                        try:
                                            if d.split('-')[0][2] == "x":
                                                # if id == id:
                                                # if float(row[ 1 ]) < 10000:
                                                try:
                                                    piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                                                                          (inputs) * (
                                                                                                  inputs - nums[0])))

                                                    # inputs = tf.where(x >= 0,
                                                    # float(row[ 1 ]) * (inputs) * (inputs - nums[ 0 ]), inputs)
                                                except:
                                                    dc = 0

                                            else:
                                                flgFirstx = True
                                                # if id == id:
                                                # if float(row[ 1 ]) < 10000:
                                                try:
                                                    piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                                                                          (inputs) * (
                                                                                                  nums[0] - inputs)))

                                                    # inputs = tf.where(x > 0 ,
                                                    # float(row[ 1 ]) * (inputs) * (nums[ 0 ] - inputs),inputs)
                                                    flgs.append(flgFirstx)
                                                except:
                                                    dc = 0

                                        except:
                                            # if id == id:
                                            # if float(row[ 1 ]) < 10000:
                                            try:
                                                piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                                                                      (inputs)))

                                                # inputs = tf.where(x >= 0, float(row[ 1 ]) * (inputs), inputs)
                                            except:
                                                dc = 0
                                try:
                                    # if id == id:
                                    if flgs.count(True) == 2:
                                        # if float(row[ 1 ])<10000:
                                        try:
                                            piecewiseFunc.append(tf.math.multiply(tf.cast(
                                                tf.math.logical_and(tf.math.less(x, nums[0]),
                                                                    tf.math.less(x, nums[1])), tf.float32),
                                                (nums[0] - inputs) * (
                                                        nums[1] - inputs)))

                                            # inputs = tf.where(x < nums[0] and x < nums[1],
                                            # float(row[ 1 ]) * (nums[ 0 ] - inputs) * (
                                            # nums[ 1 ] - inputs), inputs)
                                        except:
                                            dc = 0

                                    elif flgs.count(False) == 2:
                                        # if float(row[ 1 ]) < 10000:
                                        try:
                                            piecewiseFunc.append(tf.math.multiply(tf.cast(
                                                tf.math.logical_and(tf.math.greater(x, nums[0]),
                                                                    tf.math.greater(x, nums[1])), tf.float32),
                                                (inputs - nums[0]) * (
                                                        inputs - nums[1])))

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
                                                        piecewiseFunc.append(tf.math.multiply(tf.cast(
                                                            tf.math.logical_and(tf.math.greater(x, nums[0]),
                                                                                tf.math.less(x, nums[1])), tf.float32),
                                                            (inputs - nums[0]) * (
                                                                    nums[1] - inputs)))

                                                    # inputs = tf.where(x > nums[ 0 ] and x < nums[ 1 ],
                                                    # float(row[ 1 ]) * (inputs - nums[ 0 ]) * (
                                                    # nums[ 1 ] - inputs), inputs)
                                                    except:
                                                        dc = 0
                                                else:
                                                    # if float(row[ 1 ]) < 10000:
                                                    try:
                                                        piecewiseFunc.append(tf.math.multiply(tf.cast(
                                                            tf.math.greater(x, nums[0])
                                                            , tf.float32),
                                                            (inputs - nums[0])))

                                                        # inputs = tf.where(x > nums[0],
                                                        # float(row[ 1 ]) * (inputs - nums[ 0 ]), inputs)
                                                    except:
                                                        dc = 0
                                            else:
                                                if nums.__len__() > 1:
                                                    # if float(row[ 1 ]) < 10000:
                                                    try:
                                                        piecewiseFunc.append(tf.math.multiply(tf.cast(
                                                            tf.math.logical_and(tf.math.less(x, nums[0]),
                                                                                tf.math.greater(x, nums[1])),
                                                            tf.float32),
                                                            (nums[0] - inputs) * (
                                                                    inputs - nums[1])))

                                                        # inputs = tf.where(x < nums[ 0 ] and x > nums[1],
                                                        # float(row[ 1 ]) * (nums[ 0 ] - inputs) * (
                                                        # inputs - nums[ 1 ]), inputs)
                                                    except:
                                                        dc = 0
                                                else:
                                                    # if float(row[ 1 ]) < 10000:
                                                    try:
                                                        piecewiseFunc.append(tf.math.multiply(tf.cast(
                                                            tf.math.less(x, nums[0]), tf.float32),
                                                            (
                                                                    inputs - nums[0])))

                                                        # inputs = tf.where(x < nums[ 0 ],
                                                        # float(row[ 1 ]) * (
                                                        # inputs - nums[ 0 ]), inputs)
                                                    except:
                                                        dc = 0
                                        except:
                                            dc = 0
                                except:
                                    dc = 0

                        model = {}
                        model["id"] = id
                        model["funcs"] = piecewiseFunc
                        self.models["data"].append(model)
                # self.countTimes += 1
            # modelId = 0 if modelId[ 'args' ] == -1 else modelId[ 'args' ]

            # interc = tf.cast(x, tf.float32) + intercepts[self.modelId]
            # funcs = [ x for x in models[ 'data' ] if x[ 'id' ] == str(modelId) ][ 0 ][ 'funcs' ]
            # for f in funcs:
            # inputs = f
            # if  self.countTimes==1:
            funcs = [x for x in self.models['data'] if x['id'] == str(self.modelId)][0]['funcs']
            genFuncs = [x for x in self.models['data'] if x['id'] == 'Gen'][0]['funcs']
            self.genF = genFuncs
            intercept = self.interceptsGen if self.modelId == 'Gen' else self.intercepts[int(self.modelId)]
            # (intercept if intercept < 10000 else 0 )
            ten = tf.keras.backend.sum(funcs, keepdims=True)
            tenGen = tf.keras.backend.sum(genFuncs, keepdims=True)
            SelectedFuncs = np.sum(funcs) if self.modelId != 'Gen' else np.sum(genFuncs)
            #SelectedFuncs=tf.keras.backend.sum(funcs,keepdims=True) if len(funcs)>0 else  tf.keras.backend.sum(genFuncs,keepdims=True)
            # self.countTimes+=1
            # intercept = tf.constant()
            # constants = intercepts[ 0 if self.modelId=='Gen' else self.modelId ]
            # k_constants = keras.backend.variable(constants)
            return SelectedFuncs

        def baseline_model():
            # create model
            model = keras.models.Sequential()

            model.add(keras.layers.Dense(len(partition_labels), input_shape=(2,)))
            #try:
            model.add(keras.layers.Activation(custom_activation2))
            #except:
                #x=0
            model.add(keras.layers.Dense(5, ))
            model.add(keras.layers.Dense(1, ))

            #print(model.summary())

            # model.add(keras.layers.Activation('linear'))  # activation=custom_activation
            # Compile model
            model.compile(loss=keras.losses.mean_squared_error,
                          optimizer=keras.optimizers.Adam(), )  # experimental_run_tf_function=False )
            return model

        def baseline_model1():
            # create model
            model = keras.models.Sequential()

            model.add(keras.layers.Dense(2 + genModelKnots, input_shape=(2,)))
            model.add(keras.layers.Dense(genModelKnots, activation=custom_activation2() ))

            #model.add(keras.layers.Dense(10, input_shape=(2,)))

            model.add(keras.layers.Dense(1))

            model.compile(loss=keras.losses.mean_squared_error,
                          optimizer=keras.optimizers.Adam(), )  # experimental_run_tf_function=False )
            print(model.summary())
            return model

        seed = 7
        numpy.random.seed(seed)

        self.genF = None

        sModel = []
        sr = sp.Earth(max_degree=2)
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

            with open('./trainedModels/model_Gen_.csv', mode='w') as data:
                csvModels.append('./trainedModels/model_Gen_.csv')
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

        def customLoss(y_true, y_pred):
            return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            # * tf.keras.losses.kullback_leibler_divergence(y_true,y_pred)
            # tf.keras.losses.mean_squared_error(y_true,y_pred) *

        ##train general neural

        ########SET K MEANS INITIAL WEIGHTS TO CLUSTERING LAYER
        def customLoss1(yTrue, yPred):
            self.triRpm = preTrainedWeights(partitionsX[self.count])
            self.count += 1
            return (tf.losses.mean_squared_error(yTrue, (yPred + self.triRpm) / 2))

        srModels = []
        for idx, pCurLbl in enumerate(partition_labels):
            # maxTerms = if len(DeepCLpartitionsX) > 5000
            srM = sp.Earth(max_degree=2)
            srM.fit(np.array(partitionsX[idx]), np.array(partitionsY[idx]))
            srModels.append(srM)
        modelCount = 0
        import csv
        # csvModels = []
        ClModels = {"data": []}

        for models in srModels:
            modelSummary = str(models.summary()).split("\n")[4:]
            basisM = []
            with open('./trainedModels/model_' + str(modelCount) + '_.csv', mode='w') as data:
                csvModels.append('./trainedModels/model_' + str(modelCount) + '_.csv')
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
        self.flagGen = False

        estimator = baseline_model()

        estimator.fit(X, Y, epochs=100, validation_split=0.33,verbose=0)
        self.flagGen = True

        # validation_data=(X_test,y_test)

        def insert_intermediate_layer_in_keras(model, layer_id, new_layer):

            layers = [l for l in model.layers]

            x = layers[0].output
            for i in range(1, len(layers)):
                if i == layer_id:
                    x = new_layer(x)
                if i == len(layers) - 1:
                    x = keras.layers.Dense(1)(x)
                else:
                    x = layers[i](x)
            # x = new_layer(x)
            new_model = keras.Model(inputs=model.input, outputs=x)
            # new_model.add(new_layer)
            new_model.compile(loss='mse', optimizer=keras.optimizers.Adam())
            # .add(x)
            return new_model

        def replace_intermediate_layer_in_keras(model, layer_id, layer_id1, new_layer, new_layer1):

            layers = [l for l in model.layers]

            x = layers[0].output
            for i in range(1, len(layers)):
                if i == layer_id:
                    x = new_layer(x)

                elif i == layer_id1:
                    x = new_layer1(x)

                else:
                    # if i == len(layers) - 1:
                    # x = keras.layers.Dense(1)(x)
                    # else:
                    x = layers[i](x)

            new_model = keras.Model(inputs=model.input, outputs=x)
            new_model.compile(loss='mse', optimizer=keras.optimizers.Adam(), )
            return new_model

        NNmodels = []
        scores = []

        #estimatorGen = baseline_model1()

        #estimatorGen.fit(X, Y, epochs=100, validation_split=0.33, verbose=0)
        if len(partition_labels)>1:
            for idx, pCurLbl in enumerate(partition_labels):

                # try:
                self.modelId = idx
                self.countTimes += 1
                #estimatorCl = baseline_model()
                # modelId=idx

                # estimatorCl = baseline_moDensedel1()
                # estimatorCl.fit(np.array(DeepCLpartitionsX[ idx ]), np.array(DeepCLpartitionsY[ idx ]), epochs=100,
                # validation_split=0.33)  # validation_split=0.33
                # if idx==0:
                # estimator.add(keras.layers.Activation(custom_activation2))
                # else:
                numOfNeurons = [x for x in ClModels['data'] if x['id'] == idx][0]['funcs']
                # estimatorCl=replace_intermediate_layer_in_keras(estimator, -1 ,MyLayer(5))
                # len(DeepClpartitionLabels)+
                # estimatorCl = insert_intermediate_layer_in_keras(estimator, 1, keras.layers.Dense(numOfNeurons))
                # lr = 0.001 if np.std(np.array(partitionsX[idx])) < 30  else 0.2
                estimatorCl = keras.models.Sequential()

                estimatorCl.add(keras.layers.Dense(numOfNeurons, input_shape=(2,),))
                # try:
                #estimatorCl.add(keras.layers.Dense(numOfNeurons,activation=custom_activation2))
                # except:
                # x=0
                #estimatorCl.add(keras.layers.Dense(5, ))
                estimatorCl.add(keras.layers.Dense(1, ))
                estimatorCl.compile(loss=keras.losses.mean_squared_error,
                              optimizer=keras.optimizers.Adam(), )

                '''try:

                    estimatorCl = replace_intermediate_layer_in_keras(estimatorCl, 0, -1,
                                                                      keras.layers.Dense(numOfNeurons, input_shape=(2,)),
                                                                      keras.layers.Activation(custom_activation2))

                except:
                    self.modelId = 'Gen'
                    estimatorCl = replace_intermediate_layer_in_keras(estimatorCl, 0, -1,
                                                                      keras.layers.Dense(numOfNeurons,
                                                                                         input_shape=(2,)),
                                                                      keras.layers.Activation(custom_activation2))'''
                    # estimatorCl = insert_intermediate_layer_in_keras(estimatorGen, 1,keras.layers.Activation(custom_activation2))
                    # estimatorCl = insert_intermediate_layer_in_keras(estimator,0,keras.layers.Activation(custom_activation2))

                    # estimatorCl.add(keras.layers.Activation(custom_activation2))
                    # estimator.compile()
                    # estimator.layers[3] = custom_activation2(inputs=estimator.layers[2].output, modelId=idx) if idx ==0 else estimator.layers[3]
                    # estimator.layers[3] = custom_activation2 if idx ==3 else estimator.layers[3]

                estimatorCl.fit(np.array(partitionsX[idx]), np.array(partitionsY[idx]), epochs=100,
                                validation_split=0.33,verbose=0)  # validation_split=0.33

                #score = estimatorCl.evaluate(np.array(partitionsX[idx]), np.array(partitionsY[idx]), verbose=0)
                #print("%s: %.2f%%" % ("acc: ", score))
                #scores.append(score)
                NNmodels.append(estimatorCl)
                # except Exception as e:
                # print(str(e))
                # return
                # models[pCurLbl]=estimator
                # self._partitionsPerModel[ estimator ] = partitionsX[idx]
            # Update private models
            # models=[]

        NNmodels.append(estimator)
        self._models = NNmodels

        # Return list of models
        return estimator, None, scores, numpy.empty, None  # , estimator , DeepCLpartitionsX



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

    def createModelsFor(self, partitionsX, partitionsY, partition_labels,tri,X,Y):

        models = [ ]
        #partitionsX=np.array(partitionsX[0])
        #partitionsY = np.array(partitionsY[0])

        self.ClustersNum=len(partitionsX)
        self.modelId = -1
        #partition_labels = len(partitionsX)
        # Init model to partition map
        self._partitionsPerModel = {}
        def SplinesCoef(partitionsX, partitionsY):

           model= sp.Earth(use_fast=True)
           model.fit(partitionsX,partitionsY)

           return model.coef_


        def getFitnessOfPoint( partitions, cluster, point):
            return 1.0 / (1.0 + numpy.linalg.norm(np.mean(partitions[cluster]) - point))



        def baseline_model():
            # create model
            model = keras.models.Sequential()


            #model.add(keras.layers.Dense(2 + genModelKnots-1, input_shape=(2,)))
            model.add(keras.layers.Dense(genModelKnots - 1, input_shape=(2,)))

            model.add(keras.layers.Dense(2))

            model.add(keras.layers.Dense(1))


            model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(), )  # experimental_run_tf_function=False )
            #print(model.summary())
            return model

        seed = 7
        numpy.random.seed(seed)


        self.genF=None

        sModel=[]
        sr = sp.Earth(max_degree=1)
        sr.fit(X,Y)
        sModel.append(sr)
        import csv
        csvModels = [ ]
        genModelKnots=[]

        self.modelId = 'Gen'
        self.countTimes=0
        self.models = {"data": [ ]}
        self.intercepts = [ ]
        self.interceptsGen = 0
        self.SelectedFuncs = 0
        for models in sModel:
            modelSummary = str(models.summary()).split("\n")[ 4: ]

            with open('./trainedModels/model_Gen_.csv', mode='w') as data:
                csvModels.append('./trainedModels/model_Gen_.csv')
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [ 'Basis', 'Coeff' ])
                for row in modelSummary:
                    row = np.delete(np.array(row.split(" ")), [ i for i, x in enumerate(row.split(" ")) if x == "" ])
                    try:
                        basis = row[ 0 ]
                        pruned = row[ 1 ]
                        coeff = row[ 2 ]
                        #if basis=='x0' :continue
                        if pruned == "No":
                            data_writer.writerow([ basis, coeff ])
                            genModelKnots.append(basis)
                    except:
                        x = 0

            genModelKnots = len(genModelKnots)
            #modelCount += 1
            #models.append(autoencoder)



        srModels = []
        for idx, pCurLbl in enumerate(partition_labels):

            #maxTerms = if len(DeepCLpartitionsX) > 5000
            srM = sp.Earth(max_degree=1)
            srM.fit(np.array(partitionsX[idx]), np.array(partitionsY[idx]))
            srModels.append(srM)
        modelCount =0
        import csv
        #csvModels = []
        ClModels={"data":[]}

        for models in srModels:
            modelSummary = str(models.summary()).split("\n")[4:]
            basisM = [ ]
            with open('./trainedModels/model_'+str(modelCount)+'_.csv', mode='w') as data:
                csvModels.append('./trainedModels/model_'+str(modelCount)+'_.csv')
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                ['Basis', 'Coeff'])
                for row in modelSummary:
                    row=np.delete(np.array(row.split(" ")), [i for i, x in enumerate(row.split(" ")) if x == ""])
                    try:
                        basis = row[0]
                        pruned = row[1]
                        coeff = row[2]
                        #if basis == 'x0' : continue
                        if pruned == "No":
                            data_writer.writerow([basis, coeff])
                            basisM.append(basis)
                    except:
                        x=0
                model = {}
                model[ "id" ] = modelCount
                model[ "funcs" ] = len(basisM)
                ClModels[ "data" ].append(model)
            modelCount+=1

        self.count=0
        def extractFunctionsFromSplines(x0, x1):
            piecewiseFunc = [ ]
            self.count = self.count + 1
            for csvM in csvModels:
                if csvM!='./trainedModels/model_'+str(self.modelId)+'_.csv':
                    continue
                #id = csvM.split("_")[ 1 ]
                #piecewiseFunc = [ ]

                with open(csvM) as csv_file:
                    data = csv.reader(csv_file, delimiter=',')
                    for row in data:
                        # for d in row:
                        if [ w for w in row if w == "Basis" ].__len__() > 0:
                            continue
                        if [ w for w in row if w == "(Intercept)" ].__len__() > 0:

                            self.interceptsGen = float(row[ 1 ])
                            continue

                        if row.__len__() == 0:
                            continue
                        d = row[ 0 ]
                        if self.count==1:
                            self.intercepts.append(float(row[1]))

                        if d.split("*").__len__() == 1:
                            split = ""
                            try:
                                split = d.split('-')[ 0 ][ 2:4 ]
                                if split != "x0" and split!="x1":
                                    split = d.split('-')[ 1 ]
                                    num = float(d.split('-')[ 0 ].split('h(')[ 1 ])
                                    # if id == id:
                                    # if float(row[ 1 ]) < 10000:
                                    try:
                                        # piecewiseFunc.append(
                                        # tf.math.multiply(tf.cast(tf.math.less(x, num), tf.float32),
                                        # (num - inputs)))
                                        if split.__contains__("x0"):
                                            piecewiseFunc.append((num - x0))# * float(row[ 1 ]))
                                        if split.__contains__("x1"):
                                            piecewiseFunc.append((num - x1)) #* float(row[ 1 ]))
                                    # if id ==  self.modelId:
                                    # inputs = tf.where(x >= num, float(row[ 1 ]) * (inputs - num), inputs)
                                    except:
                                        dc = 0
                                else:
                                    ##x0 or x1
                                    split = d.split('-')[ 0 ]
                                    num = float(d.split('-')[ 1 ].split(')')[ 0 ])
                                    # if id == id:
                                    # if float(row[ 1 ]) < 10000:
                                    try:
                                        if split.__contains__("x0"):
                                            piecewiseFunc.append((x0 - num))# * float(row[ 1 ]))
                                        if split.__contains__("x1"):
                                            piecewiseFunc.append((x1 - num))# * float(row[ 1 ]))

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
                            nums = [ ]
                            flgFirstx = False
                            flgs = [ ]
                            for r in funcs:
                                try:
                                    if r.split('-')[ 0 ][ 2 ] != "x":
                                        flgFirstx = True
                                        nums.append(float(r.split('-')[ 0 ].split('h(')[ 1 ]))

                                    else:
                                        nums.append(float(r.split('-')[ 1 ].split(')')[ 0 ]))

                                    flgs.append(flgFirstx)
                                except:
                                    flgFirstx = False
                                    flgs = [ ]
                                    split=d.split('-')[ 0 ][ 2 ]
                                    try:
                                        if d.split('-')[ 0 ][ 2 ] == "x":
                                            # if id == id:
                                            # if float(row[ 1 ]) < 10000:
                                            try:

                                                if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[ 1 ].__contains__("x1"):
                                                    split="x1"
                                                if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[1 ].__contains__("x0"):
                                                    split="x0"
                                                if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[1 ].__contains__("x1"):
                                                    split="x01"
                                                if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[1 ].__contains__("x0"):
                                                    split = "x10"

                                                if split == "x0":
                                                    piecewiseFunc.append(x0 * (x0 - nums[ 0 ]) * float(row[ 1 ]))
                                                elif split=="x1":
                                                    piecewiseFunc.append(x1 * (x1 - nums[ 0 ]) * float(row[ 1 ]))
                                                elif split == "x01":
                                                    piecewiseFunc.append(x0 * (x1 - nums[ 0 ]) * float(row[ 1 ]))
                                                elif split == "x10":
                                                    piecewiseFunc.append(x1 * (x0 - nums[ 0 ]) * float(row[ 1 ]))
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

                                                if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[
                                                    1 ].__contains__("x1"):
                                                    split = "x1"
                                                if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[
                                                    1 ].__contains__("x0"):
                                                    split = "x0"
                                                if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[
                                                    1 ].__contains__("x1"):
                                                    split = "x01"
                                                if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[
                                                    1 ].__contains__("x0"):
                                                    split = "x10"

                                                if split == "x0":
                                                    piecewiseFunc.append(x0 * (nums[ 0 ] - x0) * float(row[ 1 ]))
                                                elif split=="x1":
                                                    piecewiseFunc.append(x1 * (nums[ 0 ] - x1) * float(row[ 1 ]))
                                                elif split=="x01":
                                                    piecewiseFunc.append(x0 * (nums[ 0 ] - x1) * float(row[ 1 ]))
                                                elif split=="x10":
                                                    piecewiseFunc.append(x1 * (nums[ 0 ] - x0) * float(row[ 1 ]))

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

                                        if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[ 1 ].__contains__(
                                                "x1"):
                                            split = "x1"
                                        if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[ 1 ].__contains__(
                                                "x0"):
                                            split = "x0"
                                        if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[ 1 ].__contains__(
                                                "x1"):
                                            split = "x01"
                                        if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[ 1 ].__contains__(
                                                "x0"):
                                            split = "x10"

                                        if split == "x0":
                                            piecewiseFunc.append((nums[ 0 ] - x0) * (nums[ 1 ] - x0) * float(row[ 1 ]))
                                        elif split=="x1":
                                            piecewiseFunc.append((nums[ 0 ] - x1) * (nums[ 1 ] - x1) * float(row[ 1 ]))
                                        elif split=="x01":
                                            piecewiseFunc.append((nums[ 0 ] - x0) * (nums[ 1 ] - x1) * float(row[ 1 ]))
                                        elif split=="x10":
                                            piecewiseFunc.append((nums[ 0 ] - x1) * (nums[ 1 ] - x0) * float(row[ 1 ]))

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
                                        if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[ 1 ].__contains__(
                                                "x1"):
                                            split = "x1"
                                        if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[ 1 ].__contains__(
                                                "x0"):
                                            split = "x0"
                                        if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[ 1 ].__contains__(
                                                "x1"):
                                            split = "x01"
                                        if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[ 1 ].__contains__(
                                                "x0"):
                                            split = "x10"

                                        if split=="x0":
                                            piecewiseFunc.append((x0 - nums[ 0 ]) * (x0 - nums[ 1 ]) * float(row[ 1 ]))
                                        elif split=="x1":
                                            piecewiseFunc.append((x1 - nums[ 0 ]) * (x1 - nums[ 1 ]) * float(row[ 1 ]))
                                        elif split=="x01":
                                            piecewiseFunc.append((x0 - nums[ 0 ]) * (x1 - nums[ 1 ]) * float(row[ 1 ]))
                                        elif split=="x10":
                                            piecewiseFunc.append((x1 - nums[ 0 ]) * (x0 - nums[ 1 ]) * float(row[ 1 ]))
                                        # inputs = tf.where(x > nums[ 0 ] and x > nums[ 1 ],
                                        # float(row[ 1 ]) * (inputs - nums[ 0 ]) * (
                                        # inputs - nums[ 1 ]), inputs)
                                    except:
                                        dc = 0
                                else:
                                    try:
                                        if flgs[ 0 ] == False:
                                            if nums.__len__() > 1:
                                                # if float(row[ 1 ]) < 10000:
                                                try:
                                                    if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[
                                                        2 ].__contains__("x1"):
                                                        split = "x1"
                                                    if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[
                                                        2 ].__contains__("x0"):
                                                        split = "x0"
                                                    if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[
                                                        2 ].__contains__("x1"):
                                                        split = "x01"
                                                    if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[
                                                        2 ].__contains__("x0"):
                                                        split = "x10"

                                                    if split == "x0":
                                                        piecewiseFunc.append(
                                                            (x0 - nums[ 0 ]) * (nums[ 1 ] - x0) * float(row[ 1 ]))
                                                    elif split == "x1":
                                                        piecewiseFunc.append(
                                                            (x1 - nums[ 0 ]) * (nums[ 1 ] - x1) * float(row[ 1 ]))
                                                    elif split == "x01":
                                                        piecewiseFunc.append(
                                                            (x0 - nums[ 0 ]) * (nums[ 1 ] - x1) * float(row[ 1 ]))
                                                    elif split == "x10":
                                                        piecewiseFunc.append(
                                                            (x1 - nums[ 0 ]) * (nums[ 1 ] - x0) * float(row[ 1 ]))




                                                # inputs = tf.where(x > nums[ 0 ] and x < nums[ 1 ],
                                                # float(row[ 1 ]) * (inputs - nums[ 0 ]) * (
                                                # nums[ 1 ] - inputs), inputs)
                                                except:
                                                    dc = 0
                                            else:
                                                # if float(row[ 1 ]) < 10000:
                                                try:
                                                    if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[
                                                        1 ].__contains__("x1"):
                                                        split = "x1"
                                                    if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[
                                                        1 ].__contains__("x0"):
                                                        split = "x0"
                                                    if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[
                                                        1 ].__contains__("x1"):
                                                        split = "x01"
                                                    if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[
                                                        1 ].__contains__("x0"):
                                                        split = "x10"

                                                    piecewiseFunc.append((x0 - nums[ 0 ]) * float(row[ 1 ]))



                                                    # inputs = tf.where(x > nums[0],
                                                    # float(row[ 1 ]) * (inputs - nums[ 0 ]), inputs)
                                                except:
                                                    dc = 0
                                        else:
                                            if nums.__len__() > 1:
                                                # if float(row[ 1 ]) < 10000:
                                                try:
                                                    if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[
                                                        1 ].__contains__("x1"):
                                                        split = "x1"
                                                    if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[
                                                        1 ].__contains__("x0"):
                                                        split = "x0"
                                                    if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[
                                                        1 ].__contains__("x1"):
                                                        split = "x01"
                                                    if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[
                                                        1 ].__contains__("x0"):
                                                        split = "x10"



                                                    if split == "x0":
                                                        piecewiseFunc.append(
                                                            (nums[ 0 ] - x0) * (x0 - nums[ 1 ]) * float(row[ 1 ]))
                                                    elif split == "x1":
                                                        piecewiseFunc.append(
                                                            (nums[ 0 ] - x1) * (x1 - nums[ 1 ]) * float(row[ 1 ]))
                                                    elif split == "x01":
                                                        piecewiseFunc.append(
                                                            (nums[ 0 ] - x0) * (x1 - nums[ 1 ]) * float(row[ 1 ]))
                                                    elif split == "x10":
                                                        piecewiseFunc.append(
                                                            (nums[ 0 ] - x1) * (x0 - nums[ 1 ]) * float(row[ 1 ]))
                                                    # inputs = tf.where(x < nums[ 0 ] and x > nums[1],
                                                    # float(row[ 1 ]) * (nums[ 0 ] - inputs) * (
                                                    # inputs - nums[ 1 ]), inputs)
                                                except:
                                                    dc = 0
                                            else:
                                                # if float(row[ 1 ]) < 10000:
                                                try:

                                                    if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[
                                                        1 ].__contains__("x1"):
                                                        split = "x1"
                                                    if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[
                                                        1 ].__contains__("x0"):
                                                        split = "x0"
                                                    if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[
                                                        1 ].__contains__("x1"):
                                                        split = "x01"
                                                    if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[
                                                        1 ].__contains__("x0"):
                                                        split = "x10"

                                                    if split == "x0":
                                                        piecewiseFunc.append((x0 - nums[ 0 ]) * float(row[ 1 ]))
                                                    elif split=="x1":
                                                        piecewiseFunc.append((x1 - nums[ 0 ]) * float(row[ 1 ]))
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


        self.flagGen=False

        estimator = baseline_model()
        XSplineVector=[]
        velocities = []
        vectorWeights=[]

        for i in range(0,len(X)):
            vector = extractFunctionsFromSplines(X[i][0],X[i][1])
            #XSplineVector.append(np.append(X[i], vector))
            XSplineVector.append(vector)


        XSplineVector = np.array(XSplineVector)
        weights0 =  np.mean(XSplineVector, axis=0)
        #weights1 = self.intercepts
        #weights1 = estimator.layers[0].get_weights()[0][1]
        weights = np.array(np.append(weights0.reshape(-1,1),np.asmatrix(weights0).reshape(-1,1),axis=1).reshape(2,-1))

        estimator.layers[0].set_weights([weights, np.array([0] * (genModelKnots-1))])

        estimator.fit(X, Y, epochs=100,verbose=0,validation_split=0.33,)

        self.flagGen = True
        from scipy.special import softmax
        #estimatorD = baseline_modelDeepCl()
        # dataUpdatedX = np.append(partitionsX, np.asmatrix([partitionsY]).T, axis=1)

        #input_img = keras.layers.Input(shape=(2,), name='input')
        #x = input_img
        # internal layers in encoder
        #for i in range(n_stacks - 1):
        #x =estimator.layers[ 2 ](x)

        #estimatorD.fit(X, Y, epochs=100)
        #keras.models.Model(inputs= keras.layers.Dense(input(estimator.layers[2].input)), outputs=estimator.layers[-1].output)
        #model2 = keras.models.Model(inputs=input_img, outputs=x)
        #model2 = keras.models.Model(inputs=estimator.layers[2].input, outputs=estimator.layers[-1].output)
        #model2 = estimator.layers[-2]
        #model2 =keras.models.Model(inputs=estimator.input, outputs=estimator.layers[-2].output)
        #model2.compile(optimizer=keras.optimizers.Adam(), loss= keras.losses.KLD)
        #model2.fit(X,Y,epochs=10)
        #labels =np.unique(np.argmax(estimator.predict(X),axis=1))
        #print(labels)
        NNmodels=[]
        scores=[]
        if len(partition_labels)>1:
            for idx, pCurLbl in enumerate(partition_labels):

                    self.modelId = idx
                    self.countTimes += 1

                    XSplineClusterVector=[]
                    for i in range(0, len(partitionsX[idx])):
                        vector = extractFunctionsFromSplines(partitionsX[idx][ i ][ 0 ], partitionsX[idx][ i ][ 1 ])
                        XSplineClusterVector.append(vector)

                    # X =  np.append(X, np.asmatrix([dataY]).T, axis=1)
                    XSplineClusterVector = np.array(XSplineClusterVector)

                    #estimator = baseline_model()
                    numOfNeurons = [ x for x in ClModels[ 'data' ] if x[ 'id' ] == idx ][ 0 ][ 'funcs' ]

                    estimatorCl = keras.models.Sequential()

                    #estimatorCl.add(keras.layers.Dense(numOfNeurons -1 ,input_shape=(2+numOfNeurons-1,)))
                    estimatorCl.add(keras.layers.Dense( numOfNeurons - 1, input_shape=(2 ,)))
                    estimatorCl.add(keras.layers.Dense( numOfNeurons - 1, input_shape=(2,)))
                    #estimatorCl.add(keras.layers.Dense(2))
                    estimatorCl.add(keras.layers.Dense(1, ))
                    estimatorCl.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(), )                #try:

                    weights0 = np.mean(XSplineClusterVector, axis=0)
                    # weights1 = self.intercepts
                    #weights1 = estimatorCl.layers[ 0 ].get_weights()[ 0 ][ 1 ]
                    weights = np.array(
                        np.append(weights0.reshape(-1, 1), np.asmatrix(weights0).reshape(-1, 1), axis=1).reshape(2, -1))

                    #estimatorCl.layers[ 0 ].set_weights([ weights, np.array([0]*(numOfNeurons-1))])
                        #modelId=idx

                    #estimatorCl.fit(partitionsX[idx], np.array(partitionsY[idx]),epochs=100)  # validation_split=0.33
                    #input_img = keras.layers.Input(shape=(2,), name='input')
                    #x = input_img

                    #x = estimatorCl.layers[2](x)

                    #modelCl = keras.models.Model(inputs=input_img, outputs=x)
                    # model2 = keras.models.Model(inputs=estimator.layers[2].input, outputs=estimator.layers[-1].output)

                    #estimatorCl.compile(optimizer=keras.optimizers.Adam(), loss='mse')
                    estimatorCl.fit(partitionsX[idx], np.array(partitionsY[idx]), epochs=100)

                    Clscore = estimatorCl.evaluate(np.array(partitionsX[idx]), np.array(partitionsY[idx]), verbose=0)
                    scores.append(Clscore)
                    NNmodels.append(estimatorCl)

                #self._partitionsPerModel[ estimator ] = partitionsX[idx]
        # Update private models
        #models=[]


        NNmodels.append(estimator)
        self._models = NNmodels

        # Return list of models
        return estimator, None ,scores, numpy.empty,vectorWeights #, estimator , DeepCLpartitionsX


    def getFitnessOfModelForPoint(self, model, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(self._partitionsPerModel[ model ]) - point))

class TensorFlowW(BasePartitionModeler):



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

    def createModelsFor(self, partitionsX, partitionsY, partition_labels,tri,X,Y):

        models = [ ]
        #partitionsX=np.array(partitionsX[0])
        #partitionsY = np.array(partitionsY[0])

        self.ClustersNum=len(partitionsX)
        self.modelId = -1
        #partition_labels = len(partitionsX)
        # Init model to partition map
        self._partitionsPerModel = {}


        def baseline_model():
            #create model
            model = keras.models.Sequential()



            model.add(keras.layers.LSTM(2+genModelKnots-1, input_shape=(2+genModelKnots-1,1)))

            model.add(keras.layers.Dense(genModelKnots - 1,))
            #model.add(keras.layers.Dense(genModelKnots - 2, ))
            #model.add(keras.layers.Dense(genModelKnots - 3, ))
                                         #
            #model.add(keras.layers.Dense(5, ))
            model.add(keras.layers.Dense(2, ))

            model.add(keras.layers.Dense(1,))


            #model.add(keras.layers.Activation('linear'))  # activation=custom_activation
            # Compile model
            model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(),)#experimental_run_tf_function=False )
            #print(model.summary())
            return model



        seed = 7
        numpy.random.seed(seed)


        self.genF=None

        sModel=[]
        sr = sp.Earth(max_degree=1)
        sr.fit(X,Y)
        sModel.append(sr)
        import csv
        csvModels = [ ]
        genModelKnots=[]

        self.modelId = 'Gen'
        self.countTimes=0
        self.models = {"data": [ ]}
        self.intercepts = [ ]
        self.interceptsGen = 0
        self.SelectedFuncs = 0
        for models in sModel:
            modelSummary = str(models.summary()).split("\n")[ 4: ]

            with open('./trainedModels/model_Gen_.csv', mode='w') as data:
                csvModels.append('./trainedModels/model_Gen_.csv')
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [ 'Basis', 'Coeff' ])
                for row in modelSummary:
                    row = np.delete(np.array(row.split(" ")), [ i for i, x in enumerate(row.split(" ")) if x == "" ])
                    try:
                        basis = row[ 0 ]
                        pruned = row[ 1 ]
                        coeff = row[ 2 ]
                        #if basis=='x0' :continue
                        if pruned == "No":
                            data_writer.writerow([ basis, coeff ])
                            genModelKnots.append(basis)
                    except:
                        x = 0

            genModelKnots = len(genModelKnots)
            #modelCount += 1
            #models.append(autoencoder)


        def customLoss(y_true,y_pred):

            return  tf.keras.losses.sparse_categorical_crossentropy(y_true,y_pred) + tf.keras.losses.kullback_leibler_divergence(y_true,y_pred)##+ tf.keras.losses.categorical_crossentropy(y_true,y_pred)
            #tf.keras.losses.mean_squared_error(y_true,y_pred) *
        ##train general neural


        srModels = []
        for idx, pCurLbl in enumerate(partition_labels):

            #maxTerms = if len(DeepCLpartitionsX) > 5000
            srM = sp.Earth(max_degree=1)
            srM.fit(np.array(partitionsX[idx]), np.array(partitionsY[idx]))
            srModels.append(srM)
        modelCount =0
        import csv
        #csvModels = []
        ClModels={"data":[]}

        for models in srModels:
            modelSummary = str(models.summary()).split("\n")[4:]
            basisM = [ ]
            with open('./trainedModels/model_'+str(modelCount)+'_.csv', mode='w') as data:
                csvModels.append('./trainedModels/model_'+str(modelCount)+'_.csv')
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(['Basis', 'Coeff'])
                for row in modelSummary:
                    row=np.delete(np.array(row.split(" ")), [i for i, x in enumerate(row.split(" ")) if x == ""])
                    try:
                        basis = row[0]
                        pruned = row[1]
                        coeff = row[2]
                        #if basis == 'x0' : continue
                        if pruned == "No":
                            data_writer.writerow([basis, coeff])
                            basisM.append(basis)
                    except:
                        x=0
                model = {}
                model[ "id" ] = modelCount
                model[ "funcs" ] = len(basisM)
                ClModels[ "data" ].append(model)
            modelCount+=1

        self.count=0
        def extractFunctionsFromSplines(x0, x1):
            piecewiseFunc = [ ]
            self.count = self.count + 1
            for csvM in csvModels:
                if csvM!='./trainedModels/model_'+str(self.modelId)+'_.csv':
                    continue
                #id = csvM.split("_")[ 1 ]
                #piecewiseFunc = [ ]

                with open(csvM) as csv_file:
                    data = csv.reader(csv_file, delimiter=',')
                    for row in data:
                        # for d in row:
                        if [ w for w in row if w == "Basis" ].__len__() > 0:
                            continue
                        if [ w for w in row if w == "(Intercept)" ].__len__() > 0:

                            self.interceptsGen = float(row[ 1 ])
                            continue

                        if row.__len__() == 0:
                            continue
                        d = row[ 0 ]
                        if self.count==1:
                            self.intercepts.append(float(row[1]))

                        if d.split("*").__len__() == 1:
                            split = ""
                            try:
                                split = d.split('-')[ 0 ][ 2:4 ]
                                if split != "x0" and split!="x1":
                                    split = d.split('-')[ 1 ]
                                    num = float(d.split('-')[ 0 ].split('h(')[ 1 ])
                                    # if id == id:
                                    # if float(row[ 1 ]) < 10000:
                                    try:
                                        # piecewiseFunc.append(
                                        # tf.math.multiply(tf.cast(tf.math.less(x, num), tf.float32),
                                        # (num - inputs)))
                                        if split.__contains__("x0"):
                                            piecewiseFunc.append((num - x0))# * float(row[ 1 ]))
                                        if split.__contains__("x1"):
                                            piecewiseFunc.append((num - x1)) #* float(row[ 1 ]))
                                    # if id ==  self.modelId:
                                    # inputs = tf.where(x >= num, float(row[ 1 ]) * (inputs - num), inputs)
                                    except:
                                        dc = 0
                                else:
                                    ##x0 or x1
                                    split = d.split('-')[ 0 ]
                                    num = float(d.split('-')[ 1 ].split(')')[ 0 ])
                                    # if id == id:
                                    # if float(row[ 1 ]) < 10000:
                                    try:
                                        if split.__contains__("x0"):
                                            piecewiseFunc.append((x0 - num))# * float(row[ 1 ]))
                                        if split.__contains__("x1"):
                                            piecewiseFunc.append((x1 - num))# * float(row[ 1 ]))

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
                            nums = [ ]
                            flgFirstx = False
                            flgs = [ ]
                            for r in funcs:
                                try:
                                    if r.split('-')[ 0 ][ 2 ] != "x":
                                        flgFirstx = True
                                        nums.append(float(r.split('-')[ 0 ].split('h(')[ 1 ]))

                                    else:
                                        nums.append(float(r.split('-')[ 1 ].split(')')[ 0 ]))

                                    flgs.append(flgFirstx)
                                except:
                                    flgFirstx = False
                                    flgs = [ ]
                                    split=d.split('-')[ 0 ][ 2 ]
                                    try:
                                        if d.split('-')[ 0 ][ 2 ] == "x":
                                            # if id == id:
                                            # if float(row[ 1 ]) < 10000:
                                            try:

                                                if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[ 1 ].__contains__("x1"):
                                                    split="x1"
                                                if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[1 ].__contains__("x0"):
                                                    split="x0"
                                                if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[1 ].__contains__("x1"):
                                                    split="x01"
                                                if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[1 ].__contains__("x0"):
                                                    split = "x10"

                                                if split == "x0":
                                                    piecewiseFunc.append(x0 * (x0 - nums[ 0 ]) * float(row[ 1 ]))
                                                elif split=="x1":
                                                    piecewiseFunc.append(x1 * (x1 - nums[ 0 ]) * float(row[ 1 ]))
                                                elif split == "x01":
                                                    piecewiseFunc.append(x0 * (x1 - nums[ 0 ]) * float(row[ 1 ]))
                                                elif split == "x10":
                                                    piecewiseFunc.append(x1 * (x0 - nums[ 0 ]) * float(row[ 1 ]))
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

                                                if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[
                                                    1 ].__contains__("x1"):
                                                    split = "x1"
                                                if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[
                                                    1 ].__contains__("x0"):
                                                    split = "x0"
                                                if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[
                                                    1 ].__contains__("x1"):
                                                    split = "x01"
                                                if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[
                                                    1 ].__contains__("x0"):
                                                    split = "x10"

                                                if split == "x0":
                                                    piecewiseFunc.append(x0 * (nums[ 0 ] - x0) * float(row[ 1 ]))
                                                elif split=="x1":
                                                    piecewiseFunc.append(x1 * (nums[ 0 ] - x1) * float(row[ 1 ]))
                                                elif split=="x01":
                                                    piecewiseFunc.append(x0 * (nums[ 0 ] - x1) * float(row[ 1 ]))
                                                elif split=="x10":
                                                    piecewiseFunc.append(x1 * (nums[ 0 ] - x0) * float(row[ 1 ]))

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

                                        if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[ 1 ].__contains__(
                                                "x1"):
                                            split = "x1"
                                        if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[ 1 ].__contains__(
                                                "x0"):
                                            split = "x0"
                                        if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[ 1 ].__contains__(
                                                "x1"):
                                            split = "x01"
                                        if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[ 1 ].__contains__(
                                                "x0"):
                                            split = "x10"

                                        if split == "x0":
                                            piecewiseFunc.append((nums[ 0 ] - x0) * (nums[ 1 ] - x0) * float(row[ 1 ]))
                                        elif split=="x1":
                                            piecewiseFunc.append((nums[ 0 ] - x1) * (nums[ 1 ] - x1) * float(row[ 1 ]))
                                        elif split=="x01":
                                            piecewiseFunc.append((nums[ 0 ] - x0) * (nums[ 1 ] - x1) * float(row[ 1 ]))
                                        elif split=="x10":
                                            piecewiseFunc.append((nums[ 0 ] - x1) * (nums[ 1 ] - x0) * float(row[ 1 ]))

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
                                        if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[ 1 ].__contains__(
                                                "x1"):
                                            split = "x1"
                                        if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[ 1 ].__contains__(
                                                "x0"):
                                            split = "x0"
                                        if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[ 1 ].__contains__(
                                                "x1"):
                                            split = "x01"
                                        if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[ 1 ].__contains__(
                                                "x0"):
                                            split = "x10"

                                        if split=="x0":
                                            piecewiseFunc.append((x0 - nums[ 0 ]) * (x0 - nums[ 1 ]) * float(row[ 1 ]))
                                        elif split=="x1":
                                            piecewiseFunc.append((x1 - nums[ 0 ]) * (x1 - nums[ 1 ]) * float(row[ 1 ]))
                                        elif split=="x01":
                                            piecewiseFunc.append((x0 - nums[ 0 ]) * (x1 - nums[ 1 ]) * float(row[ 1 ]))
                                        elif split=="x10":
                                            piecewiseFunc.append((x1 - nums[ 0 ]) * (x0 - nums[ 1 ]) * float(row[ 1 ]))
                                        # inputs = tf.where(x > nums[ 0 ] and x > nums[ 1 ],
                                        # float(row[ 1 ]) * (inputs - nums[ 0 ]) * (
                                        # inputs - nums[ 1 ]), inputs)
                                    except:
                                        dc = 0
                                else:
                                    try:
                                        if flgs[ 0 ] == False:
                                            if nums.__len__() > 1:
                                                # if float(row[ 1 ]) < 10000:
                                                try:
                                                    if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[
                                                        2 ].__contains__("x1"):
                                                        split = "x1"
                                                    if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[
                                                        2 ].__contains__("x0"):
                                                        split = "x0"
                                                    if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[
                                                        2 ].__contains__("x1"):
                                                        split = "x01"
                                                    if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[
                                                        2 ].__contains__("x0"):
                                                        split = "x10"

                                                    if split == "x0":
                                                        piecewiseFunc.append(
                                                            (x0 - nums[ 0 ]) * (nums[ 1 ] - x0) * float(row[ 1 ]))
                                                    elif split == "x1":
                                                        piecewiseFunc.append(
                                                            (x1 - nums[ 0 ]) * (nums[ 1 ] - x1) * float(row[ 1 ]))
                                                    elif split == "x01":
                                                        piecewiseFunc.append(
                                                            (x0 - nums[ 0 ]) * (nums[ 1 ] - x1) * float(row[ 1 ]))
                                                    elif split == "x10":
                                                        piecewiseFunc.append(
                                                            (x1 - nums[ 0 ]) * (nums[ 1 ] - x0) * float(row[ 1 ]))




                                                # inputs = tf.where(x > nums[ 0 ] and x < nums[ 1 ],
                                                # float(row[ 1 ]) * (inputs - nums[ 0 ]) * (
                                                # nums[ 1 ] - inputs), inputs)
                                                except:
                                                    dc = 0
                                            else:
                                                # if float(row[ 1 ]) < 10000:
                                                try:
                                                    if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[
                                                        1 ].__contains__("x1"):
                                                        split = "x1"
                                                    if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[
                                                        1 ].__contains__("x0"):
                                                        split = "x0"
                                                    if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[
                                                        1 ].__contains__("x1"):
                                                        split = "x01"
                                                    if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[
                                                        1 ].__contains__("x0"):
                                                        split = "x10"

                                                    piecewiseFunc.append((x0 - nums[ 0 ]) * float(row[ 1 ]))



                                                    # inputs = tf.where(x > nums[0],
                                                    # float(row[ 1 ]) * (inputs - nums[ 0 ]), inputs)
                                                except:
                                                    dc = 0
                                        else:
                                            if nums.__len__() > 1:
                                                # if float(row[ 1 ]) < 10000:
                                                try:
                                                    if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[
                                                        1 ].__contains__("x1"):
                                                        split = "x1"
                                                    if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[
                                                        1 ].__contains__("x0"):
                                                        split = "x0"
                                                    if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[
                                                        1 ].__contains__("x1"):
                                                        split = "x01"
                                                    if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[
                                                        1 ].__contains__("x0"):
                                                        split = "x10"



                                                    if split == "x0":
                                                        piecewiseFunc.append(
                                                            (nums[ 0 ] - x0) * (x0 - nums[ 1 ]) * float(row[ 1 ]))
                                                    elif split == "x1":
                                                        piecewiseFunc.append(
                                                            (nums[ 0 ] - x1) * (x1 - nums[ 1 ]) * float(row[ 1 ]))
                                                    elif split == "x01":
                                                        piecewiseFunc.append(
                                                            (nums[ 0 ] - x0) * (x1 - nums[ 1 ]) * float(row[ 1 ]))
                                                    elif split == "x10":
                                                        piecewiseFunc.append(
                                                            (nums[ 0 ] - x1) * (x0 - nums[ 1 ]) * float(row[ 1 ]))
                                                    # inputs = tf.where(x < nums[ 0 ] and x > nums[1],
                                                    # float(row[ 1 ]) * (nums[ 0 ] - inputs) * (
                                                    # inputs - nums[ 1 ]), inputs)
                                                except:
                                                    dc = 0
                                            else:
                                                # if float(row[ 1 ]) < 10000:
                                                try:

                                                    if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[
                                                        1 ].__contains__("x1"):
                                                        split = "x1"
                                                    if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[
                                                        1 ].__contains__("x0"):
                                                        split = "x0"
                                                    if d.split("-")[ 0 ].__contains__("x0") and d.split("-")[
                                                        1 ].__contains__("x1"):
                                                        split = "x01"
                                                    if d.split("-")[ 0 ].__contains__("x1") and d.split("-")[
                                                        1 ].__contains__("x0"):
                                                        split = "x10"

                                                    if split == "x0":
                                                        piecewiseFunc.append((x0 - nums[ 0 ]) * float(row[ 1 ]))
                                                    elif split=="x1":
                                                        piecewiseFunc.append((x1 - nums[ 0 ]) * float(row[ 1 ]))
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

        #extractFunctionsFromSplines(1, 1)

        self.flagGen=False

        estimator = baseline_model()
        XSplineVector=[]
        velocities = []
        vectorWeights=[]

        for i in range(0,len(X)):
            vector = extractFunctionsFromSplines(X[i][0],X[i][1])
            XSplineVector.append(np.append(X[i],vector))


        XSplineVector = np.array(XSplineVector)

        #try:
        XSplineVector = np.reshape(XSplineVector, (XSplineVector.shape[0], XSplineVector.shape[1], 1))
        estimator.fit(XSplineVector, Y, epochs=100, validation_split=0.33,verbose=0)
        #score = estimator.evaluate(np.array(XSplineVector),Y, verbose=0)
        #except:


        #estimator.fit(X, Y, epochs=100, validation_split=0.33)


        self.flagGen = True

         # validation_data=(X_test,y_test)

        def insert_intermediate_layer_in_keras(model,layer_id, new_layer):


            layers = [ l for l in model.layers ]

            x = layers[ 0 ].output
            for i in range(1, len(layers)):
                if i == layer_id:
                    x = new_layer(x)
                if i == len(layers)-1:
                    x = keras.layers.Dense(1)(x)
                else:
                    x = layers[ i ](x)
            #x = new_layer(x)
            new_model = keras.Model(inputs=model.input, outputs=x)
            #new_model.add(new_layer)
            new_model.compile(loss='mse', optimizer=keras.optimizers.Adam())
            #.add(x)
            return new_model

        def replace_intermediate_layer_in_keras(model, layer_id,layer_id1 ,new_layer,new_layer1):


            layers = [ l for l in model.layers ]

            x = layers[ 0 ].output
            for i in range(1, len(layers)):
                if i == layer_id:
                    x = new_layer(x)

                elif  i == layer_id1:
                    x = new_layer1(x)

                else:########
                    #if i == len(layers) - 1:
                        #x = keras.layers.Dense(1)(x)
                    #else:
                    x = layers[ i ](x)

            new_model = keras.Model(inputs=model.input, outputs=x)
            new_model.compile(loss='mse', optimizer=keras.optimizers.Adam(),)
            return new_model

        #for train, test in kfold.split(partitionsX[idx],partitionsY[idx]):
        #if len(partition_labels)>0:
        #models.append(estimator)
        #models={}
        #models[ 300 ] = estimator

        NNmodels=[]
        scores=[]
        if len(partition_labels) > 1:
            for idx, pCurLbl in enumerate(partition_labels):
                    #partitionsX[ idx ]=partitionsX[idx].reshape(-1,2)
                    self.modelId = idx
                    self.countTimes += 1

                    XSplineClusterVector=[]
                    for i in range(0, len(partitionsX[idx])):
                        vector = extractFunctionsFromSplines(partitionsX[idx][ i ][ 0 ], partitionsX[idx][ i ][ 1 ])
                        XSplineClusterVector.append(np.append(partitionsX[idx][i], vector))

                    # X =  np.append(X, np.asmatrix([dataY]).T, axis=1)
                    XSplineClusterVector = np.array(XSplineClusterVector)

                    #estimator = baseline_model()
                    numOfNeurons = [ x for x in ClModels[ 'data' ] if x[ 'id' ] == idx ][ 0 ][ 'funcs' ]

                    estimatorCl = keras.models.Sequential()

                    estimatorCl.add(keras.layers.LSTM(2 + numOfNeurons -1 ,input_shape=(2+numOfNeurons-1,1)))
                    estimatorCl.add(keras.layers.Dense(numOfNeurons - 1, ))
                    #estimatorCl.add(keras.layers.Dense(2))
                    estimatorCl.add(keras.layers.Dense(1, ))
                    estimatorCl.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(), )                #try:
                    XSplineClusterVector = np.reshape(XSplineClusterVector, (XSplineClusterVector.shape[0], XSplineClusterVector.shape[1], 1))
                    estimatorCl.fit(np.array(XSplineClusterVector),np.array(partitionsY[idx]),epochs=100)#validation_split=0.33

                    #Clscore = estimatorCl.evaluate(np.array(XSplineClusterVector), np.array(partitionsY[idx]), verbose=0)
                    #scores.append(Clscore)
                    #NNmodels.append([estimatorCl,'CL'])
                    NNmodels.append(estimatorCl)
                    #estimatorCl.save('./DeployedModels/estimatorCl_'+idx+'.h5')
                    #except:
                        #scores.append(score)
                        #NNmodels.append([estimator,'GEN'])

                    #np.array(XSplineClusterVector)

                    #print("%s: %.2f%%" % ("acc: ", score))

                    #except Exception as e:
                        #print(str(e))
                        #return
                    #models[pCurLbl]=estimator
                    #self._partitionsPerModel[ estimator ] = partitionsX[idx]
            # Update private models
            #models=[]


        NNmodels.append(estimator)
        #estimator.save('./DeployedModels/estimatorCl_Gen.h5')
        self._models = NNmodels

        # Return list of models
        return estimator, None ,scores, numpy.empty,vectorWeights #, estimator , DeepCLpartitionsX

    def createModelsForConv(self,partitionsX, partitionsY, partition_labels):
        ##Conv1D NN
        # Init result model list
        models = [ ]
        # Init model to partition map
        self._partitionsPerModel = {}

        TIME_PERIODS=len(partitionsX[0])
        STEP_DISTANCE = 40
        x_train, y_train = self.createSegmentsofTrData(partitionsX[ 0 ], partitionsY[ 0 ], TIME_PERIODS, STEP_DISTANCE)

        num_time_periods, num_sensors =partitionsX[0].shape[0], partitionsX[0].shape[1]
        num_classes = 1
        input_shape = (num_sensors)
        #x_train = x_train.reshape(x_train.shape[ 0 ], input_shape)


        model_m = keras.models.Sequential()
        model_m.add(keras.layers.Reshape((TIME_PERIODS,num_sensors, ), input_shape=(num_sensors,)))
        model_m.add(keras.layers.Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
        model_m.add(keras.layers.Conv1D(100, 10, activation='relu'))
        model_m.add(keras.layers.MaxPooling1D(2))
        model_m.add(keras.layers.Conv1D(160, 10, activation='relu'))
        model_m.add(keras.layers.Conv1D(160, 10, activation='relu'))
        model_m.add(keras.layers.GlobalAveragePooling1D())
        model_m.add(keras.layers.Dropout(0.3))
        model_m.add(keras.layers.Dense(num_classes, activation='softmax'))
        print(model_m.summary())

        callbacks_list = [
            keras.callbacks.ModelCheckpoint(
                filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
                monitor='val_loss', save_best_only=True),
            keras.callbacks.EarlyStopping(monitor='acc', patience=1)
        ]

        model_m.compile(loss='mean_squared_error',
                        optimizer='adam', metrics=[ 'accuracy' ])

        BATCH_SIZE = 400
        EPOCHS = 50
        for idx, pCurLbl in enumerate(partition_labels):
           #for i in range(0,x_train.shape[1]):
           history = model_m.fit(np.nan_to_num(partitionsX[idx]),np.nan_to_num(partitionsY[idx]),
                                      epochs=EPOCHS,
                                     )

        models.append(model_m)
        self._partitionsPerModel[ model_m ] = partitionsX[ idx ]

        # Update private models

        self._models = models

        # Return list of models
        return models, numpy.empty, numpy.empty


    def createModelsForS(self,partitionsX, partitionsY, partition_labels):

        # Init result model list
        models = [ ]
        # Init model to partition map
        self._partitionsPerModel = {}
        # For each label/partition
        #curModel = keras.models.Sequential()

        # This will add a fully connected neural network layer with 32#neurons, each#taking#13#inputs, and with activation function ReLU
        #curModel.add(keras.layers.Dense(1, input_dim=3, activation='relu'))

        #curModel.compile(loss='mean_squared_error',
        #                 optimizer='sgd',
        #                 metrics=[ 'mae' ])
        # Fit to data

        #input_A = keras.layers.Input(shape=partitionsX[0].shape)
        #input_B = keras.layers.Input(shape=partitionsX[1].shape)
        #input_C = keras.layers.Input(shape=partitionsX[2].shape)
        #A = keras.layers.Dense(1)(input_A)
        #A = keras.layers.Dense(1)(A)
        #B = keras.layers.concatenate([ input_A, input_B, input_C ], mode='concat')
        #B = keras.layers.Dense(1)(B)
        #C = keras.layers.concatenate([ A, B ], mode='concat')
        #C = keras.layers.Dense(1)(C)
        #B = keras.layers.concatenate([ A, B ], mode='concat')
        #B = keras.layers.Dense(1)(B)
        #A = keras.layers.Dense(1)(A)

        #curModel.fit(np.asarray([ [[0,2], [0,2] ],[[0,2], [0,2] ], [[0,2], [0,2] ] ]),
                     #np.asarray([ [[0,2], [0,2] ],[[0,2], [0,2] ], [[0,2], [0,2] ] ]), epochs=10)
        #curModel.fit(partitionsX, partitionsY, epochs=10)

        for idx, pCurLbl in enumerate(partition_labels):
            # Create a linear model
            curModel = keras.models.Sequential()
            partitionsX[ idx ] = np.reshape(partitionsX[ idx ], (partitionsX[ idx ].shape[ 0 ], 1, partitionsX[ idx ].shape[ 1 ]))

            # This will add a fully connected neural network layer with 32#neurons, each#taking#13#inputs, and with activation function ReLU
            curModel.add(keras.layers.Conv1D(2,input_shape=partitionsX[idx].shape[1:],activation='relu'))
            #curModel.add(keras.layers.LSTM(3, input_shape=partitionsX[ idx ].shape[ 1: ], activation='relu'))
            #curModel.add(keras.layers.Flatten())
            curModel.add(keras.layers.Dense(1))

            ##optimizers
            adam=keras.optimizers.Adam(lr=0.001)
            rmsprop=keras.optimizers.RMSprop(lr=0.01)
            adagrad=keras.optimizers.Adagrad(lr=0.01)
            sgd=keras.optimizers.SGD(lr=0.001)
            ######

            curModel.compile(loss='mean_squared_error',
              optimizer=adam,metrics=['mae'])
            # Fit to data
            curModel.fit(np.nan_to_num(partitionsX[idx]),np.nan_to_num(partitionsY[idx]), epochs=100)
            # Add to returned list
            models.append(curModel)
            self._partitionsPerModel[curModel] = partitionsX[idx]

        # Update private models
        self._models = models

        # Return list of models
        return models , numpy.empty ,numpy.empty


    def createModelsForX1(self, partitionsX, partitionsY, partition_labels):

        SAVE_PATH = './save'
        EPOCHS = 1
        LEARNING_RATE = 0.001
        MODEL_NAME = 'test'

        model = sp.Earth(use_fast=True)
        model.fit(partitionsX[0],partitionsY[0])
        W_1= model.coef_

        self._partitionsPerModel = {}


        ######################################################
        # Data specific constants
        n_input = 784  # data input
        n_classes = 1  #  total classes
        # Hyperparameters
        max_epochs = 10
        learning_rate = 0.5
        batch_size = 10
        seed = 0
        n_hidden = 3  # Number of neurons in the hidden layer


        # Gradient Descent optimization  for updating weights and biases
        # Execution Graph
        c_t=[]
        c_test=[]
        models=[]

        for idx, pCurLbl in enumerate(partition_labels):
            n_input = partitionsX[ idx ].shape[ 1 ]
            xs = tf.placeholder(tf.float32, [ None, n_input ])
            ys = tf.placeholder("float")

            output = self.initNN(xs, 2)
            cost = tf.reduce_mean(tf.square(output - ys))
            # our mean square error cost function
            train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
            ##

            init_op = tf.global_variables_initializer()
            #name_to_var_map = {var.op.name: var for var in tf.global_variables()}

            #name_to_var_map[ "loss/total_loss/avg" ] = name_to_var_map[
            #    "conv_classifier/loss/total_loss/avg" ]
            #del name_to_var_map[ "conv_classifier/loss/total_loss/avg" ]

            saver = tf.train.Saver()
            if idx==1:
                with tf.Session() as sess:
                    # Initiate session and initialize all vaiables
                        #################################################################
                        # output = self.initNN(xs,W_1,2)
                        sess.run(init_op)

                        for i in range(EPOCHS):
                      #for j in range(len(partitionsX[idx].shape[ 0 ]):
                          for j in range(0,len(partitionsX[idx])):

                             all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
                             tf.variables_initializer(all_variables)
                             sess.run([ cost, train ], feed_dict={xs: [partitionsX[idx][ j, : ]], ys: [partitionsY[idx][ j ]]})
                        # Run cost and train with each sample
                        #c_t.append(sess.run(cost, feed_dict={xs: partitionsX[idx], ys: partitionsY[idx]}))
                        #c_test.append(sess.run(cost, feed_dict={xs: X_test, ys: y_test}))
                        #print('Epoch :', i, 'Cost :', c_t[ i ])
                        models.append(sess)
                        #self._partitionsPerModel[sess]  = partitionsX[idx]

                        self._models = models
                        saver.save(sess, SAVE_PATH + '/' + MODEL_NAME+'_'+str(idx) + '.ckpt')
                sess.close()
        return  models , xs , output

    def getFitnessOfModelForPoint(self, model, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(self._partitionsPerModel[ model ]) - point))


class RandomForestModeler(BasePartitionModeler):
    def createModelsFor(self, partitionsX, partitionsY, partition_labels,tri,X,Y):
        models = [ ]
        # Init model to partition map
        self._partitionsPerModel = {}
        # For each label/partition
        for idx, pCurLbl in enumerate(partition_labels):
            # Create a linear model
            curModel = skl.RandomForestRegressor()
            ##HP tuning
            #random_search3 = RandomizedSearchCV(curModel, param_distributions=parameters.param_mars_dist, n_iter=4)
            # try:
            #    random_search3.fit(partitionsX[ idx ], partitionsY[ idx ])
            #    curModel.set_params(**self.report(random_search3.cv_results_))
            # except:
            #    print "Error on HP tuning"
            # Fit to data
            try:
                curModel.fit(partitionsX[ idx ], partitionsY[ idx ])
            except:
                print (idx)
            # Add to returned list
            models.append(curModel)
            self._partitionsPerModel[ curModel ] = partitionsX[ idx ]

        # Update private models
        self._models = models

        # Return list of models
        return models, numpy.empty, numpy.empty , None , None

    def getFitnessOfModelForPoint(self, model, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(self._partitionsPerModel[ model ]) - point))

class SplineRegressionModeler(BasePartitionModeler):
    def createModelsFor(self, partitionsX, partitionsY, partition_labels,tri,X,Y):
        # Init result model list
        models = [ ]
        #import scipy as sp1
        dataX = X
        dataY = Y
        #spline = sp1.interpolate.Rbf(x[0:,0], x[0:,1], y,
                                     #function='thin_plate', smooth=5, episilon=5)
        #genericModel = sp.Earth(use_fast=True)
        #genericModel.fit(x,y)


        # Init model to partition map
        #self._partitionsPerModel = {}
        self._partitionsPerModel = []
        # For each label/partition
        for idx, pCurLbl in enumerate(partition_labels):
            # Create a linear model
            #from sklearn.decomposition import PCA
            #pcaMapping = PCA(1)
            #pcaMapping.fit(partitionsX[ idx ])
            #Vmapped = pcaMapping.transform(partitionsX[ idx ])
            #dataX = Vmapped
            #if(st.pearsonr(np.asarray(dataX).reshape(1,-1)[0], partitionsY[ idx ])[0]>0.5):
                #n=1
            #else:
                #n=2
            curModel = sp.Earth()
            ##HP tuning
            #random_search3 = RandomizedSearchCV(curModel, param_distributions=parameters.param_mars_dist, n_iter=4)
            #try:
            #    random_search3.fit(partitionsX[ idx ], partitionsY[ idx ])
            #    curModel.set_params(**self.report(random_search3.cv_results_))
            #except:
            #    print "Error on HP tuning"
            # Fit to data
            #try:
            #simplicesIndexes = tri.find_simplex(partitionsX[ idx ])
            #triVerticesIndexesFiltered = [x for x in tri.vertices[simplicesIndexes] if int(x) < int(len(dataX))]
            #dataXnew=np.concatenate(dataX[tri.vertices[simplicesIndexes]])
            #dataYnew =np.concatenate(dataY[tri.vertices[simplicesIndexes]])
            curModel.fit(partitionsX[idx],partitionsY[idx])
            #plt.plot(partitionsX[ idx ], partitionsY[ idx ], 'o', markersize=3)
            #plt.plot(partitionsX[ idx ], curModel.predict(partitionsX[ idx ]), 'r--' )

            #plt.show()
            #x=1
            #except:
                #print str(Exception)
            # Add to returned list
            models.append(curModel)
            #self._partitionsPerModel[ curModel ] = partitionsX[ idx ]
            self._partitionsPerModel.append(partitionsX[idx])

        # Update private models
        self._models = models

        # Return list of models
        return models,numpy.empty ,numpy.empty,None,None

    def getBestModelForPoint(self, point):
        # mBest = None
        mBest = None
        dBestFit = 0
        # For each model
        for i in range(0,len(self._models)):
            # If it is a better for the point
            dCurFit = self.getFitnessOfModelForPoint(i, point)
            if dCurFit > dBestFit:
                # Update the selected best model and corresponding fit
                dBestFit = dCurFit
                mBest = self._models[i]

        if mBest == None:
            return self._models[0]
        else:
            return mBest

    def getFitnessOfModelForPoint(self, model, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(self._partitionsPerModel[ model ]) - point))

    def plotRegressionLine(self,partitionsX, partitionsY, partitionLabels,genericModel,modelMap):
        #xData = np.asarray(partitionsX[ :, 0 ]).T[ 0 ]
        #yData = np.asarray(partitionsY[ :, 1 ]).T[ 0 ]
        xData = np.concatenate(partitionsX)
        yData = np.concatenate(partitionsY)

        #plt.scatter(xData[:,0], xData[:,1], c=partitionLabels)

        for idx, pCurLbl in enumerate(partitionLabels):
            plt.plot(partitionsX[ idx ][:,0], partitionsY[ idx ], 'o', markersize=3)
            plt.plot(partitionsX[ idx ], modelMap[idx].predict(partitionsX[ idx ]), 'r--')

        plt.plot(xData, genericModel.predict(xData), 'b-')
        plt.show()
        x=1
       