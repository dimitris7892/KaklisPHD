import pyearth as sp
import numpy as np
import numpy.linalg
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pandas as pd
import math
import  matplotlib.pyplot as plt
from scipy.interpolate import BivariateSpline
import tensorflow as tf
from tensorflow import keras
import scipy as scipy
from tensorflow.keras.callbacks import EarlyStopping
from datetime import date, datetime
import pickle
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback


class BasePartitionModeler:

    def createModelsFor(self,partitionsX, partitionsY, partition_labels):
        pass

    def extractFunctionsFromSplines(self ,x0, x1, x2, x3, x4, x5=None, x6=None, vessel=None):
        piecewiseFunc = []
        # csvModels = ['../../../../../Danaos_ML_Project/trainedModels/model_' + str(modelId) + '_.csv']
        # for csvM in csvModels:
        # if csvM != '../../../../../Danaos_ML_Project/trainedModels/model_' + str(modelId) + '_.csv':
        # continue
        # id = csvM.split("_")[ 1 ]
        # piecewiseFunc = [ ]

        # with open(csvM) as csv_file:
        # data = csv.reader(csv_file, delimiter=',')
        csvM = './trainedModels/model_Gen_'+vessel+'.csv'
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


    def __init__(self):

        self.neptuneAPIToken = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1NjI5MDMzZS0wNjllLTRlZGMtYjBkNS0xMmI5YWU3MjQ1MjEifQ=="


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

    def extractFunctionsFromSplines(self, x0, x1, x2, x3, x4=None, x5=None, x6=None, x7=None):
        piecewiseFunc = []
        self.count = self.count + 1
        for csvM in self.csvModels:
            if csvM != './trainedModels/model_' + str(self.modelId) + '_' + self.vessel + '.csv':
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
                                    if split.__contains__("x7"):
                                        piecewiseFunc.append((num - x7))
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
                                    if split.__contains__("x7"):
                                        piecewiseFunc.append((x7 - num))

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
                                                piecewiseFunc.append(x0 * (x0 - nums[0]))  # * float(row[1]))
                                            elif split == "x1":
                                                piecewiseFunc.append(x1 * (x1 - nums[0]))  # * float(row[1]))
                                            elif split == "x01":
                                                piecewiseFunc.append(x0 * (x1 - nums[0]))  # * float(row[1]))
                                            elif split == "x10":
                                                piecewiseFunc.append(x1 * (x0 - nums[0]))  # * float(row[1]))
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
                                                piecewiseFunc.append(x0 * (nums[0] - x0))  # * float(row[1]))
                                            elif split == "x1":
                                                piecewiseFunc.append(x1 * (nums[0] - x1))  # * float(row[1]))
                                            elif split == "x01":
                                                piecewiseFunc.append(x0 * (nums[0] - x1))  # * float(row[1]))
                                            elif split == "x10":
                                                piecewiseFunc.append(x1 * (nums[0] - x0))  # * float(row[1]))

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
                                        piecewiseFunc.append((nums[0] - x0) * (nums[1] - x0))  # * float(row[1]))
                                    elif split == "x1":
                                        piecewiseFunc.append((nums[0] - x1) * (nums[1] - x1))  # * float(row[1]))
                                    elif split == "x01":
                                        piecewiseFunc.append((nums[0] - x0) * (nums[1] - x1))  # * float(row[1]))
                                    elif split == "x10":
                                        piecewiseFunc.append((nums[0] - x1) * (nums[1] - x0))  # * float(row[1]))

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

    def split_sequence(self, sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence) - 1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix][:, 0:sequence.shape[1] - 1], sequence[end_ix - 1][sequence.shape[1] - 1]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def createModelsFor(self, partitionsX, partitionsY, partition_labels, tri, X, Y, expansion, vessel, attWeights):

        import csv
        self.vessel = vessel
        self.ClustersNum = len(partitionsX)
        self.modelId = -1
        # Init model to partition map
        self._partitionsPerModel = {}

        if expansion == True  or expansion==False:
            seed = 7
            numpy.random.seed(seed)

            self.genF = None

            sModel = []
            sr = sp.Earth(max_degree=1)
            sr.fit(X, Y)
            sModel.append(sr)

            self.csvModels = []
            genModelKnots = []

            self.modelId = 'Gen'
            self.countTimes = 0
            self.models = {"data": []}
            self.intercepts = []
            self.interceptsGen = 0
            self.SelectedFuncs = 0
            for models in sModel:
                modelSummary = str(models.summary()).split("\n")[4:]

                with open('./trainedModels/model_Gen_' + vessel + '.csv', mode='w') as data:
                    self.csvModels.append('./trainedModels/model_Gen_' + vessel + '.csv')
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
                        except Exception as e:
                            print(e)

                genModelKnots = len(genModelKnots)


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
                with open('./trainedModels/model_' + str(modelCount) + '_.csv', mode='w') as data:
                    self.csvModels.append('./trainedModels/model_' + str(modelCount) + '_.csv')
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
                        except Exception as e:
                            print(e)
                    model = {}
                    model["id"] = modelCount
                    model["funcs"] = len(basisM)
                    ClModels["data"].append(model)
                modelCount += 1

            self.count = 0

        n_steps = 15
        neurons = 5 + genModelKnots -1 #if expansion == True else 5
        input_shape = 5 + genModelKnots -1 if expansion == True else 5
        print("Neurons length: "+str(neurons))

        def baseline_attention_model():
            # create model
            sequence_input = keras.layers.Input(shape=(n_steps, input_shape), dtype='float32')
            lstm, state_h, state_c = keras.layers.LSTM(units=15,

                                                       return_state=True,

                                                 return_sequences=True)(sequence_input)
            print(lstm)
            print(state_h)
            print(state_c)
            #state_h = keras.layers.Reshape((1,1,10))(state_h)



            context_vector = keras.layers.Attention()([lstm, state_h])

            print(context_vector)

            output = keras.layers.Dense(units=5,activation='softmax' )(context_vector)


            attentionModel = tf.keras.Model(inputs=sequence_input, outputs=output)

            #output = keras.layers.Dense(units=5)(output)
            output = keras.layers.Dense(units=1)(output)

            model = tf.keras.Model(inputs=sequence_input, outputs=output)
            model.compile(loss=keras.losses.mean_squared_error,
                          optimizer=keras.optimizers.Adam() )  # experimental_run_tf_function=False )
            #print(model.summary())

            return model

        print("GenModelKnots: " +str(genModelKnots))

        learnRate = 0.01

        def baseline_model():
            # create model
            model = keras.models.Sequential()

            #model.add(keras.layers.Dense(20, input_shape=(7,)))
            #model.add(keras.layers.Dense(10, input_shape=(7,)))

            #kernel_regularizer = tf.keras.regularizers.l1(0.01),
            #activity_regularizer = tf.keras.regularizers.l2(0.01),
            #bias_regularizer = tf.keras.regularizers.l2(0.01)


            model.add(keras.layers.LSTM(neurons, input_shape=(n_steps, input_shape  ,)))#
            #model.add(keras.layers.Attention())
            #model.add(keras.layers.LSTM(neurons - 5, ))
            #model.add(keras.layers.LSTM( 15, ))


            #model.add(keras.layers.Dense(30,input_shape=( 5 ,)))
            #model.add(keras.layers.Dense(20))
            #model.add(keras.layers.Dense(genModelKnots - 3,activation='relu'))
            #model.add(keras.layers.Dropout(0.2))

            model.add(keras.layers.Dense( neurons - 5, kernel_regularizer = tf.keras.regularizers.l1(0.01),

            activity_regularizer = tf.keras.regularizers.l2(0.01),
            bias_regularizer = tf.keras.regularizers.l2(0.01) ),)

            #kernel_regularizer = tf.keras.regularizers.l1(0.01),
            #activity_regularizer = tf.keras.regularizers.l2(0.01),
            #bias_regularizer = tf.keras.regularizers.l2(0.01),


            model.add(keras.layers.Dense(5,))

            model.add(keras.layers.Dense(2, ))


            #.add(keras.layers.Dense(5,kernel_regularizer = tf.keras.regularizers.l1(0.01),
            #activity_regularizer = tf.keras.regularizers.l2(0.01),
            #bias_regularizer = tf.keras.regularizers.l2(0.01)))

            model.add(keras.layers.Dense(1,))


            model.compile(loss=keras.losses.mean_squared_error,
                          optimizer=keras.optimizers.Adam(lr = learnRate) ,  )  # experimental_run_tf_function=False )
            # print(model.summary())

            return model

        self.flagGen = False
        XSplineVector = []
        velocities = []
        vectorWeights = []

        if expansion:
            for i in range(0, len(X)):
                vector = self.extractFunctionsFromSplines(X[i][0], X[i][1], X[i][2], X[i][3],X[i][4],)#X[i][5],)
                #vector = list(np.where(np.array(vector) < 0, 0, np.array(vector)))
                #vector = ([abs(k) for k in vector])
                #vector = extractFunctionsFromSplines(X[i][0], X[i][1], X[i][2], X[i][3], X[i][4])
                #vector = extractFunctionsFromSplines(X[i][0], X[i][1],X[i][2],X[i][3],X[i][4],X[i][5],X[i][6])
                #vectorNew = np.array(self.intercepts) * vector
                #vectorNew = np.array([i + self.interceptsGen for i in vectorNew])
                #XSplineVector.append(np.append(X[i],vector))
                SplineVector=np.append(X[i], vector)
                #SplineVector = X[i]
                XSplineVector.append(np.append(SplineVector, Y[i]))

            XSplineVectorGen = np.array(XSplineVector)
            raw_seq = XSplineVectorGen
        else:
            raw_seq = np.array(np.append(X, np.asmatrix([Y]).T, axis=1))
        #XSplineVectorGen = XSplineVectorGen[XSplineVectorGen[:,3].argsort()]

        # split into samples
        #if attWeights != None:
        #Xlstm = Xlstm * attWeights
        # split into samples
        Xlstm, Ylstm = self.split_sequence(raw_seq, n_steps)
        estimator = baseline_model()

        dot_img_file = '/home/dimitris/Desktop/neural.png'
        tf.keras.utils.plot_model(estimator, to_file=dot_img_file, show_shapes=True)
        #weights0 = np.mean(XSplineVector, axis=0)
        # weights1 = self.intercepts
        #weights1 = estimator.layers[0].get_weights()[0][1]
        #weights = np.array(
            #np.append(weights0.reshape(-1, 1), np.asmatrix(weights0).reshape(-1, 1), axis=1).reshape(9, -1))

        #estimator.layers[0].set_weights([weights, np.array([0] * (genModelKnots - 1))])
        epochs = 100
        learnRate = 0.01

        run = neptune.init(
            project="kaklis1992/LSTM-FOC",
            api_token= self.neptuneAPIToken,
        )  # your credentials

        params = {"lr": learnRate, "n_steps": n_steps, "epochs": epochs, "neurons": neurons}
        run["parameters"] = params

        neptune_cbk = NeptuneCallback(run=run, base_namespace="training")

        print(estimator.summary())

        print("Shape of training data: " + str(Xlstm.shape))
        print("GENERAL MODEL  ")
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, restore_best_weights=True)
        rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1,)#restore_best_weights=True
        #XSplineVectorGen = np.reshape(XSplineVectorGen, (XSplineVectorGen.shape[0],1, XSplineVectorGen.shape[1]))
        #for i in range(0,10):
        size = 3000
        X_test, Y_test = Xlstm[len(Xlstm)-size:len(Xlstm)] , Ylstm[len(Ylstm)-size:len(Ylstm)]
        Xlstm, Ylstm = Xlstm[:len(Xlstm) - size], Ylstm[:len(Ylstm) - size]
        history = estimator.fit(Xlstm, Ylstm, epochs=100 ,verbose=0 ,validation_split=0.1,  callbacks=[neptune_cbk] )#shuffle=False,batch_size=120)callbacks=[rlrop]

        eval_metrics = estimator.evaluate(X_test, Y_test, verbose=0)
        #for  metric in enumerate(eval_metrics):
        run["eval/{}".format(estimator.metrics_names[0])] = eval_metrics

        run.stop()
        #XSplineVector =np.array(XSplineVector).reshape(-1,1)
        #history = estimator.fit(XSplineVectorGen, Y, epochs=50, verbose=0, callbacks=[rlrop], validation_split=0.1,batch_size=len(XSplineVectorGen),shuffle=False)
        #estimator.reset_states()
        mse =  estimator.evaluate(Xlstm,Ylstm)

        #if attWeights == None:
        #attentionModel = tf.keras.Model(inputs = estimator.layers[0].input, outputs = estimator.layers[3].output)
        #attentionWeights = attentionModel.predict(Xlstm)
        #output = open('./attentionWeights.pkl', 'wb')
        #pickle.dump(attentionWeights, output)


        listLoss = history.history['loss']
        listValoss = history.history['val_loss']

        loss = str(np.round(math.sqrt(np.mean(listLoss))))
        valLoss = str(np.round(math.sqrt(np.mean(listValoss))))

        estimatorName = 'estimatorExpandedInput_' if expansion == True else 'estimator_'
        estimator.save('./DeployedModels/'+estimatorName + vessel + ' ' + str(
            datetime.now().strftime("%d_%m_%Y_%H:%M")) + ' '+loss+'_'+valLoss+'.h5')

        print("(MAE) loss: "+ loss)
        print("(MAE) val loss:  " +valLoss)

        print(estimator.metrics_names)

        maeListLoss  = [math.sqrt(x) for x in listLoss]
        maeListValLoss = [math.sqrt(x) for x in listValoss]

        plt.clf()
        fig, ax1 = plt.subplots(figsize=(15, 10))
        plt.plot(maeListLoss)
        plt.plot(maeListValLoss)
        plt.title('model train vs validation loss MAE: %.2f' % (np.round(math.sqrt(mse),2)) +", "+str(np.round(math.sqrt(np.mean(history.history['val_loss'])))) )
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.grid()
        plt.ylim(float(loss) - 1 if loss <= valLoss else float(valLoss) - 1, 10)
        plt.legend(['train', 'validation'], loc='upper right')
        plt.savefig('./Figures/'+estimatorName + vessel +'.eps', format = 'eps')
        #plt.show()



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
        ### IF DATA ARE CLUSTERED TRAIN ENSEBLING PREDICTION SCHEME
        if len(partition_labels) > 1:
            for idx, pCurLbl in enumerate(partition_labels):

                self.modelId = idx
                self.countTimes += 1

                XSplineClusterVector = []
                for i in range(0, len(partitionsX[idx])):
                    #vector = extractFunctionsFromSplines(partitionsX[idx][i][0], partitionsX[idx][i][1],partitionsX[idx][i][2],partitionsX[idx][i][3],partitionsX[idx][i][4],partitionsX[idx][i][5],partitionsX[idx][i][6])
                    vector = self.extractFunctionsFromSplines(partitionsX[idx][i][0], partitionsX[idx][i][1],
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
                Xlstm, Ylstm = self.split_sequence(raw_seq, n_steps)

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


                modelsCl = []
                clustersScore = []
                clustersTrScore=[]

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
                except ValueError as e:
                    return e

                estimatorCl.save('./DeployedModels/estimatorCl_'+str(idx)+'.h5')
                minClscore = min(clustersScore)
                minClTrscore = min(clustersTrScore)
                minIndCl = clustersScore.index(min(clustersScore))
                minEpoch = minIndCl + 1
                minEpochs.append(minEpoch)
                minXVars.append(np.std(partitionsX[idx]))
                minYVars.append(np.std(partitionsY[idx]))
                stdWS.append(np.std(partitionsX[idx][:,3]))
                stdSTW.append(np.std(partitionsX[idx][:, 0]))
                stdsFOC.append(np.std(partitionsY[idx]))

                clScores.append(minClscore)
                clustersTrScores.append(minClTrscore)
                clusters.append(idx+1)
                sizeTrDt.append(len(partitionsX[idx]))

                NNmodels.append(modelsCl[minIndCl] )



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

        return estimator, history, scores, numpy.empty, None  # , None , DeepCLpartitionsX

    def getFitnessOfModelForPoint(self, model, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(self._partitionsPerModel[model]) - point))
    def getFitnessOfPoint(self,partitions ,cluster, point):

        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(partitions[cluster], axis=0) - point))

