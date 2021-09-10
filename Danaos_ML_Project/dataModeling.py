import pyearth as sp
import numpy as np
import numpy.linalg
import csv
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


class BasePartitionModeler:
    def createModelsFor(self,partitionsX, partitionsY, partition_labels):
        pass

    def extractFunctionsFromSplines(self, modelId ,x0, x1, x2, x3, x4, x5=None, x6=None, vessel=None):
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

class LinearInterpolation(BasePartitionModeler):

    def createModelsFor(self, partitionsX, partitionsY, partition_labels, tri, X, Y, ):
        stw= np.array(X[:,0])
        ws = np.array(X[:,3])
        wd = list(X[:, 4])
        #dataset = np.array(np.append(stw.reshape(-1,1),np.asmatrix([ws,wd]).T,axis=1))
        #y_interp=scipy.interpolate.interpn((stw,ws,wd), Y,np.array([stw,ws,wd]).T)
        y_interp = scipy.interpolate.SmoothBivariateSpline(stw,ws,wd, Y)

        return y_interp

class PavlosInterpolation(BasePartitionModeler):

    def GetStatisticsOfVessel(self,company,vessel):

        listOfWeatherBeaufort = np.array([0,3,5,8])
        listOfWeather = [0,4.34,9,34,18.91]

        sFile = './data/'+company+'/'+vessel+'/ListOfSpeeds.csv'
        data = pd.read_csv(sFile, delimiter=',')
        listOfSpeeds = np.array(data.values)


        sFile = './data/' + company + '/'+vessel+'/ListOfCons.csv'
        data = pd.read_csv(sFile, delimiter=',')
        ListOfCons = np.array(data.values)

        sFile = './data/' + company + '/'+vessel+'/ListOfDrafts.csv'
        data = pd.read_csv(sFile, delimiter=',')
        ListOfDrafts = np.array(data.values)

        ConsProfileItem= {}
        speedIndex=0
        wsIndex=0
        wdIndex=0
        draftIndex=0
        for i in range(0,len(ListOfCons)):

            if i > 0:
                if i%80==0: ##Ballast
                    draftIndex=draftIndex+1
                if i%20==0 : #MinSpeed
                    speedIndex= speedIndex +1
                    wsIndex=0
                if i%5 ==0: #Ws Update
                    wsIndex = wsIndex +1
                    wdIndex = 0
            ConsProfileItem[i]={'speed':speedIndex,'ws':wsIndex,'wd':wdIndex,'draft':draftIndex,'foc':ListOfCons[i]}
            wdIndex = wdIndex + 1

        return listOfSpeeds , listOfWeather , ListOfCons , ListOfDrafts , ConsProfileItem

    def GetAvgCons(self,_speed, _weatherMperS,_weatherRelDir, _draft):

            listOfSpeeds, listOfWeather, ListOfCons, ListOfDrafts , ConsProfileItem = self.GetStatisticsOfVessel('MARMARAS','MT_DELTA_MARIA')
            _weatherMperS = _weatherMperS *0.514
            minSpeed = np.min(listOfSpeeds)
            maxSpeed = np.max(listOfSpeeds)
            maxConsumption = np.max(ListOfCons)
            calcAvgCons = maxConsumption #// Default for "Do not create edge" - to be divided by 24
            #relDirCode = convertWindRelDirToRelDirIndex(_weatherRelDir) #// Find relative direction
            relDirCode = 4
            exactSpeed = False
            exactWeather = False
            finalSpeedIndex = len(listOfSpeeds) - 1

            if _speed < minSpeed:
                _speed = minSpeed
            elif _speed > maxSpeed:
                _speed = maxSpeed

            # // Find draft index
            if _draft <= ListOfDrafts[1] -1:
                currDraftIndex = 0
            else:
                currDraftIndex = 1

            #// Find where it is in the list of speeds
            curspeedIndex = 0
            maxLenSpeed = 4 if currDraftIndex ==0 else 7
            minLenSpeed = 0 if currDraftIndex== 0 else  4
            for i in range(minLenSpeed,maxLenSpeed):
                curspeedIndex = i
                if _speed > listOfSpeeds[i]:
                    d=0
                elif _speed == listOfSpeeds[i]:
                    exactSpeed = True
                    break
                else:
                    break

            #// Find where it is in the list of weathers
            curweatherIndex = 0
            for i in range(0,len(listOfWeather)):
                curweatherIndex = i
                if _weatherMperS > listOfWeather[i]:
                    if i == len(listOfWeather) - 1:
                        exactWeather = True
                        break
                    #//else continue
                elif _weatherMperS == listOfWeather[i]:

                    exactWeather = True
                    break
                else:
                    if i == 0:
                        exactWeather = True #// This is for 0 BFT
                    break

            if exactSpeed and exactWeather:

                hashKey = draft + "_" + listOfSpeeds[curspeedIndex] + "_" + listOfWeather[curweatherIndex] + "_" + relDirCode
                cpi =[]
                calcAvgCons = cpi.avgCons

            elif (exactSpeed):

                prevweatherIndex = curweatherIndex - 1
                #hashKey1 = draft + "_" + listOfSpeeds[curspeedIndex] + "_" + listOfWeather[prevweatherIndex] + "_" + relDirCode
                cpi1 = [k for k in ConsProfileItem.values() if k['speed'] == curspeedIndex and k['ws']==prevweatherIndex and
                 k['wd'] == relDirCode and k['draft']==currDraftIndex][0]['foc']

                calcAvgConsPrev1 = cpi1#.avgCons
                #hashKey2 = draft + "_" + listOfSpeeds[curspeedIndex] + "_" + listOfWeather[curweatherIndex] + "_" + relDirCode
                cpi2 =  [k for k in ConsProfileItem.values() if k['speed'] == curspeedIndex and k['ws']==curweatherIndex and
                 k['wd'] == relDirCode and k['draft']==currDraftIndex][0]['foc']
                calcAvgConsCur1 = cpi2#.avgCons
                difInCons1 = calcAvgConsCur1 - calcAvgConsPrev1
                percWeatherDif = (math.pow(_weatherMperS, 3) - math.pow(listOfWeather[prevweatherIndex], 3)) / (math.pow(listOfWeather[curweatherIndex], 3) - math.pow(listOfWeather[prevweatherIndex], 3)) #// Cubic interpolation
                calcAvgCons = calcAvgConsPrev1 + difInCons1 * percWeatherDif

            elif exactWeather:

                prevspeedIndex = curspeedIndex - 1
                #hashKey3 = draft + "_" + listOfSpeeds[prevspeedIndex] + "_" + listOfWeather[curweatherIndex] + "_" + relDirCode
                cpi3 =  [k for k in ConsProfileItem.values() if k['speed'] == prevspeedIndex and k['ws']==curweatherIndex and
                 k['wd'] == relDirCode and k['draft']==currDraftIndex][0]['foc']
                calcAvgConsPrev2 = cpi3#.avgCons

                #hashKey4 = draft + "_" + listOfSpeeds[curspeedIndex] + "_" + listOfWeather[curweatherIndex] + "_" + relDirCode
                cpi4 = [k for k in ConsProfileItem.values() if k['speed'] == curspeedIndex and k['ws']==curweatherIndex and
                 k['wd'] == relDirCode and k['draft']==currDraftIndex][0]['foc']
                calcAvgConsCur2 = cpi4#.avgCons
                difInCons2 = calcAvgConsCur2 - calcAvgConsPrev2
                percSpeedDif = (math.pow(_speed, 3) - math.pow(listOfSpeeds[prevspeedIndex], 3)) / (math.pow(listOfSpeeds[curspeedIndex], 3) - math.pow(listOfSpeeds[prevspeedIndex], 3)) #// Cubic interpolation
                calcAvgCons = calcAvgConsPrev2 + difInCons2 * percSpeedDif #// Linear interpolation

            else:

                prevweatherIndex = curweatherIndex - 1
                prevspeedIndex = curspeedIndex - 1
                #hashKey3 = draft + "_" + listOfSpeeds[prevspeedIndex] + "_" + listOfWeather[prevweatherIndex] + "_" + relDirCode
                cpi3 = [k for k in ConsProfileItem.values() if k['speed'] == prevspeedIndex and k['ws']==prevweatherIndex and
                 k['wd'] == relDirCode and k['draft']==currDraftIndex][0]['foc']
                #cpi3 = []

                calcAvgConsPrev2 = cpi3#.avgCons
                #hashKey4 = draft + "_" + listOfSpeeds[curspeedIndex] + "_" + listOfWeather[curweatherIndex] + "_" + relDirCode
                cpi4 = [k for k in ConsProfileItem.values() if k['speed'] == curspeedIndex and k['ws']==curweatherIndex and
                 k['wd'] == relDirCode and k['draft']==currDraftIndex][0]['foc']
                calcAvgConsCur2 = cpi4#.avgCons
                difInCons2 = calcAvgConsCur2 - calcAvgConsPrev2
                percSpeedDif = (math.pow(_speed, 3) - math.pow(listOfSpeeds[prevspeedIndex], 3)) / (math.pow(listOfSpeeds[curspeedIndex], 3) - math.pow(listOfSpeeds[prevspeedIndex], 3)) #// Cubic interpolation
                calcAvgCons = calcAvgConsPrev2 + difInCons2 * percSpeedDif #// Linear interpolation


            return calcAvgCons[0]



    def createModelsFor(self, partitionsX, partitionsY, partition_labels, tri, X, Y):

        AvgCons = self.GetAvgCons(X[0,0],X[0,3],X[0,4],X[0,1])



        return y_interp

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



            model.add(keras.layers.Dense(5+genModelKnots-1, input_shape=(5+genModelKnots-1,)))
            #model.add(keras.layers.LSTM(2+genModelKnots - 1, input_shape=(2+ genModelKnots - 1,1)))

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
                        #if basis=='MSE:' :break
                        if pruned == "No":
                            data_writer.writerow([ basis,1 if coeff=='None' else coeff ])
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
                data_writer.writerow(['Basis', 'Coeff'])
                for row in modelSummary:
                    row=np.delete(np.array(row.split(" ")), [i for i, x in enumerate(row.split(" ")) if x == ""])
                    try:
                        basis = row[0]
                        pruned = row[1]
                        coeff = row[2]
                        if basis == 'x0' and pruned == "No":
                            f=0
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

        def extractFunctionsFromSplines(x0, x1, x2, x3, x4=None, x5=None, x6=None):
            piecewiseFunc = []
            self.count = self.count + 1
            for csvM in csvModels:
                if csvM != './trainedModels/model_' + str(self.modelId) + '_.csv':
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

        #extractFunctionsFromSplines(1, 1)

        self.flagGen=False

        estimator = baseline_model()
        XSplineVector=[]
        velocities = []
        vectorWeights=[]
        self.modelId = 'Gen'
        for i in range(0,len(X)):
            vector = extractFunctionsFromSplines(X[i][0],X[i][1],X[i][2],X[i][3],X[i][4])

            #vectorNew = np.array(self.intercepts) * vector
            #vectorNew = np.array([i + self.interceptsGen for i in vector])

            XSplineVector.append(np.append(X[i],vector))


        XSplineVector = np.array(XSplineVector)

        #try:
        #XSplineVector = np.reshape(XSplineVector, (XSplineVector.shape[0], XSplineVector.shape[1], 1))
        #es = EarlyStopping(monitor='loss', mode='min', verbose=0)
        estimator.fit(XSplineVector, Y, epochs=100, validation_split=0.33,verbose=0,)
        #score = estimator.evaluate(np.array(XSplineVector),Y, verbose=0)
        #except:


        #estimator.fit(X, Y, epochs=100, validation_split=0.33)


        self.flagGen = True

         # validation_data=(X_test,y_test)


        NNmodels=[]
        scores=[]
        XSplineClusterVectors=[]

        if len(partition_labels) > 1:
            for idx, pCurLbl in enumerate(partition_labels):
                    #partitionsX[ idx ]=partitionsX[idx].reshape(-1,2)
                    self.intercepts = []
                    self.modelId = idx
                    self.countTimes += 1
                    self.count = 0

                    XSplineClusterVector=[]
                    for i in range(0, len(partitionsX[idx])):
                        vector = extractFunctionsFromSplines(partitionsX[idx][ i ][ 0 ], partitionsX[idx][ i ][ 1 ],partitionsX[idx][ i ][ 2 ],partitionsX[idx][ i ][ 3 ]
                                                             ,partitionsX[idx][ i ][ 4 ])

                        #vectorNew = np.array(self.intercepts) * vector
                        #vectorNew = np.array([i + self.interceptsGen for i in vector])

                        XSplineClusterVector.append(np.append(partitionsX[idx][i], vector))

                    # X =  np.append(X, np.asmatrix([dataY]).T, axis=1)
                    XSplineClusterVector = np.array(XSplineClusterVector)
                    XSplineClusterVectors.append(XSplineClusterVector)
                    #estimator = baseline_model()
                    numOfNeurons = [ x for x in ClModels[ 'data' ] if x[ 'id' ] == idx ][ 0 ][ 'funcs' ]

                    estimatorCl = keras.models.Sequential()

                    #estimatorCl.add(keras.layers.LSTM(2 + numOfNeurons -1 ,input_shape=(2+numOfNeurons-1,1)))
                    estimatorCl.add(keras.layers.Dense(5 + numOfNeurons - 1 , input_shape=(5 + numOfNeurons - 1, )))
                    estimatorCl.add(keras.layers.Dense(numOfNeurons - 1,))
                    #estimatorCl.add(keras.layers.Dense(numOfNeurons - 1, ))
                    #estimatorCl.add(keras.layers.Dense(numOfNeurons - 1, ))
                    estimatorCl.add(keras.layers.Dense(2))
                    estimatorCl.add(keras.layers.Dense(1, ))
                    estimatorCl.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(), )
                    #try:
                    #XSplineClusterVector = np.reshape(XSplineClusterVector, (XSplineClusterVector.shape[0], XSplineClusterVector.shape[1], 1))

                    #es = EarlyStopping(monitor='loss', mode='min', verbose=0)
                    estimatorCl.fit(np.array(XSplineClusterVector),np.array(partitionsY[idx]),epochs=100,verbose=0, )#validation_split=0.33

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

        ###Train Weights
        def custom_loss(y_true,y_pred):
            return    tf.keras.losses.mean_squared_error(y_true,y_pred)
                      #+tf.keras.losses.kullback_leibler_divergence(y_true, y_pred) +tf.keras.losses.categorical_crossentropy(y_true,y_pred)
        NNmodels.append(estimator)
        preds=[]
        trainedWeights=None
        estimatorW = keras.models.Sequential()
        estimatorW.add(keras.layers.Dense(len(partition_labels)+1, input_shape=(len(partition_labels)+1,)))
        #estimatorW.add(keras.layers.Dense(len(partition_labels) ,))
        #estimatorW.add(keras.layers.Dense(len(partition_labels)-1, ))
        #estimatorW.add(keras.layers.Dense(len(partition_labels) - 2, ))
        estimatorW.add(keras.layers.Dense(1, ))
        estimatorW.compile(loss=custom_loss, optimizer=keras.optimizers.Adam(), )
        if len(partition_labels) > 1:
            for i in range(0, len(X)):
                pred = []
                for n in range(0, len(partitionsX)):
                        self.intercepts = []
                        #self.modelId = 'Gen' if n == len(NNmodels)-1 else n
                        self.modelId = n
                        self.countTimes += 1
                        self.count = 0
                        vector = extractFunctionsFromSplines(X[i][0], X[i][1],X[i][2],X[i][3],X[i][4])
                        #XSplinevectorNew = np.array(self.intercepts) * vector
                        #XSplinevectorNew = np.array([i + self.interceptsGen for i in XSplinevectorNew])

                        XSplinevectorNew = np.append(X[i], vector)
                        XSplinevectorNew = XSplinevectorNew.reshape(-1, XSplinevectorNew.shape[0])

                        pred.append(NNmodels[n].predict(XSplinevectorNew)[0][0])
                pred.append(estimator.predict(XSplineVector[i].reshape(-1,XSplineVector.shape[1]))[0][0])
                preds.append(pred)

            preds = np.array(preds).reshape(-1,len(partition_labels)+1)
            estimatorW.fit(preds, Y.reshape(-1,1), epochs=100, verbose=0)
            #trainedWeights = estimatorW.get_weights()[2]

        #NNmodels.append(estimator)
        #estimator.save('./DeployedModels/estimatorCl_Gen.h5')
        self._models = NNmodels

        # Return list of models
        return estimator, None ,scores, numpy.empty,estimatorW #, estimator , DeepCLpartitionsX

    def getFitnessOfModelForPoint(self, model, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(self._partitionsPerModel[ model ]) - point))

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

    def createModelsFor(self, partitionsX, partitionsY, partition_labels, tri, X, Y, expansion, vessel, ):#,numOfLayers,numOfNeurons):

        models = []
        # partitionsX=np.array(partitionsX[0])
        # partitionsY = np.array(partitionsY[0])

        self.ClustersNum = len(partitionsX)
        self.modelId = -1
        # partition_labels = len(partitionsX)
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

                with open('./trainedModels/model_Gen_' + vessel + '.csv', mode='w') as data:
                    csvModels.append('./trainedModels/model_Gen_' + vessel + '.csv')
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

            self.count = 0

        n_steps = 15
        neurons = 5 + genModelKnots -1 #if expansion == True else 5
        input_shape = 5 + genModelKnots -1 if expansion == True else 5
        print("Neurons length: "+str(neurons))
        
        def baseline_model():
            # create model
            model = keras.models.Sequential()

            #model.add(keras.layers.Dense(20, input_shape=(7,)))
            #model.add(keras.layers.Dense(10, input_shape=(7,)))

            #kernel_regularizer = tf.keras.regularizers.l1(0.01),
            #activity_regularizer = tf.keras.regularizers.l2(0.01),
            #bias_regularizer = tf.keras.regularizers.l2(0.01)

            model.add(keras.layers.LSTM(neurons, input_shape=(n_steps, input_shape  ,),))#return_sequences=True
            #model.add(keras.layers.LSTM(5 , input_shape=(n_steps, 5 ,), ))
            #model.add(keras.layers.LSTM( 5+genModelKnots-1, ))
            #model.add(keras.layers.Dense(30,input_shape=( 5 ,)))
            #model.add(keras.layers.Dense(20))
            #model.add(keras.layers.Dense(genModelKnots - 3,activation='relu'))
            #model.add(keras.layers.Dropout(0.2))

            model.add(keras.layers.Dense(genModelKnots -1, kernel_regularizer = tf.keras.regularizers.l1(0.01),
            activity_regularizer = tf.keras.regularizers.l2(0.01),
            bias_regularizer = tf.keras.regularizers.l2(0.01) ),)

            #kernel_regularizer = tf.keras.regularizers.l1(0.01),
            #activity_regularizer = tf.keras.regularizers.l2(0.01),
            #bias_regularizer = tf.keras.regularizers.l2(0.01),


            #model.add(keras.layers.Dense(5,))
            model.add(keras.layers.Dense(2, ))


            #.add(keras.layers.Dense(5,kernel_regularizer = tf.keras.regularizers.l1(0.01),
            #activity_regularizer = tf.keras.regularizers.l2(0.01),
            #bias_regularizer = tf.keras.regularizers.l2(0.01)))

            model.add(keras.layers.Dense(1,))

            model.compile(loss=keras.losses.mean_squared_error,
                          optimizer=keras.optimizers.Adam() )  # experimental_run_tf_function=False )
            # print(model.summary())

            return model


        def extractFunctionsFromSplines(x0, x1 , x2 , x3 ,x4=None , x5=None ,x6=None,x7=None):
            piecewiseFunc = []
            self.count = self.count + 1
            for csvM in csvModels:
                if csvM != './trainedModels/model_' + str(self.modelId) + '_'+vessel+'.csv':
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
                                                    piecewiseFunc.append(x0 * (x0 - nums[0]) )#* float(row[1]))
                                                elif split == "x1":
                                                    piecewiseFunc.append(x1 * (x1 - nums[0]))# * float(row[1]))
                                                elif split == "x01":
                                                    piecewiseFunc.append(x0 * (x1 - nums[0]))# * float(row[1]))
                                                elif split == "x10":
                                                    piecewiseFunc.append(x1 * (x0 - nums[0]))# * float(row[1]))
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
                                                    piecewiseFunc.append(x0 * (nums[0] - x0))# * float(row[1]))
                                                elif split == "x1":
                                                    piecewiseFunc.append(x1 * (nums[0] - x1))# * float(row[1]))
                                                elif split == "x01":
                                                    piecewiseFunc.append(x0 * (nums[0] - x1))# * float(row[1]))
                                                elif split == "x10":
                                                    piecewiseFunc.append(x1 * (nums[0] - x0))# * float(row[1]))

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
                                            piecewiseFunc.append((nums[0] - x0) * (nums[1] - x0))# * float(row[1]))
                                        elif split == "x1":
                                            piecewiseFunc.append((nums[0] - x1) * (nums[1] - x1))# * float(row[1]))
                                        elif split == "x01":
                                            piecewiseFunc.append((nums[0] - x0) * (nums[1] - x1))# * float(row[1]))
                                        elif split == "x10":
                                            piecewiseFunc.append((nums[0] - x1) * (nums[1] - x0) )#* float(row[1]))

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


        XSplineVector = []
        velocities = []
        vectorWeights = []

        if expansion == True:
            for i in range(0, len(X)):
                vector = extractFunctionsFromSplines(X[i][0], X[i][1], X[i][2], X[i][3],X[i][4],)#X[i][5],)
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
            return np.array(X), np.array(y)

            # define input sequence


        #

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
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, restore_best_weights=True)
        rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1,)#restore_best_weights=True
        #XSplineVectorGen = np.reshape(XSplineVectorGen, (XSplineVectorGen.shape[0],1, XSplineVectorGen.shape[1]))
        #for i in range(0,10):
        size = 3000
        #X_test, Y_test = Xlstm[len(Xlstm)-size:len(Xlstm)] , Ylstm[len(Ylstm)-size:len(Ylstm)]
        #Xlstm, Ylstm = Xlstm[:len(Xlstm) - size], Ylstm[:len(Ylstm) - size]
        history = estimator.fit(Xlstm, Ylstm, epochs=100 ,verbose=0 ,validation_split=0.05,)#shuffle=False,batch_size=120)callbacks=[rlrop]
        #XSplineVector =np.array(XSplineVector).reshape(-1,1)
        #history = estimator.fit(XSplineVectorGen, Y, epochs=50, verbose=0, callbacks=[rlrop], validation_split=0.1,batch_size=len(XSplineVectorGen),shuffle=False)
        #estimator.reset_states()
        mse =  estimator.evaluate(Xlstm,Ylstm)

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
        loss = str(np.round(math.sqrt(mse),2))
        valLoss = str(np.round(math.sqrt(np.mean(history.history['val_loss']))))

        estimatorName = 'estimatorExpandedInput_' if expansion == True else 'estimator_'
        estimator.save('./DeployedModels/'+estimatorName + vessel + ' ' + str(
            datetime.now().strftime("%d_%m_%Y_%H:%M")) + ' '+loss+'_'+valLoss+'.h5')

        print("LOSS: "+ loss)
        print("VAL LOSS:  " +valLoss)

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
        estimator.save('./DeployedModels/estimatorCl_Gen.h5')

        print(estimator.summary())


        #print('Accuracy: %.2f' % (accuracy * 100))

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model train vs validation loss MAE: %.2f' % (np.round(math.sqrt(mse),2)) +"  "+str(np.round(math.sqrt(np.mean(history.history['val_loss'])))) )
        plt.ylabel('loss')
        plt.xlabel('epoch')
        #plt.ylim(0,10)
        plt.legend(['train', 'validation'], loc='upper right')
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

        return estimator, history, scores, numpy.empty, None  # , None , DeepCLpartitionsX

    def getFitnessOfModelForPoint(self, model, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(self._partitionsPerModel[model]) - point))
    def getFitnessOfPoint(self,partitions ,cluster, point):

        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(partitions[cluster], axis=0) - point))

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

    def getFitnessOfModelForPoint(self, model, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(self._partitionsPerModel[model]) - point))

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
