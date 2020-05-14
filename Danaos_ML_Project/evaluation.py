import numpy as np
import dataModeling as dt
import tensorflow as tf
import sklearn.ensemble as skl
import statsmodels.api
#from statsmodels.formula.api import ols
import pandas as pd
#import scikit_posthocs as sp
from scipy import stats
from scipy.interpolate import BPoly as Bernstein
from itertools import combinations
from statsmodels.stats.multitest import multipletests
import warnings
import math
from scipy import spatial
import pyearth as sp
from scipy.spatial import Delaunay, ConvexHull
import matplotlib.pyplot as plt
import csv
import pyearth as sp
import sklearn.svm as svr
import latex
from matplotlib import rc



class Evaluation:
    def evaluate(self, train, unseen, model,genericModel):
        return 0.0

class MeanAbsoluteErrorEvaluation (Evaluation):
    '''
    Performs evaluation of the datam returning:
    errors: the list of errors over all instances
    meanError: the mean of the prediction error
    sdError: standard deviation of the error
    '''
    def evaluate(self, unseenX, unseenY, modeler,genericModel):
        lErrors = []
        for iCnt in range(np.shape(unseenX)[0]):
            pPoint = unseenX[iCnt].reshape(1, -1)#[0] # Convert to matrix
            trueVal = unseenY[iCnt]
            prediction = modeler.getBestModelForPoint(pPoint).predict(pPoint)

            lErrors.append(abs(prediction - trueVal))
        errors = np.asarray(lErrors)

        return errors, np.mean(errors), np.std(lErrors)

    def evaluateNN(self, unseenX, unseenY, modeler,output,xs):
        lErrors = []
        self.session = tf.Session()
        saver = tf.train.Saver()
       # tf.saved_model.loader.load(
       #     self.session,
       #     [ tf.saved_model.tag_constants.TRAINING ],
       #     './save')

        for iCnt in range(np.shape(unseenX)[0]):

            saver.restore(self.session, "./save/test_" + str(iCnt) + ".ckpt")
            model = modeler.getBestModelForPoint(pPoint)
            pPoint = unseenX[iCnt].reshape(1, -1)#[0] # Convert to matrix
            trueVal = unseenY[iCnt]
            prediction = self.session.run(output, feed_dict={xs: pPoint})
            lErrors.append(abs(prediction - trueVal))
            errors = np.asarray(lErrors)

        return errors, np.mean(errors), np.std(lErrors)

    def evaluateInterpolation(self, unseenX, unseenY, modeler, output, xs, genericModel, partitionsX, scores):
        for iCnt in range(np.shape(unseenX)[0]):
            pPoint = unseenX[iCnt]
            pPoint = pPoint.reshape(-1, unseenX.shape[1])

            trueVal = unseenY[iCnt]

            prediction = modeler([pPoint[0],pPoint[3]])

            error = abs(prediction - trueVal)
            lErrors.append(error)
        return errors, np.mean(errors), np.std(lErrors)

    def evaluateKerasNN1(self, unseenX, unseenY, modeler,output,xs,genericModel,partitionsX , scores):
        lErrors = []

        from tensorflow import keras
        path = '.\\'
        clusters = '30'

        count = 0
        errorStwArr=[]
        errorFoc=[]
        foc=[]
        for iCnt in range(np.shape(unseenX)[0]):
            pPoint =unseenX[iCnt]
            pPoint= pPoint.reshape(-1,unseenX.shape[1])

            trueVal = unseenY[iCnt]

            ind, fit = modeler.getBestPartitionForPoint(pPoint, partitionsX)
            #preds=[]
            #for i in range(0,len(partitionsX[ind])):
                #pPointCl = partitionsX[ind][i]
                #vector = self.extractFunctionsFromSplines(pPointCl[0], pPointCl[1], pPointCl[2], pPointCl[3],
                                                          #pPointCl[4], pPointCl[5], pPointCl[6], ind)
                #XSplineVector = np.append(pPointCl, vector)
                #XSplineVector = XSplineVector.reshape(-1, XSplineVector.shape[0])
                #pred = abs(modeler._models[ind].predict(XSplineVector))
                #preds.append(pred)
            distVec = np.mean(partitionsX[ind], axis=0) - pPoint[0]
            pPointRefined = []
            for i in range(0, pPoint.shape[1]):
                if abs(distVec[i])>10:
                    pPointRefined.append((pPoint[0][i] + (distVec[i])/2))
                else:
                    pPointRefined.append(pPoint[0][i])
            pPointRefined = np.array(pPointRefined)

            meanPointsOfCl = np.mean(partitionsX[ind],axis=0)
            meanPointsOfCl = np.mean([meanPointsOfCl,pPointRefined],axis=0)
            #####
            #vectorPpoint = self.extractFunctionsFromSplines(pPoint[0][0], pPoint[0][1], pPoint[0][2], pPoint[0][3],
                                                      #pPoint[0][4], pPoint[0][5], pPoint[0][6], ind)
            #####
            vector = self.extractFunctionsFromSplines(meanPointsOfCl[0], meanPointsOfCl[1], meanPointsOfCl[2], meanPointsOfCl[3],
                                                      meanPointsOfCl[4], meanPointsOfCl[5], meanPointsOfCl[6], ind)

            #newVector = np.mean([vector,vectorPpoint],axis=0)

            XSplineVector = np.append(meanPointsOfCl, vector)

            XSplineVector = XSplineVector.reshape(-1, XSplineVector.shape[0])
            pred = abs(modeler._models[ind].predict(XSplineVector))
            meanClpred = pred
            #pred=0
            #for i in range(0,len(partitionsX)):
                #pred +=modeler._models[i].predict(pPoint)
            #prediction = pred / len(partitionsX)
            vector = self.extractFunctionsFromSplines(pPoint[0][0], pPoint[0][1],pPoint[0][2],pPoint[0][3],pPoint[0][4],pPoint[0][5],pPoint[0][6], ind)
            XSplineVector = np.append(pPoint, vector)
            XSplineVector = XSplineVector.reshape(-1, XSplineVector.shape[0])

            vector = self.extractFunctionsFromSplines(pPoint[0][0], pPoint[0][1],pPoint[0][2],pPoint[0][3],pPoint[0][4],pPoint[0][5],pPoint[0][6], 'Gen')
            XSplineGenVector = np.append(pPoint, vector)
            XSplineGenVector = XSplineGenVector.reshape(-1, XSplineGenVector.shape[0])
            # prediction = abs(modeler._models[ 0 ].predict(XSplineVector))
            # XSplineVector = XSplineGenVector if modeler._models[ind][1]=='GEN' else XSplineVector
            #prediction = (abs(modeler._models[ind].predict(XSplineVector)) + modeler._models[len(modeler._models)-1].predict(XSplineGenVector))/2
            prediction = (meanClpred  + modeler._models[
                len(modeler._models) - 1].predict(XSplineGenVector)) / 2

            #prediction = (abs(modeler._models[ind].predict(pPoint)))
            #prediction =  modeler._models[len(modeler._models) - 1].predict(pPoint)
            #prediction = (abs(modeler._models[ind].predict(pPoint)) + modeler._models[len(modeler._models) - 1].predict(pPoint) )/ 2
            error = abs(prediction - trueVal)
            lErrors.append(error)

            errorStwArr.append(np.array(np.append(np.asmatrix(pPoint[0][0]).reshape(-1,1), np.asmatrix([error[0]]).T, axis=1)))
            errorFoc.append(abs((prediction - trueVal)/trueVal) * 100)
            foc.append(trueVal)


        errorStwArr = np.array(errorStwArr)
        errorStwArr = errorStwArr.reshape(-1, 2)
        errors = np.asarray(lErrors)
        with open('./errorPercFOC'+str(len(partitionsX))+'.csv', mode='w') as data:
            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow(
                ['FOC', 'PERC'])
            for i in range(0, len(errorFoc)):
                data_writer.writerow(
                    [foc[i],errorFoc[i][0][0]])

        with open('./errorSTW'+str(len(partitionsX))+'.csv', mode='w') as data:
            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow(
                ['STW', 'MAE'])
            for i in range(0, len(errorStwArr)):
                data_writer.writerow(
                    [errorStwArr[i][0],errorStwArr[i][1]])

        #plt.scatter(errorStwArr[:,0],errorStwArr[:,1])
        #plt.ylim(0, 2)
        #plt.title('Model loss with ' + str(1) + ' cluster(s)')
        #plt.ylabel('MAE')
        #plt.xlabel('STW')
        #plt.show()
        x = 0

        return errors, np.mean(errors), np.std(lErrors)

    def evaluateKerasNN(self, unseenX, unseenY, modeler,output,xs,genericModel,partitionsX , scores):
        lErrors = []
        with open('./meanErrorStw.csv', mode='w') as data:
            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow(['STW','DRAFT','TRIM','WS','WA','CWH','CWD','predFOC','ActualFoc','MAE'])
            for iCnt in range(np.shape(unseenX)[0]):
                pPoint =unseenX[iCnt]
                pPoint= pPoint.reshape(-1,unseenX.shape[1])


                trueVal = unseenY[iCnt]

                ind, fit = modeler.getBestPartitionForPoint(pPoint, partitionsX)
                prediction = (abs(modeler._models[ind].predict(pPoint)) + modeler._models[len(modeler._models) - 1].predict(
                    pPoint)) / 2
                error = abs(prediction - trueVal)
                lErrors.append(error)
                if(abs(prediction - trueVal)) > 5:
                    data_writer.writerow([pPoint[0][0],pPoint[0][1],pPoint[0][2],pPoint[0][3],pPoint[0][4],pPoint[0][5],pPoint[0][6],prediction[0][0],trueVal,error[0][0]])


        errors = np.asarray(lErrors)

        return errors, np.mean(errors), np.std(lErrors)

    def extractFunctionsFromSplines(self,x0, x1, x2, x3, x4, x5, x6,modelId):
        piecewiseFunc = []
        csvModels=['./model_'+str(modelId)+'_.csv']
        for csvM in csvModels:
            if csvM != './model_' + str(modelId) + '_.csv':
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
                    #if self.count == 1:
                        #self.intercepts.append(float(row[1]))

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

    def ANOVAtest(self,clusters,var,error,models,partitioners):

        df = pd.DataFrame({
                            'clusters': clusters,
                            'var': var,
                            'error':error,
                            'partitioners':partitioners,
                            'models':models
                            #'meanBearing':trFeatures[1]
                            })
        groups=[error,clusters,var]


        #data=[error,np.var(unseenX),clusters]
        print(stats.kruskal(error,clusters,var,models,partitioners))
        #print(self.kw_dunn(groups,[(0,1),(0,2)]))

        #dataf = pd.DataFrame.from_dict(df, orient='index')
        df.to_csv('./NEWres_1.csv', index=False)
        #df.melt(var_name='groups', value_name='values')
        df=pd.melt(df,var_name='groups', value_name='values')
        print(df)
        #df2=sp.posthoc_dunn(df, val_col='values', group_col='groups')
        #df2.to_csv('./kruskal.csv', index=False)
        #print(df2)


        #formula = 'error ~ C(clusters) + C(var) + C(error)'
        ##model =  ols(formula, df).fit()
        #aov_table = statsmodels.stats.anova.anova_lm(model, typ=2)
        #print(aov_table)


    def GetStatisticsOfVessel(self, company, vessel):

            listOfWeatherBeaufort = np.array([0, 3, 5, 8])
            listOfWeather = [0, 4.34, 9, 34, 18.91]

            sFile = './data/' + company + '/' + vessel + '/ListOfSpeeds.csv'
            data = pd.read_csv(sFile, delimiter=',')
            listOfSpeeds = np.array(data.values)

            sFile = './data/' + company + '/' + vessel + '/ListOfCons.csv'
            data = pd.read_csv(sFile, delimiter=',')
            ListOfCons = np.array(data.values)

            sFile = './data/' + company + '/' + vessel + '/ListOfDrafts.csv'
            data = pd.read_csv(sFile, delimiter=',')
            ListOfDrafts = np.array(data.values)

            ConsProfileItem = {}
            speedIndex = 0
            wsIndex = 0
            wdIndex = 0
            draftIndex = 0
            for i in range(0, len(ListOfCons)):

                if i > 0:
                    if i % 80 == 0:  ##Ballast
                        draftIndex = draftIndex + 1
                    if i % 20 == 0:  # MinSpeed
                        speedIndex = speedIndex + 1
                        wsIndex = 0
                    if i % 5 == 0:  # Ws Update
                        wsIndex = 0 if i % 20==0 else wsIndex +1
                        wdIndex = 0
                ConsProfileItem[i] = {'speed': speedIndex, 'ws': wsIndex, 'wd': wdIndex, 'draft': draftIndex,
                                      'foc': ListOfCons[i]}
                wdIndex = wdIndex + 1

            return listOfSpeeds, listOfWeather, ListOfCons, ListOfDrafts, ConsProfileItem

    def ConvertWDto0_180(self,_weatherRelDir):
        if _weatherRelDir > 180 and _weatherRelDir <= 225:
            _weatherRelDir = _weatherRelDir - 180
        elif _weatherRelDir > 225 and _weatherRelDir <= 270:
            _weatherRelDir = _weatherRelDir - 225
        elif _weatherRelDir > 270 and _weatherRelDir <= 316:
            _weatherRelDir = _weatherRelDir - 270
        elif _weatherRelDir > 315 and _weatherRelDir <= 360:
            _weatherRelDir = _weatherRelDir - 315

        return _weatherRelDir

    def convertWindRelDirToRelDirIndex(self,_weatherRelDir):

        _weatherRelDir = self.ConvertWDto0_180(_weatherRelDir)
        relDirCode=0
        if _weatherRelDir > 0 and _weatherRelDir <= 22.5:
            relDirCode = 0
        elif _weatherRelDir > 22.5 and _weatherRelDir <= 67.5:
            relDirCode = 1
        elif _weatherRelDir > 67.5 and _weatherRelDir <= 112.5:
            relDirCode = 2
        elif _weatherRelDir > 112.5 and _weatherRelDir <= 157.5:
            relDirCode = 3
        elif _weatherRelDir > 157.5 and _weatherRelDir <= 180:
            relDirCode = 4

        return relDirCode

    def GetAvgCons(self, _speed, _weatherMperS, _weatherRelDir, _draft,listOfSpeeds, listOfWeather, ListOfCons, ListOfDrafts, ConsProfileItem):


            _weatherMperS = _weatherMperS * 0.514
            minSpeed = np.min(listOfSpeeds)
            maxSpeed = np.max(listOfSpeeds)
            maxConsumption = np.max(ListOfCons)
            calcAvgCons = maxConsumption  # // Default for "Do not create edge" - to be divided by 24
            relDirCode = self.convertWindRelDirToRelDirIndex(_weatherRelDir) #// Find relative direction
            #relDirCode = 4
            exactSpeed = False
            exactWeather = False
            finalSpeedIndex = len(listOfSpeeds) - 1

            if _speed < minSpeed:
                _speed = minSpeed
            elif _speed > maxSpeed:
                _speed = maxSpeed

            # // Find draft index
            if _draft <= ListOfDrafts[1] - 1:
                currDraftIndex = 0
            else:
                currDraftIndex = 1

            # // Find where it is in the list of speeds
            curspeedIndex = 0
            maxLenSpeed = 4 if currDraftIndex == 0 else 8
            minLenSpeed = 0 if currDraftIndex == 0 else 4
            for i in range(minLenSpeed, maxLenSpeed):
                curspeedIndex = i
                if _speed > listOfSpeeds[i]:
                    d = 0
                elif _speed == listOfSpeeds[i]:
                    exactSpeed = True
                    break
                else:
                    break

            # // Find where it is in the list of weathers
            curweatherIndex = 0
            for i in range(0, len(listOfWeather)):
                curweatherIndex = i
                if _weatherMperS > listOfWeather[i]:
                    if i == len(listOfWeather) - 1:
                        exactWeather = True
                        break
                    # //else continue
                elif _weatherMperS == listOfWeather[i]:

                    exactWeather = True
                    break
                else:
                    if i == 0:
                        exactWeather = True  # // This is for 0 BFT
                    break

            if exactSpeed and exactWeather:

                hashKey = draft + "_" + listOfSpeeds[curspeedIndex] + "_" + listOfWeather[
                    curweatherIndex] + "_" + relDirCode
                cpi = []
                calcAvgCons = cpi.avgCons

            elif (exactSpeed):

                prevweatherIndex = curweatherIndex - 1
                # hashKey1 = draft + "_" + listOfSpeeds[curspeedIndex] + "_" + listOfWeather[prevweatherIndex] + "_" + relDirCode
                cpi1 = \
                [k for k in ConsProfileItem.values() if k['speed'] == curspeedIndex and k['ws'] == prevweatherIndex and
                 k['wd'] == relDirCode and k['draft'] == currDraftIndex][0]['foc']

                calcAvgConsPrev1 = cpi1  # .avgCons
                # hashKey2 = draft + "_" + listOfSpeeds[curspeedIndex] + "_" + listOfWeather[curweatherIndex] + "_" + relDirCode
                cpi2 = \
                [k for k in ConsProfileItem.values() if k['speed'] == curspeedIndex and k['ws'] == curweatherIndex and
                 k['wd'] == relDirCode and k['draft'] == currDraftIndex][0]['foc']
                calcAvgConsCur1 = cpi2  # .avgCons
                difInCons1 = calcAvgConsCur1 - calcAvgConsPrev1
                percWeatherDif = (math.pow(_weatherMperS, 3) - math.pow(listOfWeather[prevweatherIndex], 3)) / (
                            math.pow(listOfWeather[curweatherIndex], 3) - math.pow(listOfWeather[prevweatherIndex],
                                                                                   3))  # // Cubic interpolation
                calcAvgCons = calcAvgConsPrev1 + difInCons1 * percWeatherDif

            elif exactWeather:

                prevspeedIndex = curspeedIndex - 1
                # hashKey3 = draft + "_" + listOfSpeeds[prevspeedIndex] + "_" + listOfWeather[curweatherIndex] + "_" + relDirCode
                cpi3 = \
                [k for k in ConsProfileItem.values() if k['speed'] == prevspeedIndex and k['ws'] == curweatherIndex and
                 k['wd'] == relDirCode and k['draft'] == currDraftIndex][0]['foc']
                calcAvgConsPrev2 = cpi3  # .avgCons

                # hashKey4 = draft + "_" + listOfSpeeds[curspeedIndex] + "_" + listOfWeather[curweatherIndex] + "_" + relDirCode
                cpi4 = \
                [k for k in ConsProfileItem.values() if k['speed'] == curspeedIndex and k['ws'] == curweatherIndex and
                 k['wd'] == relDirCode and k['draft'] == currDraftIndex][0]['foc']
                calcAvgConsCur2 = cpi4  # .avgCons
                difInCons2 = calcAvgConsCur2 - calcAvgConsPrev2
                percSpeedDif = (math.pow(_speed, 3) - math.pow(listOfSpeeds[prevspeedIndex], 3)) / (
                            math.pow(listOfSpeeds[curspeedIndex], 3) - math.pow(listOfSpeeds[prevspeedIndex],
                                                                                3))  # // Cubic interpolation
                calcAvgCons = calcAvgConsPrev2 + difInCons2 * percSpeedDif  # // Linear interpolation

            else:
                flg = False
                prevweatherIndex = curweatherIndex - 1
                if currDraftIndex==1 and curspeedIndex==4:
                    prevspeedIndex = 3
                    currDraftIndex = 0
                    flg=True
                else: prevspeedIndex = curspeedIndex - 1
                # hashKey3 = draft + "_" + listOfSpeeds[prevspeedIndex] + "_" + listOfWeather[prevweatherIndex] + "_" + relDirCode
                cpi3=None
                try:
                    cpi3 = \
                    [k for k in ConsProfileItem.values() if k['speed'] == prevspeedIndex and k['ws'] == prevweatherIndex and
                     k['wd'] == relDirCode and k['draft'] == currDraftIndex][0]['foc']
                except:
                    t=0
                # cpi3 = []

                calcAvgConsPrev2 = cpi3  # .avgCons
                # hashKey4 = draft + "_" + listOfSpeeds[curspeedIndex] + "_" + listOfWeather[curweatherIndex] + "_" + relDirCode
                if flg==True:
                    currDraftIndex = 1

                try:
                    cpi4 = \
                    [k for k in ConsProfileItem.values() if k['speed'] == curspeedIndex and k['ws'] == curweatherIndex and
                     k['wd'] == relDirCode and k['draft'] == currDraftIndex][0]['foc']
                except:
                    t=0
                calcAvgConsCur2 = cpi4  # .avgCons
                difInCons2 = calcAvgConsCur2 - calcAvgConsPrev2
                percSpeedDif = (math.pow(_speed, 3) - math.pow(listOfSpeeds[prevspeedIndex], 3)) / (
                            math.pow(listOfSpeeds[curspeedIndex], 3) - math.pow(listOfSpeeds[prevspeedIndex],
                                                                                3))  # // Cubic interpolation
                calcAvgCons = calcAvgConsPrev2 + difInCons2 * percSpeedDif  # // Linear interpolation

            return calcAvgCons[0]

    def evaluatePavlosInterpolation(self, unseenX, unseenY, modeler,output,xs,genericModel,partitionsX , scores):

            lErrors = []
            foc=[]
            errorStwArr=[]
            errorFoc=[]
            listOfSpeeds, listOfWeather, ListOfCons, ListOfDrafts, ConsProfileItem = self.GetStatisticsOfVessel(
                'MARMARAS', 'MT_DELTA_MARIA')
            for iCnt in range(np.shape(unseenX)[0]):
                pPoint = unseenX[iCnt].reshape(1, -1)  # [0] # Convert to matrix
                trueVal = unseenY[iCnt]
                prediction =  self.GetAvgCons(pPoint[0][0], pPoint[0][3], pPoint[0][4], pPoint[0][1],
                                              listOfSpeeds, listOfWeather, ListOfCons, ListOfDrafts, ConsProfileItem)

                foc.append(trueVal)
                error = abs(prediction - trueVal)
                errorStwArr.append(
                    np.array(np.append(np.asmatrix(pPoint[0][0]).reshape(-1, 1), np.asmatrix([error]).T, axis=1)))
                errorFoc.append(abs((prediction - trueVal) / trueVal) * 100)

                lErrors.append(abs(prediction - trueVal))

            errorStwArr = np.array(errorStwArr)
            errorStwArr = errorStwArr.reshape(-1, 2)
            errors = np.asarray(lErrors)
            with open('./errorPercFOCPavlos' + str(len(partitionsX)) + '.csv', mode='w') as data:
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    ['FOC', 'PERC'])
                for i in range(0, len(errorFoc)):
                    data_writer.writerow(
                        [foc[i], errorFoc[i]])

            with open('./errorSTWPavlos' + str(len(partitionsX)) + '.csv', mode='w') as data:
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    ['STW', 'MAE'])
                for i in range(0, len(errorStwArr)):
                    data_writer.writerow(
                        [errorStwArr[i][0], errorStwArr[i][1]])

            return errors, np.mean(errors), np.std(lErrors)