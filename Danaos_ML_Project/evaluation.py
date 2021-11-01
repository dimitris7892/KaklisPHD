import numpy as np
import dataModeling as dt
import tensorflow as tf
import sklearn.ensemble as skl
#import statsmodels.api
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
#import latex
from numpy import array
from matplotlib import rc
from tensorflow import keras
import random
import seaborn as sns


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

    def evaluateKerasNNAvg(self, unseenX, unseenY, modeler, output, xs, genericModel, partitionsX, scores,subsetInd,type):
        lErrors = []
        self.intercepts = []
        predsXiBSpNN = []
        self.count = 0
        errorStwArr = []
        rpm = []
        errorRpm = []

        for iCnt in range(np.shape(unseenX)[0]):
            pPoint = unseenX[iCnt]
            pPoint = pPoint.reshape(-1, unseenX.shape[1])

            trueVal = unseenY[iCnt]

            # ind, fit = modeler.getBestPartitionForPoint(pPoint, partitionsX)

            # fits = modeler.getFitForEachPartitionForPoint(pPoint, partitionsX)

            preds = []
            if len(modeler._models) > 1:

                for n in range(0, len(partitionsX)):
                    self.intercepts = []
                    vector = self.extractFunctionsFromSplines(pPoint[0][0], pPoint[0][1],pPoint[0][2], pPoint[0][3], pPoint[0][4], n)
                    # XSplineVector = np.append(pPoint, vector)
                    # XSplineVector = XSplineVector.reshape(-1, XSplineVector.shape[ 0 ])

                    # XSplinevectorNew = np.array(self.intercepts) * vector
                    # XSplinevectorNew = np.array([i + self.interceptsGen for i in vector])

                    XSplinevectorNew = np.append(pPoint, vector)
                    XSplinevectorNew = XSplinevectorNew.reshape(-1, XSplinevectorNew.shape[0])
                    # try:
                    preds.append(modeler._models[n].predict(XSplinevectorNew)[0][0])
                    # except:
                    # d=0
                # weightedPreds.append(modeler._models[len(modeler._models)-1].predict(pPoint))
                #############################

                # prediction = np.average(preds ,weights=fits )
            # else:

            self.intercepts = []
            vector = self.extractFunctionsFromSplines(pPoint[0][0], pPoint[0][1], pPoint[0][2], pPoint[0][3],pPoint[0][4],'Gen')

            # XSplineGenvectorNew = np.array(self.intercepts) * vector
            # XSplineGenvectorNew = np.array([i + self.interceptsGen for i in vector])

            # XSplineGenvectorNew= np.sum(np.array(self.intercepts) * vector) + self.interceptsGen
            XSplineGenvectorNew = np.append(pPoint, vector)
            XSplineGenvectorNew = XSplineGenvectorNew.reshape(-1, XSplineGenvectorNew.shape[0])

            prediction = abs(modeler._models[len(modeler._models) - 1].predict(XSplineGenvectorNew))
            preds.append(prediction)

            # trainedWeights = genericModel
            if len(modeler._models) > 1:
                prediction = scores.predict(np.array(preds).reshape(-1, len(partitionsX) + 1))
                # prediction = (np.average(preds, weights=trainedWeights) )#+ prediction) / 2
            else:
                prediction = prediction

            lErrors.append(abs(prediction - trueVal))

            # lErrors.append(abs(prediction - trueVal))
            percError = abs((prediction - trueVal) / trueVal) * 100
            # errorStwArr.append(np.array(
            # np.append(np.asmatrix(pPoint[0][0]).reshape(-1, 1), np.asmatrix([percError[0]]).T, axis=1)))
            errorRpm.append(percError)
            rpm.append(trueVal)
            predsXiBSpNN.append(prediction[0][0])
            errorStwArr.append(pPoint[0][3])

            error = abs(prediction - trueVal)
            lErrors.append(error)

        errorStwArr = np.array(errorStwArr)
        #errorStwArr = errorStwArr.reshape(-1, 2)

        if type == 'train':
            with open('./TRAINerrorPercFOC' + str(len(partitionsX)) + '_' + str(subsetInd) + '.csv',
                      mode='w') as data:
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    ['FOC', 'PERC'])
                for i in range(0, len(errorFoc)):
                    data_writer.writerow(
                        [foc[i], errorFoc[i][0][0]])

            with open('./TRAINerrorSTW' + str(len(partitionsX)) + '_' + str(subsetInd) + '.csv', mode='w') as data:
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    ['STW', 'MAE'])
                for i in range(0, len(errorStwArr)):
                    data_writer.writerow(
                        [errorStwArr[i][0], errorStwArr[i][1]])
        else:
            with open('./TESTerrorPercFOCEnsemble' + str(len(partitionsX)) + '_' + str(0) + '.csv',
                      mode='w') as data:
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    ['FOC_PRED', 'FOC_ACT', 'PERC','STW'])
                for i in range(0, len(errorRpm)):
                    data_writer.writerow([predsXiBSpNN[i], rpm[i],  errorRpm[i][0][0],errorStwArr[i]])

        # prediction =modeler._models[ 0 ].predict(unseenX.reshape(2,2860))
        # print np.mean(abs(prediction - unseenY))
        # print("EXCEPTIONS :  "+str(count))
        print('Accuracy: ' + str(np.round(100 - np.mean(errorRpm), 2)))
        errors = np.asarray(lErrors)

        return errors, np.mean(errors), np.std(lErrors)




    def evaluateKerasNN1(self, unseenX, unseenY, modeler,output,xs,genericModel,partitionsX , scores,subsetInd,type, vessel, expansion):
        lErrors = []
        errorStwArr=[]
        errorFoc=[]
        foc=[]
        preds=[]
        candidatePoints=[]
        n_steps = 15
        self.intercepts = []
        self.count = 0



        ###############################################
        ####################################
        ###########################################
        #############
        if expansion == True:
            XSplineGenvectorNews = []
            for iCnt in range(np.shape(unseenX)[0]):
                pPoint = unseenX[iCnt]
                pPoint = pPoint.reshape(-1, unseenX.shape[1])

                # try:
                trueVal = unseenY[iCnt]

                self.intercepts = []
                vector = self.extractFunctionsFromSplines('Gen', pPoint[0][0], pPoint[0][1], pPoint[0][2], pPoint[0][3], pPoint[0][4], None, vessel)#pPoint[0][5]

                # XSplineGenvectorNew = np.array(self.intercepts) * vector
                # XSplineGenvectorNew = np.array([i + self.interceptsGen for i in XSplineGenvectorNew])

                # XSplineGenvectorNew = np.append(pPoint, vector)
                # XSplineGenvectorNews.append(XSplineGenvectorNew)
                #vectorNew = np.array([i + self.interceptsGen for i in vector])
                splineVector = np.append(pPoint, vector)
                #splineVector = pPoint

                XSplineGenvectorNews.append(np.append(splineVector, unseenY[iCnt]))
            raw_seq = np.array(XSplineGenvectorNews)
        else:
            raw_seq = np.array(np.append(unseenX, np.asmatrix([unseenY]).T, axis=1))
            # XSplineGenvectorNew = XSplineGenvectorNew.reshape(-1, XSplineGenvectorNew.shape[0])

        def split_sequence(sequence, n_steps):
            X, y = list(), list()
            for i in range(0,len(sequence)):
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


        # split into samples
        unseenXlstm, unseenYlstm = split_sequence(raw_seq, n_steps)

        if type =='test' : print("Shape of unseen data: " +str(unseenXlstm.shape))
        else: print("Shape of seen data: " +str(unseenXlstm.shape))

        for iCnt in range(np.shape(unseenXlstm)[0]):
            pPoint = unseenXlstm[iCnt]
            #pPoint =unseenX[iCnt]
            #pPoint= pPoint.reshape(-1,unseenX.shape[1])

            trueVal = unseenYlstm[iCnt]

            pPoint = np.reshape(pPoint, (1, pPoint.shape[0], pPoint.shape[1]))
            #print("  "+str(pPoint.shape))
            Genpred = modeler._models[len(modeler._models) - 1].predict(pPoint)


            prediction =  Genpred


            error = abs(prediction - trueVal)
            preds.append(prediction)
            lErrors.append(error)
            percError = (abs(prediction - trueVal) / trueVal) * 100
            if percError > 20 :
                r=0
            #errorStwArr.append(np.array(np.append(np.asmatrix(pPoint[0][0]).reshape(-1,1), np.asmatrix([percError[0] ]).T, axis=1)))
            errorFoc.append(percError)
            foc.append(trueVal)
            candidatePoints.append(pPoint)
            errorStwArr.append(pPoint[0][len(pPoint[0])-1][3])

        errorStwArr = np.array(errorStwArr)
        #errorStwArr = errorStwArr.reshape(-1, 2)
        errors = np.asarray(lErrors)

        if type == 'train':
            with open('./TRAINerrorPercFOC' + str(len(partitionsX)) + '_' + str(subsetInd) + '.csv', mode='w') as data:
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    ['FOC_pred','FOC_act' ,'PERC','stw','draft','trim','sensor_wind_speed','sensor_wind_dir','comb_waves_height','comb_waves_dir'])
                for i in range(0, len(errorFoc)):
                    data_writer.writerow(
                        [np.round( preds[i][0][0]), np.round( foc[i]), errorFoc[i][0][0]])#,candidatePoints[i][0][0], candidatePoints[i][0][1],candidatePoints[i][0][2],candidatePoints[i][0][3],candidatePoints[i][0][4],candidatePoints[i][0][5],candidatePoints[i][0][6]])

            with open('./TRAINerrorSTW' + str(len(partitionsX)) + '_' + str(subsetInd) + '.csv', mode='w') as data:
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    ['STW', 'MAE'])
                for i in range(0, len(errorStwArr)):
                    data_writer.writerow(
                        [errorStwArr[i][0], errorStwArr[i][1]])
        else:
            with open('./evalModelFiles/TESTerrorPercFOC' +'_'+ vessel +'.csv', mode='w') as data:
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    ['FOC_pred','FOC_act' ,'PERC','stw',])#'draft','trim','sensor_wind_speed','sensor_wind_dir','comb_waves_height','comb_waves_dir'
                for i in range(0, len(errorFoc)):
                    data_writer.writerow(
                        [np.round(preds[i][0][0],2),np.round(foc[i],2), errorFoc[i][0][0],errorStwArr[i]])#,candidatePoints[i][0][0], candidatePoints[i][0][1],candidatePoints[i][0][2],candidatePoints[i][0][3],candidatePoints[i][0][4],candidatePoints[i][0][5],candidatePoints[i][0][6]])


            '''with open('./TESTerrorSTW' + str(len(partitionsX)) + '_' + str(subsetInd) + '.csv', mode='w') as data:
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    ['STW', 'MAE'])
                for i in range(0, len(errorStwArr)):
                    data_writer.writerow(
                        [errorStwArr[i][0], errorStwArr[i][1]])'''


            dataErrorFoc = pd.read_csv('./evalModelFiles/TESTerrorPercFOC' +'_'+ vessel +'.csv', delimiter=',', skiprows=1)
            percError = dataErrorFoc.values

            meanPercError = np.mean(percError[:, 2])
            meanAcc = 100 - meanPercError
            print("Mean acc: " + str(np.round(meanAcc, 2)) + "% on test set of " + str(np.shape(unseenX)) + " observations")
            ####################################################################
            self.plotValidationAccuracy("DANAOS", vessel)

        return errors, np.mean(errors), np.std(lErrors)


    def plotValidationAccuracy(self, company, vessel,):

        dataErrorFoc = pd.read_csv('./evalModelFiles/TESTerrorPercFOC' +'_'+ vessel +'.csv',).values


        meanPercError = np.mean(dataErrorFoc[:, 2])
        meanAcc = np.round(100 - meanPercError, 2)

        minSpeed = np.min(dataErrorFoc[:, 3])
        i = np.floor(minSpeed)
        maxSpeed = np.ceil(np.max(dataErrorFoc[:, 3]))
        sizesSpeed = []
        speed = []
        avgActualFoc = []
        stdActualFoc = []
        avgPredFoc = []



        while i <= maxSpeed:

            speedArray = np.array([k for k in dataErrorFoc if float(k[3]) >= i - 0.25 and float(k[3]) <= i + 0.25])

            if len(speedArray) > 0:
                sizesSpeed.append(len(speedArray))
                speed.append(i)

                avgActualFoc.append(np.mean(speedArray[:, 1]) )
                avgPredFoc.append(np.mean(speedArray[:, 0]) )
                stdActualFoc.append(np.std(speedArray[:, 1]) )
            i += 0.5


        predDf_ = pd.DataFrame(
            {'speed': speed, 'pred_Foc': avgPredFoc, 'actual_Foc': avgActualFoc, 'size': sizesSpeed})

        predDf_['speed'] = [str(k) for k in predDf_['speed'].values]

        fig, ax1 = plt.subplots(figsize=(15, 10))
        color = 'tab:green'
        # bar plot creation
        y_pos = [0, 1, 5, 8, 9, 11, 12, 13, 14]
        bars = predDf_['size']
        # plt.xticks(y_pos, bars)

        ax1 = sns.barplot(x='speed', y='size', data=predDf_, palette='summer', ci="sd")
        ax1.set_xlabel('Speed Ranges (knots)', fontsize=16)
        ax1.set_ylabel('Size', fontsize=16)
        # change_width(ax1, 1.5)
        ax1.tick_params(axis='x')
        ax1.set_xticks(np.arange(9, 15, step=15.2))
        ax1.grid()
        # specify we want to share the same x-axis
        ax2 = ax1.twinx()
        color = 'tab:red'


        ax2 = sns.lineplot(x='speed', y='pred_Foc', data=predDf_, sort=False, color='blue',
                           label='neural Foc Acc: ' + str(meanAcc))
        ax2 = sns.lineplot(x='speed', y='actual_Foc', data=predDf_, sort=False, color='red', label='actual Foc')
        ax2.set_ylabel('Foc (MT/day)', fontsize=16)
        ax2.grid()
        plt.title("Accuracy on "+ "{:,.0f}".format((len(dataErrorFoc)))+" test data: ")

        plt.legend()
        plt.grid()
        plt.savefig('./Figures/' + company + '/' + vessel + '/evalPerf '+vessel+'.eps', format='eps')


    def evaluateKerasNN(self, unseenX, unseenY, modeler,output,xs,genericModel,partitionsX , scores,subsetInd,type):
        lErrors = []
        errorStwArr=[]
        foc=[]
        errorFoc=[]

        for iCnt in range(np.shape(unseenX)[0]):
                pPoint =unseenX[iCnt]
                pPoint= pPoint.reshape(-1,unseenX.shape[1])


                trueVal = unseenY[iCnt]

                ind, fit = modeler.getBestPartitionForPoint(pPoint, partitionsX)
                pPoint = np.reshape(pPoint, (pPoint.shape[0], pPoint.shape[1], 1))
                prediction = (abs(modeler._models[ind].predict(pPoint)) + modeler._models[len(modeler._models) - 1].predict(
                    pPoint)) / 2

                percError = abs((prediction - trueVal) / trueVal) * 100
                errorStwArr.append(np.array(
                    np.append(np.asmatrix(pPoint[0][0]).reshape(-1, 1), np.asmatrix([percError[0]]).T, axis=1)))
                errorFoc.append(percError)
                foc.append(trueVal)

                error = abs(prediction - trueVal)
                lErrors.append(error)

        errorStwArr = np.array(errorStwArr)
        errorStwArr = errorStwArr.reshape(-1, 2)


        if type == 'train':
                with open('./TRAINerrorPercFOC' + str(len(partitionsX)) + '_' + str(subsetInd) + '.csv',
                          mode='w') as data:
                    data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    data_writer.writerow(
                        ['FOC', 'PERC'])
                    for i in range(0, len(errorFoc)):
                        data_writer.writerow(
                            [foc[i], errorFoc[i][0][0]])

                with open('./TRAINerrorSTW' + str(len(partitionsX)) + '_' + str(subsetInd) + '.csv', mode='w') as data:
                    data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    data_writer.writerow(
                        ['STW', 'MAE'])
                    for i in range(0, len(errorStwArr)):
                        data_writer.writerow(
                            [errorStwArr[i][0], errorStwArr[i][1]])
        else:
                with open('./TESTerrorPercFOC_PERSEFONE' + str(len(partitionsX)) + '_' + str(subsetInd) + '.csv',
                          mode='w') as data:
                    data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    data_writer.writerow(
                        ['FOC', 'PERC'])
                    for i in range(0, len(errorFoc)):
                        data_writer.writerow(
                            [foc[i], errorFoc[i][0][0]])

                with open('./TESTerrorSTW' + str(len(partitionsX)) + '_' + str(subsetInd) + '.csv', mode='w') as data:
                    data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    data_writer.writerow(
                        ['STW', 'MAE'])
                    for i in range(0, len(errorStwArr)):
                        data_writer.writerow(
                            [errorStwArr[i][0], errorStwArr[i][1]])

        errors = np.asarray(lErrors)

        return errors, np.mean(errors), np.std(lErrors)

    def extractFunctionsFromSplines(self, modelId, x0, x1, x2, x3, x4, x5=None,vessel=None):
        piecewiseFunc = []
        self.count = self.count + 1
        csvModels = ['./trainedModels/model_' + str(modelId) + '_'+vessel+'.csv']
        for csvM in csvModels:
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
                                                piecewiseFunc.append(x1 * (nums[0] - x1) )#* float(row[1]))
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
                                        piecewiseFunc.append((nums[0] - x1) * (nums[1] - x0))# * float(row[1]))

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
                                        piecewiseFunc.append((x0 - nums[0]) * (x0 - nums[1]))# * float(row[1]))
                                    elif split == "x1":
                                        piecewiseFunc.append((x1 - nums[0]) * (x1 - nums[1]))# * float(row[1]))
                                    elif split == "x01":
                                        piecewiseFunc.append((x0 - nums[0]) * (x1 - nums[1]))# * float(row[1]))
                                    elif split == "x10":
                                        piecewiseFunc.append((x1 - nums[0]) * (x0 - nums[1]))# * float(row[1]))
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
                                                        (x0 - nums[0]) * (nums[1] - x0) )#* float(row[1]))
                                                elif split == "x1":
                                                    piecewiseFunc.append(
                                                        (x1 - nums[0]) * (nums[1] - x1))# * float(row[1]))
                                                elif split == "x01":
                                                    piecewiseFunc.append(
                                                        (x0 - nums[0]) * (nums[1] - x1) )#* float(row[1]))
                                                elif split == "x10":
                                                    piecewiseFunc.append(
                                                        (x1 - nums[0]) * (nums[1] - x0))# * float(row[1]))




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

                                                piecewiseFunc.append((x0 - nums[0]))# * float(row[1]))

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
                                                        (nums[0] - x0) * (x0 - nums[1]))# * float(row[1]))
                                                elif split == "x1":
                                                    piecewiseFunc.append(
                                                        (nums[0] - x1) * (x1 - nums[1]))# * float(row[1]))
                                                elif split == "x01":
                                                    piecewiseFunc.append(
                                                        (nums[0] - x0) * (x1 - nums[1]))# * float(row[1]))
                                                elif split == "x10":
                                                    piecewiseFunc.append(
                                                        (nums[0] - x1) * (x0 - nums[1]) )#* float(row[1]))
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
                                                    piecewiseFunc.append((x0 - nums[0]))# * float(row[1]))
                                                elif split == "x1":
                                                    piecewiseFunc.append((x1 - nums[0]))# * float(row[1]))
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

    def ANOVAtest(self,clusters,var,trError,error,models,partitioners):

        df = pd.DataFrame({
                            'clusters': clusters,
                            'var': var,
                            'error':error,
                            'Trerror': trError,
                            'partitioners':partitioners,
                            'models':models
                            #'meanBearing':trFeatures[1]
                            })
        groups=[error,clusters,var]


        #data=[error,np.var(unseenX),clusters]
        print(stats.kruskal(trError,error,clusters,var,models,partitioners))
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


    def GetStatisticsOfVessel(self, company, vessel,subsetInd):

            listOfWeatherBeaufort = np.array([0, 3, 5, 8])
            listOfWeather = [0, 4.34, 9, 34, 18.91]

            sFile = './data/' + company + '/' + vessel + '/ListOfSpeeds'+str(subsetInd+1)+'.csv'
            data = pd.read_csv(sFile, delimiter=',')
            listOfSpeeds = np.array(data.values)

            sFile = './data/' + company + '/' + vessel + '/ListOfCons'+str(subsetInd+1)+'.csv'
            data = pd.read_csv(sFile, delimiter=',')
            ListOfCons = np.array(data.values)

            sFile = './data/' + company + '/' + vessel + '/ListOfDrafts'+str(subsetInd+1)+'.csv'
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
                    if i == 0  or i ==4:
                        exactSpeed = True
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

            elif exactSpeed:

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

                prevweatherIndex = curweatherIndex - 1
                prevspeedIndex = curspeedIndex - 1

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
                #if flg==True:
                    #currDraftIndex = 1
                cpi4 = None
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

    def evaluatePavlosInterpolation(self, unseenX, unseenY, modeler,output,xs,genericModel,partitionsX , scores,subsetInd,type):

            lErrors = []
            foc=[]
            errorStwArr=[]
            errorFoc=[]
            preds=[]
            listOfSpeeds, listOfWeather, ListOfCons, ListOfDrafts, ConsProfileItem = self.GetStatisticsOfVessel(
                'MARMARAS', 'MT_DELTA_MARIA',subsetInd)
            for iCnt in range(np.shape(unseenX)[0]):
                pPoint = unseenX[iCnt].reshape(1, -1)  # [0] # Convert to matrix
                trueVal = unseenY[iCnt]
                prediction =  self.GetAvgCons(pPoint[0][0], pPoint[0][3], pPoint[0][4], pPoint[0][1],
                                              listOfSpeeds, listOfWeather, ListOfCons, ListOfDrafts, ConsProfileItem)

                foc.append(trueVal)
                error = abs(prediction - trueVal)
                percError = abs((prediction - trueVal) / trueVal) * 100
                errorStwArr.append(
                    np.array(np.append(np.asmatrix(pPoint[0][0]).reshape(-1, 1), np.asmatrix([percError]).T, axis=1)))
                errorFoc.append(abs((prediction - trueVal) / trueVal) * 100)
                preds.append(prediction)
                lErrors.append(abs(prediction - trueVal))

            errorStwArr = np.array(errorStwArr)
            errorStwArr = errorStwArr.reshape(-1, 2)
            errors = np.asarray(lErrors)
            if type=='train':
                with open('./TRAINerrorPercFOCPavlos' + str(subsetInd) + '.csv', mode='w') as data:
                    data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    data_writer.writerow(
                        ['FOC', 'PERC'])
                    for i in range(0, len(errorFoc)):
                        data_writer.writerow(
                            [preds[i],foc[i], errorFoc[i]])

                with open('./TRAINerrorSTWPavlos' + str(subsetInd) + '.csv', mode='w') as data:
                    data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    data_writer.writerow(
                        ['STW', 'MAE'])
                    for i in range(0, len(errorStwArr)):
                        data_writer.writerow(
                            [errorStwArr[i][0], errorStwArr[i][1]])
            else:
                with open('./TESTerrorPercFOCPavlos' + str(subsetInd) + '.csv', mode='w') as data:
                    data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    data_writer.writerow(
                        ['FOC', 'PERC'])
                    for i in range(0, len(errorFoc)):
                        data_writer.writerow(
                            [preds[i],foc[i], errorFoc[i]])

                with open('./TESTerrorSTWPavlos' + str(subsetInd) + '.csv', mode='w') as data:
                    data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    data_writer.writerow(
                        ['STW', 'MAE'])
                    for i in range(0, len(errorStwArr)):
                        data_writer.writerow(
                            [errorStwArr[i][0], errorStwArr[i][1]])

            return errors, np.mean(errors), np.std(lErrors)

    def fillExcelWithNeural(self):

        vector = np.array([[8.94,0,20,13.5,3,230]])
        vector1 = [8.94,40,1,8,1]
        vector2 = [8.94,90,1,8,1]
        vector3 =  [8.94,130,1,8,1]
        vector4 =  [8.94,167,1,8,1]

        csvM = './trainedModels/model_Gen_.csv'
        dataCsvModel = []
        dataCsvModels = []
        with open(csvM) as csv_file:
            data = csv.reader(csv_file, delimiter=',')
            for row in data:
                dataCsvModel.append(row)
        dataCsvModels.append(dataCsvModel)
        vector.reshape(-1, vector.shape[0])

        XSplineGenVector = self.extractFunctionsFromSplinesForExcel(vector[0][0], vector[0][1], vector[0][2], vector[0][3],
                                                     vector[0][4],None,None, 'Gen',dataCsvModels)
        XSplineGenVector = np.append(vector, XSplineGenVector)
        XSplineGenVector = np.array(XSplineGenVector).reshape(-1, 1)
        XSplineGenVector = np.reshape(XSplineGenVector, (XSplineGenVector.shape[1], XSplineGenVector.shape[0], 1))
        currModelerGen = keras.models.load_model('./DeployedModels/estimatorCl_Gen.h5')
        pred = currModelerGen.predict(XSplineGenVector)
        print(str(pred[0][0]))
        x=0

    def extractFunctionsFromSplinesForExcel(self, x0, x1, x2, x3, x4, x5, x6, modelId,dataCsvModels):
        piecewiseFunc = []
        # csvModels = ['../../../../../Danaos_ML_Project/trainedModels/model_' + str(modelId) + '_.csv']
        # for csvM in csvModels:
        # if csvM != '../../../../../Danaos_ML_Project/trainedModels/model_' + str(modelId) + '_.csv':
        # continue
        # id = csvM.split("_")[ 1 ]
        # piecewiseFunc = [ ]

        # with open(csvM) as csv_file:
        # data = csv.reader(csv_file, delimiter=',')
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

        return piecewiseFunc
