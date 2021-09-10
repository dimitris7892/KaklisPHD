import numpy as np
import pandas as pd
import json
from datetime import date
from sklearn.cluster import KMeans
import csv
import glob, os
from tensorflow import keras
import Danaos_ML_Project.dataModeling as dModel
import dataReading as dRead
import generateProfile as genProf
import extractLegs as exctrLegs
from contextlib import redirect_stdout

gener = genProf.BaseProfileGenerator()
dread = dRead.BaseSeriesReader()

class neuralDT:

    def __init__(self):

        self.dm = dModel.BasePartitionModeler()
        #self.currModeler = keras.models.load_model('./DeployedModels/estimator_HYUNDAI SMART 09_09_2021_13:41 3.54_4.0.h5')
        #self.currModeler = keras.models.load_model('./DeployedModels/estimator_GENOA 08_09_2021_12:13 1.7_3.0.h5')
        #self.currModeler = keras.models.load_model('./DeployedModels/estimatorExpandedInput_LEO C 08_09_2021_21:43 2.9_4.0.h5')
        #self.currModeler = keras.models.load_model('./DeployedModels/estimator_LEO C 08_09_2021_18:06 3.27_3.0.h5')
        #self.currModeler = keras.models.load_model('./DeployedModels/estimator_GENOA 09_09_2021_17:58 1.87_4.0.h5')
        self.currModeler = keras.models.load_model('./DeployedModels/estimator_GENOA 09_09_2021_21:49 1.51_3.0.h5')
        #self.currModeler = keras.models.load_model('./DeployedModels/estimator_GENOA 10_09_2021_12:00 1.58_2.0.h5')



    def evaluateNeuralOnPavlosJson(self, n_steps):

        path = './legsFromPavlos/'
        for infile in sorted(glob.glob(path + '*.json')):
            #leg = infile.split('/')[3]
            f = open(infile)
            data = json.load(f)
            for i in range(0,len(data)):

                lstmPoint = []
                stw = np.round(data[i]['stw'], 2)
                draft = data[i]['draft']
                focActual = data[i]['focActual']
                ws = data[i]['ws']
                wsBft =  gener.ConvertMSToBeaufort(ws)

                wd = data[i]['wd']
                swh = data[i]['swh']
                focStatDT = data[i]['focStatDT']
                focNeuralDT =  data[i]['focNeuralDT']
                focNeural =  data[i]['focNeural']

                # vslHead = data[i,9]

                wdVec = np.linspace(wd, wd, n_steps)
                wfVec = np.linspace(wsBft, wsBft, n_steps)
                swhVec = np.linspace(swh, swh, n_steps)
                stwVec = np.linspace(stw, stw, n_steps)
                # vslHead = np.linspace(vslHead, vslHead , n_steps)
                for k in np.arange(0, n_steps):
                    lstmPoint.append(np.array(
                        [draft, (wdVec[k]), wfVec[k],
                         stwVec[k], (swhVec[k])]))
                    # lstmPoint.append(np.array(
                    # [meanDraftLadden, (wd[k]), windF[startW]+ countW,
                    # stw[k], (swellH[startS])+ countS]))

                # lstmPoint.append(pPoint)
                lstmPoint = np.array(lstmPoint).reshape(n_steps, -1)

                XSplineVectors = []
                '''for j in range(0, len(lstmPoint)):
                    pPoint = lstmPoint[j]
                    vector, interceptsGen = self.dm.extractFunctionsFromSplines('Gen', pPoint[0], pPoint[1], pPoint[2],
                                                                                pPoint[3], pPoint[4], )#pPoint[5]
                    # vector = list(np.where(np.array(vector) < 0, 0, np.array(vector)))
                    # vector = ([abs(k) for k in vector])
                    XSplineVector = np.append(pPoint, vector)
                    XSplineVector = np.array(XSplineVector).reshape(1, -1)
                    XSplineVectors.append(XSplineVector)
                XSplineVectors = np.array(XSplineVectors).reshape(n_steps, -1)'''
                XSplineVectors = lstmPoint
                XSplineVector = XSplineVectors.reshape(1, XSplineVectors.shape[0], XSplineVectors.shape[1])
                # lstmPoint = np.array(lstmPoint).reshape(1, n_steps, -1 )
                cellValue = float(self.currModeler.predict(XSplineVector)[0][0])
                focPred = np.round((cellValue), 2)
                focPred = np.round(((focPred) / 1000) * 1440, 2)
                data[i]['focNeural'] = focPred

            with open(infile, "w") as jsonFile:
                json.dump(data, jsonFile)

    def fillDecisionTree(self, imo, vessel, dtNew, n_steps):

        #f = open('./consProfileJSON_Neural/consProfile_EXPRESS ATHENS_NeuralDT.json')
        #f = open('./legsFromPavlos/EXPRESS ATHENS_focEst_Leg_78.json')
        #jsonDict = json.load(f)

        draft = np.array([k for k in dtNew if float(k[0]) > 1 and float(k[0]) < 30])[:, 0].astype(float)

        meanDraft = np.mean(draft)
        minDraft = np.floor(np.min(draft))
        maxDraft = np.ceil(np.max(draft))
        stdDraft = np.std(draft)

        dataModel = KMeans(n_clusters=3)
        draft = draft.reshape(-1, 1)
        dataModel.fit(draft)
        # Extract centroid values
        centroids = dataModel.cluster_centers_
        draftsSorted = np.sort(centroids, axis=0)
        #draftsSorted = [8,11,14]

        wind = [22.5, 67.5, 112.5, 157.5, 180]
        #draftsSorted = np.arange(minDraft, maxDraft+1, 2)
        wind = np.arange(0, 195, 15)
        #wind = [1, 2, 3, 4, 5]
        windF = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        #swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        swellH = np.arange(0, 8.5, 0.5)
        consVelocitiesJSON = np.arange(10, 22.1, 0.1)
        consVelocitiesJSON = np.round(consVelocitiesJSON,1)

        laddenJSON = '{}'
        json_decoded = json.loads(laddenJSON)
        json_decoded['ConsumptionProfile'] = {"vessel_code": str(imo), 'vessel_name': vessel,
                                              "dateCreated": date.today().strftime("%d/%m/%Y"), "consProfilePORTS": [],
                                              "consProfile": []}
        for draft in draftsSorted:
            #draft = np.round(draft[0])
            for vel in range(0, len(consVelocitiesJSON)):

                outerItem = {"draft": draft, "speed": (consVelocitiesJSON[vel]), "cells": []}

                for w in range(0, len(windF)):

                    for s in range(0, len(swellH)):

                        for i in range(0, len(wind)-1):


                            lstmPoint = []
                            pPoint = np.array([draft, wind[i], windF[w], consVelocitiesJSON[vel], swellH[s]])
    
                            #lstmPoint.append(np.array(
                                    #[meanDraftLadden, (wind[i]), ((windF[w] + windF[w + 1]) / 2) - 1,
                                     #consVelocitiesJSON[vel], (swellH[s]) - 1]))
                            #windDir = wind[i]
                            windDirEval = np.round((wind[i] + wind[i + 1]) / 2, 2)
                            windDir = windDirEval

                            if windDir >= 0 and windDir < 15 :
                                windDir = 1
                            elif windDir >= 15 and windDir < 30:
                                windDir = 2
                            elif windDir >= 30 and windDir < 45:
                                windDir = 3
                            elif windDir >= 45 and windDir < 60:
                                windDir = 4
                            elif windDir >= 60 and windDir < 75:
                                windDir = 5
                            elif windDir >= 75 and windDir < 90 :
                                windDir = 6
                            elif windDir >= 90 and windDir < 105:
                                windDir = 7
                            elif windDir >= 105 and windDir < 120:
                                windDir = 8
                            elif windDir >= 120 and windDir < 135:
                                windDir = 9
                            elif windDir >= 135 and windDir < 150:
                                windDir = 10
                            elif windDir >= 150 and windDir < 165 :
                                windDir = 11
                            elif windDir >= 165 and windDir <= 180:
                                windDir = 12


                            wd = np.linspace(windDirEval, windDirEval, n_steps)
                            wf = np.linspace(windF[w], windF[w] , n_steps)
                            swh = np.linspace(swellH[s], swellH[s] , n_steps)
                            stw = np.linspace(consVelocitiesJSON[vel], consVelocitiesJSON[vel] , n_steps)
                            for k in np.arange(0,n_steps):
    
                                    lstmPoint.append(np.array(
                                        [draft, (wd[k]), wf[k],
                                         stw[k], (swh[k])]))
                                    #lstmPoint.append(np.array(
                                            #[meanDraftLadden, (wd[k]), windF[startW]+ countW,
                                             #stw[k], (swellH[startS])+ countS]))

                            #lstmPoint.append(pPoint)
                            lstmPoint=np.array(lstmPoint).reshape(n_steps,-1)
                            '''XSplineVectors=[]
                            for j in range(0,len(lstmPoint)):
                                    pPoint = lstmPoint[j]
                                    vector  , interceptsGen = self.dm.extractFunctionsFromSplines('Gen',pPoint[0], pPoint[1], pPoint[2], pPoint[3],pPoint[4],)
                                    #vector = list(np.where(np.array(vector) < 0, 0, np.array(vector)))
                                    #vector = ([abs(k) for k in vector])
                                    XSplineVector = np.append(pPoint, vector)
                                    XSplineVector = np.array(XSplineVector).reshape(1, -1)
                                    XSplineVectors.append(XSplineVector)
                            XSplineVectors = np.array(XSplineVectors).reshape(n_steps,-1)'''
                            XSplineVectors = lstmPoint
                            XSplineVector = XSplineVectors.reshape(1,XSplineVectors.shape[0], XSplineVectors.shape[1])
                            cellValue = float(self.currModeler.predict(XSplineVector)[0][0])
                            cellValue = np.round((cellValue),2)
                            # lt/min => MT/day
                            cellValue = np.round(((cellValue) / 1000) * 1440,2)

                            item = {"windBFT": w  , "windDir": windDir, "swell": s , "cons": cellValue }
                            outerItem['cells'].append(item)


                json_decoded['ConsumptionProfile']['consProfile'].append(outerItem)

        with open('./consProfileJSON_Neural/consProfile_'+vessel+'_NeuralDT.json', 'w') as json_file:
            json.dump(json_decoded, json_file)

    def evaluateNeuralDT(self, vessel, n_steps):

        dataSet = []
        path = './legs/' + vessel + '/'
        for infile in sorted(glob.glob(path + '*.csv')):
            preds = []
            lErrors = []
            percErrors = []
            print(str(infile))
            data = pd.read_csv(infile, sep=',', decimal='.', skiprows=0, ).values
            data = data[data[:,0]>=9]
            for i in range(0, len(data[:, 4])):
                if float(data[i, 4]) < 0:
                    data[:, 4] += 360
                data[:, 4] = dread.getRelativeDirectionWeatherVessel(float(data[i, 9]), float(data[i, 4]))

            for i in range(0, len(data[:, 4])):
                if float(data[i, 4]) > 180:
                    data[i, 4] = float(data[i, 4]) - 180

            for i in range(0,len(data)):
                foc = data[i, 2]
                stw = data[i, 0]
                if abs(data[i, 2] - data[i - 1, 2]) >= 10 and abs(data[i, 0] - data[i - 1, 0]) <= 0.5:
                    meanFocForSTW = np.mean(
                        np.array([k for k in data if k[0] >= stw - 0.25 and k[0] <= stw + 0.25])[:, 2])
                    print("foc " + str(data[i, 2]) + " value replaced with " + str(meanFocForSTW) + " for stw: " + str(
                        data[i, 0]))
                    # dataset[i, 3] = meanWS
                    data[i, 2] = meanFocForSTW
            '''meanFoc = np.mean(data[:, 2], axis=0)
            stdFoc = np.std(data[:, 2], axis=0)
            data = np.array(
                [k for k in data if (k[2] >= (meanFoc - (2 * stdFoc))) and (k[2] <= (meanFoc + (2 * stdFoc)))])'''

            data = data[:,:6]
            #data = np.array(np.append(data[:,:6],np.asmatrix([data[:,9]]).T,axis=1))
            for i in range(0, len(data)):
                data[i] = np.mean(data[i:i + 15], axis=0)

            for i in range(0,len(data)):

                lstmPoint = []
                stw = np.round(data[i,0],2)
                draft = data[i,1]
                focActual = data[i,2]
                ws = data[i,3]
                wd = data[i, 4]
                swh = data[i, 5]
                #vslHead = data[i,9]

                wd = np.linspace(wd, wd, n_steps)
                wf = np.linspace(ws, ws + 0.2, n_steps)
                swh = np.linspace(swh, swh + 0.2, n_steps)
                stw = np.linspace(stw, stw , n_steps)
                #vslHead = np.linspace(vslHead, vslHead , n_steps)
                for k in np.arange(0, n_steps):
                    lstmPoint.append(np.array(
                        [draft, (wd[k]), wf[k],
                         stw[k], (swh[k])]))
                    # lstmPoint.append(np.array(
                    # [meanDraftLadden, (wd[k]), windF[startW]+ countW,
                    # stw[k], (swellH[startS])+ countS]))

                # lstmPoint.append(pPoint)
                lstmPoint = np.array(lstmPoint).reshape(n_steps, -1)
                '''XSplineVectors = []
                for j in range(0, len(lstmPoint)):
                    pPoint = lstmPoint[j]
                    vector, interceptsGen = self.dm.extractFunctionsFromSplines('Gen', pPoint[0], pPoint[1], pPoint[2],
                                                                                pPoint[3], pPoint[4], )#pPoint[5]
                    # vector = list(np.where(np.array(vector) < 0, 0, np.array(vector)))
                    # vector = ([abs(k) for k in vector])
                    XSplineVector = np.append(pPoint, vector)
                    XSplineVector = np.array(XSplineVector).reshape(1, -1)
                    XSplineVectors.append(XSplineVector)
                XSplineVectors = np.array(XSplineVectors).reshape(n_steps, -1)'''
                XSplineVectors = lstmPoint
                XSplineVector = XSplineVectors.reshape(1, XSplineVectors.shape[0], XSplineVectors.shape[1])
                #lstmPoint = np.array(lstmPoint).reshape(1, n_steps, -1 )
                cellValue = float(self.currModeler.predict(XSplineVector)[0][0])
                focPred = np.round((cellValue), 2)

                error = abs(focActual - focPred)
                preds.append(focPred)
                lErrors.append(error)
                percError = (abs(focPred - focActual) / focActual) * 100
                percErrors.append(percError)

            meanPercError = np.mean(percErrors)
            meanABSError = np.mean(lErrors)
            stdABSError = np.std(lErrors)
            stdErrr = np.round(stdABSError / np.sqrt(np.shape(data)[0]),2)
            meanAcc = 100 - meanPercError

            print("Mean ABS Error: " + str(np.round(meanABSError, 2)) +  "+/-"+str(stdErrr)+" on test set of " + str(
                np.shape(data)) + " observations of route " + str(infile))

            print("Mean Accuracy: " + str(np.round(meanAcc, 2)) + "% on test set of " + str(
                np.shape(data)) + " observations of route "  + str(infile)+'\n')

    def scaleTrData(self, vessel):

        path = './correctedLegsForTR/' + vessel + '/'

        dataset = []
        for infile in sorted(glob.glob(path + '*.csv')):
            print(str(infile))
            leg = infile.split('/')[3]
            data = pd.read_csv(infile, sep=',', decimal='.', skiprows=0, ).values


            dataset.append(data)
        dataset = np.concatenate(dataset)
        dataset = np.array(dataset)

        xdata = np.array(np.append(dataset[:,0].reshape(-1,1), np.asmatrix([dataset[:,1], dataset[:,3], dataset[:,4], dataset[:,5]]).T, axis=1 ))
        Xscaler = MinMaxScaler()
        Xscaler.fit(xdata)

        ydata = dataset[:,2].reshape(-1,1)
        Yscaler = MinMaxScaler()
        Yscaler.fit(ydata)

        return Xscaler, Yscaler

    def writeLegs(self):

        path = './correctedLegsForTR/' + vessel + '/'
        # path = './legs/'+vessel+'/ANTWERP_LE HAVRE'

        for infile in sorted(glob.glob(path + '*.csv')):
            preds = []
            lErrors = []
            percErrors = []
            print(str(infile))
            data = pd.read_csv(infile, sep=',', decimal='.', skiprows=0, ).values
            data = data[data[:, 0] >= 10]

            for i in range(0, len(data[:, 4])):
                if float(data[i, 4]) < 0:
                    data[:, 4] += 360
                data[:, 4] = dread.getRelativeDirectionWeatherVessel(float(data[i, 6]), float(data[i, 4]))

            for i in range(0, len(data[:, 4])):
                if float(data[i, 4]) > 180:
                    data[i, 4] = float(data[i, 4]) - 180

            wfS = data[:, 3].astype(float) / (1.944)
            wsSen = []
            for i in range(0, len(wfS)):
                wsSen.append(gener.ConvertMSToBeaufort(float(float(wfS[i]))))
            data[:, 3] = wsSen

            #data = np.array(np.append(data[:, :6], np.asmatrix(data[:, 9]).T, axis=1))

            #for i in range(0, len(data)):
                #data[i] = np.mean(data[i:i + 15], axis=0)

    def evaluateNeuralDTOnFiltered(self, vessel, n_steps, expansion):

        dataSet = []
        path = './correctedLegsForTR/' + vessel + '/'
        #path = './legs/'+vessel+'/ANTWERP_LE HAVRE'
        with open('./sensAnalysis/' + vessel + '/errorStatistics.txt', 'w') as f:

            for infile in sorted(glob.glob(path + '*.csv')):
                preds = []
                lErrors = []
                percErrors = []
                focPreds = []
                leg = infile.split('/')[3]
                print(str(infile))

                #######################
                data = pd.read_csv(infile, sep=',', decimal='.', skiprows=0, ).values
                data = data[data[:, 0] >= 12]
                #data = data[data[:, 4].astype(float) > 0]

                for i in range(0, len(data[:, 4])):
                    if float(data[i, 4]) < 0:
                        data[i, 4] += 360
                    data[i, 4] = dread.getRelativeDirectionWeatherVessel(float(data[i, 9]), float(data[i, 4]))
                    data[i, 4] = np.round(data[i, 4], 2)

                for i in range(0, len(data[:, 4])):
                    if float(data[i, 4]) > 180:
                        data[i, 4] = float(data[i, 4]) - 180

                wfS = data[:, 3].astype(float)
                wsSen = []
                for i in range(0, len(wfS)):
                    wsSen.append(gener.ConvertMSToBeaufort(float(float(wfS[i]))))
                data[:, 3] = wsSen

                rollWindowData = np.array(np.append(data[:, :6], np.asmatrix(data[:, 9]).T, axis=1))

                for i in range(0, len(rollWindowData)):
                    rollWindowData[i] = np.mean(rollWindowData[i:i + 15], axis=0)

                for i in range(0,len(data)):

                    lstmPoint = []
                    stw = np.round(rollWindowData[i,0],2)
                    draft = rollWindowData[i,1]
                    focActual = rollWindowData[i,2]
                    ws = rollWindowData[i,3]
                    wd = rollWindowData[i, 4]
                    swh = rollWindowData[i, 5]
                    #vslHead = data[i,9]

                    wdVec = np.linspace(wd  , wd , n_steps)
                    wfVec = np.linspace(ws  , ws   , n_steps)
                    swhVec = np.linspace(swh  , swh , n_steps)
                    stwVec = np.linspace(stw  , stw   , n_steps)
                    #vslHead = np.linspace(vslHead, vslHead , n_steps)
                    for k in np.arange(0, n_steps):
                        lstmPoint.append(np.array(
                            [draft, (wdVec[k]), wfVec[k],
                             stwVec[k], (swhVec[k])]))
                        # lstmPoint.append(np.array(
                        # [meanDraftLadden, (wd[k]), windF[startW]+ countW,
                        # stw[k], (swellH[startS])+ countS]))

                    # lstmPoint.append(pPoint)
                    lstmPoint = np.array(lstmPoint).reshape(n_steps, -1)
                    if expansion == True:
                        XSplineVectors = []
                        for j in range(0, len(lstmPoint)):
                            pPoint = lstmPoint[j]
                            vector, interceptsGen = self.dm.extractFunctionsFromSplines('Gen', pPoint[0], pPoint[1], pPoint[2],
                                                                                        pPoint[3], pPoint[4], vessel= vessel )#pPoint[5]
                            # vector = list(np.where(np.array(vector) < 0, 0, np.array(vector)))
                            # vector = ([abs(k) for k in vector])
                            XSplineVector = np.append(pPoint, vector)
                            XSplineVector = np.array(XSplineVector).reshape(1, -1)
                            XSplineVectors.append(XSplineVector)

                        XSplineVectors = np.array(XSplineVectors).reshape(n_steps, -1)
                    else:
                        XSplineVectors = lstmPoint
                    XSplineVector = XSplineVectors.reshape(1, XSplineVectors.shape[0], XSplineVectors.shape[1])
                    #lstmPoint = np.array(lstmPoint).reshape(1, n_steps, -1 )
                    cellValue = float(self.currModeler.predict(XSplineVector)[0][0])
                    focPred = np.round((cellValue),2)

                    focPreds.append(focPred)


                    error = abs(focActual - focPred)
                    if error > 5:
                        x=0
                    preds.append(focPred)
                    lErrors.append(error)
                    percError = (abs(focPred - focActual) / focActual) * 100
                    percErrors.append(percError)


                meanPercError = np.mean(percErrors)
                meanABSError = np.mean(lErrors)
                stdABSError = np.std(lErrors)
                stdErrr = np.round(stdABSError / np.sqrt(np.shape(data)[0]),2)
                meanAcc = 100 - meanPercError

                print("Mean ABS Error: " + str(np.round(meanABSError, 2)) + "+/-" + str(
                    stdErrr) + " on test set of " + str(
                    np.shape(data)) + " observations of route " + str(infile))

                print("Mean Accuracy: " + str(np.round(meanAcc, 2)) + "% on test set of " + str(
                    np.shape(data)) + " observations of route " + str(infile) + '\n')

                with redirect_stdout(f):
                    print("Mean ABS Error: " + str(np.round(meanABSError, 2)) +  "+/-"+str(stdErrr)+" on test set of " + str(
                            np.shape(data)) + " observations of route " + str(infile))

                    print("Mean Accuracy: " + str(np.round(meanAcc, 2)) + "% on test set of " + str(
                            np.shape(data)) + " observations of route "  + str(infile)+'\n')


                with open('./sensAnalysis/'+vessel+'/preds_' +vessel +'_'+leg, mode='w') as wdata:
                    data_writer = csv.writer(wdata, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    data_writer.writerow(
                        [ 'stw','draft','ws','wd','swh','FOC_act','FOC_pred', 'dt'])
                    for i in range(0, len(data)):
                        data_writer.writerow(
                            [rollWindowData[i,0],rollWindowData[i,1], rollWindowData[i,3],rollWindowData[i,4],
                             rollWindowData[i,5], rollWindowData[i,2], focPreds[i], data[i,8]])

def loadJSONRoute():

    f = open('./tmpData/route1_4.json' )

    # returns JSON object as
    # a dictionary
    jsonDict = json.load(f)
    data = jsonDict['RouteS']['legs']['data']
    # Iterating through the json
    # list
    for i in data['emp_details']:
        print(i)

def load_data():

    genprf = genProf.BaseProfileGenerator()
    sFile = './data/DANAOS/EXPRESS ATHENS/mappedDataNew.csv'
    data = pd.read_csv(sFile).values
    wfS = data[:, 11].astype(float) / (1.944)
    wsSen = []
    for i in range(0, len(wfS)):
        wsSen.append(genprf.ConvertMSToBeaufort(float(float(wfS[i]))))
    data[:, 11] = wsSen

    # data[:, 11] = wfS

    data[:, 15] = ((data[:, 15]) / 1000) * 1440

    for i in range(0, len(data[:, 10])):
        if float(data[i, 10]) < 0:
            data[i, 10] += 360

    for i in range(0, len(data[:, 10])):
        if float(data[i, 10]) > 180:
            data[i, 10] = float(data[i, 10]) - 180  # and  float(k[8])<20

    '''for i in range(0, len(data[:,1])):
        if float(data[i,1]) > 180:
            data[i,1] = float(data[i,1]) - 180'''

    '''for i in range(0, len(vslHeadingTD)):
        if float(vslHeadingTD[i]) > 180:
            vslHeadingTD[i] = float(vslHeadingTD[i]) - 180'''

    '''for i in range(0, len(currentDirTD)):
        if float(currentDirTD[i]) > 180:
            currentDirTD[i] = float(currentDirTD[i]) - 180'''

    '''for i in range(0, len(data[:,30])):
        if float(data[i,30]) > 180:
            data[i,30] = float(data[i,30]) - 180'''
    ##################################################
    trData = np.array(
        np.append(data[:, 8].reshape(-1, 1), np.asmatrix([data[:, 10], data[:, 11], data[:, 12], data[:, 22],
                                                          data[:, 1], data[:, 26], data[:, 27], data[:, 15]]).T,
                  axis=1)).astype(float)  # data[:,26],data[:,27]
    # trData = np.nan_to_num(trData)
    # trData = np.array(np.append(data[:, 0].reshape(-1, 1),

    # np.asmatrix([data[:, 1], data[:, 3], data[:, 4], data[:, 5]]).T,
    # axis=1)).astype(float)
    # np.array(np.append(data[:,0].reshape(-1,1),np.asmatrix([data[:,1],data[:,2],data[:,3],data[:,4],data[:,7],data[:,8],data[:,9]]).T,axis=1)).astype(float)
    # np.array(np.append(data[:,8].reshape(-1,1),np.asmatrix([data[:,10],data[:,11],data[:,12],data[:,20],data[:,21],data[:,15]]).T,axis=1)).astype(float)

    meanFoc = np.mean(trData[:, ], axis=0)
    stdFoc = np.std(trData[:, ], axis=0)
    trData = np.array(
        [k for k in trData if (k >= (meanFoc - (3 * stdFoc))).all() and (k <= (meanFoc + (3 * stdFoc))).all()])

    trData = np.array([k for k in trData if
                       str(k[0]) != 'nan' and float(k[2]) >= 0 and float(k[4]) >= 0 and (float(k[3]) >= 9) and float(
                           k[8]) > 0]).astype(float)
    # trData = np.nan_to_num(trData)


    trDataPorts, trDataNoInPorts = genprf.findCloseToLandDataPoints(trData)
    trData = trDataNoInPorts

    # trData  = np.array(np.append(trData[:,0:6],np.asmatrix(trData[:,8]).T,axis=1))
    #######################################################################################
    # np.concatenate([trDataNoInPorts,trDataPorts])
    # plRes.PLotDists(trData)
    e = 0
    # X_train, X_test, y_train, y_test = train_test_split(trData[:, 0:5], trData[:, 5:], test_size=0.26, random_state=42)
    # trData1 =  trData[27000:86000, :]
    # trData2 =  trData[86000:145115,:] #HAMBURG - MUMBAI - #HAMBURG

    # for i in range(0, len(trData)):
    # trData[i] = np.mean(trData[i:i + 15], axis=0)

    '''wd = np.array([k for k in trData])[:, 1]
    for i in range(0, len(wd)):
        if float(wd[i]) > 180:
            wd[i] = float(wd[i]) - 180  # and  float(k[8])<20

    trData[:, 1] = wd'''

    '''wf = np.array([k for k in trData])[:, 2]
    for i in range(0, len(wf)):
        wf[i] = gener.ConvertMSToBeaufort(float(float(wf[i])))
    trData[:, 2] = wf'''

    # trData = trData[:44000]

    for i in range(0, len(trData)):
        trData[i] = np.mean(trData[i:i + 15], axis=0)


    np.savetxt("./tmpData/trData.csv", trData, delimiter=",")

    return  trData

def main():


    ndt = neuralDT()

    #ndt.evaluateNeuralOnPavlosJson(10)

    #ndt.extractLegsToJsonWithRollWindow("EXPRESS ATHENS", "9484948")
    #ndt.evaluateNeuralDT('EXPRESS ATHENS', 1)
    ndt.evaluateNeuralDTOnFiltered('GENOA', 10, False)
    #return

    #trData = pd.read_csv('./tmpData/trData.csv',delimiter=',').values
    #ndt.fillDecisionTree(9484948, "EXPRESS ATHENS", trData, 10)

if __name__ == "__main__":
    main()
