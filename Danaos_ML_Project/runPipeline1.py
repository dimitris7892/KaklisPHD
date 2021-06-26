import glob, os
import dataReading as dRead
#from Danaos_ML_Project import dataReading as DANdRead ##   NEWWWW
import dataReading as DANdRead
import dataPartitioning as dPart
import dataModeling as dModel
import Danaos_ML_Project.evaluation as eval
import numpy as np
import json
import seaborn as sns
import datetime
from sklearn.cluster import KMeans
from math import sqrt
from tensorflow import keras
import sys
import generateProfile
import plotResults as plotRes
import itertools
import pandas as pd
import Danaos_ML_Project.ExtractNeuralReport  as NNrep
from sklearn.model_selection import train_test_split
import random
from sklearn.decomposition import PCA
from pylab import *
import datetime
import  matplotlib.pyplot as plt
from sklearn import svm
import csv
from datetime import date
import ctypes
import clr
import clr_loader
import os
import requests
import Danaos_ML_Project.plotResults as pltRes
import Danaos_ML_Project.generateProfile as genProf
from fastapi import APIRouter
import urllib.request
import asyncio
gener = genProf.BaseProfileGenerator()
#import latex
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)
from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
from tensorflow.keras import backend as K
#K.tensorflow_backend._get_available_gpus()
from datetime import datetime
import mappingData_functions as mpf

mapping = mpf.Mapping()

def main():

    end, endU, history, future,sFile, start, startU , algs , cls = initParameters()
    # Load data
    if len(sys.argv) >1:
        algs = algs.split(',')
        cls=cls.split(',')
        print(sFile , algs , cls)
        end = int(end)
        endU = int(endU)
        history = int(history)
        future = int(future)
        start = int(start)
        startU = int(startU)

    trSetlen=[500,1000,2000,3000,5000,10000,20000,30000,40000]
    errors=[]
    trErrors=[]
    var=[]
    clusters=[]
    cutoff=np.linspace(0.1,1,111)
    subsetsX=[]
    subsetsY = []
    subsetsB=[]

    '''sFile = './data/GOLDENPORT/TRAMMO LAOURA/mappedData.csv'
    data = pd.read_csv(sFile, sep=',').values


    ##Build training data array

    # draftD = data[:,8]
    # draftD = np.where(data[:,8]==0, None, data[:,8])
    # draftDf = pd.DataFrame({'draft':draftD.astype(float)})
    # draftDf = draftDf.interpolate()
    # data[:,8] = draftDf.values.reshape(-1)

    trData = np.array(np.append(data[:, 8].reshape(-1, 1), np.asmatrix(
        [data[:, 10], data[:, 11], data[:, 12], data[:, 22], (data[:, 15]), data[:, 26], data[:, 27], data[:, 2],
         data[:, 28]]).T, axis=1))  # .astype(float)#data[:,26],data[:,27]
    trData = np.array(
        [k for k in trData if float(k[2]) >= 0 and float(k[4]) >= 0 and float(k[3]) > 2 and float(k[5]) > 0])

    timestampsWithoutMappedTlgs = np.array([k for k in trData if str(k[8]) == 'nan'])[:, 9]
    print(timestampsWithoutMappedTlgs.shape)'''
    data = pd.read_csv('./data/DANAOS/HYUNDAI SMART/HYUNDAI SMART.csv').values
    originalDeptlgDate = datetime.strptime(originalDeptlgDate, '%Y-%m-%d %H:%M:%S')
    originalArrtlgDate = datetime.strptime(originalArrtlgDate, '%Y-%m-%d %H:%M:%S')

    print("Original Dep Date: " + str(originalDeptlgDate))
    print("Original Arr Date: " + str(originalArrtlgDate))
    rawDataLeg = np.array([k for k in rawData if
                           datetime.strptime(k[8], '%Y-%m-%d %H:%M') >= originalDeptlgDate
                           and datetime.strptime(k[8], '%Y-%m-%d %H:%M') <= originalArrtlgDate])

    plRes = pltRes.ErrorGraphs()

    # Sample size - in this case ~10%
    size = 5000
    modelers=[]
    for al in algs:
        if al=='SR' : modelers.append(dModel.SplineRegressionModeler())
        elif al=='LR':modelers.append(dModel.LinearRegressionModeler())
        elif al=='RF':modelers.append(dModel.RandomForestModeler())
        elif al=='NN' :modelers.append(dModel.TensorFlow())
        elif al=='TRI' : modelers.append(dModel.TriInterpolantModeler())
        elif al == 'NNW':modelers.append(dModel.TensorFlowW())
        elif al == 'NNWD':modelers.append(dModel.TensorFlowWD())
        elif al == 'NNW1':modelers.append(dModel.TensorFlowW1())
        elif al == 'NNW2':
            modelers.append(dModel.TensorFlowW2())
        elif al == 'NNW3':
            modelers.append(dModel.TensorFlowW3())
        elif al == 'NNWCA':modelers.append(dModel.TensorFlowCA())
        elif al == 'LI':
            modelers.append(dModel.PavlosInterpolation())

    partitioners=[]
    for cl in cls:
        if cl=='KM':partitioners.append(dPart.KMeansPartitioner())
        if cl == 'KMWSWA': partitioners.append(dPart.KMeansPartitionerWS_WA())
        if cl == 'KMWHWD': partitioners.append(dPart.KMeansPartitionerWH_WD())
        if cl=='DC' :partitioners.append(dPart.DelaunayTriPartitioner())
        if cl == 'NNCL': partitioners.append(dPart.TensorFlowCl())

    print(modelers)
    ###########################################################################

    #plRes.PlotTrueVsPredLine()

    #plRes.PLotTrajectory(df,'Express Athens')
    # 2nd place BALLAST FLAG
    # 8th place DRAFT
    # 10th place rel WD
    # 11th place WF
    # 12th place SPEED
    # 15th place ME FOC 24H
    # 16th place ME FOC 24H TLGS
    # 17th place TRIM
    # 19th place SteamHours
    # 18th place STW_TLG
    # 20st place swellSWH
    # 21st place  rel swellSW Dir
    # 26th place  lat
    # 27th place  lon
    # 28th place  wsWS

    # 0 draft
    # 1 wd
    # 2 wf
    # 3 stw
    # 4 trim
    # 5 swh
    # 6 swd
    # 7 foc (MT/day)

    #data = pd.read_csv('/home/dimitris/Desktop/EUROPE.csv')
    #print(data.keys())
    x=0

    '''trdata = pd.read_csv('./data/GOLDENPORT/TRAMMO LAOURA/mappedData.csv').values
    with open('./data/TRAMMOCoor.csv', mode='w') as data:
                    data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    data_writer.writerow(['Latitude', 'Longitude'])
                    for i in range(0,len(trdata)):
                        data_writer.writerow(
                            [trdata[i][26], trdata[i][27]])

    df1 = pd.read_csv('./data/TRAMMOCoor.csv')

    plRes.PLotTrajectory(df1,'TRAMMO LAOURA ')'''

    #import mappingData_functions as mpf
    #mapping = mpf.Mapping()
    #mapping.extractLegsFromTelegrams('OCEAN_GOLD','PERSEFONE')
    #return

    DANreader = dRead.BaseSeriesReader()
    DANreader.GenericParserForDataExtraction('LEMAG', 'DANAOS', 'GENOA', driver='ORACLE',
                                             server='10.2.5.80',
                                             sid='OR12', usr='shipping', password='shipping',
                                             rawData=False, telegrams=True, companyTelegrams=False,
                                             pathOfRawData='/home/dimitris/Desktop/SEEAMAG')

    return



    data = pd.read_csv(sFile,sep=',').values

    '''for i in range(0, len(wfS)):
        wsSen.append(gener.ConvertMSToBeaufort(float(float(wfS[i]))))
    data[:, 11] = wsSen'''

    '''df = pd.DataFrame({
        'wfSensor': wfS,
    })
    sns.displot(df, x="wfSensor")'''
    ########################################################################

    '''for i in range(0, len(wfS)):
        wf.append(gener.ConvertMSToBeaufort(float(float(wfWS[i]))))'''

    '''df = pd.DataFrame({
        'wfWeatherService': wfWS,
    })
    sns.displot(df, x="wfWeatherService")'''
    #plt.show()

    '''wfWS = data[:, 28]
    wfws = []
    for i in range(0, len(wfS)):
        wfws.append(gener.ConvertMSToBeaufort(float(float(wfWS[i]))))
    data[:, 28] = wfws'''

    '''i=0
    LAT = 49.31665
    LON = -4.08136
    datetimeV='2019-12-08 10:31:00'
    dateV = datetimeV.split(" ")[0]
    hhMMss = datetimeV.split(" ")[1]
    month = dateV.split("-")[1]
    day = dateV.split("-")[2]
    year = dateV.split("-")[0]
    month = '0' + month if month.__len__() == 1 else month
    day = '0' + day if day.__len__() == 1 else day
    newDate = year + '-' + month + '-' + day
    newDate1 = year + '-' + month + '-' + day + " " + ":".join(hhMMss.split(":")[0:2])
    windSpeed ,relWindDir ,swellSWH, relSwelldir , wavesSWH , relWavesdDir ,combSWH , relCombWavesdDir =\
        DANreader.mapWeatherData(
                        data[i,1], newDate1, np.round(LAT), np.round(LON))'''

    '''for i in range(0,len(data)):
        lat = data[i,26]
        lon = data[i, 27]
        is_in_ocean = globe.is_ocean(lat,lon)
        if is_in_ocean:
            indices.append(i)

    data = np.array([data[k] for k in indices])'''


    ###########PENELOPE
    '''sFile = './data/DANAOS/EUROPE/trackRepMappedPENELOPE.csv'
    data = pd.read_csv(sFile, sep=',').values
    print(data.shape)

    data[:, 4] = data[:, 4] * 24

    wd = data[:, 2]
    for i in range(0, len(wd)):
        if float(wd[i]) > 180:
            wd[i] = float(wd[i]) - 180  # and  float(k[8])<20

    wfS = data[:, 1]
    for i in range(0, len(wfS)):
        wfS[i] = ConvertMSToBeaufort(float(float(wfS[i])))
    data[:, 1] = wfS

    trData = np.array(np.append(data[:, 5].reshape(-1, 1), np.asmatrix([data[:, 2], data[:, 1], data[:, 0], data[:, 3], (data[:, 4])]).T,axis=1))  # .astype(float)#data[:,26],data[:,27]
    trData = np.array([k for k in trData if
                       float(k[0]) >= 0 and float(k[2]) >= 0 and float(k[4]) >= 0 and float(k[3]) > 5 and float(
                           k[5]) > 0])'''

    ###################PENELOPE##################PENELOPE

    #stwDEITECH, vslHeadingTD , currentSpeedTD, currentDirTD = calculate_stw_TEIDETECH(data,129)

    vslDir = data[:, 1]
    #currentDir = data[:, 30]
    #currentSpeedKnots = data[:, 29]  # * 1.943
    #sovg = data[:, 31]
    #stwCurr, vslHeading = calculateSTW(sovg, vslDir , currentSpeedKnots, currentDir)
    #stwCurr = np.array(stwCurr)

    #data =np.array([k for k in data if k[2]=='B' or k[2]=='L'])

    wfS = data[:, 11].astype(float) / (1.944)
    wsSen = []
    for i in range(0, len(wfS)):
        wsSen.append(gener.ConvertMSToBeaufort(float(float(wfS[i]))))
    data[:, 11] = wsSen

    #data[:, 11] = wfS

    data[:, 15] = ((data[:, 15]) / 1000)* 1440

    for i in range(0, len(data[:,10])):
        if float(data[i,10]) < 0:
            data[i,10] += 360

    for i in range(0, len(data[:,10])):
        if float(data[i,10]) > 180:
            data[i,10] = float(data[i,10]) - 180  # and  float(k[8])<20

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
    trData = np.array(np.append(data[:,8].reshape(-1,1), np.asmatrix([data[:,10], data[:,11], data[:,12], data[:,22],
                                           data[:,1] ,data[:,26],data[:,27],data[:,15]]).T,axis=1)).astype(float)#data[:,26],data[:,27]
    trData = np.nan_to_num(trData)
    #trData = np.array(np.append(data[:, 0].reshape(-1, 1),

                                #np.asmatrix([data[:, 1], data[:, 3], data[:, 4], data[:, 5]]).T,
                                #axis=1)).astype(float)
    #np.array(np.append(data[:,0].reshape(-1,1),np.asmatrix([data[:,1],data[:,2],data[:,3],data[:,4],data[:,7],data[:,8],data[:,9]]).T,axis=1)).astype(float)
    #np.array(np.append(data[:,8].reshape(-1,1),np.asmatrix([data[:,10],data[:,11],data[:,12],data[:,20],data[:,21],data[:,15]]).T,axis=1)).astype(float)



    trData = np.array([k for k in trData if  str(k[0])!='nan' and  float(k[2])>=0 and float(k[4])>=0 and (float(k[3])>=8 ) and float(k[8])>0  ]).astype(float)
    #trData = np.nan_to_num(trData)
    meanFoc = np.mean(trData[:, ], axis=0)
    stdFoc = np.std(trData[:, ], axis=0)
    trData = np.array(
        [k for k in trData if (k >= (meanFoc - (3 * stdFoc))).all() and (k <= (meanFoc + (3 * stdFoc))).all()])




    genprf = generateProfile.BaseProfileGenerator()
    trDataPorts , trDataNoInPorts = genprf.findCloseToLandDataPoints(trData)
    trData = trDataNoInPorts
        #np.concatenate([trDataNoInPorts,trDataPorts])
    #plRes.PLotDists(trData)
    e=0
    #X_train, X_test, y_train, y_test = train_test_split(trData[:, 0:5], trData[:, 5:], test_size=0.26, random_state=42)
    #trData1 =  trData[27000:86000, :]
    #trData2 =  trData[86000:145115,:] #HAMBURG - MUMBAI - #HAMBURG


    #for i in range(0, len(trData)):
        #trData[i] = np.mean(trData[i:i + 15], axis=0)

    '''wd = np.array([k for k in trData])[:, 1]
    for i in range(0, len(wd)):
        if float(wd[i]) > 180:
            wd[i] = float(wd[i]) - 180  # and  float(k[8])<20

    trData[:, 1] = wd'''

    '''wf = np.array([k for k in trData])[:, 2]
    for i in range(0, len(wf)):
        wf[i] = gener.ConvertMSToBeaufort(float(float(wf[i])))
    trData[:, 2] = wf'''


    #trData = trData[:44000]


    for i in range(0, len(trData)):
        trData[i] = np.mean(trData[i:i + 15], axis=0)

    #trData = trData[:500]
    X_train, X_test, y_train, y_test = train_test_split(trData[:, 0:8], trData[:, 8], test_size=0.1, random_state=42)


    mapping.writeTrainTestData('DANAOS','HYUNDAI SMART',X_test,y_test,X_train,y_train)
    #return

    mapping.extractJSON_TestData(9229312,'HYUNDAI SMART',X_test, y_test,)
    return

    #test = np.append(X_test,y_test.reshape(-1,1),axis=1)
    #df = pd.DataFrame(test,)
    #df.to_csv('./testRaw.csv',index=False,)

    '''train = np.append(X_train, y_train.reshape(-1, 1), axis=1)

    for i in range(0, len(train)):
        train[i] = np.mean(train[i:i + 15], axis=0)

    for i in range(0, len(test)):
        test[i] = np.mean(test[i:i + 15], axis=0)

    X_train, X_test, y_train, y_test = train[:,0:5] , test[:,0:5] , train[:,5] ,test[:,5]'''

    '''i = 8.75
    maxSpeed = np.max(X_train[:,3])
    sizesSpeed = []
    AvgactualFoc = []
    dt = X_train
    speed=[]
    #dt[:, :-1] = y_train
    dt = np.hstack((dt, y_train.reshape(-1,1)))
    while i <= maxSpeed:
        # workbook._sheets[sheet].insert_rows(k+27)

        speedArray = np.array([k for k in dt if float(k[3]) >= i and float(k[3]) <= i + 0.5])

        if speedArray.__len__() > 1:
            sizesSpeed.append(speedArray.__len__())
            speed.append(i+0.25)
            AvgActualFoc = np.mean(speedArray[:,5])
        i += 0.5

    d = {'Speed': speed, 'Size': sizesSpeed}
    df  = pd.DataFrame(d)
    df.to_csv('./data/DANAOS/EXPRESS ATHENS/SpeedTrain.csv',index=False)'''



    subsetsX.append(X_train.astype(float))
    subsetsY.append(y_train.astype(float))

    unseenX = X_test.astype(float)
    unseenY = y_test.astype(float)

    '''with open('./data/EXPRESS ATHENSCoorTrain.csv', mode='w') as data:
            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow(['Latitude', 'Longitude'])
            for i in range(0,1000):
                data_writer.writerow(
                    [y_train[i][1], y_train[i][2]])

    with open('./data/EXPRESS ATHENSCoorTest.csv', mode='w') as data:
            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow(['Latitude', 'Longitude'])
            for i in range(0,1000):
                data_writer.writerow(
                    [y_test[i][1], y_test[i][2]])'''



    print(str(len(X_train)))
    print(str(len(unseenX)))

    ##################################################

    print("Number of Statistically ind. subsets for training: " + str(len(subsetsX)))

    #K=[10]
    #rangeSubs = k
    stdInU = []
    varTr = []
    models = []
    K = range(1,25)
    part = []
    subsetInd = 0

    for subsetX, subsetY in zip(subsetsX, subsetsY):

      for modeler in modelers:
        for partitioner in partitioners:
           if partitioner.__class__.__name__=='DelaunayTriPartitioner':
                 partK=np.linspace(0.7,1,4)#[0.5]
                 #np.linspace(0.2,1,11)
                     #[0.6]
           if partitioner.__class__.__name__=='KMeansPartitioner' or partitioner.__class__.__name__=='KMeansPartitionerWS_WA'\
                    or partitioner.__class__.__name__=='KMeansPartitionerWH_WD':
               if modeler.__class__.__name__ == 'PavlosInterpolation':
                 partK = [1]
               elif modeler.__class__.__name__=='TriInterpolantModeler' or modeler.__class__.__name__ == 'TensorFlow':
                 partK =[1]
               else:
                 partK=[1]
           else:
               partK=[1]
           error = {"errors": []}
           #random.seed(1)

           flagEvalTri = False
           for k in partK:
                if modeler.__class__.__name__ == 'TensorFlowW' and K==1: continue

                #if  modeler.__class__.__name__ == 'TensorFlow' and k>1: continue
                print(modeler.__class__.__name__)
                print("Reading data...")
                reader = dRead.BaseSeriesReader()
                trSize=80000

                ####################################LAROS DATA STATISTICAL TESTS
                if modeler.__class__.__name__ == 'TensorFlowWD':
                    #reader.insertDataAtDb()
                    #reader.readNewDataset()
                    reader.readExtractNewDataset('MILLENIA')
                    #data = pd.read_csv('./MT_DELTA_MARIA_data_1.csv')
                    #reader.readLarosDataFromCsvNewExtractExcels(data)
                    seriesX, targetY,unseenFeaturesX, unseenFeaturesY  , drftB6 , drftS6 , drftTargetB6 , drftTargetS6, partitionsX, partitionsY,partitionLabels = reader.readLarosDataFromCsvNew(data)
                #################

                #seriesX, targetY ,targetW= reader.readStatDifferentSubsets(data,subsetsX,subsetsY,2880)
                if modeler.__class__.__name__ != 'TensorFlowWD':
                    seriesX, targetY, = subsetX,subsetY
                    '''dataset = np.array(np.append(seriesX, np.asmatrix([targetY]).T, axis=1))

                    for i in range(0, len(dataset)):
                        dataset[i] = np.mean(dataset[i:i + 10], axis=0)

                    seriesX = dataset[:, 0:7]
                    targetY = dataset[:, 7]'''
                counter=+1

                print("Reading data... Done.")

                # Extract features

                if modeler.__class__.__name__ == 'PavlosInterpolation':
                        k=1
                #partitionsX, partitionsY , partitionLabels=X,Y,W
                #if modeler.__class__.__name__!='TensorFlow':
                # Partition data
                print("Partitioning training set...")
                partitionsX, partitionsY = seriesX , targetY
                NUM_OF_CLUSTERS =k# TODO: Read from command line
                NUM_OF_FOLDS=6
                #if modeler!='TRI':

                if modeler.__class__.__name__ != 'TensorFlowWD':
                        partitionsX, partitionsY, partitionLabels, partitionRepresentatives, partitioningModel  , centroids  = partitioner.clustering(seriesX, targetY, None ,NUM_OF_CLUSTERS, True,k)
                else:
                       #partitionLabels=23

                        partitionsX, partitionsY, partitionLabels, partitionRepresentatives, partitioningModel, tri  = partitioner.clustering(

                            seriesX, targetY, None, NUM_OF_CLUSTERS, True, k)

                        #partitionsXDB6, partitionsYDB6, partitionLabels, partitionRepresentatives, partitioningModel, tri = partitioner.clustering(
                            #drftB6, targetY, None, 25, True, k)

                        #partitionsXDBS6, partitionsYDS6, partitionLabels, partitionRepresentatives, partitioningModel, tri = partitioner.clustering(
                            #drftS6, targetY, None, 25, True, k)

                print("Partitioning training set... Done.")
                # For each partition create model
                ''''for i in range(0,len(partitionsX)):
                    with open('./DeployedModels/cluster_' + str(i) + '_.csv', mode='w') as data:
                        data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        for k in range(0,len(partitionsX[i])):


                                data_writer.writerow(
                                    [partitionsX[i][k][0], partitionsX[i][0][1],partitionsX[i][k][2],partitionsX[i][k][3],partitionsX[i][k][4],partitionsX[i][k][5],partitionsX[i][k][6]])

                    with open('./DeployedModels/cluster_foc' + str(i) + '_.csv', mode='w') as data:
                        data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        for z in range(0, len(partitionsY[i])):

                            data_writer.writerow(
                                [partitionsY[i][z]])'''


                print("Creating models per partition...")


                #if modeler.__class__.__name__ == 'TensorFlowWD':
                    #X = np.array(np.concatenate(partitionsX))
                    #Y = np.array(np.concatenate(partitionsY))
                #skip_idx1 = random.sample(range(num_linesx, num_lines), (num_lines-num_linesx) - 1000)

                #modeler.plotRegressionLine(partitionsX, partitionsY, partitionLabels,genericModel,modelMap)
                # ...and keep them in a dict, connecting label to model
                #modelMap, xs, output, genericModel =None,None,None,None

                if modeler.__class__.__name__!= 'TriInterpolantModeler' and modeler.__class__.__name__!= 'PavlosInterpolation' :
                            #and modeler.__class__.__name__ != 'TensorFlow':
                    modelMap, history,scores, output,genericModel = modeler.createModelsFor(partitionsX, partitionsY, partitionLabels,None,seriesX,targetY)
                            #, genericModel , partitionsXDC
                    #if modeler.__class__.__name__ != 'TensorFlow':
                        #modelMap = dict(zip(partitionLabels, modelMap))
                    print("Creating models per partition... Done")

                    # Get unseen data

                    stdInU.append(np.std(unseenX))
                ##
                print("Reading unseen data... Done")

                # Predict and evaluate on seen data
                print("Evaluating on seen data...")
                '''if modeler.__class__.__name__ == 'TensorFlowW1':
                    _, meanErrorTr, sdError = eval.MeanAbsoluteErrorEvaluation.evaluateKerasNN1(
                        eval.MeanAbsoluteErrorEvaluation(), seriesX,
                        targetY,
                        modeler, output, None, None, partitionsX, scores,subsetInd,'train')
                elif modeler.__class__.__name__ == 'TensorFlowW3' or modeler.__class__.__name__ == 'TensorFlowW2':
                    _, meanErrorTr, sdError = eval.MeanAbsoluteErrorEvaluation.evaluateKerasNN(
                        eval.MeanAbsoluteErrorEvaluation(), seriesX,
                        targetY,
                        modeler, output, None, None, partitionsX, scores,subsetInd,'train')
                elif modeler.__class__.__name__ == 'PavlosInterpolation':
                    _, meanErrorTr, sdError = eval.MeanAbsoluteErrorEvaluation.evaluatePavlosInterpolation(
                        eval.MeanAbsoluteErrorEvaluation(), subsetX,
                        subsetY,
                        modeler, None, None, None, partitionsX, None, subsetInd,'train')

                print("Mean absolute error on seen data: %4.2f (+/- %4.2f standard error)" % (
                meanErrorTr, sdError / sqrt(unseenY.shape[0])))'''
                print("Evaluating on seen data... Done.")

                #Predict and evaluate on unseen data
                print("Evaluating on unseen data...")


                if modeler.__class__.__name__ == 'TensorFlowW1':
                    _, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluateKerasNN1(
                            eval.MeanAbsoluteErrorEvaluation(), unseenX,
                            unseenY,
                            modeler, output, None, None, partitionsX, scores,subsetInd,'test')
                elif modeler.__class__.__name__ == 'TensorFlowW' or modeler.__class__.__name__ == 'TensorFlowW2':
                    _, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluateKerasNNAvg(
                        eval.MeanAbsoluteErrorEvaluation(), unseenX,
                        unseenY,
                        modeler, output, None, None, partitionsX, genericModel,subsetInd,'test')
                elif modeler.__class__.__name__=='PavlosInterpolation':
                    _, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluatePavlosInterpolation(
                        eval.MeanAbsoluteErrorEvaluation(), unseenX,
                        unseenY,
                        modeler, None, None, None, partitionsX, None,subsetInd,'test')



                print ("Mean absolute error on unseen data: %4.2f (+/- %4.2f standard error)"%(meanError, sdError/sqrt(unseenY.shape[0])))
                print("Evaluating on unseen data... Done.")

                print("Standard Deviation of training Data: %4.2f" % (np.std(seriesX)))
                print("Standard Deviation of unseen Data: %4.2f" % (np.std(unseenX)))
                #varExpl=np.sum(np.square(subsetX-unseenX)) / np.square(np.sum(subsetX)-np.sum(unseenX))*100
                #print("Percentage of variance in Training data explained in Unseen Dataset: %4.2f" % (
                        #(varExpl)) + " %")
                # # Evaluate performance
                numOfclusters= len(partitionsX)

                #plotRes.ErrorGraphs.PlotModelConvergence(plotRes.ErrorGraphs(),len(seriesX),len(unseenX),history,numOfclusters,meanErrorTr,sdErrorTr,meanError,sdError)

                clusters.append(numOfclusters)
                varTr.append(np.var(subsetX))
                if modeler.__class__.__name__  != 'TriInterpolantModeler':
                    models.append(modeler.__class__.__name__)
                else:
                    models.append('Analytical method')
                part.append(partitioner.__class__.__name__)
                #meanVTr.append(np.mean(s ubsetX))
                #meanBTr.append(np.mean(X[:,2]))
                errors.append(meanError)
                trErrors.append(meanError)

                err={}
                err["model"] =modeler.__class__.__name__
                err["error"] = meanError
                err["k"]=k
                error["errors"].append(err)
                if modeler.__class__.__name__ == 'TriInterpolantModeler' and numOfclusters==1 or modeler.__class__.__name__ == 'TriInterpolantModeler' \
                        and partitioner.__class__.__name__ == 'DelaunayTriPartitioner':
                    break
                if partitioner.__class__.__name__ == 'DelaunayTriPartitioner' and numOfclusters==1:
                    break
      subsetInd=subsetInd+1


    eval.MeanAbsoluteErrorEvaluation.ANOVAtest(eval.MeanAbsoluteErrorEvaluation(), clusters, varTr, trErrors,errors,models,part)

def initParameters():


    sFile = './data/DANAOS/HYUNDAI SMART/mappedData.csv'

    # Get file name
    history = 20
    future=30
    start = 10000
    end = 17000
    startU = 30000
    endU = 31000
    algs=['NNW1']

    cls=['KM']


    if len(sys.argv) > 1:
        sFile = sys.argv[1]
        history = sys.argv[2]
        start = sys.argv[3]
        end = sys.argv[4]
        algs=sys.argv[5]
        cls=sys.argv[6]
    return end, endU, history, future,sFile, start, startU , algs , cls


# # ENTRY POINT
if __name__ == "__main__":
    main()
#     v=np.round(np.random.rand(10, 3), 1)
#     v=np.append(v,v[1:3], axis=0)
#     r= np.dot(np.sum(v, axis=1), np.diag(np.random.rand(v.shape[0])))
#     print("Input:\n%s"%(str(np.append(v, np.reshape(r, (v.shape[0],1)), axis=1))))
#
#     clusterer = dPart.BoundedProximityClusterer()
#     print(clusterer.clustering(v, r, showPlot=False))
#
