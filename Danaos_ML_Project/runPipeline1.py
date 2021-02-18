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

    '''from datetime import datetime
    path = "/home/dimitris/Desktop/KaklisPHD/Danaos_ML_Project/data/OCEAN_GOLD/PERSEFONE/"
    telegrams = pd.read_csv(path+'/TELEGRAMS/PERSEFONE.csv',sep=';').values
    dataSet = []
    for infile in sorted(glob.glob(path + '*.csv')):
            data = pd.read_csv(infile, sep=',', decimal='.', skiprows=1)
            dataSet.append(data.values)
            print(str(infile))
        # if len(dataSet)>1:
    dataSet = np.concatenate(dataSet)
    dataSet[:,0] = list(map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f').replace(second=0, microsecond=0), dataSet[:,0]))
    trackReport = pd.read_csv('./data/OCEAN_GOLD/Persefone TrackReport.csv').values


    ### list of times for each day for track report
    json_decoded={'dates':[]}
    dts=[]
    for i in range(0, len(trackReport)):

        if trackReport[i, 7] == 'No data' or trackReport[i, 6] == 'Unknown': continue

        date = trackReport[i, 1]
        dt = date.split(" ")[0]
        time = date.split(" ")[1]
        day = dt.split(".")[0]
        month = dt.split(".")[1]
        year = dt.split(".")[2]
        hours = time.split(":")[0]
        minutes = int(time.split(":")[1])
        mod = minutes % 5
        minutes = minutes - mod if mod <= 2 else minutes + (mod - (mod - 1))
        minutes = str(minutes) if len(str(minutes)) > 1 else "0" + str(minutes)
        #######
        hours = str(int(hours) + 1) if minutes == '60' else hours
        hours = '00' if hours == '24' else hours
        minutes = '00' if minutes=='60' else minutes

        newTime = hours + ":" + minutes

        newDateTime = year + "-" + month + "-" + day + " " + newTime
        newDate = year + "-" + month + "-" + day
        try:
            dts.append(datetime.strptime(newDateTime, '%Y-%m-%d %H:%M'))
        except:
            print("exception")

        #speed = float(trackReport[i, 6].split(" ")[0])
        windSpeed = float(trackReport[i, 7].split("m/s")[0])
        swh = float(trackReport[i, 12].split("m")[0])
        wd = float(trackReport[i, 8].split("°")[0])
        #40° 07' 34.64"
        latDegrees = float(trackReport[i, 2].split("°")[0])
        latMinSec = trackReport[i, 2].split("°")[1]
        latMin = float(latMinSec.split("'")[0])
        latSec = float(latMinSec.split("'")[1])
        decimalLat = latDegrees + latMin /60 + latSec/3600


        timeItem = {'datetime':datetime.strptime(newDateTime, '%Y-%m-%d %H:%M'),'ws':windSpeed,'swh':swh,'wd':wd}
        json_decoded['dates'].append(timeItem)




    #with open('./data/OCEAN_GOLD/PENELOPE/' 'trackDates.json', 'w') as json_file:
        #json.dump(json_decoded, json_file)
    #return
    speeds = []
    focs = []
    windSpeeds = []
    swhs = []
    wds = []
    drafts=[]
    timestamps = []
    print(len(dataSet))
    for k in range(0, len(dataSet)):
        foc = -1
        print(k)
        dateTimeRaw = str(dataSet[k, 0])[:-3]
        dateRaw = str(dataSet[k, 0]).split(" ")[0]
        dtraw = datetime.strptime(dateTimeRaw, '%Y-%m-%d %H:%M')
        for i in range(0,len(trackReport)):

            if trackReport[i, 7] == 'No data' or trackReport[i, 6] == 'Unknown': continue

            date = trackReport[i,1]
            dt = date.split(" ")[0]
            time = date.split(" ")[1]
            day = dt.split(".")[0]
            month = dt.split(".")[1]
            year = dt.split(".")[2]
            hours = time.split(":")[0]
            minutes = int(time.split(":")[1])
            mod = minutes % 5
            minutes = minutes - mod if mod <=2 else minutes + (mod - (mod-1))
            minutes = str(minutes) if len(str(minutes))>1 else "0"+str(minutes)
            newTime = hours + ":" + minutes


            newDateTime = year + "-" + month + "-" + day + " " + newTime
            newDate = year + "-" + month + "-" + day



            if dateRaw == newDate:
                #print("found")
                keyDateTimeInTrackRep = min(dts, key=lambda x: abs(x - datetime.strptime(dateTimeRaw, '%Y-%m-%d %H:%M')))

                foc = dataSet[k, 10]
                stw=dataSet[k, 2]
                trackRepvalues = [l for l in json_decoded['dates'] if l['datetime'] == keyDateTimeInTrackRep][0]
                ws = trackRepvalues['ws']
                wd = trackRepvalues['wd']
                swh = trackRepvalues['swh']

                break

        if foc == -1 : continue
        draft = -1
        for n in range(0, len(telegrams)):
            dateTlg = str(telegrams[n, 0].split(" ")[0])
            if dateTlg == newDate.split(" ")[0]:
                draft = telegrams[n, 8]
                break

        speeds.append(stw)
        windSpeeds.append(ws)
        swhs.append(swh)
        wds.append(wd)
        timestamps.append(dateTimeRaw)
        focs.append(foc)
        ##map with tlgs for draft
        drafts.append(draft)


    df = {'speed':speeds,"ws":windSpeeds,"wd":wds,"swh":swhs,"foc":focs,'draft':drafts,'timestamp':timestamps}
    df = pd.DataFrame(df).to_csv("./data/OCEAN_GOLD/PENELOPE/trackRepMapped.csv",index=False)'''
    #return

    '''trdata = pd.read_csv('./data/GOLDENPORT/TRAMMO LAOURA/mappedData.csv').values
    with open('./data/TRAMMOCoor.csv', mode='w') as data:
                    data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    data_writer.writerow(['Latitude', 'Longitude'])
                    for i in range(0,len(trdata)):
                        data_writer.writerow(
                            [trdata[i][26], trdata[i][27]])

    df1 = pd.read_csv('./data/TRAMMOCoor.csv')

    plRes.PLotTrajectory(df1,'TRAMMO LAOURA ')'''


    DANreader = dRead.BaseSeriesReader()
    DANreader.GenericParserForDataExtraction('LEMAG', 'OCEAN_GOLD', 'PENELOPE', driver='ORACLE',
                                             server='10.2.5.80',
                                             sid='OR11', usr='oceangold', password='oceangold',
                                             rawData=False, telegrams=True, companyTelegrams=False,
                                             pathOfRawData='/home/dimitris/Desktop/SEEAMAG')

    return

    '''DANreader.GenericParserForDataExtraction('LEMAG', 'GOLDENPORT', 'TRAMMO LAOURA', driver='ORACLE',
                                             server='10.2.5.80',
                                             sid='OR12', usr='goldenport', password='goldenport',
                                             rawData=True, telegrams=True, companyTelegrams=False,
                                             pathOfRawData='/home/dimitris/Desktop/SEEAMAG')

    return'''

    indices = []
    data = pd.read_csv(sFile,sep=',').values
    #data=data[:2000]



    wfS = data[:, 11].astype(float) / (1.944)
    wsSen = []
    '''for i in range(0, len(wfS)):
        wsSen.append(gener.ConvertMSToBeaufort(float(float(wfS[i]))))
    data[:, 11] = wsSen'''

    '''df = pd.DataFrame({
        'wfSensor': wfS,
    })
    sns.displot(df, x="wfSensor")'''
    ########################################################################
    wfWS = data[:, 28]
    wfws = []
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

    def ConvertMSToBeaufort(ws):
        wsB = 0
        if ws >= 0 and ws < 0.2:
            wsB = 0
        elif ws >= 0.3 and ws < 1.5:
            wsB = 1
        elif ws >= 1.6 and ws < 3.3:
            wsB = 2
        elif ws >= 3.4 and ws < 5.4:
            wsB = 3
        elif ws >= 5.5 and ws < 7.9:
            wsB = 4

        elif ws >= 8 and ws < 10.7:
            wsB = 5
        elif ws >= 10.8 and ws < 13.8:
            wsB = 6
        elif ws >= 13.9 and ws < 17.1:
            wsB = 7
        elif ws >= 17.2 and ws < 20.7:
            wsB = 8
        elif ws >= 20.8 and ws < 24.4:
            wsB = 9
        elif ws >= 24.5 and ws < 28.4:
            wsB = 10
        elif ws >= 28.5 and ws < 32.6:
            wsB = 11
        elif ws >= 32.7:
            wsB = 12
        return wsB
    ###########PENELOPE
    sFile = './data/OCEAN_GOLD/PENELOPE/trackRepMappedPENELOPE.csv'
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
                       float(k[0]) >= 0 and float(k[2]) >= 0 and float(k[4]) >= 0 and float(k[3]) > 0 and float(
                           k[5]) > 0])

    ###################PENELOPE##################PENELOPE

    ''''#data =np.array([k for k in data if k[2]=='B' or k[2]=='L'])
    wfS = data[:, 11].astype(float) / (1.944)
    wsSen = []
    for i in range(0, len(wfS)):
        wsSen.append(gener.ConvertMSToBeaufort(float(float(wfS[i]))))
    data[:, 11] = wsSen

    data[:, 11] = wfS

    data[:, 15] = ((data[:, 15]) / 1000)* 1440
    ##################################################
    trData = np.array(np.append(data[:,8].reshape(-1,1),np.asmatrix([data[:,10],data[:,11],data[:,12],data[:,22],(data[:,15]),]).T,axis=1)).astype(float)#data[:,26],data[:,27]
    #trData = np.array(np.append(data[:, 0].reshape(-1, 1),
                                #np.asmatrix([data[:, 1], data[:, 3], data[:, 4], data[:, 5]]).T,
                                #axis=1)).astype(float)
    #np.array(np.append(data[:,0].reshape(-1,1),np.asmatrix([data[:,1],data[:,2],data[:,3],data[:,4],data[:,7],data[:,8],data[:,9]]).T,axis=1)).astype(float)
    #np.array(np.append(data[:,8].reshape(-1,1),np.asmatrix([data[:,10],data[:,11],data[:,12],data[:,20],data[:,21],data[:,15]]).T,axis=1)).astype(float)



    trData = np.array([k for k in trData if   float(k[2])>=0 and float(k[4])>=0 and (float(k[3])>10 and k[3]<=21) and float(k[5])>0  ])'''



    #plRes.PLotDists(trData)
    e=0
    #X_train, X_test, y_train, y_test = train_test_split(trData[:, 0:5], trData[:, 5:], test_size=0.26, random_state=42)
    trData1 =  trData[27000:86000, :]
    trData2 =  trData[86000:145115,:] #HAMBURG - MUMBAI - #HAMBURG
    '''with open('./data/EXPRESS ATHENSCoor1.csv', mode='w') as data:
            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow(['Latitude', 'Longitude'])
            for i in range(0,len(trData1)):
                data_writer.writerow(
                    [trData1[i][6], trData1[i][7]])

    with open('./data/EXPRESS ATHENSCoor2.csv', mode='w') as data:
            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow(['Latitude', 'Longitude'])
            for i in range(0,len(trData2)):
                data_writer.writerow(
                    [trData2[i][6], trData2[i][7]])'''

    df1 = pd.read_csv('./data/EXPRESS ATHENSCoor1.csv')
    df2 = pd.read_csv('./data/EXPRESS ATHENSCoor2.csv')
    #plRes.PLotTrajectory(df1,'EXPRESS ATHENS (2019-10-18) - (2019-12-11)')
    #plRes.PLotTrajectory(df2, 'EXPRESS ATHENS (2019-12-12) - (2020-02-05)')
    #return
    #trData = np.array([k for k in trData if k[0] >=6 and k[4] > 0])

        #np.array([k for k in trData if float(k[0])>5 and k[6]>0 and k[7]>0])

    '''meanFoc = np.mean(trData[:, :],axis=0)
    stdFoc = np.std(trData[:, :],axis=0)
    trData = np.array([k for k in trData if (k >= (meanFoc - (3 * stdFoc))).all() and (k <= (meanFoc + (3 * stdFoc))).all()])'''


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


    #for i in range(0, len(trData)):
        #trData[i] = np.mean(trData[i:i + 15], axis=0)

    X_train, X_test, y_train, y_test = train_test_split(trData[:, 0:5], trData[:, 5], test_size=0.1, random_state=42)

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
    df.to_csv('./data/DANAOS/EXPRESS ATHENS/SpeedTrain.csv',index=False)

    with open('./data/DANAOS/EXPRESS ATHENS/mappedDataTest.csv', mode='w') as data:
        data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #data_writer.writerow(['Latitude', 'Longitude'])
        for i in range(0, len(X_test)):
            data_writer.writerow(
                [X_test[i][0], X_test[i][1],X_test[i][2],X_test[i][3],X_test[i][4],y_test[i]])

    with open('./data/DANAOS/EXPRESS ATHENS/mappedDataTrain.csv', mode='w') as data:
        data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # data_writer.writerow(['Latitude', 'Longitude'])
        for i in range(0, len(X_train)):
            data_writer.writerow(
                [0, 0, 0, 0, 0, 0, 0, 0, X_train[i][0], 0,
                 X_train[i][1],
                 X_train[i][2], X_train[i][3], 0, 0, y_train[i], 0, 0,
                 0, 0,
                 X_train[i][4], 0, 0, 0, 0,
                 0,
                 0, 0])'''

    imo = "9379301"
    vessel = "PENELOPE"
    laddenJSON = '{}'
    json_decoded = json.loads(laddenJSON)
    json_decoded['ConsumptionProfile_Dataset'] = {"vessel_code": str(imo), 'vessel_name': vessel,
                                                  "dateCreated": date.today().strftime("%d/%m/%Y"), "data": []}
    for i in range(0, len(X_test)):
        item = {"draft": np.round(X_test[i][0], 2), 'stw': np.round(X_test[i][3], 2),
                "windBFT": float(np.round(X_test[i][2], 2)),
                "windDir": np.round(X_test[i][1], 2), "swell": np.round(X_test[i][4], 2),
                "cons": np.round(y_test[i], 2)}
        json_decoded['ConsumptionProfile_Dataset']["data"].append(item)
        # json_decoded['ConsumptionProfile']['consProfile'].append(outerItem)

    with open('./consProfileJSON/TestData_' + vessel + '_.json', 'w') as json_file:
        json.dump(json_decoded, json_file)


    return
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
    #unseenX = []
    #unseenY = []

    '''

    return'''
    '''trDataSwellImpact = np.array([k for k in trData if k[2] > 0 and k[2]<=2 and k[4] > 0])
    trDataWSImpact = np.array([k for k in trData if k[4] > 0 and k[4] <= 1.5 and k[2] > 0])
    trDataSTWImpact = np.array([k for k in trData if k[4] > 0 and k[4] <= 1.5 and k[2] > 0 and k[2]<=2])
    trDataWDImpact = np.array([k for k in trData if k[4] > 0 and k[4] <= 1.5 and k[2] >= 3 and k[3]>10])'''

    #trData = np.concatenate((trDataSwellImpact, trDataWSImpact, trDataSTWImpact, trDataWDImpact), axis=0)
    #X_train, y_train = trData[:,0:5] , trData[:,5]
    '''dm = dModel.BasePartitionModeler()
    currModeler = keras.models.load_model('./DeployedModels/estimatorCl_Gen.h5')

    #X_test,  y_test = trData[:, 0:5] , trData[:, 5]

    size = 0
    #X_train, X_test, y_train, y_test = trData[:len(trData)-size, 0:5],  trData[len(trData)-size:len(trData), 0:5], trData[:len(trData)-size, 5], trData[len(trData)-size:len(trData), 5]
    # train_test_split(trData[:, 0:5], trData[:, 5], test_size=0.05, random_state=42)
    # This is the size of our encoded representations
    #subsetsX.append(X_train.astype(float))
    #subsetsY.append(y_train.astype(float))

    unseenX = X_test.astype(float)
    unseenY = y_test.astype(float)'''


    '''_, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluateKerasNN1(
        eval.MeanAbsoluteErrorEvaluation(), unseenX,
        unseenY,
        currModeler, None, None, None, [1], None, 0, 'test')

    print("Mean absolute error on unseen data: %4.2f (+/- %4.2f standard error)" % (
    meanError, sdError / sqrt(unseenY.shape[0])))
    print("Evaluating on unseen data... Done.")

    #print("Standard Deviation of training Data: %4.2f" % (np.std(seriesX)))
    print("Standard Deviation of unseen Data: %4.2f" % (np.std(unseenX)))'''

    '''for i in range(1,len(data)):

        data[i,7] =np.abs( data[i,7] - data[i-1,7])'''

    '''data = np.append(data['stw'].values.reshape(-1,1),
                     np.asmatrix([
                    data['apparent_wind_speed'].values,
                    data['apparent_wind_angle'].values,
                    data['mid_draft'].values,
                    data['trim'].values,
                    data['rpm'].values]).T, axis=1)'''

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
                 partK=[9]
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

    sFile = './data/DANAOS/EXPRESS ATHENS/mappedDataNew.csv'
        #'./data/DANAOS/EXPRESS ATHENS/mappedData.csv'
        #"./neural_data/marmaras_data.csv"
    # Get file name
    history = 20
    future=30
    start = 10000
    end = 17000
    startU = 30000
    endU = 31000
    algs=['NNW']
    # ['SR','LR','RF','NN','NNW','TRI']


        #['SR','LR','RF','NN','NNW','TRI']
    cls=['KM']
    #['SR','LR','RF','NN'] algs
    #['KM','DC'] clusterers / cls

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
