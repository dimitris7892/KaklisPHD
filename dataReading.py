#from pandas import Series
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp,chisquare,chi2_contingency
import math
import random
import csv
import datetime
from dateutil.rrule import rrule, DAILY, MINUTELY

class BaseSeriesReader:
    # Imports data from time series file. Expected format: CSV with 2 fields (velocity, RPM)
    def readSeriesDataFromFile(self, path,start,end,nrows):
        # Load file
        data = pd.read_csv(
            path)#,nrows=nrows)
        dt = data.values[ 0:, 3:23 ]
        if self.__class__.__name__=='UnseenSeriesReader':
            X = dt[ start:end, 1 ]
            Y = dt[ start:end, 4 ]
            W = dt[ 129000:132000,0]
        else:
            #X = dt[ :, 1 ]
            #Y = dt[ :, 4 ]
            X = dt[ start:end, 1 ]
            Y = dt[ start:end, 4 ]
            W = dt[ 100000:103000, 0]

        return np.nan_to_num(X.astype(float)),np.nan_to_num(Y.astype(float)) , np.nan_to_num(W.astype(float))

    def readLarosDAta(self,dtFrom,dtTo):

        draft={"data": [ ]}
        draftAFT = {"data": [ ]}
        draftFORE = {"data": [ ]}
        windSpeed={"data": [ ]}
        windAngle={"data": [ ]}
        SpeedOvg={"data": [ ]}
        STW = {"data": [ ]}
        rpm={"data": [ ]}
        power={"data": [ ]}
        foc={"data": [ ]}
        foct = {"data": [ ]}
        dateD={"data":[ ]}
        with open('./MT_DELTA_MARIA_2.txt') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            line_count = 0
            for row in csv_reader:
                if row[1]=="STATUS_PARAMETER_ID":
                    continue
                id=row[1]
                date = row[4].split(" ")[0]
                hhMMss =row[ 4 ].split(" ")[ 1 ]
                newDate=date +" "+ ":".join(hhMMss.split(":")[ 0:2 ])
                dt=datetime.datetime.strptime(newDate, '%Y-%m-%d %H:%M')
                value = row[3]
                siteId=row[2]

                datt = {}
                datt[ "value" ] = 0
                datt[ "dt" ] = dt
                dateD[ "data" ].append(datt)

                if id=='21' :
                    drftA={}
                    drftA["value"]=value
                    drftA["dt"]=dt
                    draftAFT[ "data" ].append(drftA)
                if id == '67':
                    stw = {}
                    stw[ "value" ] = value
                    stw[ "dt" ] = dt
                    STW[ "data" ].append(stw)

                if id=='22':
                    drftF={}
                    drftF["value"]=value
                    drftF["dt"]=dt
                    draftFORE[ "data" ].append(drftF)

                if id=='61' :
                    drft={}
                    drft["value"]=value
                    drft["dt"]=dt
                    draft[ "data" ].append(drft)

                if id == '81':
                    wa={}
                    wa[ "value" ] = value
                    wa[ "dt" ] = dt
                    windAngle[ "data" ].append(wa)

                if id == '82' :
                    ws={}
                    ws[ "value" ] = value
                    ws[ "dt" ] = dt
                    windSpeed[ "data" ].append(ws)

                if id == '84' :
                    so={}
                    so[ "value" ] = value
                    so[ "dt" ] = dt
                    SpeedOvg[ "data" ].append(so)

                if id == '89' :
                    rpms={}
                    rpms[ "value" ] = value
                    rpms[ "dt" ] = dt
                    rpm[ "data" ].append(rpms)

                if id == '137' :
                    fo={}
                    fo[ "value" ] = value
                    fo[ "dt" ] = dt
                    foc[ "data" ].append(fo)

                if id == '136' :
                    fot={}
                    fot[ "value" ] = value
                    fot[ "dt" ] = dt
                    foct[ "data" ].append(fot)


                if id == '353' :
                    pow={}
                    pow[ "value" ] = value
                    pow[ "dt" ] = dt
                    power[ "data" ].append(pow)

                line_count+=1
                if int(id) >353 : break
            print('Processed '+str(line_count)+ ' lines.')



        with open('./MT_DELTA_MARIA_data_2.csv', mode='w') as data:
            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow(['Draft','DrfAft', 'DrftFore','WindSpeed', 'WindAngle', 'SpeedOvg','STW','Rpm' ,'M/E FOC (kg/min)',
                                 'FOC Total', 'Power','DateTime' ])
            #data_writer.writerow(
                #['Draft','DateTime' ])

            for i in range(0,len(draft['data'])):
                    dt = dateD['data'][i]['dt']
                    value=""

                    value7 = ""
                    value8 = ""
                    value9 = ""
                    value10 = ""

                    value2=""
                    value1=""
                    value6=""
                    value5=""
                    value4=""
                    value3=""

                    try:
                        value = draft['data'][i]['value']
                    except:
                        x=0
                    try:
                        value7 = draftAFT[ 'data' ][ i ][ 'value' ]
                    except:
                        x = 0

                    try:
                        value8 = draftFORE[ 'data' ][ i ][ 'value' ]
                    except:
                        x = 0

                    try:
                        value = draft[ 'data' ][ i ][ 'value' ]
                    except:
                        x = 0

                    try:
                        value9 = STW[ 'data' ][ i ][ 'value' ]
                    except:
                        x = 0

                    try:
                        value10 = foct[ 'data' ][ i ][ 'value' ]
                    except:
                        x = 0

                    try:
                        value1 = windAngle['data'][i]['value']
                    except:
                        x=0
                    try:
                        value2 = windSpeed['data'][i]['value']
                    except:
                        x=0
                    try:
                        value3 = power['data'][i]['value']
                    except:
                        x=0
                    try:
                        value4 = foc['data'][i]['value']
                    except:
                        x=0
                    try:
                        value5 = rpm['data'][i]['value']
                    except:
                        x=0
                    try:
                        value6 = SpeedOvg['data'][i]['value']
                    except:
                        x=0
                    #####
                    data_writer.writerow([value,value7,value8,value2,value1,value6,value9,value5,value4,value10,value3,str(dt)])

    def readLarosDataFromCsvNew(self, data):
        # Load file
        from sklearn import preprocessing
        from sklearn.decomposition import PCA
        #if self.__class__.__name__ == 'UnseenSeriesReader':
            #dt = data.sample(n=2880).values[ :90000:, 3:23 ]
            #dt = data.values[ 0:, 2:23 ]
        #else:
            #dt = data.values[0:,2:23]
        #dataNew = data.drop(['M/E FOC (kg/min)', 'DateTime'], axis=1)
        #names = dataNew.columns


        dataNew=data.drop(['DateTime','SpeedOvg'],axis=1)
        #'M/E FOC (kg/min)'

        #dataNew=dataNew[['Draft', 'SpeedOvg']]
        dtNew = dataNew.values[ 0:, : ].astype(float)
        dtNew = np.delete(dtNew, [ i for (i, v) in enumerate(dtNew[ 0:, 0 ]) if v <= 0 ], 0)
        dtNew = np.delete(dtNew, [ i for (i, v) in enumerate(dtNew[ 0:, 5 ]) if v <= 8 or v > 14 ], 0)
        dtNew = np.delete(dtNew, [ i for (i, v) in enumerate(dtNew[ 0:, 6 ]) if v <= 0 or (v>0 and v <=10) or v > 90], 0)
        dtNew = np.delete(dtNew, [ i for (i, v) in enumerate(dtNew[ 0:, 7 ]) if v <= 7 ], 0)
        dtNew = dtNew[~np.isnan(dtNew).any(axis=1)]

        trim = (dtNew[:,1] - dtNew[:,2])
        dtNew1 =np.array(np.append(dtNew[:,0].reshape(-1,1),np.asmatrix([np.round(trim,3),dtNew[:,3],dtNew[:,4],dtNew[:,5],dtNew[:,7]]).T,axis=1))

        ##statistics STD
        #draftSmaller6 = np.array([drft for drft in dtNew1 if drft[0]< 6])
        #draftBigger6 = np.array([ drft for drft in dtNew1 if drft[0] > 6 ])
        #########DRAFT < 6
        stw8DrftS4 = np.array([i for i in dtNew1 if i[4]>=8 and i[4]<8.5 and i[0]<6])
        stw8DrftS4=stw8DrftS4[ abs(stw8DrftS4[ :, 5 ] - np.mean(stw8DrftS4[ :, 5 ])) < 2 * np.std(stw8DrftS4[ :, 5 ]) ]

        stw9DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 8.5 and i[ 4 ] < 9 and i[ 0 ] < 4 ])
        stw9DrftS4=stw9DrftS4[ abs(stw9DrftS4[ :, 5 ] - np.mean(stw9DrftS4[ :, 5 ])) < 2 * np.std(stw9DrftS4[ :, 5 ]) ]

        stw10DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 9 and i[ 4 ] < 9.5 and i[ 0 ] < 4 ])
        stw10DrftS4 = stw10DrftS4[abs(stw10DrftS4[ :, 5 ] - np.mean(stw10DrftS4[ :, 5 ])) < 2 * np.std(stw10DrftS4[ :, 5 ]) ]

        stw11DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 9.5 and i[ 4 ] < 10 and i[ 0 ] < 4 ])
        stw11DrftS4 = stw11DrftS4[
            abs(stw11DrftS4[ :, 5 ] - np.mean(stw11DrftS4[ :, 5 ])) < 2 * np.std(stw11DrftS4[ :, 5 ]) ]

        stw12DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 10 and i[ 4 ] < 10.5 and i[ 0 ] < 4 ])
        stw12DrftS4 = stw12DrftS4[
            abs(stw12DrftS4[ :, 5 ] - np.mean(stw12DrftS4[ :, 5 ])) < 2 * np.std(stw12DrftS4[ :, 5 ]) ]

        stw13DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 10.5 and i[ 4 ] < 11 and i[ 0 ] < 4 ])
        stw13DrftS4 = stw13DrftS4[
            abs(stw13DrftS4[ :, 5 ] - np.mean(stw13DrftS4[ :, 5 ])) < 2 * np.std(stw13DrftS4[ :, 5 ]) ]

        stw14DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 11 and i[ 4 ] < 11.5 and i[ 0 ] < 4 ])
        stw14DrftS4 = stw14DrftS4[
            abs(stw14DrftS4[ :, 5 ] - np.mean(stw14DrftS4[ :, 5 ])) < 2 * np.std(stw14DrftS4[ :, 5 ]) ]

        stw15DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 11.5 and i[ 4 ] < 12 and i[ 0 ] < 4 ])
        stw15DrftS4 = stw15DrftS4[
            abs(stw15DrftS4[ :, 5 ] - np.mean(stw15DrftS4[ :, 5 ])) < 2 * np.std(stw15DrftS4[ :, 5 ]) ]

        stw16DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 12 and i[ 4 ] < 12.5 and i[ 0 ] < 4 ])
        stw16DrftS4 = stw16DrftS4[
            abs(stw16DrftS4[ :, 5 ] - np.mean(stw16DrftS4[ :, 5 ])) < 2 * np.std(stw16DrftS4[ :, 5 ]) ]

        stw17DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 12.5 and i[ 4 ] < 13 and i[ 0 ] < 4 ])
        stw17DrftS4 = stw17DrftS4[
            abs(stw17DrftS4[ :, 5 ] - np.mean(stw17DrftS4[ :, 5 ])) < 2 * np.std(stw17DrftS4[ :, 5 ]) ]

        stw18DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 13 and i[ 4 ] < 13.5 and i[ 0 ] < 4 ])
        stw18DrftS4 = stw18DrftS4[
            abs(stw18DrftS4[ :, 5 ] - np.mean(stw18DrftS4[ :, 5 ])) < 2 * np.std(stw18DrftS4[ :, 5 ]) ]

        stw19DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 13.5 and i[ 4 ] < 14 and i[ 0 ] < 4 ])
        stw19DrftS4 = stw19DrftS4[
            abs(stw19DrftS4[ :, 5 ] - np.mean(stw19DrftS4[ :, 5 ])) < 2 * np.std(stw19DrftS4[ :, 5 ]) ]

        ##################################################################################
        ######################################################
        #########DRAFT > 6
        stw8DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 8 and i[ 4 ] < 8.5 and i[ 0 ] > 4 ])
        stw8DrftB4 = stw8DrftB4[
            abs(stw8DrftB4[ :, 5 ] - np.mean(stw8DrftB4[ :, 5 ])) < 2 * np.std(stw8DrftB4[ :, 5 ]) ]

        stw9DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 8.5 and i[ 4 ] < 9 and i[ 0 ] > 4 ])
        stw9DrftB4 = stw9DrftB4[
            abs(stw9DrftB4[ :, 5 ] - np.mean(stw9DrftB4[ :, 5 ])) < 2 * np.std(stw9DrftB4[ :, 5 ]) ]

        stw10DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 9 and i[ 4 ] < 9.5 and i[ 0 ] > 4 ])
        stw10DrftB4 = stw10DrftB4[
            abs(stw10DrftB4[ :, 5 ] - np.mean(stw10DrftB4[ :, 5 ])) < 2 * np.std(stw10DrftB4[ :, 5 ]) ]

        stw11DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 9.5 and i[ 4 ] < 10 and i[ 0 ] > 4 ])
        stw11DrftB4 = stw11DrftB4[
            abs(stw11DrftB4[ :, 5 ] - np.mean(stw11DrftB4[ :, 5 ])) < 2 * np.std(stw11DrftB4[ :, 5 ]) ]

        stw12DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 10 and i[ 4 ] < 10.5 and i[ 0 ] > 4 ])
        stw12DrftB4 = stw12DrftB4[
            abs(stw12DrftB4[ :, 5 ] - np.mean(stw12DrftB4[ :, 5 ])) < 2 * np.std(stw12DrftB4[ :, 5 ]) ]

        stw13DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 10.5 and i[ 4 ] < 11 and i[ 0 ] > 4 ])
        stw13DrftB4 = stw13DrftB4[
            abs(stw13DrftB4[ :, 5 ] - np.mean(stw13DrftB4[ :, 5 ])) < 2 * np.std(stw13DrftB4[ :, 5 ]) ]

        stw14DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 11 and i[ 4 ] < 11.5 and i[ 0 ] > 4 ])
        stw14DrftB4 = stw14DrftB4[
            abs(stw14DrftB4[ :, 5 ] - np.mean(stw14DrftB4[ :, 5 ])) < 2 * np.std(stw14DrftB4[ :, 5 ]) ]

        stw15DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 11.5 and i[ 4 ] < 12 and i[ 0 ] > 4 ])
        stw15DrftB4 = stw15DrftB4[
            abs(stw15DrftB4[ :, 5 ] - np.mean(stw15DrftB4[ :, 5 ])) < 2 * np.std(stw15DrftB4[ :, 5 ]) ]

        stw16DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 12 and i[ 4 ] < 12.5 and i[ 0 ] > 4 ])
        stw16DrftB4 = stw16DrftB4[
            abs(stw16DrftB4[ :, 5 ] - np.mean(stw16DrftB4[ :, 5 ])) < 2 * np.std(stw16DrftB4[ :, 5 ]) ]

        stw17DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 12.5 and i[ 4 ] < 13 and i[ 0 ] > 4 ])
        stw17DrftB4 = stw17DrftB4[
            abs(stw17DrftB4[ :, 5 ] - np.mean(stw17DrftB4[ :, 5 ])) < 2 * np.std(stw17DrftB4[ :, 5 ]) ]

        stw18DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 13 and i[ 4 ] < 13.5 and i[ 0 ] > 4 ])
        stw18DrftB4 = stw18DrftB4[
            abs(stw18DrftB4[ :, 5 ] - np.mean(stw18DrftB4[ :, 5 ])) < 2 * np.std(stw18DrftB4[ :, 5 ]) ]

        stw19DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 13.5 and i[ 4 ] < 14 and i[ 0 ] > 4 ])
        stw19DrftB4 = stw19DrftB4[
            abs(stw19DrftB4[ :, 5 ] - np.mean(stw19DrftB4[ :, 5 ])) < 2 * np.std(stw19DrftB4[ :, 5 ]) ]

        ################################################
        concatS = np.concatenate([stw8DrftS4,stw9DrftS4,stw10DrftS4,stw11DrftS4,stw12DrftS4,stw13DrftS4,stw14DrftS4,stw15DrftS4,stw16DrftS4,stw17DrftS4,stw18DrftS4,stw19DrftS4])
        concatB = np.concatenate(
            [ stw8DrftB4, stw9DrftB4, stw10DrftB4, stw11DrftB4, stw12DrftB4, stw13DrftB4, stw14DrftB4, stw15DrftB4,
              stw16DrftB4, stw17DrftB4, stw18DrftB4, stw19DrftB4 ])
        #dtNew1 = np.concatenate([concatB,concatS])
        #partitions = np.append([concatB,concatS])
        #stdFOC4_89 = np.std(stw8DrftS4[:,5])

        concatS =[ stw8DrftS4, stw9DrftS4, stw10DrftS4, stw11DrftS4, stw12DrftS4, stw13DrftS4, stw14DrftS4, stw15DrftS4,
              stw16DrftS4, stw17DrftS4, stw18DrftS4, stw19DrftS4 ]
        concatB = [ stw8DrftB4, stw9DrftB4, stw10DrftB4, stw11DrftB4, stw12DrftB4, stw13DrftB4, stw14DrftB4, stw15DrftB4,
              stw16DrftB4, stw17DrftB4, stw18DrftB4, stw19DrftB4 ]
        #concat =[ stw8DrftS4, stw9DrftS4, stw10DrftS4, stw11DrftS4, stw12DrftS4, stw13DrftS4, stw14DrftS4, stw15DrftS4,
         #stw16DrftS4, stw17DrftS4, stw18DrftS4, stw19DrftS4,
         #stw8DrftB4, stw9DrftB4, stw10DrftB4, stw11DrftB4, stw12DrftB4, stw13DrftB4, stw14DrftB4, stw15DrftB4,
        #stw16DrftB4, stw17DrftB4, stw18DrftB4, stw19DrftB4 ]

        #partitionsX=[k[:,0:5] for k in concat]
        #partitionsY = [ k[ :, 5 ] for k in concat ]
        #draftBigger9 = np.array([ drft for drft in dtNew1 if drft[ 0 ] > 9 ])

        #stw8Drft9 = np.array([ i for i in draftBigger9 if i[ 4 ] >= 12 and i[ 4 ] < 12.5 ])

        #stdFOC9_12 = np.std(stw8Drft9[ :, 5 ])

        seriesX = dtNew1[ :, 0:5 ]
        #seriesX=dtNew[:,0:5]
        #pca = PCA()
        #pca.fit(seriesX)
        #seriesX = pca.transform(seriesX)
        #scaler = preprocessing.StandardScaler()
        # Fit your data on the scaler object
        #scaled_df = scaler.fit_transform(seriesX)
        #seriesX = scaled_df
        #scaled_df = pd.DataFrame(scaled_df, columns=names)
        #dataNew = scaled_df

        #seriesX=seriesX[ 0:, 0:3 ]
        #seriesX=preprocessing.normalize([seriesX])
        newSeriesX = []
        #for x in seriesX:
            #dmWS = x[0] * x[1]
            #waSO = x[2] * x[3]
            #newSeriesX.append([x[0],x[1],x[2],x[3],np.round(dmWS,3),np.round(waSO,3)])
        #newSeriesX = np.array(newSeriesX)
        ##seriesX = newSeriesX


        UnseenSeriesX = dtNew1[140000:141000, 0:5]
        newUnseenSeriesX=[]
        #for x in UnseenSeriesX:
            #dmWS = x[0] * x[1]
            #waSO = x[2] * x[3]
            #newUnseenSeriesX.append([x[0], x[1], x[2], x[3], np.round(dmWS, 3), np.round(waSO, 3)])
        #newUnseenSeriesX = np.array(newUnseenSeriesX)
        #UnseenSeriesX = newUnseenSeriesX
        ##RPM
        #Rpm = dtNew[0:150000,4]
        #unseenRpm = dtNew[160000:161000,4]

        ##FOC
        FOC = dtNew1[ :, 5 ]
        #scaled_df = scaler.fit_transform(FOC)
        #FOC = scaled_df
        # normalized_Y = preprocessing.normalize([FOC])

        unseenFOC = dtNew1[140000:141000, 5 ]
        #WS= np.asarray([x for x in dt[ :, 1 ] ])
        #WA = np.asarray([ x for x in dt[ :, 2 ] ])
        #SO = np.asarray([ y for y in dt[ :, 3 ] ])
        #RPM = np.asarray([ y for y in dt[ :, 4 ] ])
        #FOC = np.asarray([ y for y in dt[ :, 5 ] ])
        #POW = np.asarray([ y for y in dt[ :, 6 ] ])
        #DA = np.asarray([ np.nan_to_num(np.float(y)) for y in dt[ :, 13 ] ])
        #DF = np.asarray([ y for y in dt[ :, 0 ] ])
        WaFoc=[]
        #newArray = np.append([WS,WA,SO,RPM,POW,DF])
        partionLabels=np.arange(0,24)
        return seriesX ,FOC , UnseenSeriesX , unseenFOC  ,None,None,None,None , None,None,partionLabels#draftBigger6[:,0:5],draftSmaller6[:,0:5] , draftBigger6[:,5] , draftSmaller6[:,5]

    def readLarosDataFromCsv(self, data):
        # Load file
        #if self.__class__.__name__ == 'UnseenSeriesReader':
            #dt = data.sample(n=2880).values[ :90000:, 3:23 ]
            #dt = data.values[ 0:, 2:23 ]
        #else:
            #dt = data.values[0:,2:23]
        dt = data.values[ 0:, 2:8 ].astype(float)
        dt = dt[~np.isnan(dt).any(axis=1)]
        WS= np.asarray([x for x in dt[ :, 0 ] ])
        WA = np.asarray([ x for x in dt[ :, 1 ] ])
        SO = np.asarray([ y for y in dt[ :, 3 ] ])
        RPM = np.asarray([ y for y in dt[ :, 4 ] ])
        FOC = np.asarray([ y for y in dt[ :, 5 ] ])
        POW = np.asarray([ y for y in dt[ :, 5 ] ])
        #DA = np.asarray([ np.nan_to_num(np.float(y)) for y in dt[ :, 13 ] ])
        DF = np.asarray([ np.nan_to_num(np.float(y)) for y in dt[ :, 6 ] ])
        WaFoc=[]
        for i in range(0,len(WA)):
            prev = np.append(WA[i], FOC[i])
            WaFoc.append(prev)

        WaFoc = np.array(WaFoc).reshape(-1, 2)
        #WaFoc = np.array(np.stack([WA,FOC],axis=0))
        firstFOC =[]
        secondFOC = []
        thirdFOC = []
        fourthFOC = []
        fifthFOC = []
        for val in WaFoc:
            if  val[0]<=22.5 or (val[0]<=360 and val[0]>337.5):
                firstFOC.append(val[1])
            elif (val[0] >22.5 and val[0] < 67.5) or (val[0] >292.5 and val[0] < 247.5):
                secondFOC.append(val[1])
            elif (val[0] > 67.5 and val[0] < 112.5) or (val[0] > 247.5 and val[0] < 292.5):
                thirdFOC.append(val[1])
            elif (val[0] > 112.5 and val[0] < 157.5) or (val[0] > 202.5 and val[0] < 247.5):
                fourthFOC.append(val[1])
            elif val[0] >157.5 and val[0] < 202.5:
                fifthFOC.append(val[1])

        ##extract FOC depending on wind angle
        import matplotlib.pyplot as plt
        #arraysToplot = np.array(np.stack([firstFOC,secondFOC],axis=0))
        df = pd.DataFrame(np.array(firstFOC),
                columns = [''])
        boxplot = df.boxplot(column=[''])
        plt.ylabel('M/E FOC (kg/min)')
        plt.title("STD and outliers of FOC with 0< WA <= 22.5")
        plt.show()
        df = pd.DataFrame(np.array(secondFOC),
                          columns=[''])
        boxplot1 = df.boxplot(
            column=[''])
        plt.ylabel('M/E FOC (kg/min)')
        plt.title("STD and outliers of FOC with WA > 22.5")
        plt.show()
        x=1


        #x = [ math.sin(Lon[ i ] - Lon[ i - 1 ]) * math.cos(Lat[ i ]) for i in range(1, len(Lat)) ]
        #y = [ math.cos(Lat[ i ]) * math.sin(Lat[ i - 1 ])
              #- math.sin(Lat[ i - 1 ]) * math.sin(Lat[ i ]) * math.cos(Lon[ i ] - Lon[ i - 1 ]) for i in
              #range(1, len(Lat)) ]
        #bearing = [ ]

        #bearing.append([ math.atan2(x[ i ], y[ i ]) for i in range(0, len(x)) ])
        #bearing[ 0 ].insert(0, 0)
        #bearing = np.asarray(bearing)[ 0 ]
        #WA = np.asarray([ np.nan_to_num(np.float(y)) for y in dt[ :, 5 ] ])
        return WS, WA

    def readRandomSeriesDataFromFileAF(self, data):
        # Load file
        #if self.__class__.__name__ == 'UnseenSeriesReader':
            #dt = data.sample(n=2880).values[ :90000:, 3:23 ]
            #dt = data.values[ 0:, 2:23 ]
        #else:
            #dt = data.values[0:,2:23]
        dt = data.values[ 0:, 2:23 ][ 0:2880 ]
        PW = np.asarray([ np.nan_to_num(np.float(x)) for x in dt[ :, 0 ] ])
        X = np.asarray([ np.nan_to_num(np.float(x)) for x in dt[ :, 2 ] ])
        Y = np.asarray([ np.nan_to_num(np.float(y)) for y in dt[ :, 5 ] ])
        Lat = np.asarray([ np.nan_to_num(np.float(y)) for y in dt[ :, 19 ] ])
        Lon = np.asarray([ np.nan_to_num(np.float(y)) for y in dt[ :, 20 ] ])
        WS = np.asarray([ np.nan_to_num(np.float(y)) for y in dt[ :, 1 ] ])
        DA = np.asarray([ np.nan_to_num(np.float(y)) for y in dt[ :, 13 ] ])
        DF = np.asarray([ np.nan_to_num(np.float(y)) for y in dt[ :, 14 ] ])

        x = [ math.sin(Lon[ i ] - Lon[ i - 1 ]) * math.cos(Lat[ i ]) for i in range(1, len(Lat)) ]
        y = [ math.cos(Lat[ i ]) * math.sin(Lat[ i - 1 ])
              - math.sin(Lat[ i - 1 ]) * math.sin(Lat[ i ]) * math.cos(Lon[ i ] - Lon[ i - 1 ]) for i in
              range(1, len(Lat)) ]
        bearing = [ ]

        bearing.append([ math.atan2(x[ i ], y[ i ]) for i in range(0, len(x)) ])
        bearing[ 0 ].insert(0, 0)
        bearing = np.asarray(bearing)[ 0 ]
        WA = np.asarray([ np.nan_to_num(np.float(y)) for y in dt[ :, 5 ] ])
        if  self.__class__.__name__ == 'UnseenSeriesReader':
            return X, Y, WS, bearing
        else:
            return PW, X , Y , WS , WA , DA , DF ,bearing

    def readRandomSeriesDataFromFile(self, data,k=None):
        # Load file
        if self.__class__.__name__ == 'UnseenSeriesReader':
            dt = data.values[ 0:, 2:23 ][ 81000:82000 ]
            #dt = data.sample(n=2880).values[ :90000:, 3:23 ]
            #dt = data.values[ 0:, 2:23 ]
        else:

            dt = data.values[ 0:, 2:23 ][k*20000:(k*20000 + 20000)]
            #[0:5000]
            #dt=np.array(random.sample(dt,20000))
            #dt = data.values[0:,2:23]

        #PW = np.asarray([ np.nan_to_num(np.float(x)) for x in dt[ :, 0 ] ])
        X = np.asarray([ np.nan_to_num(np.float(x)) for x in dt[ :, 2 ] ])
        Y = np.asarray([ np.nan_to_num(np.float(y)) for y in dt[ :, 5 ] ])
        Lat = np.asarray([ np.nan_to_num(np.float(y)) for y in dt[ :, 19 ] ])
        Lon = np.asarray([ np.nan_to_num(np.float(y)) for y in dt[ :, 20 ] ])
        WS = np.asarray([ np.nan_to_num(np.float(y)) for y in dt[ :, 1 ] ])
        #DA = np.asarray([ np.nan_to_num(np.float(y)) for y in dt[ :, 13 ] ])
        #DF = np.asarray([ np.nan_to_num(np.float(y)) for y in dt[ :, 14 ] ])

        x = [ math.sin(Lon[ i ] - Lon[ i - 1 ]) * math.cos(Lat[ i ]) for i in range(1, len(Lat)) ]
        y = [ math.cos(Lat[ i ]) * math.sin(Lat[ i - 1 ])
              - math.sin(Lat[ i - 1 ]) * math.sin(Lat[ i ]) * math.cos(Lon[ i ] - Lon[ i - 1 ]) for i in
              range(1, len(Lat)) ]
        bearing = [ ]

        bearing.append([ math.atan2(x[ i ], y[ i ]) for i in range(0, len(x)) ])
        bearing[ 0 ].insert(0, 0)
        bearing = np.asarray(bearing)[ 0 ]
        #WA = np.asarray([ np.nan_to_num(np.float(y)) for y in dt[ :, 5 ] ])

        return  X , Y , WS  ,bearing


    def readStatDifferentSubsets(self,data,subsetsX,subsetsY ,k,rows):


        X, Y, WS, bearing = self.readRandomSeriesDataFromFile(data,k)

        flag = False
        if np.std(X) >0 :
            candidateSetDist = X
            for i in range(0 ,len(subsetsX)):
                dataDist=ks_2samp(subsetsX[i],candidateSetDist)
                if dataDist[1] > 0.05:
                    flag = True
            if flag == False or k==0:
                return X,Y , WS , bearing
            else: return [],[],[],[]

class UnseenSeriesReader (BaseSeriesReader):
    # Imports data from time series file. Expected format: CSV with 2 fields (velocity, RPM)
   pass