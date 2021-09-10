import glob, os
from scipy import stats
import numpy as np
import dataReading
import pandas as pd
from pylab import *
from datetime import date, datetime
import generateProfile as genProf
import json
import csv
import seaborn as sns
gener = genProf.BaseProfileGenerator()
dread = dataReading.BaseSeriesReader()

class preProcessLegs:

    def __init__(self, knotsFlag):

        self.knotsFLag = knotsFlag
    
    def extractLegsFromRawCleanedData(self, vessel, dataType):

        wsIndSensor = 11 if dataType == 'raw' else 3
        wsIndWS = 28 if dataType == 'raw' else 10
        focInd = 15 if dataType == 'raw' else 2
        stwInd = 12 if dataType == 'raw' else 0
        draftInd = 8 if dataType == 'raw' else 1
        wdInd = 10 if dataType == 'raw' else 4
        swhInd = 20 if dataType == 'raw' else 5
        dtInd = 0 if dataType == 'raw' else 8
        bearInd = 1 if dataType == 'raw' else 9
        latInd = 26 if dataType == 'raw' else 6
        lonInd = 27 if dataType == 'raw' else 7


        dataset = pd.read_csv('./consProfileJSON_Neural/cleaned_'+dataType+'_'+vessel+'.csv').values

        path = './legs/' + vessel + '/'
        datasetLegs = []
        for infile in sorted(glob.glob(path + '*.csv')):
            print(str(infile))
            leg = infile.split('/')[3]
            dataLegs = pd.read_csv(infile, sep=',', decimal='.', skiprows=0, ).values
            arrDate = dataLegs[len(dataLegs)-1,8]
            depDate = dataLegs[0,8]
            
            rawLeg = np.array([k for k in dataset if k[dtInd] >= depDate and k[dtInd] <= arrDate])

            dfNew = pd.DataFrame(
                {'stw': rawLeg[:, stwInd], 'draft': rawLeg[:, draftInd], 'foc': rawLeg[:, focInd],
                 'ws': rawLeg[:, wsIndSensor],
                 'wd': rawLeg[:, wdInd], 'swh': rawLeg[:, swhInd], 'lat': rawLeg[:, latInd], 'lon': rawLeg[:, lonInd], 'dt': rawLeg[:, dtInd],
                 'vslHeading': rawLeg[:, bearInd], 'wsWS': rawLeg[:, wsIndWS]})

            dfNew.to_csv('./correctedLegsForTR/' + vessel + '/' + leg, index=False)
    
    def correctData(self, vessel):

        tlgs = pd.read_csv('./data/DANAOS/' + vessel + '/TELEGRAMS/'+vessel+'.csv', delimiter=';').values
        data = pd.read_csv('./data/DANAOS/' + vessel + '/mappedData.csv', delimiter=',').values

        vslCode = 113 if vessel == 'HYUNDAI SMART' else 129
        vslCode = 127 if vessel=='GENOA' else 0
        ## STW TEIDETECH CORRECTION
        if vessel == 'GENOA_':
            #stwTeidetech = pd.read_csv('./data/DANAOS/TEIDETECH/stwTEIDETECH_LEOC_EUROPE.csv')
            stwTeidetech = pd.read_csv('./data/DANAOS/TEIDETECH/TEIDETECH_.csv')
            stwTeidetech_Leoc = stwTeidetech['VESSEL_CODE'] == vslCode
            stwTeidetech_Leoc = stwTeidetech[stwTeidetech_Leoc].values

            ttime = stwTeidetech_Leoc[:, 1]

            ttimeDt = list(map(lambda x: str(datetime.strptime(x, '%d/%m/%Y %H:%M:%S').replace(second=0, microsecond=0)),
                               ttime))

            ttimeDt = np.array([k[:-3] for k in ttimeDt])
            ttimeDtDf = pd.DataFrame({'date': ttimeDt, 'stw': stwTeidetech_Leoc[:, 11]})

            ttimeRaw  = list(map(lambda s: s[:-3], data[:, 0]))

            ttimeNew = ttimeDtDf[ttimeDtDf['date'].isin(ttimeRaw)]
            #print(trData.shape)
            stwTD = np.nan_to_num(ttimeNew['stw'].values.astype(float))
            df = pd.DataFrame({'stw':data[:,12], 'stwTD':stwTD})
            sns.displot(df, x='stw')
            sns.displot(df, x='stwTD')
            #plt.show()
            data[:,12] = stwTD

        countDrft = 0
        for i in range(0,len(data)):

            if np.isnan(float(data[i,8])):
                datetimeV = str(data[i, 0])

                dateV = datetimeV.split(" ")[0]
                hhMMss = datetimeV.split(" ")[1]
                month = dateV.split("-")[1]
                day = dateV.split("-")[2]
                year = dateV.split("-")[0]
                month = '0' + month if month.__len__() == 1 else month
                day = '0' + day if day.__len__() == 1 else day
                newDate = year + '-' + month + '-' + day

                telegramRow = np.array([row for row in tlgs if str(row[0]).split(" ")[0] == newDate])

                drftTlg = telegramRow[0][8] if len(telegramRow) > 0 else -1
                if drftTlg == -1 :
                    drftTlg = data[i-1,8]
                data[i,8]  = drftTlg
                countDrft += 1

        print("Found " + str(countDrft)+ " nan Drft values")
        pd.DataFrame(data).to_csv('./data/DANAOS/' + vessel + '/mappedData_.csv', header=False,
                                  index=False)

    def returnCleanedDatasetForTR(self, path, dataType):

        wsIndSensor = 11 if dataType == 'raw' else 3
        wsIndWS = 28 if dataType == 'raw' else 10
        focInd = 15 if dataType == 'raw' else 2
        stwInd = 12 if dataType == 'raw' else 0
        draftInd = 8 if dataType == 'raw' else 1
        wdInd = 10 if dataType == 'raw' else 4
        swhInd = 20 if dataType == 'raw' else 5
        dtInd = 0 if dataType == 'raw' else 8
        bearInd = 1 if dataType == 'raw' else 9

        dataset = pd.read_csv(path).values
        #dataset = dataset[dataset[:, wdInd].astype(float) > 0]
        if dataType == 'raw':

            for i in range(0, len(dataset[:, wdInd])):
                if float(dataset[i, wdInd]) < 0:
                    dataset[i, wdInd] += 360
                dataset[i, wdInd] = dread.getRelativeDirectionWeatherVessel(float(dataset[i, bearInd]), float(dataset[i, wdInd]))
                dataset[i, wdInd] = np.round(dataset[i, wdInd], 2)

            for i in range(0, len(dataset[:, wdInd])):
                if float(dataset[i, wdInd]) > 180:
                    dataset[i, wdInd] = float(dataset[i, wdInd]) - 180

            ###convert wind speed from knots to  m/s to beaufort
            #dataset[:, wsIndSensor] = dataset[:, wsIndSensor].astype(float) / 1.944

            #dataset, ind = self.replaceOutliers(dataset, dataset, 'ws', dataType, True, False)

            wsSen = []
            for i in range(0, len(dataset[:, wsIndSensor])):
                wsSen.append(gener.ConvertMSToBeaufort(float(dataset[i, wsIndSensor])))
            dataset[:, wsIndSensor] = wsSen


        draft = dataset[:, draftInd].astype(float).reshape(-1, 1)
        relWd = np.round(dataset[:, wdInd].astype(float), 2)
        wsBft = dataset[:, wsIndSensor].astype(int)
        stw = dataset[:, stwInd].astype(float)
        swh = dataset[:, swhInd].astype(float)
        foc = dataset[:, focInd].astype(int)

        trData = np.array(
            np.append(draft, np.asmatrix([relWd, wsBft, stw, swh, foc]).T, axis=1)).astype(float)


        #last check for negative values
        trData = np.array([k for k in trData if k[0] >= 0 if k[1] >= 0 and k[2] >= 0 and k[3] >= 12 and  k[4] >= 0 and k[5] > 0 ])



        for i in range(0, len(trData)):
            trData[i] = np.mean(trData[i:i + 15], axis=0)

        dfDrft = pd.DataFrame({'draft': trData[:, 0]})
        dfWD = pd.DataFrame({'wd': trData[:, 1]})
        dfWs = pd.DataFrame({'ws': trData[:, 2]})
        dfStw = pd.DataFrame({'stw': trData[:, 3]})
        dfSwh = pd.DataFrame({'swh': trData[:, 4]})
        dfFoc = pd.DataFrame({'foc': trData[:, 5]})
        sns.displot(dfDrft, x='draft')
        sns.displot(dfWD, x='wd')
        sns.displot(dfWs, x='ws')
        sns.displot(dfStw, x='stw')
        sns.displot(dfSwh, x='swh')
        sns.displot(dfFoc, x='foc')
        #plt.show()
        return trData

    def cleanDataset(self, data, datatype, vessel):


        stwInd = 12 if datatype == 'raw' else 0
        draftInd = 8 if datatype == 'raw' else 1
        wsIndSensor = 11 if datatype == 'raw' else 3
        focInd = 15 if datatype == 'raw' else 2
        
        
        
        if datatype == 'raw':

            
            ###convert wind speed from knots to  m/s to beaufort
            if self.knotsFLag == True: wfMS = data[:, wsIndSensor].astype(float) / 1.944

            data, ind = self.replaceOutliers(data, data, 'ws', 'raw', True, False)



            
        f = open('./consProfileJSON_Neural/tableWithMeanValues_'+vessel+'.json', )

        # returns JSON object as
        # a dictionary
        meanValuesJson = json.load(f)
        meanValues = meanValuesJson['meanConsValuesProfile']['meanConsValues']

        minStwTresh = 10
        data = data[data[:,stwInd].astype(float) >= minStwTresh]
        indicesToBeDeleted = []
        count = 0
        countNotFound = 0
        countStd = 0
        focThreshArr = np.array([k for k in data if k[stwInd] >= minStwTresh - 0.25 and k[stwInd] <= minStwTresh + 0.25 and
                  k[wsIndSensor] >= 0 and k[wsIndSensor] <= 5.5])[:, focInd]
        minFocTreshHold = np.round(np.mean(focThreshArr))
        stdFocTreshHold = np.std(focThreshArr)

        for i in range(0,len(data)):

            stw = float(data[i,stwInd])
            draft = float(data[i,draftInd])
            ws = gener.ConvertMSToBeaufort(float(data[i, wsIndSensor]))
            #ws = float(data[i, wsIndSensor])
            foc = float(data[i, focInd])

            ## search dict


            #dicStw = meanValues.get(stw, meanValues[min(data.keys(), key=lambda k: abs(k - stw))])

            #listOfDictsForSTW_DRFT = [k for k in meanValues if abs(stw - k['speed']) <= 0.5 and abs(draft - k['draft']) <= 2]
            listOfDictsForStw = min(meanValues, key=lambda x:abs(x["speed"] - stw))
            listOfDictsForStw_Drft = min(listOfDictsForStw['drafts'], key=lambda x: abs(x["draft"] - draft))['cells']

            if len(listOfDictsForStw_Drft) > 0:

                listOfWS = [k for k in listOfDictsForStw_Drft if ws == k['windBFT']]

                if len(listOfWS) > 0:
                    meanCons = listOfWS[0]['meanCons']
                    stdCons = listOfWS[0]['stdCons']
                else:

                    #if  foc < minFocTreshHold:
                    indicesToBeDeleted.append(i)
                    countNotFound += 1
                    continue
            else:

                #if foc < minFocTreshHold:
                indicesToBeDeleted.append(i)
                countNotFound += 1
                continue
                #print(ws)


            if (foc >= meanCons +  1.5 * stdCons) or (foc <= meanCons - 1.5 * stdCons) or foc < minFocTreshHold:
                foundOutlier = True

                if meanCons < minFocTreshHold - stdFocTreshHold:
                    indicesToBeDeleted.append(i)
                    #meanConsOfPrev15 = np.round(np.mean(data[i - 15:i, focInd]), 2)
                    #data[i, focInd] = meanConsOfPrev15
                else:
                    data[i, focInd] = meanCons

                    #data[i, focInd] =  meanConsOfPrev15
                    #indicesToBeDeleted.append(i)

                count += 1

        data = np.delete(data, indicesToBeDeleted, axis=0)
        pd.DataFrame(data).to_csv('./consProfileJSON_Neural/cleaned_'+datatype+'_'+vessel+'.csv',header = False, index = False)
        # realFocValue = dataset[i,2]  if foundOutlier == False else realFocValue
        print("\nFound " + str(count) + " FOC outliers")
        print("Deleted " + str(len(indicesToBeDeleted)) + " instances")
        print("Not Found " + str(countNotFound) + " FOC values or " + str((countNotFound / len(data)) * 100) + ' % of the dataset' )

    def buildTableWithMeanValues(self, data, vessel):

        data, ind = self.replaceOutliers(data, data, 'ws', 'raw', True, False)


        #data[:, 11] = data[:, 11].astype(float) / (1.944)

        for i in range(0, len(data[:, 11])):
            data[i,11] = gener.ConvertMSToBeaufort(float(data[i,11]))


        wind = [1, 2, 3, 4, 5]
        windFs = np.arange(0, 10, 1)
        swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        maxSpeed = max(data[:,12])
        stws = np.arange(10, maxSpeed, 0.25)
        #drafts = np.arange(6, 19, 2)
        drafts = np.arange(min(data[:,8]), max(data[:,8])+1, 2)
        drafts = [8,11,14]
        count = 0

        laddenJSON = '{}'
        json_decoded = json.loads(laddenJSON)
        json_decoded['meanConsValuesProfile'] = {
                                              "dateCreated": date.today().strftime("%d/%m/%Y"),
                                              "meanConsValues": []}
        for i in range(0,len(stws)-1):

            outerItem = {"speed": np.round((stws[i] + stws[i + 1]) / 2, 2), "drafts": []}

            for d in range(0, len(drafts) - 1):

                #outerItem = {"draft": str(drafts[d]) +'-' +str(drafts[d + 1]), "speed": str(stws[i])+'-' + str(stws[i+1]), "cells": []}
                innerItem = {"draft": np.round( (drafts[d] + drafts[d + 1] )/2,2),
                              "cells": []}

                outerItem['drafts'].append(innerItem)

                for w in range(0, len(windFs)-1):



                    focForSTW = np.array([k for k in data if k[12] >= stws[i] and k[12] < stws[i+1]
                                                              and k[8] >= drafts[d] and k[8] < drafts[d+1]
                                                              and k[11] >= windFs[w] and k[11] <  windFs[w+1]   ])

                    #focForSTW = np.array([k for k in dataRaw if k[0] >= stw - 0.2 and k[0] <= stw + 0.2])[:, 2]
                    if len(focForSTW) > 30:
                        meanFocForSTW = np.round(np.mean(focForSTW[:,15]),2)
                        stdFocForSTW = np.round(np.std(focForSTW[:,15]),2)
                    else:
                        count += 1
                        continue

                    item = {"windBFT": w, "meanCons": meanFocForSTW, "stdCons": stdFocForSTW}
                    innerItem['cells'].append(item)

                json_decoded['meanConsValuesProfile']['meanConsValues'].append(outerItem)

        with open('./consProfileJSON_Neural/tableWithMeanValues_' + vessel + '.json', 'w') as json_file:
            json.dump(json_decoded, json_file)

        #return self.dataToReturn

    def extractLegsToJsonWithRollWindow(self, vessel, imo):

        legsExtracted = []
        path = './correctedLegsForTR/' + vessel + '/'
        for infile in sorted(glob.glob(path + '*.csv')):

            leg = infile.split('/')[3]
            rawDataLeg = pd.read_csv(infile, sep=',', decimal='.', skiprows=0, ).values
            if len(rawDataLeg) == 0: continue
            for i in range(0, len(rawDataLeg[:, 4])):
                if float(rawDataLeg[i, 4]) < 0:
                    rawDataLeg[i, 4] = abs(rawDataLeg[i, 4])
                rawDataLeg[i, 4] = dread.getRelativeDirectionWeatherVessel(float(rawDataLeg[i, 9]),
                                                                           float(rawDataLeg[i, 4]))
                rawDataLeg[i, 4] = np.round(rawDataLeg[i, 4], 2)

            for i in range(0, len(rawDataLeg[:, 4])):
                if float(rawDataLeg[i, 4]) > 180:
                    rawDataLeg[i, 4] = float(rawDataLeg[i, 4]) - 180

            '''wfS = rawDataLeg[:, 3].astype(float)
            wsSen = []
            for i in range(0, len(wfS)):
                wsSen.append(gener.ConvertMSToBeaufort(float(float(wfS[i]))))
            rawDataLeg[:, 3] = wsSen'''

            legDepartureDate = rawDataLeg[0, 8]
            legArrivalDate = rawDataLeg[len(rawDataLeg) - 1, 8]
            portDeparure = leg.split('_')[0]
            portArrival = leg.split('_')[1]
            lat_lon_Dep = (rawDataLeg[0][6], rawDataLeg[0][7])
            lat_lon_Arr = (rawDataLeg[len(rawDataLeg) - 1][6], rawDataLeg[len(rawDataLeg) - 1][7])

            leg = {'departure': str(legDepartureDate),
                   'arrival': str(legArrivalDate),
                   'depPort': portDeparure,
                   'arrPort': portArrival,
                   'lat_lon_Dep': lat_lon_Dep,
                   'lat_lon_Arr': lat_lon_Arr,
                   'data': []}

            rollWindowData = np.array(
                np.append(rawDataLeg[:, :6], np.asmatrix([rawDataLeg[:, 9], rawDataLeg[:, 10]]).T, axis=1))

            for i in range(0, len(rollWindowData)):
                rollWindowData[i] = np.mean(rollWindowData[i:i + 15], axis=0)

            with open('./rollWindLegs/' + vessel + '/' + portDeparure + '_' + portArrival + '_leg.csv',
                      mode='w') as dataw:
                data_writer = csv.writer(dataw, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    ['stw', 'draft', 'foc', 'ws', 'wd', 'swh', 'lat', 'lon', 'dt'])

                for i in range(0, len(rollWindowData)):
                    draft = np.round(rollWindowData[i][1], 2)
                    ws = np.round(rollWindowData[i][3], 2)
                    wd = np.round(rollWindowData[i][4], 2)
                    stw = np.round(rollWindowData[i][0], 2)
                    foc = np.round(rollWindowData[i][2], 2)
                    foc = np.round(((foc) / 1000) * 1440, 2)
                    swh = np.round(rollWindowData[i][5], 2)
                    vslDir = np.round(rollWindowData[i][6], 2)
                    wsWS = np.round(rollWindowData[i][7], 2)

                    dt = str(rawDataLeg[i][8])
                    lat = rawDataLeg[i][6]
                    lon = rawDataLeg[i][7]

                    item = {'datetime': dt, 'ws': ws, 'swh'
                    : swh, 'wd': wd, 'stw': stw, 'foc': foc, 'lat': lat, 'lon': lon, 'draft': draft}
                    leg['data'].append(item)
                    data_writer.writerow([stw, draft, foc, ws, wd, swh, lat, lon, dt])

            legsExtracted.append(leg)

        jsonFile = {'vessel': vessel,
                    'imo': imo,
                    'legs': legsExtracted
                    }

        with open('./legs/' + vessel + '/legs_' + vessel + '_' + str(datetime.now().strftime("%d_%m_%Y %H_%M")) + '_.json',
                  'w') as outfile:
            json.dump(jsonFile, outfile)

    def concatLegs_PrepareForTR(self, vessel, path):

        dataSet = []
        # path = './correctedLegsForTR/' + vessel + '/'
        dataset = []
        for infile in sorted(glob.glob(path + '*.csv')):
            print(str(infile))
            leg = infile.split('/')[3]
            data = pd.read_csv(infile, sep=',', decimal='.', skiprows=0, ).values

            dataset.append(data)
        dataset = np.concatenate(dataset)
        dataset = np.array(dataset)

        ###sort dataset by datetime
        #dataset = dataset[np.argsort(dataset[:, 8])]
        ##convert wd to rel wind dir ###
        #dataset = dataset[dataset[:, 4].astype(float) > 0]

        for i in range(0, len(dataset[:, 4])):
            if float(dataset[i, 4]) < 0:
                dataset[i, 4] = abs(dataset[i, 4])
            dataset[i, 4] = dread.getRelativeDirectionWeatherVessel(abs(float(dataset[i, 9])), float(dataset[i, 4]))
            dataset[i, 4] = np.round(dataset[i, 4], 2)

        for i in range(0, len(dataset[:, 4])):
            if float(dataset[i, 4]) > 180:
                dataset[i, 4] = float(dataset[i, 4]) - 180

        ###convert wind speed from m/s to beaufort
        wfS = dataset[:, 3].astype(float)
        wsSen = []
        for i in range(0, len(wfS)):
            wsSen.append(gener.ConvertMSToBeaufort(float(float(wfS[i]))))
        dataset[:, 3] = wsSen

        wfWS = dataset[:, 10].astype(float)
        wsWS = []
        for i in range(0, len(wfWS)):
            wsWS.append(gener.ConvertMSToBeaufort(float(float(wfWS[i]))))
        dataset[:, 10] = wsWS
        ##sort bydatetime
        # dataset = dataset[np.argsort(dataset[:, 8])]
        ## ################## draft, wd, ws ,stw, swh, foc
        draft = dataset[:, 1].astype(float).reshape(-1, 1)
        relWd = np.round(dataset[:, 4].astype(float), 2)
        wsBft = dataset[:, 3].astype(int)
        stw = dataset[:, 0].astype(float)
        swh = dataset[:, 5].astype(float)
        foc = dataset[:, 2].astype(int)



        trData = np.array(
            np.append(draft, np.asmatrix([relWd, wsBft, stw, swh, foc]).T, axis=1))  # .astype(float)

        # trData = replaceOutliers(trData, None,'ws')
        # last check for negative values
        trData = np.array(
            [k for k in trData if k[0] >= 0 if k[1] >= 0 and k[2] >= 0 and k[3] >= 11 and k[4] >= 0 and k[5] > 0])

        '''dfNew = pd.DataFrame(
            {'draft': trData[:, 0], 'relWd': trData[:, 1], 'wsBft': trData[:, 2], 'wsBftWS': dataset[:, 10],
             'stw': trData[:, 3],
             'swh': trData[:, 4], 'foc': trData[:, 5], 'dt': dataset[:, 8]})

        dfNew.to_csv('./consProfileJSON_Neural/cleaned_' + vessel + '.csv', index=False)'''

        for i in range(0, len(trData)):
            trData[i] = np.mean(trData[i:i + 15], axis=0)

        return trData

    def concatLegs(self, vessel, path):


        dataSet = []
        #path = './correctedLegsForTR/' + vessel + '/'
        dataset = []
        for infile in sorted(glob.glob(path + '*.csv')):
            print(str(infile))
            leg = infile.split('/')[3]
            data = pd.read_csv(infile, sep=',', decimal='.', skiprows=0, ).values

            dataset.append(data)
        dataset = np.concatenate(dataset)
        dataset = np.array(dataset)

        # dataset = dataset.astype(float)
        ##sort by datetime
        #dataset = dataset[np.argsort(dataset[:, 8])]
        dataset, indToBeDeleted = self.replaceOutliers(dataset, dataset, 'ws', 'legs', False)

        dfNew = pd.DataFrame({'stw': dataset[:, 0], 'draft': dataset[:, 1], 'foc': dataset[:, 2],
             'ws': dataset[:, 3],
             'wd': dataset[:, 4], 'swh': dataset[:, 5], 'lat': dataset[:, 6], 'lon': dataset[:, 7], 'dt': dataset[:, 8],
             'vslHeading': dataset[:, 9], 'wsWS':dataset[:,10]})

        dfNew.to_csv('./legs/concatLegs_' + vessel + '.csv', index=False)

        return pd.read_csv('./legs/concatLegs_' + vessel + '.csv').values

    def correctWS_Beaufort(self, vessel):

        path = './legs/' + vessel + '/'
        dataRaw = pd.read_csv('./legs/concatLegs_' + vessel + '.csv', delimiter=',').values

        dataset = []
        print("Running WS correction for files . . .")
        for infile in sorted(glob.glob(path + '*.csv')):
            print(str(infile))
            leg = infile.split('/')[3]
            data = pd.read_csv(infile, sep=',', decimal='.', skiprows=0, ).values

            ##convert sensor wind speed to BF
            wfS = data[:, 3].astype(float) / (1.944) if self.knotsFLag == True else  data[:, 3].astype(float) / (1.944)
            wsSen = []
            for i in range(0, len(wfS)):
                wsSen.append(gener.ConvertMSToBeaufort(float(float(wfS[i]))))
            data[:, 3] = wsSen

            ##convert weather service wind speed to BF
            wfWS = data[:, 10].astype(float)
            wsWS = []
            for i in range(0, len(wfWS)):
                wsWS.append(gener.ConvertMSToBeaufort(float(float(wfWS[i]))))
            data[:, 10] = wsWS

            ##correct wind speed
            data = self.replaceOutliers(data, dataRaw, 'ws')

            ##write leg data with corrected wind speed
            dfNew = pd.DataFrame(
                {'stw': data[:, 0], 'draft': data[:, 1], 'foc': data[:, 2],
                 'ws': data[:, 3],
                 'wd': data[:, 4], 'swh': data[:, 5], 'lat': data[:, 6], 'lon': data[:, 7], 'dt': data[:, 8],
                 'vslHeading': data[:, 9], 'wsWS': data[:, 10]})

            dfNew.to_csv('./correctedLegsForTR/' + vessel + '/' + leg, index=False)

    def replaceOutliers(self, dataset, dataRaw, type, dataType, convertToBft, rollWind=None):

        indicesToBeDeleted = []
        ws_indicesToBeDeleted = []
        ##define indices of columns
        wsIndSensor = 11 if dataType == 'raw' else 3
        wsIndWS = 28 if dataType == 'raw' else 10
        focInd =  15 if dataType == 'raw' else 2
        stwInd = 12 if dataType == 'raw' else 0
        draftInd = 8 if dataType == 'raw' else 1

        if type == 'foc':
            minStwTresh = 12
            minFocTreshHold = np.round(np.mean(np.array([k for k in dataRaw if
                                                         k[stwInd] >= minStwTresh - 0.25 and k[
                                                             stwInd] <= minStwTresh + 0.25 and
                                                         k[wsIndSensor] >= 0 and k[wsIndSensor] <= 5.5])[:, focInd]))

        if dataType == 'legs':

            wsIndWS = 7 if rollWind == True else 10

        if type == 'ws' and convertToBft == True:

            ##convert sensor wind speed sensor  to m/s
            wfS = dataset[:, wsIndSensor].astype(float) / (1.944) if self.knotsFLag == True else  dataset[:, wsIndSensor].astype(float)
            dataset[:, wsIndSensor] = np.round(wfS,2)



        if type == 'ws':

            countWS =0
            meanWsSens = np.mean(dataRaw[:, wsIndSensor])
            stdWSsSens = np.std(dataRaw[:, wsIndSensor])

            for i in range(0 , len(dataset)):
                ##ws correction

                wsSens = dataset[i, wsIndSensor]
                if wsSens >= meanWsSens + 3 * stdWSsSens or  wsSens <= meanWsSens - 3 * stdWSsSens:
                    #print("found outlier!")
                    ws_indicesToBeDeleted.append(i)
                    continue


                wsSensPrev = dataset[i-1, wsIndSensor]
                wsWS = dataset[i, wsIndWS]

                sensor_ws_diff = np.round((abs(wsSens - wsWS)))
                sensor_prev_diff = np.round((abs(wsSens - wsSensPrev)))

                if sensor_ws_diff >= 10 or sensor_prev_diff > 10:

                    #if dataset[i , wsIndSensor] > 0:
                    # print("ws "+str(dataset[i,3])+" value replaced with " + str(dataset[i,10]))
                    newWS = int (np.mean([wsSens , wsWS]))
                    dataset[i ,wsIndSensor] = newWS
                    #else:
                        #if i >= 5:
                            #dataset[i ,wsIndSensor] = np.mean(dataset[ i - 5:i , wsIndSensor])
                        #else:
                            #dataset[i, wsIndSensor] = dataset[i ,wsIndWS]
                    countWS += 1
            dataset = np.delete(dataset, ws_indicesToBeDeleted, axis=0)
            print("Found " + str(len(ws_indicesToBeDeleted)) + " WS outliers")
            print("Corrected " + str(countWS) + " WS values")
        else:
            count = 0


            for i in range(0, len(dataset)):
                ##foc check #######
                foundOutlier = False
                foc = dataset[i ,focInd]
                stw = dataset[i ,stwInd]
                ws = dataset[i, wsIndSensor]
                drft = dataset[i, draftInd]
                # realprevFoc = dataset[i-1,2] if i==1 else realFocValue
                # replacedPrevFoc = dataset[i-1,2]
                try:
                    focForSTW = np.array([k for k in dataRaw if k[stwInd] >= stw - 0.25 and k[stwInd] <= stw + 0.25
                                          and k[draftInd] >= drft - 2 and k[draftInd] <= drft + 2
                                          and k[wsIndSensor] >= ws - 2 and k[wsIndSensor] <= ws + 2  ])[:, focInd]

                    #focForSTW = np.array([k for k in dataRaw if k[0] >= stw - 0.2 and k[0] <= stw + 0.2])[:, 2]


                except:

                    #focForSTW = np.array([k for k in dataRaw if k[stwInd] >= stw - 0.25 and k[stwInd] <= stw + 0.25
                                          #and k[draftInd] >= drft - 1.5 and k[draftInd] <= drft + 1.5
                                         #])[:, focInd]
                    indicesToBeDeleted.append(i)

                    print("exception! for stw:" + str(stw) + " and ws: " +str(ws))
                    continue

                meanFocForSTW = np.mean(focForSTW)
                stdFocForSTW = np.std(focForSTW)
                # meanFocForSTW = np.mean(np.array([k for k in dataRaw if k[12] >= stw - 0.25 and k[12] <= stw + 0.25])[:, 15])
                # if  (abs(foc - meanFocForSTW) >= 15 ) or (abs(foc - realprevFoc) >= 10  and abs(stw - dataset[i-1,0]) < 1) or  \
                # (abs(foc - replacedPrevFoc) >= 10  and abs(stw - dataset[i-1,0]) < 1)
                if foc >= meanFocForSTW + stdFocForSTW or foc <= meanFocForSTW - stdFocForSTW :#or foc < minFocTreshHold:
                    foundOutlier = True
                    # print("foc "+str(dataset[i,2])+" value replaced with " + str(meanFocForSTW) +" for stw: " + str(dataset[i,0]))

                    indicesToBeDeleted.append(i)

                    count += 1

            dataset = np.delete(dataset, indicesToBeDeleted, axis=0)
            # realFocValue = dataset[i,2]  if foundOutlier == False else realFocValue
            print("Found " + str(count) + " FOC outliers")

        return dataset , indicesToBeDeleted

    def extractDataFromLegs(self, dataRaw, path, vessel, dataType):

        ###process raw data
        wsIndSensor = 11 if dataType == 'raw' else 3
        wsIndWS = 28 if dataType == 'raw' else 10
        focInd = 15 if dataType == 'raw' else 2
        stwInd = 12 if dataType == 'raw' else 0
        draftInd = 8 if dataType == 'raw' else 1
        wdInd = 10 if dataType == 'raw' else 4
        swhInd = 20 if dataType == 'raw' else 5
        dtInd = 0 if dataType == 'raw' else 8
        bearInd = 1 if dataType == 'raw' else 9
        #wfS = dataRaw[:, 11].astype(float) / (1.944)
        #dataRaw[:, 11] = np.round(wfS,2)
        '''wsSen = []
        for i in range(0, len(wfS)):
            wsSen.append(gener.ConvertMSToBeaufort(float(float(wfS[i]))))
        dataRaw[:, 11] = wsSen
        dataRaw[:, 15] = ((dataRaw[:, 15]) / 1000) * 1440'''
        #dataRaw = np.array([k for k in dataRaw if k[12] >= 10])
        #############################

        #self.correctWS_Beaufort(vessel)
        #self.concatLegs('./legs/EXPRESS ATHENS/')

        print("\n Running leg exctraction and FOC outlier detection . . .")
        #path = './legs/' + vessel + '/'
        #path = './correctedLegsForTR/' + vessel + '/'

        #dataRaw = pd.read_csv('./legs/concatLegs_' + vessel + '.csv', delimiter=',').values

        dataset = []
        for infile in sorted(glob.glob(path + '*.csv')):
            print(str(infile))
            leg = infile.split('/')[3]
            data = pd.read_csv(infile, sep=',', decimal='.', skiprows=0, ).values

            '''meanFoc = np.mean(data[:, 2], axis=0)
            stdFoc = np.std(data[:, 2], axis=0)
            data = np.array(
                [k for k in data if (k[2] >= (meanFoc - (2 * stdFoc))) and (k[2] <= (meanFoc + (2 * stdFoc)))])'''

            data = data[data[:, stwInd].astype(float) >= 12]


            ##correct wind speed

            '''rollWindData = np.array(np.append(data[:,:6], np.asmatrix([data[:,9],data[:,10]]).T, axis=1))

            for i in range(0, len(rollWindData)):
                 rollWindData[i] = np.mean(rollWindData[i:i + 15], axis=0)'''

            data, ind  = self.replaceOutliers(data, dataRaw, 'ws', dataType, True,  False)
            data, indicesToBeDeleted = self.replaceOutliers(data, dataRaw, 'foc', dataType, False, False)

            #data = np.delete(data, indicesToBeDeleted, axis=0)
            # data = data.astype(float)

            dfNew = pd.DataFrame(
                {'stw': data[:, 0], 'draft': data[:, 1], 'foc': data[:, 2],
                 'ws': data[:, 3],
                 'wd': data[:, 4], 'swh': data[:, 5], 'lat': data[:, 6], 'lon': data[:, 7], 'dt': data[:, 8],
                 'vslHeading': data[:, 6], 'wsWS': data[:,7]})


            dfNew.to_csv('./correctedLegsForTR/' + vessel + '/' + leg, index=False)

            '''dfNew = pd.DataFrame({'stw': np.round(data[:, 0],2), 'draft': np.round(data[:, 1],2), 'foc': np.round(data[:, 2],2), 'ws': np.round(data[:, 3],2),
                                          'wd': np.round(data[:, 4],2), 'swh': np.round(data[:, 5],2),
                                          'vslHeading': np.round(data[:, 6],2) })'''

            dataset.append(data)
        dataset = np.concatenate(dataset)
        dataset = np.array(dataset)

        ###sort dataset by datetime
        ##convert wd to rel wind dir ###


        for i in range(0, len(dataset[:, wdInd])):
            if float(dataset[i, wdInd]) < 0:
                dataset[i, wdInd] = abs(dataset[i, wdInd])
            dataset[i, wdInd] = dread.getRelativeDirectionWeatherVessel(abs(float(dataset[i, bearInd])), float(dataset[i, wdInd]))
            dataset[i, wdInd] = np.round(dataset[i,wdInd],2)

        for i in range(0, len(dataset[:, wdInd])):
            if float(dataset[i, wdInd]) > 180:
                dataset[i, wdInd] = float(dataset[i, wdInd]) - 180

        ###convert wind speed from m/s to beaufort
        wfS = dataset[:, wsIndSensor].astype(float)
        wsSen = []
        for i in range(0, len(wfS)):
            wsSen.append(gener.ConvertMSToBeaufort(float(float(wfS[i]))))
        dataset[:, wsIndSensor] = wsSen
    
        wfWS = dataset[:, wsIndWS].astype(float)
        wsWS = []
        for i in range(0, len(wfWS)):
            wsWS.append(gener.ConvertMSToBeaufort(float(float(wfWS[i]))))
        dataset[:, wsIndWS] = wsWS

        ##sort bydatetime
        dataset = dataset[np.argsort(dataset[:, dtInd])]
        ## ################## draft, wd, ws ,stw, swh, foc

        draft = np.round(dataset[:, draftInd].astype(float).reshape(-1, 1),2)
        relWd = np.round(dataset[:, wdInd].astype(float),2)
        wsBft = dataset[:, wsIndSensor].astype(int)
        wsBftWS = dataset[:, wsIndWS].astype(int)
        stw = dataset[:, stwInd].astype(float)
        swh = dataset[:, swhInd].astype(float)
        foc = dataset[:,focInd].astype(int)
        dt = dataset[:,dtInd]

        trData = np.array(
            np.append(draft, np.asmatrix([relWd, wsBft , stw, swh, foc]).T, axis=1))#.astype(float)

        # trData = replaceOutliers(trData, None,'ws')

        dfNew = pd.DataFrame(
            {'draft': trData[:, 0], 'relWd': trData[:, 1], 'wsBft': trData[:, 2], 'wsBftWS': wsBftWS,
             'stw': trData[:, 3],
             'swh': trData[:, 4], 'foc': trData[:, 5], 'dt': dt })

        dfNew.to_csv('./consProfileJSON_Neural/cleaned_' + vessel+'.csv' , index=False)

        #sns.displot(dfNew, x='wsBft')
        #sns.displot(dfNew, x='wsBftWS')
        #plt.show()

        for i in range(0, len(trData)):
            trData[i] = np.mean(trData[i:i + 15], axis=0)

        #trData[:, 5] = np.round(trData[:, 5])
        #trData[:, 2] = np.round(trData[:, 2])

        return trData


def main():

  vessel = 'GENOA'

  preLegs = preProcessLegs(knotsFlag=False)

  #preLegs.extractLegsFromRawCleanedData(vessel, 'legs')
  #return

  preLegs.correctData('GENOA')
  #return
  #preLegs.extractLegsToJsonWithRollWindow(vessel, 9229312)

  dataRaw = pd.read_csv('./data/DANAOS/'+vessel+'/mappedData_.csv', delimiter=',').values
  preLegs.buildTableWithMeanValues(dataRaw, vessel)

  #return
  #dataRaw = preLegs.concatLegs(vessel, './legs/' + vessel + '/')
  dataRaw = pd.read_csv('./data/DANAOS/' + vessel + '/mappedData_.csv', delimiter=',').values
  preLegs.cleanDataset(dataRaw, "raw", vessel)

  #return

  #dataRaw = preLegs.concatLegs(vessel, './legs/'+vessel+'/')
  #
  #preLegs.extractDataFromLegs(dataRaw, './legs/'+vessel+'/', vessel, 'legs')

if __name__ == "__main__":
    main()


'''preLeg = preProcessLegs()
vessel = 'EXPRESS ATHENS'
#data = pd.read_csv('./data/DANAOS/' + vessel + '/mappedDataNew.csv', delimiter=',').values

#path = './correctedLegsForTR/' + vessel + '/'
path = './legs/' + vessel + '/'
preLeg.concatLegs(path)

data = pd.read_csv('./legs/concatLegs_' + vessel + '.csv', delimiter=',').values

preLeg.cleanDataset(data, 'legs')'''
#preLeg.buildTableWithMeanValues(data, vessel)