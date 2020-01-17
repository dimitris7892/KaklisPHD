#from pandas import Series
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp,chisquare,chi2_contingency
import math
import pyearth as sp
import matplotlib
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

    def readNewDataset(self):

        ####STATSTICAL ANALYSIS PENELOPE

        data = pd.read_csv('./Mapped_data/PERSEFONE_mapped.csv')
        data1 = pd.read_csv('./Mapped_data/PERSEFONE_mapped_1.csv')
        data2 = pd.read_csv('./Mapped_data/PERSEFONE_mapped_2.csv')
        data3 = pd.read_csv('./Mapped_data/PERSEFONE_mapped_3.csv')
        data4 = pd.read_csv('./Mapped_data/PERSEFONE_mapped_4.csv')
        data5 = pd.read_csv('./Mapped_data/PERSEFONE_mapped_5.csv')
        data6 = pd.read_csv('./Mapped_data/PERSEFONE_mapped_6.csv')
        data7 = pd.read_csv('./Mapped_data/PERSEFONE_mapped_7.csv')
        #data8 = pd.read_csv('./Mapped_data/PERSEFONE_mapped_8.csv')
        #data9 = pd.read_csv('./Mapped_data/PERSEFONE_mapped_9.csv')
        #data10 = pd.read_csv('./Mapped_data/PERSEFONE_mapped_10.csv')
        dtNew = data.values#.astype(float)
        dtNew1 = data1.values#.astype(float)
        dtNew2 = data2.values
        dtNew3 = data3.values
        dtNew4 = data4.values
        dtNew5 = data5.values
        dtNew6 = data6.values
        dtNew7 = data7.values
        #dtNew8 = data8.values
        #dtNew9 = data9.values
        #dtNew10 = data10.values





        dataNew = np.concatenate([ dtNew , dtNew1 , dtNew2, dtNew3 , dtNew4 , dtNew5 , dtNew6  , dtNew7 ])
        ballastDt = np.array([k for k in dataNew if k[5]=='B' if k[4]>0.8])[:,0:5].astype(float)
        ladenDt = np.array([ k for k in dataNew if k[ 5 ] == 'L'  ])[ :, 0:5 ].astype(float)
        ####################
        ####################
        ####################
        dataV = pd.read_csv('./data/PENELOPE_1-31_03_19.csv')
        dataV=dataV.drop([ 't_stamp', 'vessel status' ], axis=1)
        dtNew = dataV.values.astype(float)

        windSpeedVmapped = [ ]
        windDirsVmapped = [ ]
        draftsVmapped = [ ]
        with open('./windSpeedMapped.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                windSpeedVmapped.append(float(row[ 0 ]))
        with open('./windDirMapped.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                windDirsVmapped.append(float(row[ 0 ]))

        with open('./draftsMapped.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                draftsVmapped.append(float(row[ 0 ]))



        newMappedData = np.array(
            np.append(dtNew[ :, : ], np.asmatrix([ np.array(windSpeedVmapped), np.array(windDirsVmapped) ]).T,
                      axis=1)).astype(float)
        Vcourse = newMappedData[ :, 2 ]
        Vstw = newMappedData[ :, 1 ]
        windDir = newMappedData[ :, 27 ]
        windSpeed = newMappedData[ :, 26 ]
        ###projection of vector wind speed direction vector into vessel speed direction orthogonal system of coordinates
        ##convert vessel speed (knots) to m/s
        relWindSpeeds = [ ]
        relWindDirs = [ ]
        for i in range(0, len(Vcourse)):
            x = np.array([ Vstw[ i ] * 0.514, Vcourse[ i ] ])
            y = np.array([ windSpeed[ i ], windDir[ i ] ])
            ##convert cartesian coordinates to polar
            x = np.array([ x[ 0 ] * np.cos(x[ 1 ]), x[ 0 ] * np.sin(x[ 1 ]) ])
            y = np.array([ y[ 0 ] * np.cos(y[ 1 ]), y[ 0 ] * np.sin(y[ 1 ]) ])

            x_norm = np.sqrt(sum(x ** 2))

            # Apply the formula as mentioned above
            # for projecting a vector onto the orthogonal vector n
            # find dot product using np.dot()
            proj_of_y_on_x = (np.dot(y, x) / x_norm ** 2) * x

            vecproj = np.dot(y, x) / np.dot(y, y) * y
            relWindSpeed = proj_of_y_on_x[ 0 ] + windSpeed[ i ]
            if relWindSpeed > 40:
                d = 0
            relDir = windDir[ i ] - Vcourse[ i ]
            relDir += 360 if relDir < 0 else relDir
            relWindDirs.append(relDir)
            relWindSpeeds.append(relWindSpeed)
        newMappedData = np.array(
            np.append(dtNew[ :, : ],
                      np.asmatrix([ np.array(relWindSpeeds), np.array(relWindDirs), np.array(draftsVmapped) ]).T,
                      axis=1)).astype(float)

        #####################
        ############################MAP TELEGRAM DATES FOR DRAFT



        x = 0

    def readExtractNewDataset(self):

        ####################
        ####################
        ####################
        dataV = pd.read_csv('./data/Persefone/PERSEFONE_1-31_03_19.csv')
        #dataV=dataV.drop([ 't_stamp', 'vessel status' ], axis=1)
        dtNew = dataV.values#.astype(float)


        data = pd.read_csv('./export telegram.csv', sep='\t')
        DdtNew = data.values[0:, :]
        dateTimesW = []
        ws = []

        for i in range(0, len(DdtNew[:, 1])):
            if DdtNew[i, 0] == 'T004':
                date = DdtNew[i, 1].split(" ")[0]
                month = date.split('/')[1]
                day = date.split('/')[0]
                year = date.split('/')[2]

                hhMMss = DdtNew[i, 1].split(" ")[1]
                newDate = year + "-" + month + "-" + day + " " + ":".join(hhMMss.split(":")[0:2])
                dt = datetime.datetime.strptime(newDate, '%Y-%m-%d %H:%M')
                # if DdtNew[ i, 7 ] != 'No data' and DdtNew[ i, 8 ] != 'No data':
                dateTimesW.append(dt)

        dateTimesV = []
        for row in dtNew[:, 0]:
            date = row.split(" ")[0]
            hhMMss = row.split(" ")[1]
            newDate = date + " " + ":".join(hhMMss.split(":")[0:2])
            dtV = datetime.datetime.strptime(newDate, '%Y-%m-%d %H:%M')
            #################################
            date = str(dtV).split(" ")[0]
            month = date.split('-')[1]

            day = date.split('-')[2]
            #hhMMssV = str(dateTimesV[i]).split(" ")[1]
            year = date.split('-')[0]

            ##find dateTimesV at sea
            filteredDTWs = [d for d in dateTimesW if month == str(d).split(' ')[0].split('-')[1] and day ==
                            str(d).split(' ')[0].split('-')[2] and year == str(d).split(' ')[0].split('-')[0]]
            dateTimesV.append(dtV)
            #for filteredDate in filteredDTWs:
                #date = str(filteredDate).split(" ")[0]
                #month = date.split('-')[1]
                #month = month[1] if month[0] == '0' else month
                #day = date.split('-')[2]
                #day = day[1] if day[0] == '0' else day
                #year = date.split('-')[0]
                #hhMMss = str(filteredDate).split(" ")[ 1 ]

                #newDate = day + "/" + month + "/" + year + " " + ":".join(hhMMss.split(":"))
                #tlgType=np.array(data.loc[ data[ 'telegram_date' ] == newDate ].values)[ 0 ][ 2 ]
                #if tlgType == 'A' or tlgType=='N':

            ############################

        draftsVmapped = [ ]
        ballastLadenFlagsVmapped = [ ]
        for i in range(0, len(dateTimesV)):

            datesWs = [ ]

            date = str(dateTimesV[ i ]).split(" ")[ 0 ]
            month = date.split('-')[ 1 ]

            day = date.split('-')[ 2 ]
            hhMMssV = str(dateTimesV[ i ]).split(" ")[1]
            year = date.split('-')[ 0 ]
            dt = dateTimesV[ i ]

            filteredDTWs = [ d for d in dateTimesW if month == str(d).split(' ')[ 0 ].split('-')[ 1 ] and day ==
                             str(d).split(' ')[ 0 ].split('-')[ 2 ] and year == str(d).split(' ')[ 0 ].split('-')[ 0 ] ]

            date2numFiltered = {}
            sorted = [ ]

            for k in range(0, len(filteredDTWs)):
                # dtDiff[filteredDTWs[k]]=abs((filteredDTWs[k]-dt).total_seconds() / 60.0)
                date = str(filteredDTWs[ k ]).split(" ")[ 0 ]
                month = date.split('-')[ 1 ]
                month = month[ 1 ] if month[ 0 ] == '0' else month
                day = date.split('-')[ 2 ]
                day = day[ 1 ] if day[ 0 ] == '0' else day
                year = date.split('-')[ 0 ]
                hhMMss = str(filteredDTWs[ k ]).split(" ")[ 1 ]
                # newDate = month + "/" + day + "/" + year + " " + ":".join(hhMMss.split(":")[ 0:2 ])
                newDate = day + "/" + month + "/" + year + " " + ":".join(hhMMss.split(":"))
                datesWs.append(newDate)

                date2numFilteredList = [ ]

                sorted.append(newDate)
                # date2numFiltered.append(matplotlib.dates.date2num(k))

                # dtFiltered[ filteredDTWs[ k ] ] = matplotlib.dates.date2num(k)
                # dt2num = matplotlib.dates.date2num(dt)
                # date2numFiltered.append(dt2num)
            date = str(dt).split(" ")[ 0 ]
            month = date.split('-')[ 1 ]
            month = month[ 1 ] if month[ 0 ] == '0' else month
            day = date.split('-')[ 2 ]
            day = day[ 1 ] if day[ 0 ] == '0' else day
            year = date.split('-')[ 0 ]
            hhMMss = hhMMssV
            # newDate = month + "/" + day + "/" + year + " " + ":".join(hhMMss.split(":")[ 0:2 ])
            newDate = day + "/" + month + "/" + year + " " + ":".join(hhMMss.split(":"))
            sorted.append(newDate)
            sorted.sort()
            for i in range(0, len(sorted)):
                if sorted[ i ] == newDate:
                    try:
                        if i > 0:
                            if i == len(sorted) - 1:
                                previous = sorted[ i - 1 ]
                                blFlag = np.array(data.loc[ data[ 'telegram_date' ] == previous ].values)[ 0 ][ 49 ]
                                if blFlag == 'B':
                                    ballastLadenFlagsVmapped.append('B')
                                else:
                                    ballastLadenFlagsVmapped.append('L')
                            else:
                                try:
                                    previous = sorted[ i - 1 ]
                                    next = sorted[ i + 1 ]

                                    # date1 = {k: v for k, v in dtFiltered.items() if v == previous}
                                    # date2 = {k: v for k, v in dtFiltered.items() if v == next}
                                    blFlag1 = np.array(data.loc[ data[ 'telegram_date' ] == previous ].values)[ 0 ][ 49 ]
                                    blFlag2 = np.array(data.loc[ data[ 'telegram_date' ] == next ].values)[ 0 ][ 49 ]
                                except:
                                    x = 0
                                if blFlag1 == 'L' and blFlag2 == 'L':
                                    ballastLadenFlagsVmapped.append('L')
                                elif blFlag1 == 'L' and blFlag2 == 'B':
                                    ballastLadenFlagsVmapped.append('L')
                                elif blFlag1 == 'B' and blFlag2 == 'L':
                                    ballastLadenFlagsVmapped.append('B')
                                else:
                                    ballastLadenFlagsVmapped.append('B')

                        elif i == 0:
                            next = sorted[ i + 1 ]
                            blFlag = np.array(data.loc[ data[ 'telegram_date' ] == next ].values)[ 0 ][ 49 ]
                            if blFlag == 'B':
                                ballastLadenFlagsVmapped.append('B')
                            else:
                                ballastLadenFlagsVmapped.append('L')
                        elif i == len(sorted) - 1:
                            previous = sorted[ i - 1 ]
                            blFlag = np.array(data.loc[ data[ 'telegram_date' ] == previous ].values)[ 0 ][ 49 ]
                            if blFlag == 'B':
                                ballastLadenFlagsVmapped.append('B')
                            else:
                                ballastLadenFlagsVmapped.append('L')
                    except:
                        d=0
                        #ballastLadenFlagsVmapped.append('N')
                            # firstMin = min(dtDiff.items(), key=lambda x: x[1])
            # del dtDiff[ firstMin[ 0 ] ]
            # secMin = min(dtDiff.items(), key=lambda x: x[ 1 ] )

            drafts = [ ]
            for date in datesWs:

                    drftAFT = str(np.nan_to_num(np.array(data.loc[ data[ 'telegram_date' ] == date ].values)[ 0 ][ 27 ]))
                    drftFORE = str(np.nan_to_num(np.array(data.loc[ data[ 'telegram_date' ] == date ].values)[ 0 ][ 28 ]))

                    drftAFT = float(drftAFT.split(',')[ 0 ] + "." + drftAFT.split(',')[ 1 ]) if drftAFT.__contains__(
                            ',') else float(drftAFT)
                    drftFORE = float(
                            drftFORE.split(',')[ 0 ] + "." + drftFORE.split(',')[ 1 ]) if drftFORE.__contains__(
                            ',') else float(drftFORE)

                    drft = (drftAFT + drftFORE) / 2
                    drafts.append(float(drft))
            if len(datesWs)>1:
                x=0
            try:
                y = np.array(drafts)
                x = matplotlib.dates.date2num(filteredDTWs)
                f = sp.Earth()
                f.fit(x.reshape(-1, 1), y.reshape(-1, 1))
                pred = f.predict(np.array(matplotlib.dates.date2num(dt)).reshape(-1, 1))
                draftsVmapped.append(pred)
            except:
                x=0
                    #if np.nan_to_num(np.array(data.loc[ data[ 'telegram_date' ] == date ].values)[ 0 ][ 27 ]).__len__()==0 or \
                        #np.nan_to_num(np.array(data.loc[ data[ 'telegram_date' ] == date ].values)[ 0 ][ 28 ]).__len__() == 0:
                if drafts.__len__()==0:
                    draftsVmapped.append([0])


        ##########

        windSpeedVmapped = [ ]
        windDirsVmapped = [ ]

        dateTimesV = [ ]
        for row in dtNew[ :, 0 ]:
            date = row.split(" ")[ 0 ]
            hhMMss = row.split(" ")[ 1 ]
            newDate = date + " " + ":".join(hhMMss.split(":")[ 0:2 ])
            dt = datetime.datetime.strptime(newDate, '%Y-%m-%d %H:%M')
            dateTimesV.append(dt)

        data = pd.read_excel('./data/Persefone/Persefone TrackReport.xls')
        WdtNew = data.values[ 0:, : ]
        dateTimesW = [ ]
        ws = [ ]

        for i in range(0, len(WdtNew[ :, 1 ])):
            date = WdtNew[ i, 1 ].split(" ")[ 0 ]
            month = date.split('.')[ 1 ]
            day = date.split('.')[ 0 ]
            year = date.split('.')[ 2 ]

            hhMMss = WdtNew[ i, 1 ].split(" ")[ 1 ]
            newDate = year + "-" + month + "-" + day + " " + ":".join(hhMMss.split(":")[ 0:2 ])
            dt = datetime.datetime.strptime(newDate, '%Y-%m-%d %H:%M')
            if WdtNew[ i, 7 ] != 'No data' and WdtNew[ i, 8 ] != 'No data':
                dateTimesW.append(dt)

        for i in range(0, len(dateTimesV)):

            dtFiltered = {}
            datesWs = [ ]

            date = str(dateTimesV[ i ]).split(" ")[ 0 ]
            month = date.split('-')[ 1 ]
            day = date.split('-')[ 2 ]
            year = date.split('-')[ 0 ]
            dt = dateTimesV[ i ]

            filteredDTWs = [ d for d in dateTimesW if month == str(d).split(' ')[ 0 ].split('-')[ 1 ] and day ==
                             str(d).split(' ')[ 0 ].split('-')[ 2 ] and year == str(d).split(' ')[ 0 ].split('-')[ 0 ] ]
            date2numFiltered = {}

            for k in range(0, len(filteredDTWs)):
                date = str(filteredDTWs[ k ]).split(" ")[ 0 ]
                month = date.split('-')[ 1 ]
                day = date.split('-')[ 2 ]
                year = date.split('-')[ 0 ]
                hhMMss = str(filteredDTWs[ k ]).split(" ")[ 1 ]
                newDate = day + "." + month + "." + year + " " + ":".join(hhMMss.split(":")[ 0:2 ])

                datesWs.append(newDate)

            windSpeeds = [ ]
            windDirs = [ ]
            for date in datesWs:
                ws = np.array(data.loc[ data[ 'Date' ] == date + " Z" ].values)[ 0 ][ 7 ]
                wd = np.array(data.loc[ data[ 'Date' ] == date + " Z" ].values)[ 0 ][ 8 ]
                if ws != 'No data':
                    windSpeeds.append(float(ws.split('m')[ 0 ]))
                if wd != 'No data':
                    windDirs.append(float(wd[ :len(wd) - 1 ]))
            try:
                y = np.array(windSpeeds)
                x = matplotlib.dates.date2num(filteredDTWs)
                f = sp.Earth(max_degree=2)
                f.fit(x.reshape(-1, 1), y.reshape(-1, 1))
                pred = f.predict(np.array(matplotlib.dates.date2num(dt)).reshape(-1, 1))
                windSpeedVmapped.append(pred)
            except:
                if windSpeeds.__len__() == 0:
                    windSpeedVmapped.append(None)

            try:
                y = np.array(windDirs)
                x = matplotlib.dates.date2num(filteredDTWs)
                f = sp.Earth(max_degree=2)

                f.fit(x.reshape(-1, 1), y.reshape(-1, 1))
                pred = f.predict(np.array(matplotlib.dates.date2num(dt)).reshape(-1, 1))
                # f=interp1d(list(x), list(y), kind='cubic')
                windDirsVmapped.append(pred)
            except:
                if windDirs.__len__()==0:
                    windDirsVmapped.append(None)
        # pred = f(matplotlib.dates.date2num(dt))
        # data = pd.read_excel('./Penelope TrackReport.xls')
        # WdtNew = data.values[ 0:, : ]
        # print(pred)


        dataV = pd.read_csv('./data/Persefone/PERSEFONE_1-31_03_19.csv')
        dataV=dataV.drop([ 't_stamp', 'vessel status' ], axis=1)
        dtNew = dataV.values.astype(float)

        newMappedData = np.array(
            np.append(dtNew[ :, : ], np.asmatrix([ np.array(windSpeedVmapped).reshape(-1), np.array(windDirsVmapped).reshape(-1) , np.array(ballastLadenFlagsVmapped)[0:len(draftsVmapped)] ]).T,
        axis=1))
        Vcourse = np.array(newMappedData[ :, 2 ]).astype(float)
        Vstw = np.array(newMappedData[ :, 1 ]).astype(float)
        windDir = np.array(newMappedData[ :, 26 ]).astype(float)
        windSpeed = np.array(newMappedData[ :, 25 ]).astype(float)
        ###projection of vector wind speed direction vector into vessel speed direction orthogonal system of coordinates
        ##convert vessel speed (knots) to m/s
        relWindSpeeds = [ ]
        relWindDirs = [ ]
        for i in range(0, len(Vcourse)):
            x = np.array([ Vstw[ i ] * 0.514, Vcourse[ i ] ])
            y = np.array([ windSpeed[ i ], windDir[ i ] ])
            ##convert cartesian coordinates to polar
            x = np.array([x[0] * np.cos(x[1]),x[0]* np.sin(x[1])])
            y = np.array([ y[ 0 ] * np.cos(y[ 1 ]), y[ 0 ] * np.sin(y[ 1 ])])

            x_norm = np.sqrt(sum(x ** 2))

            # Apply the formula as mentioned above
            # for projecting a vector onto the orthogonal vector n
            # find dot product using np.dot()
            proj_of_y_on_x = (np.dot(y, x) / x_norm ** 2) *x

            vecproj =  np.dot(y, x) / np.dot(y, y) * y
            relWindSpeed = proj_of_y_on_x[ 0 ] + windSpeed[ i ]
            if relWindSpeed > 40:
                d=0
            relDir = windDir[ i ] - Vcourse[ i ]
            relDir += 360 if relDir < 0 else relDir
            relWindDirs.append(relDir)
            relWindSpeeds.append(relWindSpeed)
        newMappedData = np.array(
            np.append(dtNew[ :, : ],
                      np.asmatrix([ np.array(relWindSpeeds), np.array(relWindDirs), np.array(draftsVmapped).reshape(-1) ,np.array(ballastLadenFlagsVmapped)[0:len(draftsVmapped)]]).T,
                      axis=1))#.astype(float)

        #####################
        ############################MAP TELEGRAM DATES FOR DRAFT

        ####STATSTICAL ANALYSIS PENELOPE
        with open('./PERSEFONE_mapped_7.csv', mode='w') as dataw:
            data_writer = csv.writer(dataw, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow([ 'STW',  'WindSpeed', 'WindAngle', 'Draft'
                                     , 'M/E FOC (MT/days)' ,'B/L Flag'])
            for k in range(0, len(newMappedData)):
                data_writer.writerow([ newMappedData[ k ][ 1 ],newMappedData[ k ][ 25 ],newMappedData[ k ][ 26 ],newMappedData[ k ][ 27 ],newMappedData[ k ][ 8 ] ,newMappedData[ k ][ 28 ] ])
        x = 0

    def readLarosDAta(self,dtFrom,dtTo):
        UNK = {"data": []}
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
        print(sum(1 for line in open('./MT_DELTA_MARIA.txt')))
        with open('./MT_DELTA_MARIA.txt') as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=';')
            line_count = 0
            for row in csv_reader:
             try:
                    if row[1] == "STATUS_PARAMETER_ID":
                        continue
                    id = row[1]
                    date = row[4].split(" ")[0]
                    hhMMss = row[4].split(" ")[1]
                    newDate = date + " " + ":".join(hhMMss.split(":")[0:2])
                    dt = datetime.datetime.strptime(newDate, '%Y-%m-%d %H:%M')
                    value = row[3]
                    siteId=row[2]
                    if dt >  datetime.datetime.strptime('2018-05-20', '%Y-%m-%d'):
                        print(dt)
                    datt = {}
                    datt[ "value" ] = 0
                    datt[ "dt" ] = dt
                    dateD[ "data" ].append(datt)

                    if id == '7':
                        unknown = {}
                        unknown["value"] = value
                        unknown["dt"] = dt
                        UNK["data"].append(unknown)

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
                    #print('Processed '+str(line_count)+ ' lines.')
             except:
                p=0
                print('Processed ' + str(line_count) + ' lines.')



        with open('./MT_DELTA_MARIA_data_1.csv', mode='w') as data:
            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow(['Draft','DrfAft', 'DrftFore','WindSpeed', 'WindAngle', 'SpeedOvg','STW','Rpm' ,'M/E FOC (kg/min)',
                                 'FOC Total', 'Power','DateTime' ])
            #data_writer.writerow(
                #['Draft','DateTime' ])

            for i in range(0,len(UNK['data'])):
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

    def readLarosDataFromCsvNewExtractExcels(self, data):
        # Load file
        dataNew = data.drop(['SpeedOvg'], axis=1)
        dtNew = dataNew.values[484362:569386, :]

        daysSincelastPropClean = []
        daysSinceLastDryDock = []


        with open('./daysSincelastPropClean.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if row.__len__()==1:
                    daysSincelastPropClean.append(float(row[0]))
        with open('./daysSinceLastDryDock.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if row.__len__() == 1:
                    daysSinceLastDryDock.append(float(row[0]))

        daysSincelastPropClean = np.array(daysSincelastPropClean).reshape(-1,).astype(float)
        daysSinceLastDryDock = np.array(daysSinceLastDryDock).reshape(-1,).astype(float)

        dataNew = data.drop(['SpeedOvg', 'DateTime'], axis=1)
        dtNew = dataNew.values[ 484362:569386, : ].astype(float)

        draftMean = (dtNew[:, 1] + dtNew[:, 2]) / 2
        dtNew[:, 0] = np.round(draftMean, 3)

        #dtNew = dtNew[~np.isnan(dtNew).any(axis=1)]

        trim = (dtNew[:, 1] - dtNew[:, 2])

        dtNew1 = np.array(np.append(dtNew[:, 0].reshape(-1, 1), np.asmatrix(
            [np.round(trim, 3), dtNew[:, 3], dtNew[:, 4], dtNew[:, 5],dtNew[:, 6], daysSincelastPropClean , daysSinceLastDryDock ,  dtNew[:, 7]] ).T, axis=1))

        #dtNew1 = np.delete(dtNew1, [i for (i, v) in enumerate(dtNew1[0:, 0]) if v <= 6], 0)
        dtNew1 = np.delete(dtNew1, [i for (i, v) in enumerate(dtNew1[0:, 4]) if v <=0], 0)

        rpmB = np.array([i for i in dtNew1 if i[5] > 90 and i[7] < 100 ])

        #dtNew1 = np.delete(dtNew1, [i for (i, v) in enumerate(dtNew1[0:, 5]) if v <= 0 or v > 90], 0)  # or (v>0 and v <=10)
        # dtNew = np.delete(dtNew, [ i for (i, v) in enumerate(dtNew[ 0:, 7 ]) ], 0) # if v <= 7

        ##statistics STD
        # draftSmaller6 = np.array([drft for drft in dtNew1 if drft[0]< 6])
        # draftBigger6 = np.array([ drft for drft in dtNew1 if drft[0] > 6 ])
        #########DRAFT < 9
        drftStw8 = []
        #stw8DrftS4 = np.array([i for i in dtNew1 if i[4] >= 8 and i[4] < 8.5 and i[0] <= 9])
        with open('./MT_DELTA_MARIA_draftS_9_analysis.csv', mode='w') as rows:
            for dr in range(6, 9):
                stw8DrftS4 = np.array([i for i in dtNew1 if i[4] >= 8 and i[4] < 8.5 and i[0] > dr and i[0] <= dr + 1])
                stw8DrftS4 = np.delete(stw8DrftS4, [i for (i, v) in enumerate(stw8DrftS4[:, 8]) if v < (
                np.mean(stw8DrftS4[:, 8]) - np.std(stw8DrftS4[:, 8])) or v > np.mean(stw8DrftS4[:, 8]) + np.std(
                    stw8DrftS4[:, 8])], 0)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 8 < STW < 8.5                   '], )
                data_writer.writerow([], )
                data_writer.writerow(['Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW','RPM'
                                     ,'DaysSinceLastPropCleaning','DaysSinceLastDryDock', 'M/E FOC (kg/min)'])
                for data in stw8DrftS4:
                    data_writer.writerow([data[0], data[1], data[2], data[3], data[4], data[5],data[6],data[7],data[8]])
                data_writer.writerow([], )
            data_writer.writerow([], )
            data_writer.writerow([], )
            #drftStw8.append(stw8DrftS4)

            drftStw9 = []
            stw9DrftS4 = np.array([i for i in dtNew1 if i[4] >= 8.5 and i[4] < 9 and i[0] <= 9])

            for dr in range(6, 9):
                stw9DrftS4 = np.array([i for i in dtNew1 if i[4] >= 8.5 and i[4] < 9 and i[0] > dr and i[0] <= dr + 1])
                stw9DrftS4 = np.delete(stw9DrftS4, [i for (i, v) in enumerate(stw9DrftS4[:, 8]) if v < (
                np.mean(stw9DrftS4[:, 8]) - np.std(stw9DrftS4[:, 8])) or v > np.mean(stw9DrftS4[:, 8]) + np.std(
                    stw9DrftS4[:, 8])], 0)
                drftStw9.append(stw9DrftS4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 8.5 < STW < 9                   '], )
                data_writer.writerow([], )
                data_writer.writerow(['Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW','RPM'
                                     ,'DaysSinceLastPropCleaning','DaysSinceLastDryDock', 'M/E FOC (kg/min)'])
                for data in stw9DrftS4:
                    data_writer.writerow([data[0], data[1], data[2], data[3], data[4], data[5],data[6],data[7],data[8]])
                data_writer.writerow([], )
            data_writer.writerow([], )
            data_writer.writerow([], )

            drftStw10 = []
            stw10DrftS4 = np.array([i for i in dtNew1 if i[4] >= 9 and i[4] < 9.5 and i[0] <= 9])

            for dr in range(6, 9):
                stw10DrftS4 = np.array([i for i in dtNew1 if i[4] >= 9 and i[4] < 9.5 and i[0] > dr and i[0] <= dr + 1])
                stw10DrftS4 = np.delete(stw10DrftS4, [i for (i, v) in enumerate(stw10DrftS4[:, 8]) if v < (
                np.mean(stw10DrftS4[:, 8]) - np.std(stw10DrftS4[:, 8])) or v > np.mean(stw10DrftS4[:, 8]) + np.std(
                    stw10DrftS4[:, 8])], 0)
                drftStw10.append(stw10DrftS4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 9 < STW < 9.5                   '], )
                data_writer.writerow([], )
                data_writer.writerow(['Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW','RPM'
                                     ,'DaysSinceLastPropCleaning','DaysSinceLastDryDock', 'M/E FOC (kg/min)'])
                for data in stw10DrftS4:
                    data_writer.writerow([data[0], data[1], data[2], data[3], data[4], data[5],data[6],data[7],data[8]])
                data_writer.writerow([], )
            data_writer.writerow([], )
            data_writer.writerow([], )

            drftStw11 = []
            stw11DrftS4 = np.array([i for i in dtNew1 if i[4] >= 9.5 and i[4] < 10 and i[0] <= 9])

            for dr in range(6, 9):
                stw11DrftS4 = np.array(
                    [i for i in dtNew1 if i[4] >= 9.5 and i[4] < 10 and i[0] > dr and i[0] <= dr + 1])
                stw11DrftS4 = np.delete(stw10DrftS4, [i for (i, v) in enumerate(stw11DrftS4[:, 8]) if v < (
                np.mean(stw11DrftS4[:, 8]) - np.std(stw11DrftS4[:, 8])) or v > np.mean(stw11DrftS4[:, 8]) + np.std(
                    stw11DrftS4[:, 8])], 0)
                drftStw11.append(stw11DrftS4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 9.5 < STW < 10                   '], )
                data_writer.writerow([], )
                data_writer.writerow(['Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW','RPM'
                                     ,'DaysSinceLastPropCleaning','DaysSinceLastDryDock', 'M/E FOC (kg/min)'])
                for data in stw11DrftS4:
                    data_writer.writerow([data[0], data[1], data[2], data[3], data[4], data[5],data[6],data[7],data[8]])
                data_writer.writerow([], )
            data_writer.writerow([], )
            data_writer.writerow([], )

            drftStw12 = []
            stw12DrftS4 = np.array([i for i in dtNew1 if i[4] >= 10 and i[4] < 10.5 and i[0] <= 9])

            for dr in range(6, 9):
                stw12DrftS4 = np.array(
                    [i for i in dtNew1 if i[4] >= 10 and i[4] < 10.5 and i[0] > dr and i[0] <= dr + 1])
                stw12DrftS4 = np.delete(stw12DrftS4, [i for (i, v) in enumerate(stw12DrftS4[:, 8]) if v < (
                np.mean(stw12DrftS4[:, 8]) - np.std(stw12DrftS4[:, 8])) or v > np.mean(stw12DrftS4[:, 8]) + np.std(
                    stw12DrftS4[:, 8])], 0)
                drftStw12.append(stw11DrftS4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 10 < STW < 10.5                   '], )
                data_writer.writerow([], )
                data_writer.writerow(['Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW','RPM'
                                     ,'DaysSinceLastPropCleaning','DaysSinceLastDryDock', 'M/E FOC (kg/min)'])
                for data in stw12DrftS4:
                    data_writer.writerow([data[0], data[1], data[2], data[3], data[4], data[5],data[6],data[7],data[8]])
                data_writer.writerow([], )
            data_writer.writerow([], )
            data_writer.writerow([], )

            drftStw13 = []
            #stw13DrftS4 = np.array([i for i in dtNew1 if i[4] >= 10.5 and i[4] < 11 and i[0] <= 9])

            for dr in range(6, 9):
                stw13DrftS4 = np.array(
                    [i for i in dtNew1 if i[4] >= 10.5 and i[4] < 11 and i[0] > dr and i[0] <= dr + 1])
                stw13DrftS4 = np.delete(stw13DrftS4, [i for (i, v) in enumerate(stw13DrftS4[:, 8]) if v < (
                np.mean(stw13DrftS4[:, 8]) - np.std(stw13DrftS4[:, 8])) or v > np.mean(stw13DrftS4[:, 8]) + np.std(
                    stw13DrftS4[:, 8])], 0)
                drftStw13.append(drftStw13)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 10.5 < STW < 11                   '], )
                data_writer.writerow([], )
                data_writer.writerow(['Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW','RPM'
                                     ,'DaysSinceLastPropCleaning','DaysSinceLastDryDock', 'M/E FOC (kg/min)'])
                for data in stw13DrftS4:
                    data_writer.writerow([data[0], data[1], data[2], data[3], data[4], data[5],data[6],data[7],data[8]])
                data_writer.writerow([], )
            data_writer.writerow([], )
            data_writer.writerow([], )

            drftStw14 = []
            #stw14DrftS4 = np.array([i for i in dtNew1 if i[4] >= 11 and i[4] < 11.5 and i[0] <= 9])

            for dr in range(6, 9):
                stw14DrftS4 = np.array(
                    [i for i in dtNew1 if i[4] >= 11 and i[4] < 11.5 and i[0] > dr and i[0] <= dr + 1])
                stw14DrftS4 = np.delete(stw14DrftS4, [i for (i, v) in enumerate(stw14DrftS4[:, 8]) if v < (
                np.mean(stw14DrftS4[:, 8]) - np.std(stw14DrftS4[:, 8])) or v > np.mean(stw14DrftS4[:, 8]) + np.std(
                    stw14DrftS4[:, 8])], 0)
                drftStw14.append(stw14DrftS4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 11 < STW < 11.5                   '], )
                data_writer.writerow([], )
                data_writer.writerow(['Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW','RPM'
                                     ,'DaysSinceLastPropCleaning','DaysSinceLastDryDock', 'M/E FOC (kg/min)'])
                for data in stw14DrftS4:
                    data_writer.writerow([data[0], data[1], data[2], data[3], data[4], data[5],data[6],data[7],data[8]])
                data_writer.writerow([], )
            data_writer.writerow([], )
            data_writer.writerow([], )

            drftStw15 = []
            #stw15DrftS4 = np.array([i for i in dtNew1 if i[4] >= 11.5 and i[4] < 12 and i[0] <= 9])

            for dr in range(6, 9):
                stw15DrftS4 = np.array(
                    [i for i in dtNew1 if i[4] >= 11.5 and i[4] < 12 and i[0] > dr and i[0] <= dr + 1])
                stw15DrftS4 = np.delete(stw15DrftS4, [i for (i, v) in enumerate(stw15DrftS4[:, 8]) if v < (
                np.mean(stw15DrftS4[:, 8]) - np.std(stw15DrftS4[:, 8])) or v > np.mean(stw15DrftS4[:, 8]) + np.std(
                    stw15DrftS4[:, 8])], 0)
                drftStw15.append(stw15DrftS4)
                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 11.5 < STW < 12                   '], )
                data_writer.writerow([], )
                data_writer.writerow(['Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW','RPM'
                                     ,'DaysSinceLastPropCleaning','DaysSinceLastDryDock', 'M/E FOC (kg/min)'])
                for data in stw15DrftS4:
                    data_writer.writerow([data[0], data[1], data[2], data[3], data[4], data[5],data[6],data[7],data[8]])
                data_writer.writerow([], )
            data_writer.writerow([], )
            data_writer.writerow([], )

            drftStw16 = []
            #stw16DrftS4 = np.array([i for i in dtNew1 if i[4] >= 12 and i[4] < 12.5 and i[0] <= 9])

            for dr in range(6, 9):
                stw16DrftS4 = np.array(
                    [i for i in dtNew1 if i[4] >= 12 and i[4] < 12.5 and i[0] > dr and i[0] <= dr + 1])
                stw16DrftS4 = np.delete(stw16DrftS4, [i for (i, v) in enumerate(stw16DrftS4[:, 8]) if v < (
                np.mean(stw16DrftS4[:, 8]) - np.std(stw16DrftS4[:, 8])) or v > np.mean(stw16DrftS4[:, 8]) + np.std(
                    stw16DrftS4[:, 8])], 0)
                drftStw16.append(stw16DrftS4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 12 < STW < 12.5                   '], )
                data_writer.writerow([], )
                data_writer.writerow(['Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW','RPM'
                                     ,'DaysSinceLastPropCleaning','DaysSinceLastDryDock', 'M/E FOC (kg/min)'])
                for data in stw16DrftS4:
                    data_writer.writerow([data[0], data[1], data[2], data[3], data[4], data[5]])
                data_writer.writerow([], )
            data_writer.writerow([], )
            data_writer.writerow([], )

            drftStw17 = []
            #stw17DrftS4 = np.array([i for i in dtNew1 if i[4] >= 12.5 and i[4] < 13 and i[0] <= 9])

            for dr in range(6, 9):
                stw17DrftS4 = np.array(
                    [i for i in dtNew1 if i[4] >= 12 and i[4] < 12.5 and i[0] > dr and i[0] <= dr + 1])
                stw17DrftS4 = np.delete(stw17DrftS4, [i for (i, v) in enumerate(stw17DrftS4[:, 8]) if v < (
                np.mean(stw17DrftS4[:, 8]) - np.std(stw17DrftS4[:, 8])) or v > np.mean(stw17DrftS4[:, 8]) + np.std(
                    stw17DrftS4[:, 8])], 0)
                drftStw17.append(drftStw17)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 12 < STW < 12.5                   '], )
                data_writer.writerow([], )
                data_writer.writerow(['Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW','RPM'
                                     ,'DaysSinceLastPropCleaning','DaysSinceLastDryDock', 'M/E FOC (kg/min)'])
                for data in stw17DrftS4:
                    data_writer.writerow([data[0], data[1], data[2], data[3], data[4], data[5],data[6],data[7],data[8]])
                data_writer.writerow([], )
            data_writer.writerow([], )
            data_writer.writerow([], )

            drftStw18 = []
            #stw18DrftS4 = np.array([i for i in dtNew1 if i[4] >= 13 and i[4] < 13.5 and i[0] <= 9])

            for dr in range(6, 9):
                stw18DrftS4 = np.array(
                    [i for i in dtNew1 if i[4] >= 13 and i[4] < 13.5 and i[0] > dr and i[0] <= dr + 1])

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 13 < STW < 13.5                   '], )
                if stw18DrftS4.shape[0]!=0:
                    stw18DrftS4 = np.delete(stw18DrftS4, [i for (i, v) in enumerate(stw18DrftS4[:, 8]) if v < (
                np.mean(stw18DrftS4[:, 8]) - np.std(stw18DrftS4[:, 8])) or v > np.mean(stw18DrftS4[:, 8]) + np.std(
                    stw18DrftS4[:, 8])], 0)
                    drftStw18.append(stw18DrftS4)

                    data_writer.writerow([], )
                    data_writer.writerow(['Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW',
                                          'M/E FOC (kg/min)','DaysSinceLastPropCleaning','DaysSinceLastDryDock'])
                    for data in stw18DrftS4:
                        data_writer.writerow([data[0], data[1], data[2], data[3], data[4], data[5],data[6],data[7],data[8]])
                    data_writer.writerow([], )
                data_writer.writerow([], )
                data_writer.writerow([], )

            drftStw19 = []
            #stw19DrftS4 = np.array([i for i in dtNew1 if i[4] >= 13.5 and i[4] < 14 and i[0] <= 9])

            for dr in range(6, 9):
                stw19DrftS4 = np.array(
                    [i for i in dtNew1 if i[4] >= 13.5 and i[4] < 14 and i[0] > dr and i[0] <= dr + 1])
                data_writer.writerow(
                    [str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 13.5 < STW < 14                   '], )

                if stw19DrftS4.shape[0]!=0:
                    stw19DrftS4 = np.delete(stw19DrftS4, [i for (i, v) in enumerate(stw19DrftS4[:, 8]) if v < (
                    np.mean(stw19DrftS4[:, 8]) - np.std(stw19DrftS4[:, 8])) or v > np.mean(stw19DrftS4[:, 8]) + np.std(
                        stw19DrftS4[:, 8])], 0)
                    drftStw19.append(stw19DrftS4)

                    data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                    data_writer.writerow([], )
                    data_writer.writerow(['Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW','RPM'
                                         ,'DaysSinceLastPropCleaning','DaysSinceLastDryDock', 'M/E FOC (kg/min)'])
                    for data in stw19DrftS4:
                        data_writer.writerow([data[0], data[1], data[2], data[3], data[4], data[5],data[6],data[7],data[8]])
                data_writer.writerow([], )
            data_writer.writerow([], )
            data_writer.writerow([], )

        ########################################################################################################################################
        ########################################################################################################################################
        ########################################################################################################################################
        ########################################################################################################################################


        drftStw8 = []
        #stw8DrftB4 = np.array([i for i in dtNew1 if i[4] >= 8 and i[4] < 8.5 and i[0] > 9])
        with open('./MT_DELTA_MARIA_draftB_9_analysis.csv', mode='w') as rows:
            for dr in range(9, 14):
                stw8DrftB4 = np.array([i for i in dtNew1 if i[4] >= 8 and i[4] < 8.5 and i[0] > dr and i[0] <= dr + 1])
                stw8DrftB4 = np.delete(stw8DrftB4, [i for (i, v) in enumerate(stw8DrftB4[:, 8]) if v < (
                np.mean(stw8DrftB4[:, 8]) - np.std(stw8DrftB4[:, 8])) or v > np.mean(stw8DrftB4[:, 8]) + np.std(
                    stw8DrftB4[:, 8])], 0)
                drftStw8.append(stw8DrftB4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 8 < STW < 8.5                   '], )
                data_writer.writerow([], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW','RPM'
                                     ,'DaysSinceLastPropCleaning','DaysSinceLastDryDock', 'M/E FOC (kg/min)'])
                for data in stw8DrftB4:
                    data_writer.writerow([data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]])
                data_writer.writerow([], )
            data_writer.writerow([], )
            data_writer.writerow([], )

        #stw8DrftB4 = np.concatenate(stw8DrftB4)

            drftStw9 = []
            #stw9DrftB4 = np.array([i for i in dtNew1 if i[4] >= 8.5 and i[4] < 9 and i[0] > 9])
            for dr in range(9, 14):
                stw9DrftB4 = np.array([i for i in dtNew1 if i[4] >= 8.5 and i[4] < 9 and i[0] > dr and i[0] <= dr + 1])
                stw9DrftB4 = np.delete(stw9DrftB4, [i for (i, v) in enumerate(stw9DrftB4[:, 8]) if v < (
                np.mean(stw9DrftB4[:, 8]) - np.std(stw9DrftB4[:, 8])) or v > np.mean(stw9DrftB4[:, 8]) + np.std(
                    stw9DrftB4[:, 8])], 0)
                drftStw9.append(stw9DrftB4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 8.5 < STW < 9                   '], )
                data_writer.writerow([], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW','RPM'
                                     ,'DaysSinceLastPropCleaning','DaysSinceLastDryDock', 'M/E FOC (kg/min)'])
                for data in stw9DrftB4:
                    data_writer.writerow([data[0], data[1], data[2], data[3], data[4], data[5],data[6],data[7],data[8]])
                data_writer.writerow([], )
            data_writer.writerow([], )
            data_writer.writerow([], )

        #stw9DrftB4 = np.concatenate(drftStw9)

        #drftStw10 = []
        #stw10DrftB4 = np.array([i for i in dtNew1 if i[4] >= 9 and i[4] < 9.5 and i[0] > 9])
            for dr in range(9, 14):
                stw10DrftB4 = np.array([i for i in dtNew1 if i[4] >= 9 and i[4] < 9.5 and i[0] > dr and i[0] <= dr + 1])
                stw10DrftB4 = np.delete(stw10DrftB4, [i for (i, v) in enumerate(stw10DrftB4[:, 8]) if v < (
                np.mean(stw10DrftB4[:, 8]) - np.std(stw10DrftB4[:, 8])) or v > np.mean(stw10DrftB4[:, 8]) + np.std(
                    stw10DrftB4[:, 8])], 0)
                drftStw10.append(stw10DrftB4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 9 < STW < 9.5                  '], )
                data_writer.writerow([], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW','RPM'
                                     ,'DaysSinceLastPropCleaning','DaysSinceLastDryDock', 'M/E FOC (kg/min)'])
                for data in stw10DrftB4:
                    data_writer.writerow([data[0], data[1], data[2], data[3], data[4], data[5],data[6],data[7],data[8]])
                data_writer.writerow([], )
            data_writer.writerow([], )
            data_writer.writerow([], )
        #stw10DrftB4 = np.concatenate(drftStw10)

        #drftStw11 = []
        #stw11DrftB4 = np.array([i for i in dtNew1 if i[4] >= 9.5 and i[4] < 10 and i[0] > 9])
            for dr in range(9, 14):
                stw11DrftB4 = np.array([i for i in dtNew1 if i[4] >= 9.5 and i[4] < 10 and i[0] > dr and i[0] <= dr + 1])
                stw11DrftB4 = np.delete(stw11DrftB4, [i for (i, v) in enumerate(stw11DrftB4[:, 8]) if v < (
                np.mean(stw11DrftB4[:, 8]) - np.std(stw11DrftB4[:, 8])) or v > np.mean(stw11DrftB4[:, 8]) + np.std(
                    stw11DrftB4[:, 8])], 0)
                drftStw11.append(stw11DrftB4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 9.5 < STW < 10                   '], )
                data_writer.writerow([], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW','RPM'
                                     ,'DaysSinceLastPropCleaning','DaysSinceLastDryDock', 'M/E FOC (kg/min)'])
                for data in stw11DrftB4:
                    data_writer.writerow([data[0], data[1], data[2], data[3], data[4], data[5],data[6],data[7],data[8]])
                data_writer.writerow([], )
            data_writer.writerow([], )
            data_writer.writerow([], )

        #stw11DrftB4 = np.concatenate(drftStw11)

        #drftStw12 = []
        #stw12DrftB4 = np.array([i for i in dtNew1 if i[4] >= 10 and i[4] < 10.5 and i[0] > 9])
            for dr in range(9, 14):
                stw12DrftB4 = np.array([i for i in dtNew1 if i[4] >= 10 and i[4] < 10.5 and i[0] > dr and i[0] <= dr + 1])
                stw12DrftB4 = np.delete(stw12DrftB4, [i for (i, v) in enumerate(stw12DrftB4[:, 8]) if v < (
                np.mean(stw12DrftB4[:, 8]) - np.std(stw12DrftB4[:, 8])) or v > np.mean(stw12DrftB4[:, 8]) + np.std(
                    stw12DrftB4[:, 8])], 0)
                drftStw12.append(drftStw12)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 10 < STW < 10.5                   '], )
                data_writer.writerow([], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW','RPM'
                                     ,'DaysSinceLastPropCleaning','DaysSinceLastDryDock', 'M/E FOC (kg/min)'])
                for data in stw12DrftB4:
                    data_writer.writerow([data[0], data[1], data[2], data[3], data[4], data[5],data[6],data[7],data[8]])
                data_writer.writerow([], )
            data_writer.writerow([], )
            data_writer.writerow([], )
        #stw12DrftB4 = np.concatenate(drftStw12)

        #drftStw13 = []
        #stw13DrftB4 = np.array([i for i in dtNew1 if i[4] >= 10.5 and i[4] < 11 and i[0] > 9])
            for dr in range(9, 14):
                stw13DrftB4 = np.array([i for i in dtNew1 if i[4] >= 10.5 and i[4] < 11 and i[0] > dr and i[0] <= dr + 1])
                stw13DrftB4 = np.delete(stw13DrftB4, [i for (i, v) in enumerate(stw13DrftB4[:, 8]) if v < (
                np.mean(stw13DrftB4[:, 8]) - np.std(stw13DrftB4[:, 8])) or v > np.mean(stw13DrftB4[:, 8]) + np.std(
                    stw13DrftB4[:, 8])], 0)
                drftStw13.append(stw13DrftB4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 10.5 < STW < 11                 '], )
                data_writer.writerow([], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW','RPM'
                                     ,'DaysSinceLastPropCleaning','DaysSinceLastDryDock', 'M/E FOC (kg/min)'])
                for data in stw13DrftB4:
                    data_writer.writerow([data[0], data[1], data[2], data[3], data[4], data[5],data[6],data[7],data[8]])
                data_writer.writerow([], )
            data_writer.writerow([], )
            data_writer.writerow([], )
        #stw13DrftB4 = np.concatenate(drftStw13)

        #drftStw14 = []
        #stw14DrftB4 = np.array([i for i in dtNew1 if i[4] >= 11 and i[4] < 11.5 and i[0] > 9])
            for dr in range(9, 14):
                stw14DrftB4 = np.array([i for i in dtNew1 if i[4] >= 11 and i[4] < 11.5 and i[0] > dr and i[0] <= dr + 1])
                stw14DrftB4 = np.delete(stw14DrftB4, [i for (i, v) in enumerate(stw14DrftB4[:, 8]) if v < (
                np.mean(stw14DrftB4[:, 8]) - np.std(stw14DrftB4[:, 8])) or v > np.mean(stw14DrftB4[:, 8]) + np.std(
                    stw14DrftB4[:, 8])], 0)
                drftStw14.append(stw14DrftB4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 11 < STW < 11.5                   '], )
                data_writer.writerow([], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW','RPM'
                                     ,'DaysSinceLastPropCleaning','DaysSinceLastDryDock', 'M/E FOC (kg/min)'])
                for data in stw14DrftB4:
                    data_writer.writerow([data[0], data[1], data[2], data[3], data[4], data[5],data[6],data[7],data[8]])
                data_writer.writerow([], )
            data_writer.writerow([], )
            data_writer.writerow([], )
        #stw14DrftB4 = np.concatenate(drftStw14)

        #drftStw15 = []
        #stw15DrftB4 = np.array([i for i in dtNew1 if i[4] >= 11.5 and i[4] < 12 and i[0] > 9])
            for dr in range(9, 14):
                stw15DrftB4 = np.array([i for i in dtNew1 if i[4] >= 11.5 and i[4] < 12 and i[0] > dr and i[0] <= dr + 1])
                stw15DrftB4 = np.delete(stw15DrftB4, [i for (i, v) in enumerate(stw15DrftB4[:, 8]) if v < (
                np.mean(stw15DrftB4[:, 8]) - np.std(stw15DrftB4[:, 8])) or v > np.mean(stw15DrftB4[:, 8]) + np.std(
                    stw15DrftB4[:, 8])], 0)
                drftStw15.append(stw15DrftB4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 11.5 < STW < 12                  '], )
                data_writer.writerow([], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW','RPM'
                                     ,'DaysSinceLastPropCleaning','DaysSinceLastDryDock', 'M/E FOC (kg/min)'])
                for data in stw15DrftB4:
                    data_writer.writerow([data[0], data[1], data[2], data[3], data[4], data[5],data[6],data[7],data[8]])
                data_writer.writerow([], )
            data_writer.writerow([], )
            data_writer.writerow([], )

        #stw15DrftB4 = np.concatenate(drftStw15)

        #drftStw16 = []
        #stw16DrftB4 = np.array([i for i in dtNew1 if i[4] >= 12 and i[4] < 12.5 and i[0] > 9])
            for dr in range(9, 14):
                stw16DrftB4 = np.array([i for i in dtNew1 if i[4] >= 12 and i[4] < 12.5 and i[0] > dr and i[0] <= dr + 1])
                stw16DrftB4 = np.delete(stw16DrftB4, [i for (i, v) in enumerate(stw16DrftB4[:, 8]) if v < (
                np.mean(stw16DrftB4[:, 8]) - np.std(stw16DrftB4[:, 8])) or v > np.mean(stw16DrftB4[:, 8]) + np.std(
                    stw16DrftB4[:, 8])], 0)
                drftStw16.append(stw16DrftB4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 12 < STW < 12.5                   '], )
                data_writer.writerow([], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW','RPM'
                                     ,'DaysSinceLastPropCleaning','DaysSinceLastDryDock', 'M/E FOC (kg/min)'])
                for data in stw16DrftB4:
                    data_writer.writerow([data[0], data[1], data[2], data[3], data[4], data[5],data[6],data[7],data[8]])
                data_writer.writerow([], )
            data_writer.writerow([], )
            data_writer.writerow([], )

        #stw16DrftB4 = np.concatenate(drftStw16)

        #drftStw17 = []
        #stw17DrftB4 = np.array([i for i in dtNew1 if i[4] >= 12.5 and i[4] < 13 and i[0] > 9])
            for dr in range(9, 14):
                stw17DrftB4 = np.array([i for i in dtNew1 if i[4] >= 12.5 and i[4] < 13 and i[0] > dr and i[0] <= dr + 1])
                data_writer.writerow(
                    [str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 12.5 < STW < 13                 '], )

                if stw17DrftB4.shape[0]!=0:
                    stw17DrftB4 = np.delete(stw17DrftB4, [i for (i, v) in enumerate(stw17DrftB4[:, 8]) if v < (
                    np.mean(stw17DrftB4[:, 8]) - np.std(stw17DrftB4[:, 8])) or v > np.mean(stw17DrftB4[:, 8]) + np.std(
                        stw17DrftB4[:, 8])], 0)
                    drftStw17.append(stw17DrftB4)

                    data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                    data_writer.writerow([], )
                    data_writer.writerow(['Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW',
                                          'M/E FOC (kg/min)', 'DaysSinceLastPropCleaning', 'DaysSinceLastDryDock'])
                    for data in stw17DrftB4:
                        data_writer.writerow([data[0], data[1], data[2], data[3], data[4], data[5],data[6],data[7],data[8]])
                data_writer.writerow([], )
            data_writer.writerow([], )
            data_writer.writerow([], )

        #stw17DrftB4 = np.concatenate(drftStw17)

        #drftStw18 = []
        #stw18DrftB4 = np.array([i for i in dtNew1 if i[4] >= 13 and i[4] < 13.5 and i[0] > 9])
            for dr in range(9, 14):
                stw18DrftB4 = np.array([i for i in dtNew1 if i[4] >= 13 and i[4] < 13.5 and i[0] > dr and i[0] <= dr + 1])
                data_writer.writerow(
                    [str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 13 < STW < 13.5                 '], )

                if stw18DrftB4.shape[0] != 0:
                    stw18DrftB4 = np.delete(stw18DrftB4, [i for (i, v) in enumerate(stw18DrftB4[:, 8]) if v < (
                    np.mean(stw18DrftB4[:, 8]) - np.std(stw18DrftB4[:, 8])) or v > np.mean(stw18DrftB4[:, 8]) + np.std(
                        stw18DrftB4[:, 8])], 0)
                    drftStw18.append(stw18DrftB4)

                    data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                    data_writer.writerow([], )
                    data_writer.writerow(['Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW',
                                          'M/E FOC (kg/min)', 'DaysSinceLastPropCleaning', 'DaysSinceLastDryDock'])
                    for data in stw18DrftB4:
                        data_writer.writerow([data[0], data[1], data[2], data[3], data[4], data[5],data[6],data[7],data[8]])
                data_writer.writerow([], )
            data_writer.writerow([], )
            data_writer.writerow([], )

        #stw18DrftB4 = np.concatenate(drftStw18)

        #drftStw19 = []
        #stw19DrftB4 = np.array([i for i in dtNew1 if i[4] >= 13.5 and i[4] < 14 and i[0] > 9])
            for dr in range(9, 14):
                stw19DrftB4 = np.array([i for i in dtNew1 if i[4] >= 13.5 and i[4] < 14 and i[0] > dr and i[0] <= dr + 1])
                data_writer.writerow(
                    [str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 13.5 < STW < 14                 '], )
                if stw19DrftB4.shape[0] != 0:
                    stw19DrftB4 = np.delete(stw19DrftB4, [i for (i, v) in enumerate(stw19DrftB4[:, 8]) if v < (
                    np.mean(stw19DrftB4[:, 8]) - np.std(stw19DrftB4[:, 8])) or v > np.mean(stw19DrftB4[:, 8]) + np.std(
                        stw19DrftB4[:, 8])], 0)
                    drftStw19.append(stw19DrftB4)

                    data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                    data_writer.writerow([], )
                    data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW','RPM'
                                         ,'DaysSinceLastPropCleaning','DaysSinceLastDryDock', 'M/E FOC (kg/min)'])
                    for data in stw19DrftB4:
                        data_writer.writerow([data[0], data[1], data[2], data[3], data[4], data[5],data[6],data[7],data[8]])
                data_writer.writerow([], )
            data_writer.writerow([], )
            data_writer.writerow([], )

        #tw19DrftB4 = np.concatenate(drftStw19)

        x=0

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
        dtNew = dataNew.values[ 484362:569386, : ].astype(float)

        draftMean = (dtNew[:, 1] + dtNew[:, 2]) / 2
        dtNew[:, 0] = np.round(draftMean,3)

        dtNew = np.delete(dtNew, [ i for (i, v) in enumerate(dtNew[ 0:, 0 ]) if v <= 6 ], 0)
        dtNew = np.delete(dtNew, [ i for (i, v) in enumerate(dtNew[ 0:, 5 ]) if v <= 8 or v > 14 ], 0)
        dtNew = np.delete(dtNew, [ i for (i, v) in enumerate(dtNew[ 0:, 6 ]) if v <= 0  or v > 90], 0) # or (v>0 and v <=10)
        #dtNew = np.delete(dtNew, [ i for (i, v) in enumerate(dtNew[ 0:, 7 ]) ], 0) # if v <= 7

        dtNew = dtNew[~np.isnan(dtNew).any(axis=1)]

        trim = (dtNew[:,1] - dtNew[:,2])


        dtNew1 =np.array(np.append(dtNew[:,0].reshape(-1,1),np.asmatrix([np.round(trim,3),dtNew[:,3],dtNew[:,4],dtNew[:,5],dtNew[:,7]]).T,axis=1))

        ##statistics STD
        #draftSmaller6 = np.array([drft for drft in dtNew1 if drft[0]< 6])
        #draftBigger6 = np.array([ drft for drft in dtNew1 if drft[0] > 6 ])
        #########DRAFT < 9
        drftStw8 = []
        #stw8DrftS4 = np.array([i for i in dtNew1 if i[4]>=8 and i[4]<8.5 and i[0]<=9])

        for dr in range(6,9):
            stw8DrftS4 =np.array([ i for i in dtNew1  if i[4]>=8 and i[4]<8.5 and i[ 0 ] > dr and  i[ 0 ] <= dr+1 ])
            if stw8DrftS4.shape[0] != 0:
                stw8DrftS4 = np.delete(stw8DrftS4, [ i for (i, v) in enumerate(stw8DrftS4[:,5]) if  v < (np.mean(stw8DrftS4[ :, 5 ]) - np.std(stw8DrftS4[ :, 5 ])) or   v > np.mean(stw8DrftS4[ :, 5 ]) + np.std(stw8DrftS4[ :, 5 ]) ], 0)
                drftStw8.append(stw8DrftS4)
        stw8DrftS4 = np.concatenate(drftStw8)

        drftStw9 = []
        stw9DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 8.5 and i[ 4 ] < 9 and i[ 0 ] <= 9 ])

        for dr in range(6,9):
                stw9DrftS4 =np.array([ i for i in dtNew1  if i[ 4 ] >= 8.5 and i[ 4 ] < 9 and i[ 0 ] > dr and  i[ 0 ] <= dr+1 ])
                stw9DrftS4 = np.delete(stw9DrftS4, [ i for (i, v) in enumerate(stw9DrftS4[:,5]) if  v < (np.mean(stw9DrftS4[ :, 5 ]) - np.std(stw9DrftS4[ :, 5 ])) or   v > np.mean(stw9DrftS4[ :, 5 ]) + np.std(stw9DrftS4[ :, 5 ]) ], 0)
                drftStw9.append(stw9DrftS4)

        stw9DrftS4 = np.concatenate(drftStw9)

        drftStw10 = []
        #stw10DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 9 and i[ 4 ] < 9.5 and i[ 0 ] <= 9 ])

        for dr in range(6,9):
                stw10DrftS4 =np.array([ i for i in dtNew1  if i[ 4 ] >= 9 and i[ 4 ] < 9.5 and i[ 0 ] > dr and  i[ 0 ] <= dr+1 ])
                stw10DrftS4 = np.delete(stw10DrftS4, [ i for (i, v) in enumerate(stw10DrftS4[:,5]) if  v < (np.mean(stw10DrftS4[ :, 5 ]) - np.std(stw10DrftS4[ :, 5 ])) or   v > np.mean(stw10DrftS4[ :, 5 ]) + np.std(stw10DrftS4[ :, 5 ]) ], 0)
                drftStw10.append(stw10DrftS4)

        stw10DrftS4 = np.concatenate(drftStw10)

        drftStw11 = []
        #stw11DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 9.5 and i[ 4 ] < 10 and i[ 0 ] <= 9 ])

        for dr in range(6,9):
                stw11DrftS4 =np.array([ i for i in dtNew1  if i[ 4 ] >= 9.5 and i[ 4 ] < 10 and i[ 0 ] > dr and  i[ 0 ] <= dr+1 ])
                stw11DrftS4 = np.delete(stw10DrftS4, [ i for (i, v) in enumerate(stw11DrftS4[:,5]) if  v < (np.mean(stw11DrftS4[ :, 5 ]) - np.std(stw11DrftS4[ :, 5 ])) or   v > np.mean(stw11DrftS4[ :, 5 ]) + np.std(stw11DrftS4[ :, 5 ]) ], 0)
                drftStw11.append(stw11DrftS4)

        stw11DrftS4 = np.concatenate(drftStw11)

        drftStw12 = []
        #stw12DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 10 and i[ 4 ] < 10.5 and i[ 0 ] <= 9 ])

        for dr in range(6,9):
                stw12DrftS4 =np.array([ i for i in dtNew1  if i[ 4 ] >= 10 and i[ 4 ] < 10.5 and i[ 0 ] > dr and  i[ 0 ] <= dr+1 ])
                stw12DrftS4 = np.delete(stw12DrftS4, [ i for (i, v) in enumerate(stw12DrftS4[:,5]) if  v < (np.mean(stw12DrftS4[ :, 5 ]) - np.std(stw12DrftS4[ :, 5 ])) or   v > np.mean(stw12DrftS4[ :, 5 ]) + np.std(stw12DrftS4[ :, 5 ]) ], 0)
                drftStw12.append(stw12DrftS4)

        stw12DrftS4 = np.concatenate(drftStw12)

        drftStw13 = []
        #stw13DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 10.5 and i[ 4 ] < 11 and i[ 0 ] <= 9 ])

        for dr in range(6,9):
                stw13DrftS4 =np.array([ i for i in dtNew1  if i[ 4 ] >= 10.5 and i[ 4 ] < 11 and i[ 0 ] > dr and  i[ 0 ] <= dr+1 ])
                stw13DrftS4 = np.delete(stw13DrftS4, [ i for (i, v) in enumerate(stw13DrftS4[:,5]) if  v < (np.mean(stw13DrftS4[ :, 5 ]) - np.std(stw13DrftS4[ :, 5 ])) or   v > np.mean(stw13DrftS4[ :, 5 ]) + np.std(stw13DrftS4[ :, 5 ]) ], 0)
                drftStw13.append(stw13DrftS4)

        stw13DrftS4 = np.concatenate(drftStw13)

        drftStw14 = []
        #stw14DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 11 and i[ 4 ] < 11.5 and i[ 0 ] <= 9 ])

        for dr in range(6,9):
                stw14DrftS4 =np.array([ i for i in dtNew1  if i[ 4 ] >= 11 and i[ 4 ] < 11.5 and i[ 0 ] > dr and  i[ 0 ] <= dr+1 ])
                stw14DrftS4 = np.delete(stw14DrftS4, [ i for (i, v) in enumerate(stw14DrftS4[:,5]) if  v < (np.mean(stw14DrftS4[ :, 5 ]) - np.std(stw14DrftS4[ :, 5 ])) or   v > np.mean(stw14DrftS4[ :, 5 ]) + np.std(stw14DrftS4[ :, 5 ]) ], 0)
                drftStw14.append(stw14DrftS4)

        stw14DrftS4 = np.concatenate(drftStw14)


        drftStw15 = []
        stw15DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 11.5 and i[ 4 ] < 12 and i[ 0 ] <= 9 ])

        for dr in range(6,9):
                stw15DrftS4 =np.array([ i for i in dtNew1  if i[ 4 ] >= 11.5 and i[ 4 ] < 12 and i[ 0 ] > dr and  i[ 0 ] <= dr+1 ])
                stw15DrftS4 = np.delete(stw15DrftS4, [ i for (i, v) in enumerate(stw15DrftS4[:,5]) if  v < (np.mean(stw15DrftS4[ :, 5 ]) - np.std(stw15DrftS4[ :, 5 ])) or   v > np.mean(stw15DrftS4[ :, 5 ]) + np.std(stw15DrftS4[ :, 5 ]) ], 0)
                drftStw15.append(stw15DrftS4)

        stw15DrftS4 = np.concatenate(drftStw15)

        drftStw16 = []
        #stw16DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 12 and i[ 4 ] < 12.5 and i[ 0 ] <= 9 ])
        for dr in range(6,9):
                stw16DrftS4 =np.array([ i for i in dtNew1  if i[ 4 ] >= 12 and i[ 4 ] < 12.5 and i[ 0 ] > dr and  i[ 0 ] <= dr+1 ])
                stw16DrftS4 = np.delete(stw16DrftS4, [ i for (i, v) in enumerate(stw16DrftS4[:,5]) if  v < (np.mean(stw16DrftS4[ :, 5 ]) - np.std(stw16DrftS4[ :, 5 ])) or   v > np.mean(stw16DrftS4[ :, 5 ]) + np.std(stw16DrftS4[ :, 5 ]) ], 0)
                drftStw16.append(stw16DrftS4)

        stw16DrftS4 = np.concatenate(drftStw16)

        drftStw17 = []
        #stw17DrftS4 = np.array([i for i in dtNew1 if i[4] >= 12.5 and i[4] < 13 and i[0] <= 9])

        for dr in range(6,9):
                stw17DrftS4 =np.array([ i for i in dtNew1  if i[ 4 ] >= 12 and i[ 4 ] < 12.5 and i[ 0 ] > dr and  i[ 0 ] <= dr+1 ])
                if stw17DrftS4.shape[0] != 0:
                    stw17DrftS4 = np.delete(stw17DrftS4, [ i for (i, v) in enumerate(stw17DrftS4[:,5]) if  v < (np.mean(stw17DrftS4[ :, 5 ]) - np.std(stw17DrftS4[ :, 5 ])) or   v > np.mean(stw17DrftS4[ :, 5 ]) + np.std(stw17DrftS4[ :, 5 ]) ], 0)
                    drftStw17.append(stw17DrftS4)

        stw17DrftS4 = np.concatenate(drftStw17)

        drftStw18 = []
        #stw18DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 13 and i[ 4 ] < 13.5 and i[ 0 ] <= 9 ])
        for dr in range(6,9):
            stw18DrftS4 =np.array([ i for i in dtNew1  if i[ 4 ] >= 13 and i[ 4 ] < 13.5 and i[ 0 ] > dr and  i[ 0 ] <= dr+1 ])
            if stw18DrftS4.shape[0] != 0:
                stw18DrftS4 = np.delete(stw18DrftS4, [ i for (i, v) in enumerate(stw18DrftS4[:,5]) if  v < (np.mean(stw18DrftS4[ :, 5 ]) - np.std(stw18DrftS4[ :, 5 ])) or   v > np.mean(stw18DrftS4[ :, 5 ]) + np.std(stw18DrftS4[ :, 5 ]) ], 0)
                drftStw18.append(stw18DrftS4)

        stw18DrftS4 = np.concatenate(drftStw18)


        drftStw19 = []
       # stw19DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 13.5 and i[ 4 ] < 14 and i[ 0 ] <= 9 ])
        for dr in range(6,9):
                stw19DrftS4 =np.array([ i for i in dtNew1  if i[ 4 ] >= 13.5 and i[ 4 ] < 14 and i[ 0 ] > dr and  i[ 0 ] <= dr+1 ])
                if stw19DrftS4.shape[0] != 0:
                    stw19DrftS4 = np.delete(stw19DrftS4, [ i for (i, v) in enumerate(stw19DrftS4[:,5]) if  v < (np.mean(stw19DrftS4[ :, 5 ]) - np.std(stw19DrftS4[ :, 5 ])) or   v > np.mean(stw19DrftS4[ :, 5 ]) + np.std(stw19DrftS4[ :, 5 ]) ], 0)
                    drftStw19.append(stw19DrftS4)

        stw19DrftS4 = np.concatenate(drftStw19)


        ##################################################################################
        ##################################################################################
        ##################################################################################
        #########DRAFT > 9
        drftStw8=[]
        stw8DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 8 and i[ 4 ] < 8.5 and i[ 0 ] > 9   ])
        for dr in range(9,14):
                stw8DrftB4 =np.array([ i for i in dtNew1 if i[ 4 ] >= 8 and i[ 4 ] < 8.5 and i[ 0 ] > dr and  i[ 0 ] <= dr+1 ])
                stw8DrftB4 = np.delete(stw8DrftB4, [ i for (i, v) in enumerate(stw8DrftB4[:,5]) if  v < (np.mean(stw8DrftB4[ :, 5 ]) - np.std(stw8DrftB4[ :, 5 ])) or   v > np.mean(stw8DrftB4[ :, 5 ]) + np.std(stw8DrftB4[ :, 5 ]) ], 0)
                drftStw8.append(stw8DrftB4)
        stw8DrftB4 = np.concatenate(drftStw8)

        drftStw9 = []
        stw9DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 8.5 and i[ 4 ] <9 and i[ 0 ] > 9 ])
        for dr in range(9,14):
                stw9DrftB4 =np.array([ i for i in dtNew1 if i[ 4 ] >= 8.5 and i[ 4 ] <9  and i[ 0 ] > dr and  i[ 0 ] <= dr+1 ])
                stw9DrftB4 = np.delete(stw9DrftB4, [i for (i, v) in enumerate(stw9DrftB4[:, 5]) if v < (np.mean(stw9DrftB4[:, 5]) - np.std(stw9DrftB4[:, 5])) or v > np.mean(stw9DrftB4[:, 5]) + np.std(stw9DrftB4[:, 5])], 0)
                drftStw9.append(stw9DrftB4)

        stw9DrftB4 = np.concatenate(drftStw9)

        drftStw10=[]
        stw10DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 9 and i[ 4 ] < 9.5 and i[ 0 ] > 9 ])
        for dr in range(9,14):
                stw10DrftB4 =np.array([ i for i in dtNew1 if i[ 4 ] >= 9 and i[ 4 ] < 9.5  and i[ 0 ] > dr and  i[ 0 ] <= dr+1 ])
                stw10DrftB4 = np.delete(stw10DrftB4, [i for (i, v) in enumerate(stw10DrftB4[:, 5]) if v < (np.mean(stw10DrftB4[:, 5]) - np.std(stw10DrftB4[:, 5])) or v > np.mean(stw10DrftB4[:, 5]) + np.std(stw10DrftB4[:, 5])], 0)
                drftStw10.append(stw10DrftB4)
        stw10DrftB4 = np.concatenate(drftStw10)

        drftStw11 = []
        stw11DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 9.5 and i[ 4 ] < 10 and i[ 0 ] > 9 ])
        for dr in range(9,14):
                stw11DrftB4 =np.array([ i for i in dtNew1 if i[ 4 ] >= 9.5 and i[ 4 ] < 10 and i[ 0 ] > dr and  i[ 0 ] <= dr+1 ])
                stw11DrftB4 = np.delete(stw11DrftB4, [i for (i, v) in enumerate(stw11DrftB4[:, 5]) if v < (np.mean(stw11DrftB4[:, 5]) - np.std(stw11DrftB4[:, 5])) or v > np.mean(stw11DrftB4[:, 5]) + np.std(stw11DrftB4[:, 5])], 0)
                drftStw11.append(stw11DrftB4)
        stw11DrftB4 = np.concatenate(drftStw11)

        drftStw12 = []
        stw12DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 10 and i[ 4 ] < 10.5 and i[ 0 ] > 9 ])
        for dr in range(9,14):
            stw12DrftB4 =np.array([ i for i in dtNew1 if i[ 4 ] >= 10 and i[ 4 ] < 10.5 and i[ 0 ] > dr and  i[ 0 ] <= dr+1 ])
            stw12DrftB4 = np.delete(stw12DrftB4, [i for (i, v) in enumerate(stw12DrftB4[:, 5]) if v < (np.mean(stw12DrftB4[:, 5]) - np.std(stw12DrftB4[:, 5])) or v > np.mean(stw12DrftB4[:, 5]) + np.std(stw12DrftB4[:, 5])], 0)
            drftStw12.append(stw12DrftB4)

        stw12DrftB4 = np.concatenate(drftStw12)

        drftStw13 = []
        stw13DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 10.5 and i[ 4 ] < 11 and i[ 0 ] > 9 ])
        for dr in range(9,14):
            stw13DrftB4 =np.array([ i for i in dtNew1 if i[ 4 ] >= 10.5 and i[ 4 ] < 11 and i[ 0 ] > dr and  i[ 0 ] <= dr+1 ])
            stw13DrftB4 = np.delete(stw13DrftB4, [i for (i, v) in enumerate(stw13DrftB4[:, 5]) if v < (np.mean(stw13DrftB4[:, 5]) - np.std(stw13DrftB4[:, 5])) or v > np.mean(stw13DrftB4[:, 5]) + np.std(stw13DrftB4[:, 5])], 0)
            drftStw13.append(stw13DrftB4)
        stw13DrftB4 = np.concatenate(drftStw13)

        drftStw14 = []
        stw14DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 11 and i[ 4 ] < 11.5 and i[ 0 ] > 9 ])
        for dr in range(9,14):
            stw14DrftB4 =np.array([ i for i in dtNew1 if i[ 4 ] >= 11 and i[ 4 ] < 11.5 and i[ 0 ] > dr and  i[ 0 ] <= dr+1 ])
            stw14DrftB4 = np.delete(stw14DrftB4, [i for (i, v) in enumerate(stw14DrftB4[:, 5]) if v < (np.mean(stw14DrftB4[:, 5]) - np.std(stw14DrftB4[:, 5])) or v > np.mean(stw14DrftB4[:, 5]) + np.std(stw14DrftB4[:, 5])], 0)
            drftStw14.append(stw14DrftB4)
        stw14DrftB4 = np.concatenate(drftStw14)

        drftStw15 = []
        stw15DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 11.5 and i[ 4 ] < 12 and i[ 0 ] > 9 ])
        for dr in range(9,14):
            stw15DrftB4 =np.array([ i for i in dtNew1 if i[ 4 ] >= 11.5 and i[ 4 ] < 12 and i[ 0 ] > dr and  i[ 0 ] <= dr+1 ])
            stw15DrftB4 = np.delete(stw15DrftB4, [i for (i, v) in enumerate(stw15DrftB4[:, 5]) if v < (np.mean(stw15DrftB4[:, 5]) - np.std(stw15DrftB4[:, 5])) or v > np.mean(stw15DrftB4[:, 5]) + np.std(stw15DrftB4[:, 5])], 0)
            drftStw15.append(stw15DrftB4)
        stw15DrftB4 = np.concatenate(drftStw15)

        drftStw16 = []
        stw16DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 12 and i[ 4 ] < 12.5 and i[ 0 ] > 9 ])
        for dr in range(9,14):
            stw16DrftB4 =np.array([ i for i in dtNew1 if i[ 4 ] >= 12 and i[ 4 ] < 12.5 and i[ 0 ] > dr and  i[ 0 ] <= dr+1 ])
            stw16DrftB4 = np.delete(stw16DrftB4, [i for (i, v) in enumerate(stw16DrftB4[:, 5]) if v < (np.mean(stw16DrftB4[:, 5]) - np.std(stw16DrftB4[:, 5])) or v > np.mean(stw16DrftB4[:, 5]) + np.std(stw16DrftB4[:, 5])], 0)
            drftStw16.append(stw16DrftB4)
        stw16DrftB4 = np.concatenate(drftStw16)

        drftStw17 = []
        stw17DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 12.5 and i[ 4 ] < 13 and i[ 0 ] > 9 ])
        for dr in range(9,14):
            stw17DrftB4 =np.array([ i for i in dtNew1 if i[ 4 ] >= 12.5 and i[ 4 ] < 13 and i[ 0 ] > dr and  i[ 0 ] <= dr+1 ])
            if stw17DrftB4.shape[0]!=0:
                stw17DrftB4 = np.delete(stw17DrftB4, [i for (i, v) in enumerate(stw17DrftB4[:, 5]) if v < (np.mean(stw17DrftB4[:, 5]) - np.std(stw17DrftB4[:, 5])) or v > np.mean(stw17DrftB4[:, 5]) + np.std(stw17DrftB4[:, 5])], 0)
                drftStw17.append(stw17DrftB4)
        stw17DrftB4 = np.concatenate(drftStw17)

        drftStw18 = []
        stw18DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 13 and i[ 4 ] < 13.5 and i[ 0 ] > 9 ])
        for dr in range(9,14):
            stw18DrftB4 =np.array([ i for i in dtNew1 if i[ 4 ] >= 13 and i[ 4 ] < 13.5  and i[ 0 ] > dr and  i[ 0 ] <= dr+1 ])
            if stw18DrftB4.shape[0] != 0:
                stw18DrftB4 = np.delete(stw18DrftB4, [i for (i, v) in enumerate(stw18DrftB4[:, 5]) if v < (np.mean(stw18DrftB4[:, 5]) - np.std(stw18DrftB4[:, 5])) or v > np.mean(stw18DrftB4[:, 5]) + np.std(stw18DrftB4[:, 5])], 0)
                drftStw18.append(stw18DrftB4)
        stw18DrftB4 = np.concatenate(drftStw18)

        drftStw19 = []
        stw19DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 13.5 and i[ 4 ] < 14 and i[ 0 ] > 9 ])
        for dr in range(9,14):
            stw19DrftB4 =np.array([ i for i in dtNew1 if i[ 4 ] >= 13.5 and i[ 4 ] < 14 and i[ 0 ] > dr and  i[ 0 ] <= dr+1 ])
            if stw19DrftB4.shape[0] != 0:
                stw19DrftB4 = np.delete(stw19DrftB4, [i for (i, v) in enumerate(stw19DrftB4[:, 5]) if v < (np.mean(stw19DrftB4[:, 5]) - np.std(stw19DrftB4[:, 5])) or v > np.mean(stw19DrftB4[:, 5]) + np.std(stw19DrftB4[:, 5])], 0)
                drftStw19.append(stw19DrftB4)
        stw19DrftB4 = np.concatenate(drftStw19)


        ################################################
        concatS = np.concatenate([stw8DrftS4,stw9DrftS4,stw10DrftS4,stw11DrftS4,stw12DrftS4,stw13DrftS4,stw14DrftS4,stw15DrftS4,stw16DrftS4,stw17DrftS4,stw18DrftS4,stw19DrftS4])
        concatB = np.concatenate(
            [ stw8DrftB4, stw9DrftB4, stw10DrftB4, stw11DrftB4, stw12DrftB4, stw13DrftB4, stw14DrftB4, stw15DrftB4,
              stw16DrftB4, stw17DrftB4, stw18DrftB4, stw19DrftB4 ])
        dtNew1 = np.concatenate([concatB,concatS])
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


        UnseenSeriesX = dtNew1[10000:11000, 0:5]
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

        unseenFOC = dtNew1[10000:11000, 5 ]
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