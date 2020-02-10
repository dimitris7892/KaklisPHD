import numpy as np
import pandas as pd
from scipy.stats import ks_2samp,chisquare,chi2_contingency
import math
import pyearth as sp
import matplotlib
import random
#import pyodbc
import csv
import datetime
from dateutil.rrule import rrule, DAILY, MINUTELY

class BaseSeriesReader:

    def readNewDataset(self):
        ####STATSTICAL ANALYSIS PENELOPE

        data = pd.read_csv('./Mapped_data/PERSEFONE_mapped_new.csv')
        data1 = pd.read_csv('./Mapped_data/PERSEFONE_mapped_new_1.csv')
        data2 = pd.read_csv('./Mapped_data/PERSEFONE_mapped_new_2.csv')
        data3 = pd.read_csv('./Mapped_data/PERSEFONE_mapped_3.csv')
        data4 = pd.read_csv('./Mapped_data/PERSEFONE_mapped_4.csv')
        data5 = pd.read_csv('./Mapped_data/PERSEFONE_mapped_5.csv')
        data6 = pd.read_csv('./Mapped_data/PERSEFONE_mapped_6.csv')
        data7 = pd.read_csv('./Mapped_data/PERSEFONE_mapped_7.csv')
        data8 = pd.read_csv('./Mapped_data/PERSEFONE_mapped_8.csv')
        data9 = pd.read_csv('./Mapped_data/PERSEFONE_mapped_9.csv')
        data10 = pd.read_csv('./Mapped_data/PERSEFONE_mapped_10.csv')

        dtNew = data.values  # .astype(float)
        dtNew1 = data1.values  # .astype(float)
        dtNew2 = data2.values
        dtNew3 = data3.values
        dtNew4 = data4.values
        dtNew5 = data5.values
        dtNew6 = data6.values
        dtNew7 = data7.values

        dtNew8 = data8.values
        dtNew9 = data9.values
        dtNew10 = data10.values

        dataNew = np.concatenate([ dtNew, dtNew1, dtNew2 ])
        ballastDt = np.array([ k for k in dataNew if k[ 5 ] == 'B' ])[ :, 0:5 ].astype(float)
        ladenDt = np.array([ k for k in dataNew if k[ 5 ] == 'L' ])[ :, 0:5 ].astype(float)
        ####################
        ####################
        ####################
        dataV = pd.read_csv('./data/PENELOPE_1-28_02_19.csv')

        dataV = dataV.drop([ 't_stamp', 'vessel status' ], axis=1)
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

            relDir = windDir[ i ] - Vcourse[ i ]
            relDir += 360 if relDir < 0 else relDir
            #################################################
            relWindDirs.append(relDir)
            relWindSpeeds.append(relWindSpeed)
        newMappedData = np.array(
            np.append(dtNew[ :, : ],
                      np.asmatrix([ np.array(relWindSpeeds), np.array(relWindDirs), np.array(draftsVmapped) ]).T,
                      axis=1)).astype(float)

        #####################
        ############################MAP TELEGRAM DATES FOR DRAFT

        x = 0


    def GenericParserForDataExtraction(self, company, systemType, fileType, granularity, fileName):
        if systemType == 'LAROS':

            UNK = {"data": [ ]}
            draft = {"data": [ ]}
            draftAFT = {"data": [ ]}
            draftFORE = {"data": [ ]}
            windSpeed = {"data": [ ]}
            windAngle = {"data": [ ]}
            SpeedOvg = {"data": [ ]}
            STW = {"data": [ ]}
            rpm = {"data": [ ]}
            power = {"data": [ ]}
            foc = {"data": [ ]}
            foct = {"data": [ ]}
            dateD = {"data": [ ]}
            latitude = {"data": [ ]}
            longitude = {"data": [ ]}
            vcourse = {"data": [ ]}
            # print(sum(1 for line in open('.data/LAROS/'+fileName)))
            with open('./data/LAROS/' + fileName) as csv_file:

                csv_reader = csv.reader(csv_file, delimiter=';')
                line_count = 0
                for row in csv_reader:
                    try:
                        if row[ 1 ] == "STATUS_PARAMETER_ID":
                            continue
                        id = row[ 1 ]
                        date = row[ 4 ].split(" ")[ 0 ]
                        hhMMss = row[ 4 ].split(" ")[ 1 ]
                        newDate = date + " " + ":".join(hhMMss.split(":")[ 0:2 ])
                        dt = datetime.datetime.strptime(newDate, '%Y-%m-%d %H:%M')
                        value = row[ 3 ]
                        siteId = row[ 2 ]
                        # if dt > datetime.datetime.strptime('2018-05-20', '%Y-%m-%d'):
                        # print(dt)
                        datt = {}
                        datt[ "value" ] = 0
                        datt[ "dt" ] = dt
                        dateD[ "data" ].append(datt)

                        if id == '7':
                            unknown = {}
                            unknown[ "value" ] = value
                            unknown[ "dt" ] = dt
                            UNK[ "data" ].append(unknown)

                        if id == '35':
                            lat = {}
                            lat[ "value" ] = value
                            lat[ "dt" ] = dt
                            latitude[ "data" ].append(lat)
                        if id == '38':
                            lon = {}
                            lon[ "value" ] = value
                            lon[ "dt" ] = dt
                            longitude[ "data" ].append(lon)

                        if id == '65':
                            course = {}
                            course[ "value" ] = value
                            course[ "dt" ] = dt
                            vcourse[ "data" ].append(course)

                        if id == '21':
                            drftA = {}
                            drftA[ "value" ] = value
                            drftA[ "dt" ] = dt
                            draftAFT[ "data" ].append(drftA)

                        if id == '67':
                            stw = {}
                            stw[ "value" ] = value
                            stw[ "dt" ] = dt
                            STW[ "data" ].append(stw)

                        if id == '22':
                            drftF = {}
                            drftF[ "value" ] = value
                            drftF[ "dt" ] = dt
                            draftFORE[ "data" ].append(drftF)

                        if id == '61':
                            drft = {}
                            drft[ "value" ] = value
                            drft[ "dt" ] = dt
                            draft[ "data" ].append(drft)

                        if id == '81':
                            wa = {}
                            wa[ "value" ] = value
                            wa[ "dt" ] = dt
                            windAngle[ "data" ].append(wa)

                        if id == '82':
                            ws = {}
                            ws[ "value" ] = value
                            ws[ "dt" ] = dt
                            windSpeed[ "data" ].append(ws)

                        if id == '84':
                            so = {}
                            so[ "value" ] = value
                            so[ "dt" ] = dt
                            SpeedOvg[ "data" ].append(so)

                        if id == '89':
                            rpms = {}
                            rpms[ "value" ] = value
                            rpms[ "dt" ] = dt
                            rpm[ "data" ].append(rpms)

                        if id == '137':
                            fo = {}
                            fo[ "value" ] = value * 144  ## converion of kg/h to MT/day
                            fo[ "dt" ] = dt
                            foc[ "data" ].append(fo)

                        if id == '136':
                            fot = {}
                            fot[ "value" ] = value
                            fot[ "dt" ] = dt
                            foct[ "data" ].append(fot)

                        if id == '353':
                            pow = {}
                            pow[ "value" ] = value
                            pow[ "dt" ] = dt
                            power[ "data" ].append(pow)

                        line_count += 1

                        if int(id) > 353: break

                        # print('Processed '+str(line_count)+ ' lines.')
                    except:
                        p = 0
                        print('Processed ' + str(line_count) + ' lines.')
            LAT = ""
            LON = ""
            dt = ""
            with open(fileName + '.csv', mode='w') as data:
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [ 'Lat', 'Lon', 'DateTime', 'Vessel Course', 'Speed Ovg', 'Draft', 'Trim', 'WindSpeed', 'WindAngle',
                      'STW', 'Rpm', 'M/E FOC (MT/day)' ])

                for i in range(0, len(UNK[ 'data' ])):
                    dt = dateD[ 'data' ][ i ][ 'dt' ]
                    DRAFT = ""
                    LAT = ""
                    LON = ""
                    COURSE = ""
                    DRAFT_AFT = ""
                    DRAFT_FORE = ""
                    TRIM = ""
                    RPM = ""
                    SOVG = ""
                    SPEED_TW = ""
                    FOC = ""
                    WA = ""
                    WS = ""

                    try:
                        LAT = latitude[ 'data' ][ i ][ 'value' ]
                    except:
                        x = 0

                    try:
                        LON = longitude[ 'data' ][ i ][ 'value' ]
                    except:
                        x = 0
                    try:
                        COURSE = vcourse[ 'data' ][ i ][ 'value' ]
                    except:
                        x = 0

                    try:
                        DRAFT = draft[ 'data' ][ i ][ 'value' ]
                    except:
                        x = 0
                    try:
                        DRAFT_AFT = draftAFT[ 'data' ][ i ][ 'value' ]
                    except:
                        x = 0

                    try:
                        DRAFT_FORE = draftFORE[ 'data' ][ i ][ 'value' ]
                    except:
                        x = 0

                    try:
                        SPEED_TW = STW[ 'data' ][ i ][ 'value' ]
                    except:
                        x = 0

                    try:
                        FOC = foct[ 'data' ][ i ][ 'value' ]
                    except:
                        x = 0

                    try:
                        WA = windAngle[ 'data' ][ i ][ 'value' ]
                    except:
                        x = 0
                    try:
                        WS = windSpeed[ 'data' ][ i ][ 'value' ]
                    except:
                        x = 0

                    try:
                        RPM = rpm[ 'data' ][ i ][ 'value' ]
                    except:
                        x = 0
                    try:
                        SOVG = SpeedOvg[ 'data' ][ i ][ 'value' ]
                    except:
                        x = 0
                    #####
                    TRIM = DRAFT_AFT - DRAFT_FORE

                    data_writer.writerow(
                        [ LAT, LON, str(dt), COURSE, SOVG, DRAFT, TRIM, WS, WA, SPEED_TW, RPM, FOC ])

            date = str(dt).split(" ")[ 0 ]
            day = '0' + date.split("/")[ 0 ] if date.split("/")[ 0 ].__len__() == 1 else date.split("/")[ 0 ]
            month = '0' + date.split("/")[ 1 ] if date.split("/")[ 1 ].__len__() == 1 else date.split("/")[ 1 ]
            year = date.split("/")[ 2 ]
            hhMM = str(dt).split(" ")[ 1 ]
            h = int(hhMM.split(":")[ 0 ])
            M = int(hhMM.split(":")[ 1 ])
            if int(M) > 30:
                h = + 1
            if h % 3 == 2:
                h = + 1
            else:
                h = h - h % 3
            dbDate = year + month + day + str(h)

            conn = pyodbc.connect('DRIVER={SQL Server};SERVER=WEATHERSERVER;'
                                  'DATABASE=WeatherHistoryDB;'
                                  'UID=sa;'
                                  'PWD=sa1!')

            cursor = conn.cursor()
            cursor.execute('SELECT * FROM WeatherHistoryDB.dbo.' + dbDate + ' WHERE lat=' + LAT + 'and lon=' + LON)

            conn = pyodbc.connect('DRIVER={SQL Server};SERVER=WEATHERSERVER_DEV;'
                                  'DATABASE=millenia;'
                                  'UID=sa;'
                                  'PWD=sa1!')

            cursor = conn.cursor()
            cursor.execute('SELECT * FROM millenia.dbo.data')


    def readExtractNewDataset(self, company):
        ####################
        ####################
        ####################
        if company == 'MILLENIA':

            draft = {"data": [ ]}
            course = {"data": [ ]}
            trim = {"data": [ ]}
            windSpeed = {"data": [ ]}
            windAngle = {"data": [ ]}
            blFlag = {"data": [ ]}
            STW = {"data": [ ]}
            rpm = {"data": [ ]}

            foc = {"data": [ ]}

            dateD = {"data": [ ]}

            data = pd.read_csv('./data/MARIA_M_TLG.csv', sep='\t')
            # data = data.values[0:, :]

            rows = data.values[ :, : ]
            for srow in rows:
                row = srow[ 0 ].split(';')
                for id in range(0, len(row)):

                    if id == 0:
                        dt = {}
                        dt[ "value" ] = row[ id ]
                        dateD[ "data" ].append(dt)

                    if id == 13:
                        drft = {}
                        drft[ "value" ] = row[ id ].replace(',', '.') if ',' in row[ id ] else row[ id ]
                        draft[ "data" ].append(drft)

                    if id == 8:
                        stw = {}
                        stw[ "value" ] = row[ id ].replace(',', '.') if ',' in row[ id ] else row[ id ]
                        STW[ "data" ].append(stw)

                    if id == 12:
                        wa = {}
                        wa[ "value" ] = row[ id ].replace(',', '.') if ',' in row[ id ] else row[ id ]
                        windAngle[ "data" ].append(wa)

                    if id == 11:
                        ws = {}
                        ws[ "value" ] = row[ id ].replace(',', '.') if ',' in row[ id ] else row[ id ]
                        windSpeed[ "data" ].append(ws)

                    if id == 10:
                        vcourse = {}
                        vcourse[ "value" ] = row[ id ].replace(',', '.') if ',' in row[ id ] else row[ id ]
                        course[ "data" ].append(vcourse)

                    if id == 8:
                        rpms = {}
                        rpms[ "value" ] = row[ id ].replace(',', '.') if ',' in row[ id ] else row[ id ]
                        rpm[ "data" ].append(rpms)

                    if id == 15:
                        fo = {}
                        fo[ "value" ] = row[ id ].replace(',', '.') if ',' in row[ id ] else row[ id ]
                        foc[ "data" ].append(fo)

                    if id == 1:
                        bl = {}
                        bl[ "value" ] = row[ id ]
                        blFlag[ "data" ].append(bl)

                    if id == 14:
                        vtrim = {}
                        vtrim[ "value" ] = row[ id ].replace(',', '.') if ',' in row[ id ] else row[ id ]
                        trim[ "data" ].append(vtrim)

            with open('./data/MARIA_M.csv', mode='w') as data:
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [ 'DateTime', 'BL Flag', 'Draft', 'Course', 'WindSpeed', 'WindAngle', 'STW', 'Rpm', 'Trim',
                      'M/E FOC (MT/day)',
                      ])

                for i in range(0, len(dateD[ 'data' ])):
                    dt = dateD[ 'data' ][ i ][ 'value' ]

                    bl = blFlag[ 'data' ][ i ][ 'value' ]
                    vcourse = course[ 'data' ][ i ][ 'value' ]
                    drft = draft[ 'data' ][ i ][ 'value' ]

                    stw = STW[ 'data' ][ i ][ 'value' ]

                    wa = windAngle[ 'data' ][ i ][ 'value' ]

                    ws = windSpeed[ 'data' ][ i ][ 'value' ]

                    vfoc = foc[ 'data' ][ i ][ 'value' ]
                    vtrim = trim[ 'data' ][ i ][ 'value' ]
                    vrpm = rpm[ 'data' ][ i ][ 'value' ]

                    data_writer.writerow(
                        [ dt, bl, drft, vcourse, ws, wa, stw, vrpm, vtrim, vfoc ])

        dataV = pd.read_csv('./data/MARIA_M.csv')
        dataV = dataV.drop([ 'DateTime' ], axis=1)
        dtNew = dataV.values
        ballastDt = np.array([ k for k in dtNew if k[ 0 ] == 'B' ])[ :, 1: ].astype(float)
        ladenDt = np.array([ k for k in dtNew if k[ 0 ] == 'L' ])[ :, 1: ].astype(float)
        x = 0

        if company == 'OCEAN_GOLD':
            dataV = pd.read_csv('./data/PERSEFONE/PERSEFONE_1-30_06_19.csv')
            # dataV=dataV.drop([ 't_stamp', 'vessel status' ], axis=1)
            dtNew = dataV.values  # .astype(float)

            data = pd.read_csv('./export telegram.csv', sep='\t')
            DTLGtNew = data.values[ 0:, : ]
            dateTimesW = [ ]
            ws = [ ]
            arrivalsDate = [ ]
            departuresDate = [ ]

            for i in range(0, len(DTLGtNew[ :, 1 ])):
                if DTLGtNew[ i, 0 ] == 'T004':
                    tlgType = DTLGtNew[ i, 2 ]

                    date = DTLGtNew[ i, 1 ].split(" ")[ 0 ]
                    month = date.split('/')[ 1 ]
                    day = date.split('/')[ 0 ]
                    year = date.split('/')[ 2 ]

                    hhMMss = DTLGtNew[ i, 1 ].split(" ")[ 1 ]
                    newDate = year + "-" + month + "-" + day + " " + ":".join(hhMMss.split(":")[ 0:2 ])
                    dtObj = datetime.datetime.strptime(newDate, '%Y-%m-%d %H:%M')
                    if tlgType == 'A':
                        arrivalsDate.append(dtObj)
                        for k in range(i, len(DTLGtNew[ :, 1 ])):
                            if DTLGtNew[ k, 0 ] == 'T004':
                                if DTLGtNew[ k, 2 ] == 'D':
                                    date = DTLGtNew[ k, 1 ].split(" ")[ 0 ]
                                    month = date.split('/')[ 1 ]
                                    day = date.split('/')[ 0 ]
                                    year = date.split('/')[ 2 ]

                                    hhMMss = DTLGtNew[ k, 1 ].split(" ")[ 1 ]
                                    newDate = year + "-" + month + "-" + day + " " + ":".join(hhMMss.split(":")[ 0:2 ])
                                    dtObj = datetime.datetime.strptime(newDate, '%Y-%m-%d %H:%M')
                                    departuresDate.append(dtObj)
                                    break
                    # elif tlgType == 'D':
                    # departuresDate.append(dtObj)

                    dt = datetime.datetime.strptime(newDate, '%Y-%m-%d %H:%M')
                    # if DdtNew[ i, 7 ] != 'No data' and DdtNew[ i, 8 ] != 'No data':
                    dateTimesW.append(dt)

            dateTimesV = [ ]
            dateTimesVstr = [ ]
            for row in dtNew[ :, 0 ]:
                date = row.split(" ")[ 0 ]
                hhMMss = row.split(" ")[ 1 ]
                newDate = date + " " + ":".join(hhMMss.split(":")[ 0:2 ])
                dtV = datetime.datetime.strptime(newDate, '%Y-%m-%d %H:%M')
                #################################
                date = str(dtV).split(" ")[ 0 ]
                month = date.split('-')[ 1 ]

                day = date.split('-')[ 2 ]
                # hhMMssV = str(dateTimesV[i]).split(" ")[1]
                year = date.split('-')[ 0 ]

                ##find dateTimesV at sea
                filteredDTWs = [ d for d in dateTimesW if month == str(d).split(' ')[ 0 ].split('-')[ 1 ] and day ==
                                 str(d).split(' ')[ 0 ].split('-')[ 2 ] and year == str(d).split(' ')[ 0 ].split('-')[ 0 ] ]
                for i in range(0, len(arrivalsDate)):
                    if dtV > arrivalsDate[ i ] and dtV < departuresDate[ i ]:
                        dateTimesV.append(dtV)
                        dateTimesVstr.append(str(dtV))
                # for filteredDate in filteredDTWs:
                # date = str(filteredDate).split(" ")[0]
                # month = date.split('-')[1]
                # month = month[1] if month[0] == '0' else month
                # day = date.split('-')[2]
                # day = day[1] if day[0] == '0' else day
                # year = date.split('-')[0]
                # hhMMss = str(filteredDate).split(" ")[ 1 ]

                # newDate = day + "/" + month + "/" + year + " " + ":".join(hhMMss.split(":"))
                # tlgType=np.array(data.loc[ data[ 'telegram_date' ] == newDate ].values)[ 0 ][ 2 ]
                # if tlgType == 'A' or tlgType=='N':
                # x=0

                ############################

            draftsVmapped = [ ]
            ballastLadenFlagsVmapped = [ ]
            for i in range(0, len(dateTimesV)):

                datesWs = [ ]

                date = str(dateTimesV[ i ]).split(" ")[ 0 ]
                month = date.split('-')[ 1 ]

                day = date.split('-')[ 2 ]

                hhMMssV = str(dateTimesV[ i ]).split(" ")[ 1 ]

                year = date.split('-')[ 0 ]
                dt = dateTimesV[ i ]
                day = '0' + day if day.__len__() == 1 else day
                month = '0' + month if month.__len__() == 1 else month
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
                                        blFlag1 = np.array(data.loc[ data[ 'telegram_date' ] == previous ].values)[ 0 ][
                                            49 ]
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
                                # day =
                                # day = float(day)-1
                                # filteredDTWs=[]
                                # for i in range(0,len(dateTimesW)):
                                # yearW =  str(dateTimesW[i]).split(' ')[ 0 ].split('-')[ 0 ]
                                # monthW =  str(dateTimesW[i]).split(' ')[ 0 ].split('-')[ 1 ]
                                # dayW = str(dateTimesW[i]).split(' ')[ 0 ].split('-')[ 2 ]
                                # while filteredDTWs.__len__()==0:
                                # if month==monthW and day ==dayW and year==yearW:
                                # filteredDTWs.append(dateTimesW[i])
                                newDate = dateTimesV[ i ] - datetime.timedelta(1)

                                date = str(newDate).split(" ")[ 0 ]
                                month = date.split('-')[ 1 ]

                                day = date.split('-')[ 2 ]

                                hhMMssV = str(newDate).split(" ")[ 1 ]

                                year = date.split('-')[ 0 ]
                                dt = dateTimesV[ i ]
                                day = '0' + day if day.__len__() == 1 else day
                                month = '0' + month if month.__len__() == 1 else month

                                filteredDTWsBL = [ d for d in dateTimesW if
                                                   month == str(d).split(' ')[ 0 ].split('-')[ 1 ] and day ==
                                                   str(d).split(' ')[ 0 ].split('-')[ 2 ] and year ==
                                                   str(d).split(' ')[ 0 ].split('-')[ 0 ] ]

                                # next = sorted[ i + 1 ]
                                previous = str(filteredDTWsBL[ len(filteredDTWsBL) - 1 ])
                                dtBL = previous.split(' ')[ 0 ]
                                month = dtBL.split('-')[ 1 ]
                                month = month[ 1 ] if month[ 0 ] == '0' else month
                                day = dtBL.split('-')[ 2 ]
                                day = day[ 1 ] if day[ 0 ] == '0' else day
                                year = dtBL.split('-')[ 0 ]
                                hhMMss = str(previous).split(" ")[ 1 ]
                                previous = day + "/" + month + "/" + year + " " + ":".join(hhMMss.split(":"))

                                blFlag = np.array(data.loc[ data[ 'telegram_date' ] == previous ].values)[ 0 ][ 49 ]
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
                        except Exception as e:

                            d = 0
                            print(e)
                            ballastLadenFlagsVmapped.append('N')

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
                if len(datesWs) > 1:
                    dx = 0
                try:
                    y = np.array(drafts)
                    x = matplotlib.dates.date2num(filteredDTWs)
                    f = sp.Earth()
                    f.fit(x.reshape(-1, 1), y.reshape(-1, 1))
                    pred = f.predict(np.array(matplotlib.dates.date2num(dt)).reshape(-1, 1))
                    draftsVmapped.append(pred)
                except Exception as e:
                    dx = 0
                    # if np.nan_to_num(np.array(data.loc[ data[ 'telegram_date' ] == date ].values)[ 0 ][ 27 ]).__len__()==0 or \
                    # np.nan_to_num(np.array(data.loc[ data[ 'telegram_date' ] == date ].values)[ 0 ][ 28 ]).__len__() == 0:
                    print(e)
                    if drafts.__len__() == 0:
                        draftsVmapped.append([ 0 ])

            ##########

            windSpeedVmapped = [ ]
            windDirsVmapped = [ ]

            # dateTimesV = [ ]
            # for row in dtNew[ :, 0 ]:
            # date = row.split(" ")[ 0 ]
            # hhMMss = row.split(" ")[ 1 ]
            # newDate = date + " " + ":".join(hhMMss.split(":")[ 0:2 ])
            # dt = datetime.datetime.strptime(newDate, '%Y-%m-%d %H:%M')
            # dateTimesV.append(dt)

            data = pd.read_excel('./data/Persefone/Persefone TrackReport.xls')

            WdtNew = data.values[ 0:, : ]
            dateTimesW = [ ]
            ws = [ ]
            newDataset = [ ]
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

            for i in range(0, len(dtNew)):

                # dtFiltered = {}
                #########################
                if str(dtNew[ i ][ 0 ].split(".")[ 0 ]) in dateTimesVstr:
                    newDataset.append(dtNew[ i, : ].reshape(1, -1))

            for i in range(0, len(dateTimesV)):

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

                    if windDirs.__len__() == 0:
                        windDirsVmapped.append(None)

            # pred = f(matplotlib.dates.date2num(dt))
            # data = pd.read_excel('./Penelope TrackReport.xls')
            # WdtNew = data.values[ 0:, : ]
            # print(pred)

            dataV = pd.read_csv('./data/Persefone/PERSEFONE_1-30_06_19.csv')
            newDataset = np.array(newDataset).reshape(-1, dtNew.shape[ 1 ])

            dataV = dataV.drop([ 't_stamp', 'vessel status' ], axis=1)
            dtNew = dataV.values.astype(float)

            newMappedData = np.array(

                np.append(newDataset, np.asmatrix(
                    [ np.array(windSpeedVmapped).reshape(-1), np.array(windDirsVmapped).reshape(-1),
                      np.array(ballastLadenFlagsVmapped)[ 0:len(draftsVmapped) ] ]).T,
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
                x = np.array([ x[ 0 ] * np.cos(x[ 1 ]), x[ 0 ] * np.sin(x[ 1 ]) ])

                y = np.array([ y[ 0 ] * np.cos(y[ 1 ]), y[ 0 ] * np.sin(y[ 1 ]) ])

                x_norm = np.sqrt(sum(x ** 2))

                # Apply the formula as mentioned above
                # for projecting a vector onto the orthogonal vector n
                # find dot product using np.dot()
                proj_of_y_on_x = (np.dot(y, x) / x_norm ** 2) * x

                # vecproj =  np.dot(y, x) / np.dot(y, y) * y
                relWindSpeed = proj_of_y_on_x[ 0 ] + windSpeed[ i ]
                if relWindSpeed > 40:
                    d = 0
                relDir = windDir[ i ] - Vcourse[ i ]
                relDir += 360 if relDir < 0 else relDir
                relWindDirs.append(relDir)
                relWindSpeeds.append(relWindSpeed)
            newMappedData = np.array(
                np.append(newDataset,

                          np.asmatrix([ np.array(relWindSpeeds), np.array(relWindDirs), np.array(draftsVmapped).reshape(-1),
                                        np.array(ballastLadenFlagsVmapped)[ 0:len(draftsVmapped) ] ]).T,

                          axis=1))  # .astype(float)

            #####################
            ############################MAP TELEGRAM DATES FOR DRAFT

            ####STATSTICAL ANALYSIS PENELOPE

            with open('./PERSEFONE_mapped_new_3.csv', mode='w') as dataw:

                data_writer = csv.writer(dataw, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow([ 'STW', 'WindSpeed', 'WindAngle', 'Draft'
                                         , 'M/E FOC (MT/days)', 'B/L Flag' ])
                for k in range(0, len(newMappedData)):
                    data_writer.writerow([ newMappedData[ k ][ 1 ], newMappedData[ k ][ 25 ], newMappedData[ k ][ 26 ],
                                           newMappedData[ k ][ 27 ], newMappedData[ k ][ 8 ], newMappedData[ k ][ 28 ] ])

            x = 0


    def readLarosDAta(self, dtFrom, dtTo):
        UNK = {"data": [ ]}
        draft = {"data": [ ]}
        draftAFT = {"data": [ ]}
        draftFORE = {"data": [ ]}
        windSpeed = {"data": [ ]}
        windAngle = {"data": [ ]}
        SpeedOvg = {"data": [ ]}
        STW = {"data": [ ]}
        rpm = {"data": [ ]}
        power = {"data": [ ]}
        foc = {"data": [ ]}
        foct = {"data": [ ]}
        dateD = {"data": [ ]}
        ListTimeStamps = [ ]

        drftFList = [ ]
        drftAList = [ ]
        drftList = [ ]
        STWList = [ ]
        POWList = [ ]
        RPMList = [ ]
        WAList = [ ]
        WSList = [ ]
        SOVGList = [ ]
        FOCList = [ ]
        FOCTList = [ ]
        UNKList = [ ]

        timeStampsCounter = 0
        drftFCounter = 0
        drftACounter = 0
        drftCounter = 0
        STWCounter = 0
        POWCounter = 0
        RPMCounter = 0
        WACounter = 0
        WSCounter = 0
        SOVGCounter = 0
        FOCCounter = 0
        FOCTCounter = 0
        UNKCounter = 0

        # print(sum(1 for line in open('./MT_DELTA_MARIA.txt')))
        with open('./MT_DELTA_MARIA.txt') as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=';')
            line_count = 0

            for row in csv_reader:
                try:

                    if row[ 1 ] == "STATUS_PARAMETER_ID":
                        continue
                    if row[ 2 ] == '122':
                        x = 0
                    id = row[ 1 ]
                    date = row[ 4 ].split(" ")[ 0 ]
                    hhMMss = row[ 4 ].split(" ")[ 1 ]
                    newDate = date + " " + ":".join(hhMMss.split(":")[ 0:2 ])
                    dt = datetime.datetime.strptime(newDate, '%Y-%m-%d %H:%M')

                    value = row[ 3 ]
                    siteId = row[ 2 ]
                    # if dt >  datetime.datetime.strptime('2018-05-25', '%Y-%m-%d'):

                    if id == "7":
                        if ListTimeStamps != [ ]:
                            if dt != ListTimeStamps[ timeStampsCounter - 1 ]:
                                datt = {}
                                datt[ "value" ] = 0
                                datt[ "dt" ] = dt
                                dateD[ "data" ].append(datt)
                                print(dt)
                                ListTimeStamps.append(dt)
                                timeStampsCounter = timeStampsCounter + 1
                        else:
                            datt = {}
                            datt[ "value" ] = 0
                            datt[ "dt" ] = dt
                            dateD[ "data" ].append(datt)
                            print(dt)
                            ListTimeStamps.append(dt)
                            timeStampsCounter = timeStampsCounter + 1

                    if id == '7':
                        if UNKList != [ ]:
                            if dt != UNKList[ UNKCounter - 1 ]:
                                unknown = {}
                                unknown[ "value" ] = value
                                unknown[ "dt" ] = dt
                                UNK[ "data" ].append(unknown)
                                UNKList.append(dt)
                                UNKCounter = UNKCounter + 1
                        else:
                            unknown = {}
                            unknown[ "value" ] = value
                            unknown[ "dt" ] = dt
                            UNK[ "data" ].append(unknown)
                            UNKList.append(dt)
                            UNKCounter = UNKCounter + 1

                    if id == '21':
                        if drftAList != [ ]:
                            if dt != drftAList[ drftACounter - 1 ]:
                                drftA = {}
                                drftA[ "value" ] = value
                                drftA[ "dt" ] = dt
                                draftAFT[ "data" ].append(drftA)
                                drftAList.append(dt)
                                drftACounter = drftACounter + 1
                        else:
                            drftA = {}
                            drftA[ "value" ] = value
                            drftA[ "dt" ] = dt
                            draftAFT[ "data" ].append(drftA)
                            drftAList.append(dt)
                            drftACounter = drftACounter + 1

                    if id == '67':
                        if STWList != [ ]:
                            if dt != STWList[ STWCounter - 1 ]:
                                stw = {}
                                stw[ "value" ] = value
                                stw[ "dt" ] = dt
                                STW[ "data" ].append(stw)
                                STWList.append(dt)
                                STWCounter = STWCounter + 1
                        else:
                            stw = {}
                            stw[ "value" ] = value
                            stw[ "dt" ] = dt
                            STW[ "data" ].append(stw)
                            STWList.append(dt)
                            STWCounter = STWCounter + 1

                    if id == '22':
                        if drftFList != [ ]:
                            if dt != drftFList[ drftFCounter - 1 ]:
                                drftF = {}
                                drftF[ "value" ] = value
                                drftF[ "dt" ] = dt
                                draftFORE[ "data" ].append(drftF)
                                drftFList.append(dt)
                                drftFCounter = drftFCounter + 1
                        else:
                            drftF = {}
                            drftF[ "value" ] = value
                            drftF[ "dt" ] = dt
                            draftFORE[ "data" ].append(drftF)
                            drftFList.append(dt)
                            drftFCounter = drftFCounter + 1

                    if id == '61':
                        if drftList != [ ]:
                            if dt != drftList[ drftCounter - 1 ]:
                                drft = {}
                                drft[ "value" ] = value
                                drft[ "dt" ] = dt
                                draft[ "data" ].append(drft)
                                drftList.append(dt)
                                drftCounter = drftCounter + 1
                        else:
                            drft = {}
                            drft[ "value" ] = value
                            drft[ "dt" ] = dt
                            draft[ "data" ].append(drft)
                            drftList.append(dt)
                            drftCounter = drftCounter + 1

                    if id == '81':
                        if WAList != [ ]:
                            if dt != WAList[ WACounter - 1 ]:
                                wa = {}
                                wa[ "value" ] = value
                                wa[ "dt" ] = dt
                                windAngle[ "data" ].append(wa)
                                WAList.append(dt)
                                WACounter = WACounter + 1
                        else:
                            wa = {}
                            wa[ "value" ] = value
                            wa[ "dt" ] = dt
                            windAngle[ "data" ].append(wa)
                            WAList.append(dt)
                            WACounter = WACounter + 1

                    if id == '82':
                        if WSList != [ ]:
                            if dt != WSList[ WSCounter - 1 ]:
                                ws = {}
                                ws[ "value" ] = value
                                ws[ "dt" ] = dt
                                windSpeed[ "data" ].append(ws)
                                WSList.append(dt)
                                WSCounter = WSCounter + 1
                        else:
                            ws = {}
                            ws[ "value" ] = value
                            ws[ "dt" ] = dt
                            windSpeed[ "data" ].append(ws)
                            WSList.append(dt)
                            WSCounter = WSCounter + 1

                    if id == '84':
                        if SOVGList != [ ]:
                            if dt != SOVGList[ SOVGCounter - 1 ]:
                                so = {}
                                so[ "value" ] = value
                                so[ "dt" ] = dt
                                SpeedOvg[ "data" ].append(so)
                                SOVGList.append(dt)
                                SOVGCounter = SOVGCounter + 1
                        else:
                            so = {}
                            so[ "value" ] = value
                            so[ "dt" ] = dt
                            SpeedOvg[ "data" ].append(so)
                            SOVGList.append(dt)
                            SOVGCounter = SOVGCounter + 1

                    if id == '89':
                        if RPMList != [ ]:
                            if dt != RPMList[ RPMCounter - 1 ]:
                                rpms = {}
                                rpms[ "value" ] = value
                                rpms[ "dt" ] = dt
                                rpm[ "data" ].append(rpms)
                                RPMList.append(dt)
                                RPMCounter = RPMCounter + 1
                        else:
                            rpms = {}
                            rpms[ "value" ] = value
                            rpms[ "dt" ] = dt
                            rpm[ "data" ].append(rpms)
                            RPMList.append(dt)
                            RPMCounter = RPMCounter + 1

                    if id == '137':
                        if FOCList != [ ]:
                            if dt != FOCList[ FOCCounter - 1 ]:
                                fo = {}
                                fo[ "value" ] = value
                                fo[ "dt" ] = dt
                                foc[ "data" ].append(fo)
                                FOCList.append(dt)
                                FOCCounter = FOCCounter + 1
                        else:
                            fo = {}
                            fo[ "value" ] = value
                            fo[ "dt" ] = dt
                            foc[ "data" ].append(fo)
                            FOCList.append(dt)
                            FOCCounter = FOCCounter + 1

                    if id == '136':
                        if FOCTList != [ ]:
                            if dt != FOCTList[ FOCTCounter - 1 ]:
                                fot = {}
                                fot[ "value" ] = value
                                fot[ "dt" ] = dt
                                foct[ "data" ].append(fot)
                                FOCTList.append(dt)
                                FOCTCounter = FOCTCounter + 1
                        else:
                            fot = {}
                            fot[ "value" ] = value
                            fot[ "dt" ] = dt
                            foct[ "data" ].append(fot)
                            FOCTList.append(dt)
                            FOCTCounter = FOCTCounter + 1

                    if id == '353':
                        if POWList != [ ]:
                            if dt != POWList[ POWCounter - 1 ]:
                                pow = {}
                                pow[ "value" ] = value
                                pow[ "dt" ] = dt
                                power[ "data" ].append(pow)
                                POWList.append(dt)
                                POWCounter = POWCounter + 1
                        else:
                            pow = {}
                            pow[ "value" ] = value
                            pow[ "dt" ] = dt
                            power[ "data" ].append(pow)
                            POWList.append(dt)
                            POWCounter = POWCounter + 1

                    line_count += 1

                    if int(id) > 353:
                        break

                    # print('Processed '+str(line_count)+ ' lines.')
                except Exception as e:
                    p = 0
                    print(str(e))
        try:
            # with open('./WSraw.csv', mode='w') as data:
            # ws_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # ws_writer.writerow(['Field', 'DateTime'])
            # print(len(windSpeed["data"]))
            # for i in range(0, len(windSpeed["data"])):
            # ws_writer.writerow([windSpeed["data"][i]['value'], windSpeed["data"][i]['dt']])
            # return
            endDate = str(dateD[ "data" ][ len(dateD[ "data" ]) - 1 ][ "dt" ])
            startDate = str(dateD[ "data" ][ 0 ][ "dt" ])

            continuousDTS = (pd.DataFrame(columns=[ 'NULL' ],
                                          index=pd.date_range(startDate, endDate,
                                                              freq='1T'))
                # .between_time('00:00', '15:11')
                .index.strftime('%Y-%m-%d %H:%M:%S')
                .tolist()
                )

            with open('./data/LAROS/DateTime.csv', mode='w') as data:
                dt_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                dt_writer.writerow([ 'DateTimes' ])
                for i in range(0, len(continuousDTS)):
                    dt_writer.writerow([ continuousDTS[ i ] ])

            try:
                with open('./data/LAROS/Draft.csv', mode='w') as data:
                    draft_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    draft_writer.writerow([ 'Field', 'DateTime' ])

                    # for i in range(0, len(draft["data"])):

                    # draft_writer.writerow([draft["data"][i]['value'], draft["data"][i]['dt']])

                    k = 0
                    for i in range(0, len(continuousDTS)):
                        if k > 0:
                            k = k + 1 if str(draft[ "data" ][ k ][ 'dt' ]) == str(draft[ "data" ][ k - 1 ][ 'dt' ]) else k
                        if str(draft[ "data" ][ k ][ 'dt' ]) == continuousDTS[ i ]:
                            draft_writer.writerow([ draft[ "data" ][ k ][ 'value' ], draft[ "data" ][ k ][ 'dt' ] ])
                            k = k + 1
                            if k == 238:
                                c = 0
                        else:
                            # k = k - 1 if k > 0 else 0
                            # if dateD[ "data" ][ i ][ 'dt' ] > draft[ "data" ][ k ][ 'dt' ]:
                            # k=k+1
                            # draft_writer.writerow([ draft[ "data" ][ k ][ 'value' ], dateD[ "data" ][ i ][ 'dt' ] ])
                            # k=k+1
                            # else:
                            draft_writer.writerow([ "", continuousDTS[ i ] ])

                            # for i in range(0, len(draft[ "data" ])):
                            # draft_writer.writerow([ draft[ "data" ][ i ][ 'value' ], draft[ "data" ][ i ][ 'dt' ] ])
            except:
                d = 0

            try:
                with open('./data/LAROS/WS.csv', mode='w') as data:
                    ws_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    ws_writer.writerow([ 'Field', 'DateTime' ])

                    k = 0
                    for i in range(0, len(continuousDTS)):
                        if k > 0:
                            k = k + 1 if str(windSpeed[ "data" ][ k ][ 'dt' ]) == str(
                                windSpeed[ "data" ][ k - 1 ][ 'dt' ]) else k
                        if str(windSpeed[ "data" ][ k ][ 'dt' ]) == continuousDTS[ i ]:
                            ws_writer.writerow([ windSpeed[ "data" ][ k ][ 'value' ], windSpeed[ "data" ][ k ][ 'dt' ] ])
                            k = k + 1
                        else:
                            # k = k - 1 if k > 0 else 0
                            # if dateD["data"][i]['dt'] > windSpeed["data"][k]['dt']:
                            # k = k + 1
                            # ws_writer.writerow([windSpeed["data"][k]['value'], dateD["data"][i]['dt']])
                            # k = k + 1
                            # else:
                            ws_writer.writerow([ "", continuousDTS[ i ] ])

                            # for i in range(0, len(windSpeed[ "data" ])):
                            # ws_writer.writerow([ windSpeed[ "data" ][ i ][ 'value' ], windSpeed[ "data" ][ i ][ 'dt' ] ])
            except:
                d = 0

            try:
                with open('./data/LAROS/DraftAFT.csv', mode='w') as data:
                    draftA_writer = csv.writer(data, delimiter=',', quotechar='"',
                                               quoting=csv.QUOTE_MINIMAL)
                    draftA_writer.writerow([ 'Field', 'DateTime' ])

                    k = 0
                    for i in range(0, len(continuousDTS)):
                        if str(dateD[ "data" ][ i ][ 'dt' ]) == str(
                                datetime.datetime.strptime('2018-10-09 13:01', '%Y-%m-%d %H:%M')):
                            print(dt)

                        if k > 0:
                            k = k + 1 if str(draftAFT[ "data" ][ k ][ 'dt' ]) == str(
                                draftAFT[ "data" ][ k - 1 ][ 'dt' ]) else k
                        if str(draftAFT[ "data" ][ k ][ 'dt' ]) == continuousDTS[ i ]:
                            draftA_writer.writerow([ draftAFT[ "data" ][ k ][ 'value' ], draftAFT[ "data" ][ k ][ 'dt' ] ])
                            k = k + 1
                        else:
                            # k = k - 1 if k > 0 else 0
                            # if dateD[ "data" ][ i ][ 'dt' ] > draftAFT[ "data" ][ k ][ 'dt' ]:
                            # k=k+1
                            # draftA_writer.writerow([ draftAFT[ "data" ][ k ][ 'value' ], dateD[ "data" ][ i ][ 'dt' ] ])
                            # k=k+1
                            # else:

                            draftA_writer.writerow([ "", continuousDTS[ i ] ])
            except:
                ex = 0
                # for i in range(0, len(draftAFT[ "data" ])):
                # draftA_writer.writerow([ draftAFT[ "data" ][ i ][ 'value' ], draftAFT[ "data" ][ i ][ 'dt' ] ])

            try:
                with open('./data/LAROS/DraftFORE.csv', mode='w') as data:
                    draftF_writer = csv.writer(data, delimiter=',', quotechar='"',
                                               quoting=csv.QUOTE_MINIMAL)
                    draftF_writer.writerow([ 'Field', 'DateTime' ])

                    k = 0
                    for i in range(0, len(continuousDTS)):
                        if k > 0:
                            k = k + 1 if str(draftFORE[ "data" ][ k ][ 'dt' ]) == str(
                                draftFORE[ "data" ][ k - 1 ][ 'dt' ]) else k
                        if str(draftFORE[ "data" ][ k ][ 'dt' ]) == continuousDTS[ i ]:
                            draftF_writer.writerow(
                                [ draftFORE[ "data" ][ k ][ 'value' ], draftFORE[ "data" ][ k ][ 'dt' ] ])
                            k = k + 1
                        else:
                            # k = k-1 if k > 0 else 0
                            # if dateD[ "data" ][ i ][ 'dt' ] > draftFORE[ "data" ][ k ][ 'dt' ]:
                            # k=k+1
                            # draftF_writer.writerow([ draftFORE[ "data" ][ k ][ 'value' ], dateD[ "data" ][ i ][ 'dt' ] ])
                            # k=k+1
                            # else:
                            draftF_writer.writerow([ "", continuousDTS[ i ] ])
                        # k = +1

                # for i in range(0, len(draftFORE[ "data" ])):
                # draftF_writer.writerow([ draftFORE[ "data" ][ i ][ 'value' ], draftFORE[ "data" ][ i ][ 'dt' ] ])
            except:
                d = 0

            try:
                with open('./data/LAROS/WA.csv', mode='w') as data:
                    wa_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    wa_writer.writerow([ 'Field', 'DateTime' ])
                    k = 0
                    for i in range(0, len(continuousDTS)):
                        if k > 0:
                            k = k + 1 if str(windAngle[ "data" ][ k ][ 'dt' ]) == str(
                                windAngle[ "data" ][ k - 1 ][ 'dt' ]) else k
                        if str(windAngle[ "data" ][ k ][ 'dt' ]) == continuousDTS[ i ]:
                            wa_writer.writerow([ windAngle[ "data" ][ k ][ 'value' ], windAngle[ "data" ][ k ][ 'dt' ] ])
                            k = k + 1
                        else:
                            # k = k - 1 if k > 0 else 0
                            # if dateD["data"][i]['dt'] > windAngle["data"][k]['dt']:
                            # k = k + 1
                            # wa_writer.writerow([windAngle["data"][k]['value'], dateD["data"][i]['dt']])
                            # k = k + 1
                            # else:
                            wa_writer.writerow([ "", continuousDTS[ i ] ])
                            # for i in range(0, len(windAngle[ "data" ])):
                            # wa_writer.writerow([ windAngle[ "data" ][ i ][ 'value' ], windAngle[ "data" ][ i ][ 'dt' ] ])
            except:
                d = 0

            try:
                with open('./data/LAROS/STW.csv', mode='w') as data:
                    stw_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    stw_writer.writerow([ 'Field', 'DateTime' ])

                    k = 0
                    for i in range(0, len(continuousDTS)):
                        if k > 0:
                            k = k + 1 if str(STW[ "data" ][ k ][ 'dt' ]) == str(
                                STW[ "data" ][ k - 1 ][ 'dt' ]) else k
                        if str(STW[ "data" ][ k ][ 'dt' ]) == continuousDTS[ i ]:
                            stw_writer.writerow([ STW[ "data" ][ k ][ 'value' ], STW[ "data" ][ k ][ 'dt' ] ])
                            k = k + 1
                        else:
                            # k = k - 1 if k > 0 else 0
                            # if dateD[ "data" ][ i ][ 'dt' ] > STW[ "data" ][ k ][ 'dt' ]:
                            # k=k+1
                            # stw_writer.writerow([ STW[ "data" ][ k ][ 'value' ], dateD[ "data" ][ i ][ 'dt' ] ])
                            # k=k+1
                            # else:
                            stw_writer.writerow([ "", continuousDTS[ i ] ])

                # for i in range(0, len(STW[ "data" ])):
                # stw_writer.writerow([ STW[ "data" ][ i ][ 'value' ], STW[ "data" ][ i ][ 'dt' ] ])
            except:
                d = 0

            try:
                with open('./data/LAROS/RPM.csv', mode='w') as data:
                    rpm_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    rpm_writer.writerow([ 'Field', 'DateTime' ])
                    k = 0
                    for i in range(0, len(continuousDTS)):
                        if k > 0:
                            k = k + 1 if str(rpm[ "data" ][ k ][ 'dt' ]) == str(
                                rpm[ "data" ][ k - 1 ][ 'dt' ]) else k
                        if str(rpm[ "data" ][ k ][ 'dt' ]) == continuousDTS[ i ]:
                            rpm_writer.writerow([ rpm[ "data" ][ k ][ 'value' ], rpm[ "data" ][ k ][ 'dt' ] ])
                            k = k + 1
                        else:
                            # k = k - 1 if k > 0 else 0
                            # if dateD[ "data" ][ i ][ 'dt' ] > rpm[ "data" ][ k ][ 'dt' ]:
                            # k=k+1
                            # rpm_writer.writerow([ rpm[ "data" ][ k ][ 'value' ], dateD[ "data" ][ i ][ 'dt' ] ])
                            # k=k+1
                            # else:
                            rpm_writer.writerow([ "", continuousDTS[ i ] ])

                # for i in range(0, len(rpm[ "data" ])):
                # rpm_writer.writerow([ rpm[ "data" ][ i ][ 'value' ], rpm[ "data" ][ i ][ 'dt' ] ])
            except:
                d = 0

            try:
                with open('./data/LAROS/FOC.csv', mode='w') as data:
                    foc_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    foc_writer.writerow([ 'Field', 'DateTime' ])

                    k = 0
                    for i in range(0, len(continuousDTS[ i ])):
                        if k > 0:
                            k = k + 1 if str(foc[ "data" ][ k ][ 'dt' ]) == str(
                                foc[ "data" ][ k - 1 ][ 'dt' ]) else k
                        if str(foc[ "data" ][ k ][ 'dt' ]) == continuousDTS[ i ]:
                            foc_writer.writerow([ foc[ "data" ][ k ][ 'value' ], foc[ "data" ][ k ][ 'dt' ] ])
                            k = k + 1
                        else:
                            # k = k - 1 if k > 0 else 0
                            # if dateD[ "data" ][ i ][ 'dt' ] > foc[ "data" ][ k ][ 'dt' ]:
                            # k = k + 1
                            # foc_writer.writerow(
                            # [ foc[ "data" ][ k ][ 'value' ], dateD[ "data" ][ i ][ 'dt' ] ])
                            # k = k + 1
                            # else:
                            foc_writer.writerow([ "", continuousDTS[ i ] ])

                # for i in range(0, len(foc[ "data" ])):
                # foc_writer.writerow([ foc[ "data" ][ i ][ 'value' ], foc[ "data" ][ i ][ 'dt' ] ])
            except:
                d = 0

            try:
                with open('./data/LAROS/FOCt.csv', mode='w') as data:
                    foct_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    foct_writer.writerow([ 'Field', 'DateTime' ])
                    k = 0
                    for i in range(0, len(continuousDTS)):
                        if k > 0:
                            k = k + 1 if str(foct[ "data" ][ k ][ 'dt' ]) == str(
                                foct[ "data" ][ k - 1 ][ 'dt' ]) else k
                        if str(foct[ "data" ][ k ][ 'dt' ]) == continuousDTS[ i ]:
                            foct_writer.writerow([ foct[ "data" ][ k ][ 'value' ], foct[ "data" ][ k ][ 'dt' ] ])
                            k = k + 1
                        else:
                            # k = k - 1 if k > 0 else 0
                            # if dateD[ "data" ][ i ][ 'dt' ] > foct[ "data" ][ k ][ 'dt' ]:
                            # k = k + 1
                            # foct_writer.writerow(
                            # [ foct[ "data" ][ k ][ 'value' ], dateD[ "data" ][ i ][ 'dt' ] ])
                            # k = k + 1
                            # else:
                            foct_writer.writerow([ "", continuousDTS[ i ] ])

                # for i in range(0, len(foct[ "data" ])):
                # foct_writer.writerow([ foct[ "data" ][ i ][ 'value' ], foct[ "data" ][ i ][ 'dt' ] ])
            except:
                d = 0

            try:
                with open('./data/LAROS/SOVG.csv', mode='w') as data:
                    sovg_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    sovg_writer.writerow([ 'Field', 'DateTime' ])
                    k = 0
                    for i in range(0, len(continuousDTS)):
                        if k > 0:
                            k = k + 1 if str(SpeedOvg[ "data" ][ k ][ 'dt' ]) == str(
                                SpeedOvg[ "data" ][ k - 1 ][ 'dt' ]) else k
                        if str(SpeedOvg[ "data" ][ k ][ 'dt' ]) == continuousDTS[ i ]:
                            sovg_writer.writerow([ SpeedOvg[ "data" ][ k ][ 'value' ], SpeedOvg[ "data" ][ k ][ 'dt' ] ])
                            k = k + 1
                        else:
                            # k = k - 1 if k > 0 else 0
                            # if dateD[ "data" ][ i ][ 'dt' ] > SpeedOvg[ "data" ][ k ][ 'dt' ]:
                            # k = k + 1
                            # sovg_writer.writerow(
                            # [ SpeedOvg[ "data" ][ k ][ 'value' ], dateD[ "data" ][ i ][ 'dt' ] ])
                            # k = k + 1
                            # else:
                            sovg_writer.writerow([ "", continuousDTS[ i ] ])
                    # sovg_writer.writerow([ SpeedOvg[ "data" ][ i ][ 'value' ], SpeedOvg[ "data" ][ i ][ 'dt' ] ])
            except:
                d = 0

            try:
                with open('./data/LAROS/POW.csv', mode='w') as data:
                    pow_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    pow_writer.writerow([ 'Field', 'DateTime' ])
                    k = 0
                    for i in range(0, len(continuousDTS)):
                        if k > 0:
                            k = k + 1 if str(power[ "data" ][ k ][ 'dt' ]) == str(
                                power[ "data" ][ k - 1 ][ 'dt' ]) else k
                        if str(power[ "data" ][ k ][ 'dt' ]) == continuousDTS[ i ]:
                            pow_writer.writerow([ power[ "data" ][ k ][ 'value' ], power[ "data" ][ k ][ 'dt' ] ])
                            k = k + 1
                        else:
                            # k=k - 1 if k > 0 else 0
                            # if dateD[ "data" ][ i ][ 'dt' ] > power[ "data" ][ k ][ 'dt' ]:
                            # k = k + 1
                            # pow_writer.writerow(
                            # [ power[ "data" ][ k ][ 'value' ], dateD[ "data" ][ i ][ 'dt' ] ])
                            # k = k + 1
                            # else:
                            pow_writer.writerow([ "", continuousDTS[ i ] ])
            except:
                d = 0
        except Exception as e:
            print(str(e))


    def ExtractLAROSDataset(self):
        draft = {"data": [ ]}
        draftAFT = {"data": [ ]}
        draftFORE = {"data": [ ]}
        windSpeed = {"data": [ ]}
        windAngle = {"data": [ ]}
        SpeedOvg = {"data": [ ]}
        STW = {"data": [ ]}
        rpm = {"data": [ ]}
        power = {"data": [ ]}
        foc = {"data": [ ]}
        foct = {"data": [ ]}
        dateD = {"data": [ ]}

        drftFList = [ ]
        drftAList = [ ]
        drftList = [ ]
        STWList = [ ]
        POWList = [ ]
        RPMList = [ ]
        WAList = [ ]
        WSList = [ ]
        SOVGList = [ ]
        FOCList = [ ]
        FOCTList = [ ]

        with open('./MT_DELTA_MARIA_data_NEW.csv', mode='w') as data:
            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow(
                [ 'Draft', 'DrfAft', 'DrftFore', 'WindSpeed', 'WindAngle', 'SpeedOvg', 'STW', 'Rpm', 'M/E FOC (kg/min)',
                  'FOC Total', 'Power', 'DateTime' ])
            # data_writer.writerow(
            # ['Draft','DateTime' ])
            drftA = {}
            with open('./data/LAROS/DraftAFT.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if row != [ ]:
                        if row[ 0 ] != 'Field':
                            drftA[ row[ 1 ] ] = row[ 0 ]
                            drftAList.append(row[ 0 ])
                            # drftA[ "dt" ] =row[1]
                            # draftAFT[ "data" ].append(drftA)

            drftF = {}
            with open('./data/LAROS/DraftFORE.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if row != [ ]:
                        if row[ 0 ] != 'Field':
                            drftF[ row[ 1 ] ] = row[ 0 ]
                            drftFList.append(row[ 0 ])

                            # drftF[ "dt" ] = row[1]
                            # draftFORE[ "data" ].append(drftF)
            stw = {}
            with open('./data/LAROS/STW.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if row != [ ]:
                        if row[ 0 ] != 'Field':
                            stw[ row[ 1 ] ] = row[ 0 ]
                            STWList.append(row[ 0 ])
                            # stw[ "dt" ] = row[1]
                            # STW[ "data" ].append(stw)
            wa = {}
            with open('./data/LAROS/WA.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if row != [ ]:
                        if row[ 0 ] != 'Field':
                            wa[ row[ 1 ] ] = row[ 0 ]
                            WAList.append(row[ 0 ])
                            # wa[ "dt" ] = row[1]
                            # windAngle[ "data" ].append(wa)
            ws = {}
            with open('./data/LAROS/WS.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if row != [ ]:
                        if row[ 0 ] != 'Field':
                            ws[ row[ 1 ] ] = row[ 0 ]
                            WSList.append(row[ 0 ])
                            # ws[ "dt" ] =
                            # windSpeed[ "data" ].append(ws)
            so = {}
            with open('./data/LAROS/SOVG.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if row != [ ]:
                        if row[ 0 ] != 'Field':
                            so[ row[ 1 ] ] = row[ 0 ]
                            SOVGList.append(row[ 0 ])
                            # so[ "dt" ] = row[1]
                            # SpeedOvg[ "data" ].append(so)
            rpms = {}
            with open('./data/LAROS/RPM.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if row != [ ]:
                        if row[ 0 ] != 'Field':
                            rpms[ row[ 1 ] ] = row[ 0 ]
                            RPMList.append(row[ 0 ])
                            # rpms[ "dt" ] = row[1]
                            # rpm[ "data" ].append(rpms)
            fo = {}
            with open('./data/LAROS/FOC.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if row != [ ]:
                        if row[ 0 ] != 'Field':
                            fo[ row[ 1 ] ] = row[ 0 ]
                            FOCList.append(row[ 0 ])
                            # fo[ "dt" ] = row[1]
                            # foc[ "data" ].append(fo)
            fot = {}
            with open('./data/LAROS/FOCt.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if row != [ ]:
                        if row[ 0 ] != 'Field':
                            fot[ row[ 1 ] ] = row[ 0 ]
                            FOCTList.append(row[ 0 ])
                            # fot[ "dt" ] = row[1]
                            # foct[ "data" ].append(fot)
            pow = {}
            with open('./data/LAROS/POW.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if row != [ ]:
                        if row[ 0 ] != 'Field':
                            pow[ row[ 1 ] ] = row[ 0 ]
                            POWList.append(row[ 0 ])
                            # pow[ "dt" ] = row[1]
                            # power[ "data" ].append(pow)

            drft = {}
            drftCount = 0
            with open('./data/LAROS/Draft.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if row != [ ]:
                        if row[ 0 ] != 'Field':
                            drft[ row[ 1 ] ] = row[ 0 ]
                            drftList.append(row[ 0 ])
                            # drft[ "dt" ] = row[1]
                            # draft[ "data" ].append(drft)
                            drftCount = drftCount + 1
                        # if drftCount > 1000 : break

            dateD = [ ]
            dateCount = 0
            with open('./data/LAROS/DateTime.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if row != [ ]:
                        if row[ 0 ] != 'DateTimes':
                            # datt[ "value" ] = row[0]
                            dateD.append(row[ 0 ])
                            dateCount = dateCount + 1
                            # if dateCount>1000: break
                            # dateD[ "data" ].append(datt)

            from itertools import islice

            for i in range(0, len(dateD)):
                dt = dateD[ i ]
                print(str(dt))
                value = ""

                value7 = ""
                value8 = ""
                value9 = ""
                value10 = ""

                value2 = ""
                value1 = ""
                value6 = ""
                value5 = ""
                value4 = ""
                value3 = ""
                try:

                    # if i > 5:
                    # drftSliced = dict(list(drft.items())[i-5:i + 5]).items()
                    # else:
                    # drftSliced = dict(list(drft.items())[i:i + 5]).items()

                    # drftList=list(dict(filter(lambda elem: elem[ 0 ] == dt, dict(islice(drftSliced.items(), i+10)).items())).values())
                    # drftList = list(dict(filter(lambda elem: elem[0] == dt,
                    # drftSliced)).values())
                    # if drftList.__len__()>0: value = drftList[0]
                    try:
                        value = drftList[ i ]
                    except:
                        r = 0

                    # if draft["data"][i]['dt']==dt:
                    # value=draft['data'][ i ][ 'value' ]
                    # value = [k for k in draft[ "data" ] if k['dt']==dt][0]['value']
                    # drftAList = list(dict(filter(lambda elem: elem[ 0 ] == dt, drftA.items())).values())
                    # if drftAList.__len__() > 0:
                    try:
                        value7 = drftAList[ i ]
                    except:
                        r = 0
                    # if draftAFT["data"][i]['dt']==dt:
                    # value7=draftAFT['data'][ i ][ 'value' ]
                    # value7 = [k for k in draftAFT[ "data" ] if k['dt']==dt][0]['value']

                    # drftFList = list(dict(filter(lambda elem: elem[ 0 ] == dt, drftF.items())).values())
                    # if drftFList.__len__() > 0: value7  = drftFList[0]
                    try:
                        value8 = drftFList[ i ]
                    except:
                        r = 0

                    # if draftFORE[ "data" ][ i ][ 'dt' ] == dt:
                    # value8 = draftFORE[ 'data' ][ i ][ 'value' ]
                    # value8 =[k for k in draftFORE[ "data" ] if k['dt']==dt][0]['value']

                    # STWList = list(dict(filter(lambda elem: elem[ 0 ] == dt, stw.items())).values())
                    # if STWList.__len__() > 0:
                    try:
                        value9 = STWList[ i ]
                    except:
                        r = 0

                    # if STW[ "data" ][ i ][ 'dt' ] == dt:
                    # value9 = STW[ 'data' ][ i ][ 'value' ]
                    # value9 =[k for k in STW[ "data" ] if k['dt']==dt][0]['value']
                    # FOCTList = list(dict(filter(lambda elem: elem[ 0 ] == dt, fot.items())).values())
                    # if FOCTList.__len__() > 0:
                    try:
                        value10 = FOCTList[ i ]
                    except:
                        r = 0

                    # if foct[ "data" ][ i ][ 'dt' ] == dt:
                    # value10 = foct[ 'data' ][ i ][ 'value' ]
                    # value10 = [k for k in foct[ "data" ] if k['dt']==dt][0]['value']

                    # WAList = list(dict(filter(lambda elem: elem[ 0 ] == dt, wa.items())).values())
                    # if WAList.__len__() > 0:
                    try:
                        value1 = WAList[ i ]
                    except:
                        r = 0
                    # if windAngle[ "data" ][ i ][ 'dt' ] == dt:
                    # value1 = windAngle[ 'data' ][ i ][ 'value' ]
                    # value1 =[k for k in windAngle[ "data" ] if k['dt']==dt][0]['value']

                    # WSList = list(dict(filter(lambda elem: elem[ 0 ] == dt, ws.items())).values())
                    # if WSList.__len__() > 0:
                    try:
                        value2 = WSList[ i ]
                    except:
                        r = 0
                    # if windSpeed[ "data" ][ i ][ 'dt' ] == dt:
                    # value2 = windSpeed[ 'data' ][ i ][ 'value' ]
                    # value2 = [k for k in windSpeed[ "data" ] if k['dt']==dt][0]['value']
                    # POWList = list(dict(filter(lambda elem: elem[ 0 ] == dt, pow.items())).values())
                    # if POWList.__len__() > 0:
                    try:
                        value3 = POWList[ i ]
                    except:
                        r = 0

                    # if power[ "data" ][ i ][ 'dt' ] == dt:
                    # value3 = power[ 'data' ][ i ][ 'value' ]
                    # value3 = [k for k in power[ "data" ] if k['dt']==dt][0]['value']
                    # POWList = list(dict(filter(lambda elem: elem[ 0 ] == dt, pow.items())).values())
                    # if POWList.__len__() > 0:
                    try:
                        value3 = POWList[ i ]
                    except:
                        r = 0

                    # FOCList = list(dict(filter(lambda elem: elem[ 0 ] == dt, foc.items())).values())
                    # if FOCList.__len__() > 0:
                    try:
                        value4 = FOCList[ i ]
                    except:
                        r = 0
                    # if foc[ "data" ][ i ][ 'dt' ] == dt:
                    # value4 = foc[ 'data' ][ i ][ 'value' ]
                    # value4 =[k for k in foc[ "data" ] if k['dt']==dt][0]['value']

                    # RPMList = list(dict(filter(lambda elem: elem[ 0 ] == dt, rpm.items())).values())
                    # if RPMList.__len__() > 0:
                    try:
                        value5 = RPMList[ i ]
                    except:
                        r = 0

                    # if rpm[ "data" ][ i ][ 'dt' ] == dt:
                    # value5 = rpm[ 'data' ][ i ][ 'value' ]
                    # value5 = [k for k in rpm[ "data" ] if k['dt']==dt][0]['value']

                    # SOVGList = list(dict(filter(lambda elem: elem[ 0 ] == dt, so.items())).values())
                    # if SOVGList.__len__() > 0:
                    try:
                        value6 = SOVGList[ i ]
                    except:
                        r = 0
                    # if SpeedOvg[ "data" ][ i ][ 'dt' ] == dt:
                    # value6 = SpeedOvg[ 'data' ][ i ][ 'value' ]
                    # value6 =[k for k in SpeedOvg[ "data" ] if k['dt']==dt][0]['value']

                #####
                except Exception as e:
                    print(str(e))
                data_writer.writerow(
                    [ value, value7, value8, value2, value1, value6, value9, value5, value4, value10, value3, str(dt) ])


    def readLarosDataFromCsvNewExtractExcels(self, data):
        # Load file
        dataNew = data.drop([ 'SpeedOvg' ], axis=1)

        dtNew = dataNew.values[ 484362:569386, : ]

        daysSincelastPropClean = [ ]
        daysSinceLastDryDock = [ ]

        with open('./daysSincelastPropClean.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if row.__len__() == 1:
                    daysSincelastPropClean.append(float(row[ 0 ]))
        with open('./daysSinceLastDryDock.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if row.__len__() == 1:
                    daysSinceLastDryDock.append(float(row[ 0 ]))

        daysSincelastPropClean = np.array(daysSincelastPropClean).reshape(-1, ).astype(float)
        daysSinceLastDryDock = np.array(daysSinceLastDryDock).reshape(-1, ).astype(float)

        dataNew = data.drop([ 'SpeedOvg', 'DateTime' ], axis=1)

        dtNew = dataNew.values[ 484362:569386, : ].astype(float)

        draftMean = (dtNew[ :, 1 ] + dtNew[ :, 2 ]) / 2
        dtNew[ :, 0 ] = np.round(draftMean, 3)

        # dtNew = dtNew[~np.isnan(dtNew).any(axis=1)]

        trim = (dtNew[ :, 1 ] - dtNew[ :, 2 ])

        dtNew1 = np.array(np.append(dtNew[ :, 0 ].reshape(-1, 1), np.asmatrix(
            [ np.round(trim, 3), dtNew[ :, 3 ], dtNew[ :, 4 ], dtNew[ :, 5 ], dtNew[ :, 6 ], daysSincelastPropClean,
              daysSinceLastDryDock, dtNew[ :, 7 ] ]).T, axis=1))

        # dtNew1 = np.delete(dtNew1, [i for (i, v) in enumerate(dtNew1[0:, 0]) if v <= 6], 0)
        dtNew1 = np.delete(dtNew1, [ i for (i, v) in enumerate(dtNew1[ 0:, 4 ]) if v <= 0 ], 0)

        rpmB = np.array([ i for i in dtNew1 if i[ 5 ] > 90 and i[ 7 ] < 100 ])

        # dtNew1 = np.delete(dtNew1, [i for (i, v) in enumerate(dtNew1[0:, 5]) if v <= 0 or v > 90], 0)  # or (v>0 and v <=10)

        # dtNew = np.delete(dtNew, [ i for (i, v) in enumerate(dtNew[ 0:, 7 ]) ], 0) # if v <= 7

        ##statistics STD
        # draftSmaller6 = np.array([drft for drft in dtNew1 if drft[0]< 6])
        # draftBigger6 = np.array([ drft for drft in dtNew1 if drft[0] > 6 ])
        #########DRAFT < 9
        drftStw8 = [ ]

        # stw8DrftS4 = np.array([i for i in dtNew1 if i[4] >= 8 and i[4] < 8.5 and i[0] <= 9])

        with open('./MT_DELTA_MARIA_draftS_9_analysis.csv', mode='w') as rows:
            for dr in range(6, 9):
                stw8DrftS4 = np.array(
                    [ i for i in dtNew1 if i[ 4 ] >= 8 and i[ 4 ] < 8.5 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
                stw8DrftS4 = np.delete(stw8DrftS4, [ i for (i, v) in enumerate(stw8DrftS4[ :, 8 ]) if v < (
                        np.mean(stw8DrftS4[ :, 8 ]) - np.std(stw8DrftS4[ :, 8 ])) or v > np.mean(
                    stw8DrftS4[ :, 8 ]) + np.std(
                    stw8DrftS4[ :, 8 ]) ], 0)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                data_writer.writerow(
                    [ str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 8 < STW < 8.5                   ' ], )
                data_writer.writerow([ ], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW', 'RPM'
                                         , 'DaysSinceLastPropCleaning', 'DaysSinceLastDryDock', 'M/E FOC (kg/min)' ])
                for data in stw8DrftS4:
                    data_writer.writerow(
                        [ data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ], data[ 4 ], data[ 5 ], data[ 6 ], data[ 7 ],
                          data[ 8 ] ])
                data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            # drftStw8.append(stw8DrftS4)

            drftStw9 = [ ]
            stw9DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 8.5 and i[ 4 ] < 9 and i[ 0 ] <= 9 ])

            for dr in range(6, 9):
                stw9DrftS4 = np.array(
                    [ i for i in dtNew1 if i[ 4 ] >= 8.5 and i[ 4 ] < 9 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
                stw9DrftS4 = np.delete(stw9DrftS4, [ i for (i, v) in enumerate(stw9DrftS4[ :, 8 ]) if v < (
                        np.mean(stw9DrftS4[ :, 8 ]) - np.std(stw9DrftS4[ :, 8 ])) or v > np.mean(
                    stw9DrftS4[ :, 8 ]) + np.std(
                    stw9DrftS4[ :, 8 ]) ], 0)
                drftStw9.append(stw9DrftS4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [ str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 8.5 < STW < 9                   ' ], )
                data_writer.writerow([ ], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW', 'RPM'
                                         , 'DaysSinceLastPropCleaning', 'DaysSinceLastDryDock', 'M/E FOC (kg/min)' ])
                for data in stw9DrftS4:
                    data_writer.writerow(
                        [ data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ], data[ 4 ], data[ 5 ], data[ 6 ], data[ 7 ],
                          data[ 8 ] ])
                data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            data_writer.writerow([ ], )

            drftStw10 = [ ]
            stw10DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 9 and i[ 4 ] < 9.5 and i[ 0 ] <= 9 ])

            for dr in range(6, 9):
                stw10DrftS4 = np.array(
                    [ i for i in dtNew1 if i[ 4 ] >= 9 and i[ 4 ] < 9.5 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
                stw10DrftS4 = np.delete(stw10DrftS4, [ i for (i, v) in enumerate(stw10DrftS4[ :, 8 ]) if v < (
                        np.mean(stw10DrftS4[ :, 8 ]) - np.std(stw10DrftS4[ :, 8 ])) or v > np.mean(
                    stw10DrftS4[ :, 8 ]) + np.std(
                    stw10DrftS4[ :, 8 ]) ], 0)
                drftStw10.append(stw10DrftS4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [ str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 9 < STW < 9.5                   ' ], )
                data_writer.writerow([ ], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW', 'RPM'
                                         , 'DaysSinceLastPropCleaning', 'DaysSinceLastDryDock', 'M/E FOC (kg/min)' ])
                for data in stw10DrftS4:
                    data_writer.writerow(
                        [ data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ], data[ 4 ], data[ 5 ], data[ 6 ], data[ 7 ],
                          data[ 8 ] ])
                data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            data_writer.writerow([ ], )

            drftStw11 = [ ]
            stw11DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 9.5 and i[ 4 ] < 10 and i[ 0 ] <= 9 ])

            for dr in range(6, 9):
                stw11DrftS4 = np.array(
                    [ i for i in dtNew1 if i[ 4 ] >= 9.5 and i[ 4 ] < 10 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
                stw11DrftS4 = np.delete(stw10DrftS4, [ i for (i, v) in enumerate(stw11DrftS4[ :, 8 ]) if v < (
                        np.mean(stw11DrftS4[ :, 8 ]) - np.std(stw11DrftS4[ :, 8 ])) or v > np.mean(
                    stw11DrftS4[ :, 8 ]) + np.std(
                    stw11DrftS4[ :, 8 ]) ], 0)
                drftStw11.append(stw11DrftS4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [ str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 9.5 < STW < 10                   ' ], )
                data_writer.writerow([ ], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW', 'RPM'
                                         , 'DaysSinceLastPropCleaning', 'DaysSinceLastDryDock', 'M/E FOC (kg/min)' ])
                for data in stw11DrftS4:
                    data_writer.writerow(
                        [ data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ], data[ 4 ], data[ 5 ], data[ 6 ], data[ 7 ],
                          data[ 8 ] ])
                data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            data_writer.writerow([ ], )

            drftStw12 = [ ]
            stw12DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 10 and i[ 4 ] < 10.5 and i[ 0 ] <= 9 ])

            for dr in range(6, 9):
                stw12DrftS4 = np.array(
                    [ i for i in dtNew1 if i[ 4 ] >= 10 and i[ 4 ] < 10.5 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
                stw12DrftS4 = np.delete(stw12DrftS4, [ i for (i, v) in enumerate(stw12DrftS4[ :, 8 ]) if v < (
                        np.mean(stw12DrftS4[ :, 8 ]) - np.std(stw12DrftS4[ :, 8 ])) or v > np.mean(
                    stw12DrftS4[ :, 8 ]) + np.std(
                    stw12DrftS4[ :, 8 ]) ], 0)
                drftStw12.append(stw11DrftS4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [ str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 10 < STW < 10.5                   ' ], )
                data_writer.writerow([ ], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW', 'RPM'
                                         , 'DaysSinceLastPropCleaning', 'DaysSinceLastDryDock', 'M/E FOC (kg/min)' ])
                for data in stw12DrftS4:
                    data_writer.writerow(
                        [ data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ], data[ 4 ], data[ 5 ], data[ 6 ], data[ 7 ],
                          data[ 8 ] ])
                data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            data_writer.writerow([ ], )

            drftStw13 = [ ]
            # stw13DrftS4 = np.array([i for i in dtNew1 if i[4] >= 10.5 and i[4] < 11 and i[0] <= 9])

            for dr in range(6, 9):
                stw13DrftS4 = np.array(
                    [ i for i in dtNew1 if i[ 4 ] >= 10.5 and i[ 4 ] < 11 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
                stw13DrftS4 = np.delete(stw13DrftS4, [ i for (i, v) in enumerate(stw13DrftS4[ :, 8 ]) if v < (
                        np.mean(stw13DrftS4[ :, 8 ]) - np.std(stw13DrftS4[ :, 8 ])) or v > np.mean(
                    stw13DrftS4[ :, 8 ]) + np.std(
                    stw13DrftS4[ :, 8 ]) ], 0)
                drftStw13.append(drftStw13)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [ str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 10.5 < STW < 11                   ' ], )
                data_writer.writerow([ ], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW', 'RPM'
                                         , 'DaysSinceLastPropCleaning', 'DaysSinceLastDryDock', 'M/E FOC (kg/min)' ])
                for data in stw13DrftS4:
                    data_writer.writerow(
                        [ data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ], data[ 4 ], data[ 5 ], data[ 6 ], data[ 7 ],
                          data[ 8 ] ])
                data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            data_writer.writerow([ ], )

            drftStw14 = [ ]
            # stw14DrftS4 = np.array([i for i in dtNew1 if i[4] >= 11 and i[4] < 11.5 and i[0] <= 9])

            for dr in range(6, 9):
                stw14DrftS4 = np.array(
                    [ i for i in dtNew1 if i[ 4 ] >= 11 and i[ 4 ] < 11.5 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
                stw14DrftS4 = np.delete(stw14DrftS4, [ i for (i, v) in enumerate(stw14DrftS4[ :, 8 ]) if v < (
                        np.mean(stw14DrftS4[ :, 8 ]) - np.std(stw14DrftS4[ :, 8 ])) or v > np.mean(
                    stw14DrftS4[ :, 8 ]) + np.std(
                    stw14DrftS4[ :, 8 ]) ], 0)
                drftStw14.append(stw14DrftS4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [ str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 11 < STW < 11.5                   ' ], )
                data_writer.writerow([ ], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW', 'RPM'
                                         , 'DaysSinceLastPropCleaning', 'DaysSinceLastDryDock', 'M/E FOC (kg/min)' ])
                for data in stw14DrftS4:
                    data_writer.writerow(
                        [ data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ], data[ 4 ], data[ 5 ], data[ 6 ], data[ 7 ],
                          data[ 8 ] ])
                data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            data_writer.writerow([ ], )

            drftStw15 = [ ]
            # stw15DrftS4 = np.array([i for i in dtNew1 if i[4] >= 11.5 and i[4] < 12 and i[0] <= 9])

            for dr in range(6, 9):
                stw15DrftS4 = np.array(
                    [ i for i in dtNew1 if i[ 4 ] >= 11.5 and i[ 4 ] < 12 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
                stw15DrftS4 = np.delete(stw15DrftS4, [ i for (i, v) in enumerate(stw15DrftS4[ :, 8 ]) if v < (
                        np.mean(stw15DrftS4[ :, 8 ]) - np.std(stw15DrftS4[ :, 8 ])) or v > np.mean(
                    stw15DrftS4[ :, 8 ]) + np.std(
                    stw15DrftS4[ :, 8 ]) ], 0)
                drftStw15.append(stw15DrftS4)
                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [ str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 11.5 < STW < 12                   ' ], )
                data_writer.writerow([ ], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW', 'RPM'
                                         , 'DaysSinceLastPropCleaning', 'DaysSinceLastDryDock', 'M/E FOC (kg/min)' ])
                for data in stw15DrftS4:
                    data_writer.writerow(
                        [ data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ], data[ 4 ], data[ 5 ], data[ 6 ], data[ 7 ],
                          data[ 8 ] ])
                data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            data_writer.writerow([ ], )

            drftStw16 = [ ]
            # stw16DrftS4 = np.array([i for i in dtNew1 if i[4] >= 12 and i[4] < 12.5 and i[0] <= 9])

            for dr in range(6, 9):
                stw16DrftS4 = np.array(
                    [ i for i in dtNew1 if i[ 4 ] >= 12 and i[ 4 ] < 12.5 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
                stw16DrftS4 = np.delete(stw16DrftS4, [ i for (i, v) in enumerate(stw16DrftS4[ :, 8 ]) if v < (
                        np.mean(stw16DrftS4[ :, 8 ]) - np.std(stw16DrftS4[ :, 8 ])) or v > np.mean(
                    stw16DrftS4[ :, 8 ]) + np.std(
                    stw16DrftS4[ :, 8 ]) ], 0)
                drftStw16.append(stw16DrftS4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [ str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 12 < STW < 12.5                   ' ], )
                data_writer.writerow([ ], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW', 'RPM'
                                         , 'DaysSinceLastPropCleaning', 'DaysSinceLastDryDock', 'M/E FOC (kg/min)' ])
                for data in stw16DrftS4:
                    data_writer.writerow([ data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ], data[ 4 ], data[ 5 ] ])
                data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            data_writer.writerow([ ], )

            drftStw17 = [ ]
            # stw17DrftS4 = np.array([i for i in dtNew1 if i[4] >= 12.5 and i[4] < 13 and i[0] <= 9])

            for dr in range(6, 9):
                stw17DrftS4 = np.array(
                    [ i for i in dtNew1 if i[ 4 ] >= 12 and i[ 4 ] < 12.5 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
                stw17DrftS4 = np.delete(stw17DrftS4, [ i for (i, v) in enumerate(stw17DrftS4[ :, 8 ]) if v < (
                        np.mean(stw17DrftS4[ :, 8 ]) - np.std(stw17DrftS4[ :, 8 ])) or v > np.mean(
                    stw17DrftS4[ :, 8 ]) + np.std(
                    stw17DrftS4[ :, 8 ]) ], 0)
                drftStw17.append(drftStw17)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [ str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 12 < STW < 12.5                   ' ], )
                data_writer.writerow([ ], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW', 'RPM'
                                         , 'DaysSinceLastPropCleaning', 'DaysSinceLastDryDock', 'M/E FOC (kg/min)' ])
                for data in stw17DrftS4:
                    data_writer.writerow(
                        [ data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ], data[ 4 ], data[ 5 ], data[ 6 ], data[ 7 ],
                          data[ 8 ] ])
                data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            data_writer.writerow([ ], )

            drftStw18 = [ ]
            # stw18DrftS4 = np.array([i for i in dtNew1 if i[4] >= 13 and i[4] < 13.5 and i[0] <= 9])

            for dr in range(6, 9):
                stw18DrftS4 = np.array(
                    [ i for i in dtNew1 if i[ 4 ] >= 13 and i[ 4 ] < 13.5 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(

                    [ str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 13 < STW < 13.5                   ' ], )
                if stw18DrftS4.shape[ 0 ] != 0:
                    stw18DrftS4 = np.delete(stw18DrftS4, [ i for (i, v) in enumerate(stw18DrftS4[ :, 8 ]) if v < (
                            np.mean(stw18DrftS4[ :, 8 ]) - np.std(stw18DrftS4[ :, 8 ])) or v > np.mean(
                        stw18DrftS4[ :, 8 ]) + np.std(
                        stw18DrftS4[ :, 8 ]) ], 0)
                    drftStw18.append(stw18DrftS4)

                    data_writer.writerow([ ], )
                    data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW',
                                           'M/E FOC (kg/min)', 'DaysSinceLastPropCleaning', 'DaysSinceLastDryDock' ])
                    for data in stw18DrftS4:
                        data_writer.writerow(
                            [ data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ], data[ 4 ], data[ 5 ], data[ 6 ], data[ 7 ],
                              data[ 8 ] ])
                    data_writer.writerow([ ], )
                data_writer.writerow([ ], )
                data_writer.writerow([ ], )

            drftStw19 = [ ]
            # stw19DrftS4 = np.array([i for i in dtNew1 if i[4] >= 13.5 and i[4] < 14 and i[0] <= 9])

            for dr in range(6, 9):
                stw19DrftS4 = np.array(
                    [ i for i in dtNew1 if i[ 4 ] >= 13.5 and i[ 4 ] < 14 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
                data_writer.writerow(
                    [ str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 13.5 < STW < 14                   ' ], )

                if stw19DrftS4.shape[ 0 ] != 0:
                    stw19DrftS4 = np.delete(stw19DrftS4, [ i for (i, v) in enumerate(stw19DrftS4[ :, 8 ]) if v < (
                            np.mean(stw19DrftS4[ :, 8 ]) - np.std(stw19DrftS4[ :, 8 ])) or v > np.mean(
                        stw19DrftS4[ :, 8 ]) + np.std(
                        stw19DrftS4[ :, 8 ]) ], 0)
                    drftStw19.append(stw19DrftS4)

                    data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                    data_writer.writerow([ ], )
                    data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW', 'RPM'
                                             , 'DaysSinceLastPropCleaning', 'DaysSinceLastDryDock', 'M/E FOC (kg/min)' ])
                    for data in stw19DrftS4:
                        data_writer.writerow(
                            [ data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ], data[ 4 ], data[ 5 ], data[ 6 ], data[ 7 ],
                              data[ 8 ] ])
                data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            data_writer.writerow([ ], )

        ########################################################################################################################################
        ########################################################################################################################################
        ########################################################################################################################################
        ########################################################################################################################################

        drftStw8 = [ ]
        # stw8DrftB4 = np.array([i for i in dtNew1 if i[4] >= 8 and i[4] < 8.5 and i[0] > 9])
        with open('./MT_DELTA_MARIA_draftB_9_analysis.csv', mode='w') as rows:
            for dr in range(9, 14):
                stw8DrftB4 = np.array(
                    [ i for i in dtNew1 if i[ 4 ] >= 8 and i[ 4 ] < 8.5 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
                stw8DrftB4 = np.delete(stw8DrftB4, [ i for (i, v) in enumerate(stw8DrftB4[ :, 8 ]) if v < (
                        np.mean(stw8DrftB4[ :, 8 ]) - np.std(stw8DrftB4[ :, 8 ])) or v > np.mean(
                    stw8DrftB4[ :, 8 ]) + np.std(
                    stw8DrftB4[ :, 8 ]) ], 0)
                drftStw8.append(stw8DrftB4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [ str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 8 < STW < 8.5                   ' ], )
                data_writer.writerow([ ], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW', 'RPM'
                                         , 'DaysSinceLastPropCleaning', 'DaysSinceLastDryDock', 'M/E FOC (kg/min)' ])
                for data in stw8DrftB4:
                    data_writer.writerow(
                        [ data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ], data[ 4 ], data[ 5 ], data[ 6 ], data[ 7 ],
                          data[ 8 ] ])
                data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            data_writer.writerow([ ], )

            # stw8DrftB4 = np.concatenate(stw8DrftB4)

            drftStw9 = [ ]
            # stw9DrftB4 = np.array([i for i in dtNew1 if i[4] >= 8.5 and i[4] < 9 and i[0] > 9])
            for dr in range(9, 14):
                stw9DrftB4 = np.array(
                    [ i for i in dtNew1 if i[ 4 ] >= 8.5 and i[ 4 ] < 9 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
                stw9DrftB4 = np.delete(stw9DrftB4, [ i for (i, v) in enumerate(stw9DrftB4[ :, 8 ]) if v < (
                        np.mean(stw9DrftB4[ :, 8 ]) - np.std(stw9DrftB4[ :, 8 ])) or v > np.mean(
                    stw9DrftB4[ :, 8 ]) + np.std(
                    stw9DrftB4[ :, 8 ]) ], 0)
                drftStw9.append(stw9DrftB4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [ str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 8.5 < STW < 9                   ' ], )
                data_writer.writerow([ ], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW', 'RPM'
                                         , 'DaysSinceLastPropCleaning', 'DaysSinceLastDryDock', 'M/E FOC (kg/min)' ])
                for data in stw9DrftB4:
                    data_writer.writerow(
                        [ data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ], data[ 4 ], data[ 5 ], data[ 6 ], data[ 7 ],
                          data[ 8 ] ])
                data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            data_writer.writerow([ ], )

            # stw9DrftB4 = np.concatenate(drftStw9)

            # drftStw10 = []
            # stw10DrftB4 = np.array([i for i in dtNew1 if i[4] >= 9 and i[4] < 9.5 and i[0] > 9])
            for dr in range(9, 14):
                stw10DrftB4 = np.array(
                    [ i for i in dtNew1 if i[ 4 ] >= 9 and i[ 4 ] < 9.5 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
                stw10DrftB4 = np.delete(stw10DrftB4, [ i for (i, v) in enumerate(stw10DrftB4[ :, 8 ]) if v < (
                        np.mean(stw10DrftB4[ :, 8 ]) - np.std(stw10DrftB4[ :, 8 ])) or v > np.mean(
                    stw10DrftB4[ :, 8 ]) + np.std(
                    stw10DrftB4[ :, 8 ]) ], 0)
                drftStw10.append(stw10DrftB4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [ str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 9 < STW < 9.5                  ' ], )
                data_writer.writerow([ ], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW', 'RPM'
                                         , 'DaysSinceLastPropCleaning', 'DaysSinceLastDryDock', 'M/E FOC (kg/min)' ])
                for data in stw10DrftB4:
                    data_writer.writerow(
                        [ data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ], data[ 4 ], data[ 5 ], data[ 6 ], data[ 7 ],
                          data[ 8 ] ])
                data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            # stw10DrftB4 = np.concatenate(drftStw10)

            # drftStw11 = []
            # stw11DrftB4 = np.array([i for i in dtNew1 if i[4] >= 9.5 and i[4] < 10 and i[0] > 9])
            for dr in range(9, 14):
                stw11DrftB4 = np.array(
                    [ i for i in dtNew1 if i[ 4 ] >= 9.5 and i[ 4 ] < 10 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
                stw11DrftB4 = np.delete(stw11DrftB4, [ i for (i, v) in enumerate(stw11DrftB4[ :, 8 ]) if v < (
                        np.mean(stw11DrftB4[ :, 8 ]) - np.std(stw11DrftB4[ :, 8 ])) or v > np.mean(
                    stw11DrftB4[ :, 8 ]) + np.std(
                    stw11DrftB4[ :, 8 ]) ], 0)
                drftStw11.append(stw11DrftB4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [ str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 9.5 < STW < 10                   ' ], )
                data_writer.writerow([ ], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW', 'RPM'
                                         , 'DaysSinceLastPropCleaning', 'DaysSinceLastDryDock', 'M/E FOC (kg/min)' ])
                for data in stw11DrftB4:
                    data_writer.writerow(
                        [ data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ], data[ 4 ], data[ 5 ], data[ 6 ], data[ 7 ],
                          data[ 8 ] ])
                data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            data_writer.writerow([ ], )

            # stw11DrftB4 = np.concatenate(drftStw11)

            # drftStw12 = []
            # stw12DrftB4 = np.array([i for i in dtNew1 if i[4] >= 10 and i[4] < 10.5 and i[0] > 9])
            for dr in range(9, 14):
                stw12DrftB4 = np.array(
                    [ i for i in dtNew1 if i[ 4 ] >= 10 and i[ 4 ] < 10.5 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
                stw12DrftB4 = np.delete(stw12DrftB4, [ i for (i, v) in enumerate(stw12DrftB4[ :, 8 ]) if v < (
                        np.mean(stw12DrftB4[ :, 8 ]) - np.std(stw12DrftB4[ :, 8 ])) or v > np.mean(
                    stw12DrftB4[ :, 8 ]) + np.std(
                    stw12DrftB4[ :, 8 ]) ], 0)
                drftStw12.append(drftStw12)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [ str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 10 < STW < 10.5                   ' ], )
                data_writer.writerow([ ], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW', 'RPM'
                                         , 'DaysSinceLastPropCleaning', 'DaysSinceLastDryDock', 'M/E FOC (kg/min)' ])
                for data in stw12DrftB4:
                    data_writer.writerow(
                        [ data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ], data[ 4 ], data[ 5 ], data[ 6 ], data[ 7 ],
                          data[ 8 ] ])
                data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            # stw12DrftB4 = np.concatenate(drftStw12)

            # drftStw13 = []
            # stw13DrftB4 = np.array([i for i in dtNew1 if i[4] >= 10.5 and i[4] < 11 and i[0] > 9])
            for dr in range(9, 14):
                stw13DrftB4 = np.array(
                    [ i for i in dtNew1 if i[ 4 ] >= 10.5 and i[ 4 ] < 11 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
                stw13DrftB4 = np.delete(stw13DrftB4, [ i for (i, v) in enumerate(stw13DrftB4[ :, 8 ]) if v < (
                        np.mean(stw13DrftB4[ :, 8 ]) - np.std(stw13DrftB4[ :, 8 ])) or v > np.mean(
                    stw13DrftB4[ :, 8 ]) + np.std(
                    stw13DrftB4[ :, 8 ]) ], 0)
                drftStw13.append(stw13DrftB4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [ str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 10.5 < STW < 11                 ' ], )
                data_writer.writerow([ ], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW', 'RPM'
                                         , 'DaysSinceLastPropCleaning', 'DaysSinceLastDryDock', 'M/E FOC (kg/min)' ])
                for data in stw13DrftB4:
                    data_writer.writerow(
                        [ data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ], data[ 4 ], data[ 5 ], data[ 6 ], data[ 7 ],
                          data[ 8 ] ])
                data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            # stw13DrftB4 = np.concatenate(drftStw13)

            # drftStw14 = []
            # stw14DrftB4 = np.array([i for i in dtNew1 if i[4] >= 11 and i[4] < 11.5 and i[0] > 9])
            for dr in range(9, 14):
                stw14DrftB4 = np.array(
                    [ i for i in dtNew1 if i[ 4 ] >= 11 and i[ 4 ] < 11.5 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
                stw14DrftB4 = np.delete(stw14DrftB4, [ i for (i, v) in enumerate(stw14DrftB4[ :, 8 ]) if v < (
                        np.mean(stw14DrftB4[ :, 8 ]) - np.std(stw14DrftB4[ :, 8 ])) or v > np.mean(
                    stw14DrftB4[ :, 8 ]) + np.std(
                    stw14DrftB4[ :, 8 ]) ], 0)
                drftStw14.append(stw14DrftB4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [ str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 11 < STW < 11.5                   ' ], )
                data_writer.writerow([ ], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW', 'RPM'
                                         , 'DaysSinceLastPropCleaning', 'DaysSinceLastDryDock', 'M/E FOC (kg/min)' ])
                for data in stw14DrftB4:
                    data_writer.writerow(
                        [ data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ], data[ 4 ], data[ 5 ], data[ 6 ], data[ 7 ],
                          data[ 8 ] ])
                data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            # stw14DrftB4 = np.concatenate(drftStw14)

            # drftStw15 = []
            # stw15DrftB4 = np.array([i for i in dtNew1 if i[4] >= 11.5 and i[4] < 12 and i[0] > 9])
            for dr in range(9, 14):
                stw15DrftB4 = np.array(
                    [ i for i in dtNew1 if i[ 4 ] >= 11.5 and i[ 4 ] < 12 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
                stw15DrftB4 = np.delete(stw15DrftB4, [ i for (i, v) in enumerate(stw15DrftB4[ :, 8 ]) if v < (
                        np.mean(stw15DrftB4[ :, 8 ]) - np.std(stw15DrftB4[ :, 8 ])) or v > np.mean(
                    stw15DrftB4[ :, 8 ]) + np.std(
                    stw15DrftB4[ :, 8 ]) ], 0)
                drftStw15.append(stw15DrftB4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [ str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 11.5 < STW < 12                  ' ], )
                data_writer.writerow([ ], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW', 'RPM'
                                         , 'DaysSinceLastPropCleaning', 'DaysSinceLastDryDock', 'M/E FOC (kg/min)' ])
                for data in stw15DrftB4:
                    data_writer.writerow(
                        [ data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ], data[ 4 ], data[ 5 ], data[ 6 ], data[ 7 ],
                          data[ 8 ] ])
                data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            data_writer.writerow([ ], )

            # stw15DrftB4 = np.concatenate(drftStw15)

            # drftStw16 = []
            # stw16DrftB4 = np.array([i for i in dtNew1 if i[4] >= 12 and i[4] < 12.5 and i[0] > 9])
            for dr in range(9, 14):
                stw16DrftB4 = np.array(
                    [ i for i in dtNew1 if i[ 4 ] >= 12 and i[ 4 ] < 12.5 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
                stw16DrftB4 = np.delete(stw16DrftB4, [ i for (i, v) in enumerate(stw16DrftB4[ :, 8 ]) if v < (
                        np.mean(stw16DrftB4[ :, 8 ]) - np.std(stw16DrftB4[ :, 8 ])) or v > np.mean(
                    stw16DrftB4[ :, 8 ]) + np.std(
                    stw16DrftB4[ :, 8 ]) ], 0)
                drftStw16.append(stw16DrftB4)

                data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [ str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 12 < STW < 12.5                   ' ], )
                data_writer.writerow([ ], )
                data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW', 'RPM'
                                         , 'DaysSinceLastPropCleaning', 'DaysSinceLastDryDock', 'M/E FOC (kg/min)' ])
                for data in stw16DrftB4:
                    data_writer.writerow(
                        [ data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ], data[ 4 ], data[ 5 ], data[ 6 ], data[ 7 ],
                          data[ 8 ] ])
                data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            data_writer.writerow([ ], )

            # stw16DrftB4 = np.concatenate(drftStw16)

            # drftStw17 = []
            # stw17DrftB4 = np.array([i for i in dtNew1 if i[4] >= 12.5 and i[4] < 13 and i[0] > 9])
            for dr in range(9, 14):
                stw17DrftB4 = np.array(
                    [ i for i in dtNew1 if i[ 4 ] >= 12.5 and i[ 4 ] < 13 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
                data_writer.writerow(
                    [ str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 12.5 < STW < 13                 ' ], )

                if stw17DrftB4.shape[ 0 ] != 0:
                    stw17DrftB4 = np.delete(stw17DrftB4, [ i for (i, v) in enumerate(stw17DrftB4[ :, 8 ]) if v < (
                            np.mean(stw17DrftB4[ :, 8 ]) - np.std(stw17DrftB4[ :, 8 ])) or v > np.mean(
                        stw17DrftB4[ :, 8 ]) + np.std(
                        stw17DrftB4[ :, 8 ]) ], 0)
                    drftStw17.append(stw17DrftB4)

                    data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                    data_writer.writerow([ ], )
                    data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW',
                                           'M/E FOC (kg/min)', 'DaysSinceLastPropCleaning', 'DaysSinceLastDryDock' ])
                    for data in stw17DrftB4:
                        data_writer.writerow(
                            [ data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ], data[ 4 ], data[ 5 ], data[ 6 ], data[ 7 ],
                              data[ 8 ] ])
                data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            data_writer.writerow([ ], )

            # stw17DrftB4 = np.concatenate(drftStw17)

            # drftStw18 = []
            # stw18DrftB4 = np.array([i for i in dtNew1 if i[4] >= 13 and i[4] < 13.5 and i[0] > 9])
            for dr in range(9, 14):
                stw18DrftB4 = np.array(
                    [ i for i in dtNew1 if i[ 4 ] >= 13 and i[ 4 ] < 13.5 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
                data_writer.writerow(
                    [ str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 13 < STW < 13.5                 ' ], )

                if stw18DrftB4.shape[ 0 ] != 0:
                    stw18DrftB4 = np.delete(stw18DrftB4, [ i for (i, v) in enumerate(stw18DrftB4[ :, 8 ]) if v < (
                            np.mean(stw18DrftB4[ :, 8 ]) - np.std(stw18DrftB4[ :, 8 ])) or v > np.mean(
                        stw18DrftB4[ :, 8 ]) + np.std(
                        stw18DrftB4[ :, 8 ]) ], 0)
                    drftStw18.append(stw18DrftB4)

                    data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                    data_writer.writerow([ ], )
                    data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW',
                                           'M/E FOC (kg/min)', 'DaysSinceLastPropCleaning', 'DaysSinceLastDryDock' ])
                    for data in stw18DrftB4:
                        data_writer.writerow(
                            [ data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ], data[ 4 ], data[ 5 ], data[ 6 ], data[ 7 ],
                              data[ 8 ] ])
                data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            data_writer.writerow([ ], )

            # stw18DrftB4 = np.concatenate(drftStw18)

            # drftStw19 = []
            # stw19DrftB4 = np.array([i for i in dtNew1 if i[4] >= 13.5 and i[4] < 14 and i[0] > 9])
            for dr in range(9, 14):
                stw19DrftB4 = np.array(
                    [ i for i in dtNew1 if i[ 4 ] >= 13.5 and i[ 4 ] < 14 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
                data_writer.writerow(
                    [ str(dr) + ' < DRAFT < ' + str(dr + 1) + '   ' + ' 13.5 < STW < 14                 ' ], )
                if stw19DrftB4.shape[ 0 ] != 0:
                    stw19DrftB4 = np.delete(stw19DrftB4, [ i for (i, v) in enumerate(stw19DrftB4[ :, 8 ]) if v < (
                            np.mean(stw19DrftB4[ :, 8 ]) - np.std(stw19DrftB4[ :, 8 ])) or v > np.mean(
                        stw19DrftB4[ :, 8 ]) + np.std(
                        stw19DrftB4[ :, 8 ]) ], 0)
                    drftStw19.append(stw19DrftB4)

                    data_writer = csv.writer(rows, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                    data_writer.writerow([ ], )
                    data_writer.writerow([ 'Draft', 'Trim', 'WindSpeed', 'WindAngle', 'STW', 'RPM'
                                             , 'DaysSinceLastPropCleaning', 'DaysSinceLastDryDock', 'M/E FOC (kg/min)' ])
                    for data in stw19DrftB4:
                        data_writer.writerow(
                            [ data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ], data[ 4 ], data[ 5 ], data[ 6 ], data[ 7 ],
                              data[ 8 ] ])
                data_writer.writerow([ ], )
            data_writer.writerow([ ], )
            data_writer.writerow([ ], )

        # tw19DrftB4 = np.concatenate(drftStw19)

        x = 0


    def readLarosDataFromCsvNew(self, data):
        # Load file
        from sklearn import preprocessing
        from sklearn.decomposition import PCA
        # if self.__class__.__name__ == 'UnseenSeriesReader':
        # dt = data.sample(n=2880).values[ :90000:, 3:23 ]
        # dt = data.values[ 0:, 2:23 ]
        # else:
        # dt = data.values[0:,2:23]
        # dataNew = data.drop(['M/E FOC (kg/min)', 'DateTime'], axis=1)
        # names = dataNew.columns

        dataNew = data.drop([ 'DateTime', 'SpeedOvg' ], axis=1)
        # 'M/E FOC (kg/min)'

        # dataNew=dataNew[['Draft', 'SpeedOvg']]

        dtNew = dataNew.values[ 484362:569386, : ].astype(float)

        draftMean = (dtNew[ :, 1 ] + dtNew[ :, 2 ]) / 2
        dtNew[ :, 0 ] = np.round(draftMean, 3)

        dtNew = np.delete(dtNew, [ i for (i, v) in enumerate(dtNew[ 0:, 0 ]) if v <= 6 ], 0)
        dtNew = np.delete(dtNew, [ i for (i, v) in enumerate(dtNew[ 0:, 5 ]) if v <= 8 or v > 14 ], 0)
        dtNew = np.delete(dtNew, [ i for (i, v) in enumerate(dtNew[ 0:, 6 ]) if v <= 0 or v > 90 ],
                          0)  # or (v>0 and v <=10)
        # dtNew = np.delete(dtNew, [ i for (i, v) in enumerate(dtNew[ 0:, 7 ]) ], 0) # if v <= 7

        dtNew = dtNew[ ~np.isnan(dtNew).any(axis=1) ]

        trim = (dtNew[ :, 1 ] - dtNew[ :, 2 ])

        dtNew1 = np.array(np.append(dtNew[ :, 0 ].reshape(-1, 1), np.asmatrix(
            [ np.round(trim, 3), dtNew[ :, 3 ], dtNew[ :, 4 ], dtNew[ :, 5 ], dtNew[ :, 7 ] ]).T, axis=1))

        ##statistics STD
        # draftSmaller6 = np.array([drft for drft in dtNew1 if drft[0]< 6])
        # draftBigger6 = np.array([ drft for drft in dtNew1 if drft[0] > 6 ])
        #########DRAFT < 9
        drftStw8 = [ ]
        # stw8DrftS4 = np.array([i for i in dtNew1 if i[4]>=8 and i[4]<8.5 and i[0]<=9])

        for dr in range(6, 9):
            stw8DrftS4 = np.array(
                [ i for i in dtNew1 if i[ 4 ] >= 8 and i[ 4 ] < 8.5 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
            if stw8DrftS4.shape[ 0 ] != 0:
                stw8DrftS4 = np.delete(stw8DrftS4, [ i for (i, v) in enumerate(stw8DrftS4[ :, 5 ]) if v < (
                        np.mean(stw8DrftS4[ :, 5 ]) - np.std(stw8DrftS4[ :, 5 ])) or v > np.mean(
                    stw8DrftS4[ :, 5 ]) + np.std(stw8DrftS4[ :, 5 ]) ], 0)
                drftStw8.append(stw8DrftS4)
        stw8DrftS4 = np.concatenate(drftStw8)

        drftStw9 = [ ]
        stw9DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 8.5 and i[ 4 ] < 9 and i[ 0 ] <= 9 ])

        for dr in range(6, 9):
            stw9DrftS4 = np.array(
                [ i for i in dtNew1 if i[ 4 ] >= 8.5 and i[ 4 ] < 9 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
            stw9DrftS4 = np.delete(stw9DrftS4, [ i for (i, v) in enumerate(stw9DrftS4[ :, 5 ]) if v < (
                    np.mean(stw9DrftS4[ :, 5 ]) - np.std(stw9DrftS4[ :, 5 ])) or v > np.mean(stw9DrftS4[ :, 5 ]) + np.std(
                stw9DrftS4[ :, 5 ]) ], 0)
            drftStw9.append(stw9DrftS4)

        stw9DrftS4 = np.concatenate(drftStw9)

        drftStw10 = [ ]
        # stw10DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 9 and i[ 4 ] < 9.5 and i[ 0 ] <= 9 ])

        for dr in range(6, 9):
            stw10DrftS4 = np.array(
                [ i for i in dtNew1 if i[ 4 ] >= 9 and i[ 4 ] < 9.5 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
            stw10DrftS4 = np.delete(stw10DrftS4, [ i for (i, v) in enumerate(stw10DrftS4[ :, 5 ]) if v < (
                    np.mean(stw10DrftS4[ :, 5 ]) - np.std(stw10DrftS4[ :, 5 ])) or v > np.mean(
                stw10DrftS4[ :, 5 ]) + np.std(stw10DrftS4[ :, 5 ]) ], 0)
            drftStw10.append(stw10DrftS4)

        stw10DrftS4 = np.concatenate(drftStw10)

        drftStw11 = [ ]
        # stw11DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 9.5 and i[ 4 ] < 10 and i[ 0 ] <= 9 ])

        for dr in range(6, 9):
            stw11DrftS4 = np.array(
                [ i for i in dtNew1 if i[ 4 ] >= 9.5 and i[ 4 ] < 10 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
            stw11DrftS4 = np.delete(stw10DrftS4, [ i for (i, v) in enumerate(stw11DrftS4[ :, 5 ]) if v < (
                    np.mean(stw11DrftS4[ :, 5 ]) - np.std(stw11DrftS4[ :, 5 ])) or v > np.mean(
                stw11DrftS4[ :, 5 ]) + np.std(stw11DrftS4[ :, 5 ]) ], 0)
            drftStw11.append(stw11DrftS4)

        stw11DrftS4 = np.concatenate(drftStw11)

        drftStw12 = [ ]
        # stw12DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 10 and i[ 4 ] < 10.5 and i[ 0 ] <= 9 ])

        for dr in range(6, 9):
            stw12DrftS4 = np.array(
                [ i for i in dtNew1 if i[ 4 ] >= 10 and i[ 4 ] < 10.5 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
            stw12DrftS4 = np.delete(stw12DrftS4, [ i for (i, v) in enumerate(stw12DrftS4[ :, 5 ]) if v < (
                    np.mean(stw12DrftS4[ :, 5 ]) - np.std(stw12DrftS4[ :, 5 ])) or v > np.mean(
                stw12DrftS4[ :, 5 ]) + np.std(stw12DrftS4[ :, 5 ]) ], 0)
            drftStw12.append(stw12DrftS4)

        stw12DrftS4 = np.concatenate(drftStw12)

        drftStw13 = [ ]
        # stw13DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 10.5 and i[ 4 ] < 11 and i[ 0 ] <= 9 ])

        for dr in range(6, 9):
            stw13DrftS4 = np.array(
                [ i for i in dtNew1 if i[ 4 ] >= 10.5 and i[ 4 ] < 11 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
            stw13DrftS4 = np.delete(stw13DrftS4, [ i for (i, v) in enumerate(stw13DrftS4[ :, 5 ]) if v < (
                    np.mean(stw13DrftS4[ :, 5 ]) - np.std(stw13DrftS4[ :, 5 ])) or v > np.mean(
                stw13DrftS4[ :, 5 ]) + np.std(stw13DrftS4[ :, 5 ]) ], 0)
            drftStw13.append(stw13DrftS4)

        stw13DrftS4 = np.concatenate(drftStw13)

        drftStw14 = [ ]
        # stw14DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 11 and i[ 4 ] < 11.5 and i[ 0 ] <= 9 ])

        for dr in range(6, 9):
            stw14DrftS4 = np.array(
                [ i for i in dtNew1 if i[ 4 ] >= 11 and i[ 4 ] < 11.5 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
            stw14DrftS4 = np.delete(stw14DrftS4, [ i for (i, v) in enumerate(stw14DrftS4[ :, 5 ]) if v < (
                    np.mean(stw14DrftS4[ :, 5 ]) - np.std(stw14DrftS4[ :, 5 ])) or v > np.mean(
                stw14DrftS4[ :, 5 ]) + np.std(stw14DrftS4[ :, 5 ]) ], 0)
            drftStw14.append(stw14DrftS4)

        stw14DrftS4 = np.concatenate(drftStw14)

        drftStw15 = [ ]
        stw15DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 11.5 and i[ 4 ] < 12 and i[ 0 ] <= 9 ])

        for dr in range(6, 9):
            stw15DrftS4 = np.array(
                [ i for i in dtNew1 if i[ 4 ] >= 11.5 and i[ 4 ] < 12 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
            stw15DrftS4 = np.delete(stw15DrftS4, [ i for (i, v) in enumerate(stw15DrftS4[ :, 5 ]) if v < (
                    np.mean(stw15DrftS4[ :, 5 ]) - np.std(stw15DrftS4[ :, 5 ])) or v > np.mean(
                stw15DrftS4[ :, 5 ]) + np.std(stw15DrftS4[ :, 5 ]) ], 0)
            drftStw15.append(stw15DrftS4)

        stw15DrftS4 = np.concatenate(drftStw15)

        drftStw16 = [ ]
        # stw16DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 12 and i[ 4 ] < 12.5 and i[ 0 ] <= 9 ])
        for dr in range(6, 9):
            stw16DrftS4 = np.array(
                [ i for i in dtNew1 if i[ 4 ] >= 12 and i[ 4 ] < 12.5 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
            stw16DrftS4 = np.delete(stw16DrftS4, [ i for (i, v) in enumerate(stw16DrftS4[ :, 5 ]) if v < (
                    np.mean(stw16DrftS4[ :, 5 ]) - np.std(stw16DrftS4[ :, 5 ])) or v > np.mean(
                stw16DrftS4[ :, 5 ]) + np.std(stw16DrftS4[ :, 5 ]) ], 0)
            drftStw16.append(stw16DrftS4)

        stw16DrftS4 = np.concatenate(drftStw16)

        drftStw17 = [ ]
        # stw17DrftS4 = np.array([i for i in dtNew1 if i[4] >= 12.5 and i[4] < 13 and i[0] <= 9])

        for dr in range(6, 9):
            stw17DrftS4 = np.array(
                [ i for i in dtNew1 if i[ 4 ] >= 12 and i[ 4 ] < 12.5 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
            if stw17DrftS4.shape[ 0 ] != 0:
                stw17DrftS4 = np.delete(stw17DrftS4, [ i for (i, v) in enumerate(stw17DrftS4[ :, 5 ]) if v < (
                        np.mean(stw17DrftS4[ :, 5 ]) - np.std(stw17DrftS4[ :, 5 ])) or v > np.mean(
                    stw17DrftS4[ :, 5 ]) + np.std(stw17DrftS4[ :, 5 ]) ], 0)
                drftStw17.append(stw17DrftS4)

        stw17DrftS4 = np.concatenate(drftStw17)

        drftStw18 = [ ]
        # stw18DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 13 and i[ 4 ] < 13.5 and i[ 0 ] <= 9 ])
        for dr in range(6, 9):
            stw18DrftS4 = np.array(
                [ i for i in dtNew1 if i[ 4 ] >= 13 and i[ 4 ] < 13.5 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
            if stw18DrftS4.shape[ 0 ] != 0:
                stw18DrftS4 = np.delete(stw18DrftS4, [ i for (i, v) in enumerate(stw18DrftS4[ :, 5 ]) if v < (
                        np.mean(stw18DrftS4[ :, 5 ]) - np.std(stw18DrftS4[ :, 5 ])) or v > np.mean(
                    stw18DrftS4[ :, 5 ]) + np.std(stw18DrftS4[ :, 5 ]) ], 0)
                drftStw18.append(stw18DrftS4)

        stw18DrftS4 = np.concatenate(drftStw18)

        drftStw19 = [ ]
        # stw19DrftS4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 13.5 and i[ 4 ] < 14 and i[ 0 ] <= 9 ])
        for dr in range(6, 9):
            stw19DrftS4 = np.array(
                [ i for i in dtNew1 if i[ 4 ] >= 13.5 and i[ 4 ] < 14 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
            if stw19DrftS4.shape[ 0 ] != 0:
                stw19DrftS4 = np.delete(stw19DrftS4, [ i for (i, v) in enumerate(stw19DrftS4[ :, 5 ]) if v < (
                        np.mean(stw19DrftS4[ :, 5 ]) - np.std(stw19DrftS4[ :, 5 ])) or v > np.mean(
                    stw19DrftS4[ :, 5 ]) + np.std(stw19DrftS4[ :, 5 ]) ], 0)
                drftStw19.append(stw19DrftS4)

        stw19DrftS4 = np.concatenate(drftStw19)

        ##################################################################################
        ##################################################################################
        ##################################################################################
        #########DRAFT > 9
        drftStw8 = [ ]
        stw8DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 8 and i[ 4 ] < 8.5 and i[ 0 ] > 9 ])
        for dr in range(9, 14):
            stw8DrftB4 = np.array(
                [ i for i in dtNew1 if i[ 4 ] >= 8 and i[ 4 ] < 8.5 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
            stw8DrftB4 = np.delete(stw8DrftB4, [ i for (i, v) in enumerate(stw8DrftB4[ :, 5 ]) if v < (
                    np.mean(stw8DrftB4[ :, 5 ]) - np.std(stw8DrftB4[ :, 5 ])) or v > np.mean(stw8DrftB4[ :, 5 ]) + np.std(
                stw8DrftB4[ :, 5 ]) ], 0)
            drftStw8.append(stw8DrftB4)
        stw8DrftB4 = np.concatenate(drftStw8)

        drftStw9 = [ ]
        stw9DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 8.5 and i[ 4 ] < 9 and i[ 0 ] > 9 ])
        for dr in range(9, 14):
            stw9DrftB4 = np.array(
                [ i for i in dtNew1 if i[ 4 ] >= 8.5 and i[ 4 ] < 9 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
            stw9DrftB4 = np.delete(stw9DrftB4, [ i for (i, v) in enumerate(stw9DrftB4[ :, 5 ]) if v < (
                    np.mean(stw9DrftB4[ :, 5 ]) - np.std(stw9DrftB4[ :, 5 ])) or v > np.mean(stw9DrftB4[ :, 5 ]) + np.std(
                stw9DrftB4[ :, 5 ]) ], 0)
            drftStw9.append(stw9DrftB4)

        stw9DrftB4 = np.concatenate(drftStw9)

        drftStw10 = [ ]
        stw10DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 9 and i[ 4 ] < 9.5 and i[ 0 ] > 9 ])
        for dr in range(9, 14):
            stw10DrftB4 = np.array(
                [ i for i in dtNew1 if i[ 4 ] >= 9 and i[ 4 ] < 9.5 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
            stw10DrftB4 = np.delete(stw10DrftB4, [ i for (i, v) in enumerate(stw10DrftB4[ :, 5 ]) if v < (
                    np.mean(stw10DrftB4[ :, 5 ]) - np.std(stw10DrftB4[ :, 5 ])) or v > np.mean(
                stw10DrftB4[ :, 5 ]) + np.std(stw10DrftB4[ :, 5 ]) ], 0)
            drftStw10.append(stw10DrftB4)
        stw10DrftB4 = np.concatenate(drftStw10)

        drftStw11 = [ ]
        stw11DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 9.5 and i[ 4 ] < 10 and i[ 0 ] > 9 ])
        for dr in range(9, 14):
            stw11DrftB4 = np.array(
                [ i for i in dtNew1 if i[ 4 ] >= 9.5 and i[ 4 ] < 10 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
            stw11DrftB4 = np.delete(stw11DrftB4, [ i for (i, v) in enumerate(stw11DrftB4[ :, 5 ]) if v < (
                    np.mean(stw11DrftB4[ :, 5 ]) - np.std(stw11DrftB4[ :, 5 ])) or v > np.mean(
                stw11DrftB4[ :, 5 ]) + np.std(stw11DrftB4[ :, 5 ]) ], 0)
            drftStw11.append(stw11DrftB4)
        stw11DrftB4 = np.concatenate(drftStw11)

        drftStw12 = [ ]
        stw12DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 10 and i[ 4 ] < 10.5 and i[ 0 ] > 9 ])
        for dr in range(9, 14):
            stw12DrftB4 = np.array(
                [ i for i in dtNew1 if i[ 4 ] >= 10 and i[ 4 ] < 10.5 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
            stw12DrftB4 = np.delete(stw12DrftB4, [ i for (i, v) in enumerate(stw12DrftB4[ :, 5 ]) if v < (
                    np.mean(stw12DrftB4[ :, 5 ]) - np.std(stw12DrftB4[ :, 5 ])) or v > np.mean(
                stw12DrftB4[ :, 5 ]) + np.std(stw12DrftB4[ :, 5 ]) ], 0)
            drftStw12.append(stw12DrftB4)

        stw12DrftB4 = np.concatenate(drftStw12)

        drftStw13 = [ ]
        stw13DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 10.5 and i[ 4 ] < 11 and i[ 0 ] > 9 ])
        for dr in range(9, 14):
            stw13DrftB4 = np.array(
                [ i for i in dtNew1 if i[ 4 ] >= 10.5 and i[ 4 ] < 11 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
            stw13DrftB4 = np.delete(stw13DrftB4, [ i for (i, v) in enumerate(stw13DrftB4[ :, 5 ]) if v < (
                    np.mean(stw13DrftB4[ :, 5 ]) - np.std(stw13DrftB4[ :, 5 ])) or v > np.mean(
                stw13DrftB4[ :, 5 ]) + np.std(stw13DrftB4[ :, 5 ]) ], 0)
            drftStw13.append(stw13DrftB4)
        stw13DrftB4 = np.concatenate(drftStw13)

        drftStw14 = [ ]
        stw14DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 11 and i[ 4 ] < 11.5 and i[ 0 ] > 9 ])
        for dr in range(9, 14):
            stw14DrftB4 = np.array(
                [ i for i in dtNew1 if i[ 4 ] >= 11 and i[ 4 ] < 11.5 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
            stw14DrftB4 = np.delete(stw14DrftB4, [ i for (i, v) in enumerate(stw14DrftB4[ :, 5 ]) if v < (
                    np.mean(stw14DrftB4[ :, 5 ]) - np.std(stw14DrftB4[ :, 5 ])) or v > np.mean(
                stw14DrftB4[ :, 5 ]) + np.std(stw14DrftB4[ :, 5 ]) ], 0)
            drftStw14.append(stw14DrftB4)
        stw14DrftB4 = np.concatenate(drftStw14)

        drftStw15 = [ ]
        stw15DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 11.5 and i[ 4 ] < 12 and i[ 0 ] > 9 ])
        for dr in range(9, 14):
            stw15DrftB4 = np.array(
                [ i for i in dtNew1 if i[ 4 ] >= 11.5 and i[ 4 ] < 12 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
            stw15DrftB4 = np.delete(stw15DrftB4, [ i for (i, v) in enumerate(stw15DrftB4[ :, 5 ]) if v < (
                    np.mean(stw15DrftB4[ :, 5 ]) - np.std(stw15DrftB4[ :, 5 ])) or v > np.mean(
                stw15DrftB4[ :, 5 ]) + np.std(stw15DrftB4[ :, 5 ]) ], 0)
            drftStw15.append(stw15DrftB4)
        stw15DrftB4 = np.concatenate(drftStw15)

        drftStw16 = [ ]
        stw16DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 12 and i[ 4 ] < 12.5 and i[ 0 ] > 9 ])
        for dr in range(9, 14):
            stw16DrftB4 = np.array(
                [ i for i in dtNew1 if i[ 4 ] >= 12 and i[ 4 ] < 12.5 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
            stw16DrftB4 = np.delete(stw16DrftB4, [ i for (i, v) in enumerate(stw16DrftB4[ :, 5 ]) if v < (
                    np.mean(stw16DrftB4[ :, 5 ]) - np.std(stw16DrftB4[ :, 5 ])) or v > np.mean(
                stw16DrftB4[ :, 5 ]) + np.std(stw16DrftB4[ :, 5 ]) ], 0)
            drftStw16.append(stw16DrftB4)
        stw16DrftB4 = np.concatenate(drftStw16)

        drftStw17 = [ ]
        stw17DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 12.5 and i[ 4 ] < 13 and i[ 0 ] > 9 ])
        for dr in range(9, 14):
            stw17DrftB4 = np.array(
                [ i for i in dtNew1 if i[ 4 ] >= 12.5 and i[ 4 ] < 13 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
            if stw17DrftB4.shape[ 0 ] != 0:
                stw17DrftB4 = np.delete(stw17DrftB4, [ i for (i, v) in enumerate(stw17DrftB4[ :, 5 ]) if v < (
                        np.mean(stw17DrftB4[ :, 5 ]) - np.std(stw17DrftB4[ :, 5 ])) or v > np.mean(
                    stw17DrftB4[ :, 5 ]) + np.std(stw17DrftB4[ :, 5 ]) ], 0)
                drftStw17.append(stw17DrftB4)
        stw17DrftB4 = np.concatenate(drftStw17)

        drftStw18 = [ ]
        stw18DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 13 and i[ 4 ] < 13.5 and i[ 0 ] > 9 ])
        for dr in range(9, 14):
            stw18DrftB4 = np.array(
                [ i for i in dtNew1 if i[ 4 ] >= 13 and i[ 4 ] < 13.5 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
            if stw18DrftB4.shape[ 0 ] != 0:
                stw18DrftB4 = np.delete(stw18DrftB4, [ i for (i, v) in enumerate(stw18DrftB4[ :, 5 ]) if v < (
                        np.mean(stw18DrftB4[ :, 5 ]) - np.std(stw18DrftB4[ :, 5 ])) or v > np.mean(
                    stw18DrftB4[ :, 5 ]) + np.std(stw18DrftB4[ :, 5 ]) ], 0)
                drftStw18.append(stw18DrftB4)
        stw18DrftB4 = np.concatenate(drftStw18)

        drftStw19 = [ ]
        stw19DrftB4 = np.array([ i for i in dtNew1 if i[ 4 ] >= 13.5 and i[ 4 ] < 14 and i[ 0 ] > 9 ])
        for dr in range(9, 14):
            stw19DrftB4 = np.array(
                [ i for i in dtNew1 if i[ 4 ] >= 13.5 and i[ 4 ] < 14 and i[ 0 ] > dr and i[ 0 ] <= dr + 1 ])
            if stw19DrftB4.shape[ 0 ] != 0:
                stw19DrftB4 = np.delete(stw19DrftB4, [ i for (i, v) in enumerate(stw19DrftB4[ :, 5 ]) if v < (
                        np.mean(stw19DrftB4[ :, 5 ]) - np.std(stw19DrftB4[ :, 5 ])) or v > np.mean(
                    stw19DrftB4[ :, 5 ]) + np.std(stw19DrftB4[ :, 5 ]) ], 0)
                drftStw19.append(stw19DrftB4)
        stw19DrftB4 = np.concatenate(drftStw19)

        ################################################
        concatS = np.concatenate(
            [ stw8DrftS4, stw9DrftS4, stw10DrftS4, stw11DrftS4, stw12DrftS4, stw13DrftS4, stw14DrftS4, stw15DrftS4,
              stw16DrftS4, stw17DrftS4, stw18DrftS4, stw19DrftS4 ])
        concatB = np.concatenate(
            [ stw8DrftB4, stw9DrftB4, stw10DrftB4, stw11DrftB4, stw12DrftB4, stw13DrftB4, stw14DrftB4, stw15DrftB4,
              stw16DrftB4, stw17DrftB4, stw18DrftB4, stw19DrftB4 ])
        dtNew1 = np.concatenate([ concatB, concatS ])
        # partitions = np.append([concatB,concatS])
        # stdFOC4_89 = np.std(stw8DrftS4[:,5])

        concatS = [ stw8DrftS4, stw9DrftS4, stw10DrftS4, stw11DrftS4, stw12DrftS4, stw13DrftS4, stw14DrftS4, stw15DrftS4,
                    stw16DrftS4, stw17DrftS4, stw18DrftS4, stw19DrftS4 ]
        concatB = [ stw8DrftB4, stw9DrftB4, stw10DrftB4, stw11DrftB4, stw12DrftB4, stw13DrftB4, stw14DrftB4, stw15DrftB4,
                    stw16DrftB4, stw17DrftB4, stw18DrftB4, stw19DrftB4 ]
        # concat =[ stw8DrftS4, stw9DrftS4, stw10DrftS4, stw11DrftS4, stw12DrftS4, stw13DrftS4, stw14DrftS4, stw15DrftS4,
        # stw16DrftS4, stw17DrftS4, stw18DrftS4, stw19DrftS4,
        # stw8DrftB4, stw9DrftB4, stw10DrftB4, stw11DrftB4, stw12DrftB4, stw13DrftB4, stw14DrftB4, stw15DrftB4,
        # stw16DrftB4, stw17DrftB4, stw18DrftB4, stw19DrftB4 ]

        # partitionsX=[k[:,0:5] for k in concat]
        # partitionsY = [ k[ :, 5 ] for k in concat ]
        # draftBigger9 = np.array([ drft for drft in dtNew1 if drft[ 0 ] > 9 ])

        # stw8Drft9 = np.array([ i for i in draftBigger9 if i[ 4 ] >= 12 and i[ 4 ] < 12.5 ])

        # stdFOC9_12 = np.std(stw8Drft9[ :, 5 ])

        seriesX = dtNew1[ :, 0:5 ]
        # seriesX=dtNew[:,0:5]
        # pca = PCA()
        # pca.fit(seriesX)
        # seriesX = pca.transform(seriesX)
        # scaler = preprocessing.StandardScaler()
        # Fit your data on the scaler object
        # scaled_df = scaler.fit_transform(seriesX)
        # seriesX = scaled_df
        # scaled_df = pd.DataFrame(scaled_df, columns=names)
        # dataNew = scaled_df

        # seriesX=seriesX[ 0:, 0:3 ]
        # seriesX=preprocessing.normalize([seriesX])
        newSeriesX = [ ]
        # for x in seriesX:
        # dmWS = x[0] * x[1]
        # waSO = x[2] * x[3]
        # newSeriesX.append([x[0],x[1],x[2],x[3],np.round(dmWS,3),np.round(waSO,3)])
        # newSeriesX = np.array(newSeriesX)
        ##seriesX = newSeriesX

        UnseenSeriesX = dtNew1[ 10000:11000, 0:5 ]
        newUnseenSeriesX = [ ]
        # for x in UnseenSeriesX:
        # dmWS = x[0] * x[1]
        # waSO = x[2] * x[3]
        # newUnseenSeriesX.append([x[0], x[1], x[2], x[3], np.round(dmWS, 3), np.round(waSO, 3)])
        # newUnseenSeriesX = np.array(newUnseenSeriesX)
        # UnseenSeriesX = newUnseenSeriesX
        ##RPM
        # Rpm = dtNew[0:150000,4]
        # unseenRpm = dtNew[160000:161000,4]

        ##FOC
        FOC = dtNew1[ :, 5 ]
        # scaled_df = scaler.fit_transform(FOC)
        # FOC = scaled_df
        # normalized_Y = preprocessing.normalize([FOC])

        unseenFOC = dtNew1[ 10000:11000, 5 ]
        # WS= np.asarray([x for x in dt[ :, 1 ] ])
        # WA = np.asarray([ x for x in dt[ :, 2 ] ])
        # SO = np.asarray([ y for y in dt[ :, 3 ] ])
        # RPM = np.asarray([ y for y in dt[ :, 4 ] ])
        # FOC = np.asarray([ y for y in dt[ :, 5 ] ])
        # POW = np.asarray([ y for y in dt[ :, 6 ] ])
        # DA = np.asarray([ np.nan_to_num(np.float(y)) for y in dt[ :, 13 ] ])
        # DF = np.asarray([ y for y in dt[ :, 0 ] ])
        WaFoc = [ ]
        # newArray = np.append([WS,WA,SO,RPM,POW,DF])
        partionLabels = np.arange(0, 24)
        return seriesX, FOC, UnseenSeriesX, unseenFOC, None, None, None, None, None, None, partionLabels  # draftBigger6[:,0:5],draftSmaller6[:,0:5] , draftBigger6[:,5] , draftSmaller6[:,5]


    def readLarosDataFromCsv(self, data):
        # Load file
        # if self.__class__.__name__ == 'UnseenSeriesReader':
        # dt = data.sample(n=2880).values[ :90000:, 3:23 ]
        # dt = data.values[ 0:, 2:23 ]
        # else:
        # dt = data.values[0:,2:23]
        dt = data.values[ 0:, 2:8 ].astype(float)
        dt = dt[ ~np.isnan(dt).any(axis=1) ]
        WS = np.asarray([ x for x in dt[ :, 0 ] ])
        WA = np.asarray([ x for x in dt[ :, 1 ] ])
        SO = np.asarray([ y for y in dt[ :, 3 ] ])
        RPM = np.asarray([ y for y in dt[ :, 4 ] ])
        FOC = np.asarray([ y for y in dt[ :, 5 ] ])
        POW = np.asarray([ y for y in dt[ :, 5 ] ])
        # DA = np.asarray([ np.nan_to_num(np.float(y)) for y in dt[ :, 13 ] ])
        DF = np.asarray([ np.nan_to_num(np.float(y)) for y in dt[ :, 6 ] ])
        WaFoc = [ ]
        for i in range(0, len(WA)):
            prev = np.append(WA[ i ], FOC[ i ])
            WaFoc.append(prev)

        WaFoc = np.array(WaFoc).reshape(-1, 2)
        # WaFoc = np.array(np.stack([WA,FOC],axis=0))
        firstFOC = [ ]
        secondFOC = [ ]
        thirdFOC = [ ]
        fourthFOC = [ ]
        fifthFOC = [ ]
        for val in WaFoc:
            if val[ 0 ] <= 22.5 or (val[ 0 ] <= 360 and val[ 0 ] > 337.5):
                firstFOC.append(val[ 1 ])
            elif (val[ 0 ] > 22.5 and val[ 0 ] < 67.5) or (val[ 0 ] > 292.5 and val[ 0 ] < 247.5):
                secondFOC.append(val[ 1 ])
            elif (val[ 0 ] > 67.5 and val[ 0 ] < 112.5) or (val[ 0 ] > 247.5 and val[ 0 ] < 292.5):
                thirdFOC.append(val[ 1 ])
            elif (val[ 0 ] > 112.5 and val[ 0 ] < 157.5) or (val[ 0 ] > 202.5 and val[ 0 ] < 247.5):
                fourthFOC.append(val[ 1 ])
            elif val[ 0 ] > 157.5 and val[ 0 ] < 202.5:
                fifthFOC.append(val[ 1 ])

        ##extract FOC depending on wind angle
        import matplotlib.pyplot as plt
        # arraysToplot = np.array(np.stack([firstFOC,secondFOC],axis=0))
        df = pd.DataFrame(np.array(firstFOC),
                          columns=[ '' ])
        boxplot = df.boxplot(column=[ '' ])
        plt.ylabel('M/E FOC (kg/min)')
        plt.title("STD and outliers of FOC with 0< WA <= 22.5")
        plt.show()
        df = pd.DataFrame(np.array(secondFOC),
                          columns=[ '' ])
        boxplot1 = df.boxplot(
            column=[ '' ])
        plt.ylabel('M/E FOC (kg/min)')
        plt.title("STD and outliers of FOC with WA > 22.5")
        plt.show()
        x = 1

        # x = [ math.sin(Lon[ i ] - Lon[ i - 1 ]) * math.cos(Lat[ i ]) for i in range(1, len(Lat)) ]
        # y = [ math.cos(Lat[ i ]) * math.sin(Lat[ i - 1 ])
        # - math.sin(Lat[ i - 1 ]) * math.sin(Lat[ i ]) * math.cos(Lon[ i ] - Lon[ i - 1 ]) for i in
        # range(1, len(Lat)) ]
        # bearing = [ ]

        # bearing.append([ math.atan2(x[ i ], y[ i ]) for i in range(0, len(x)) ])
        # bearing[ 0 ].insert(0, 0)
        # bearing = np.asarray(bearing)[ 0 ]
        # WA = np.asarray([ np.nan_to_num(np.float(y)) for y in dt[ :, 5 ] ])
        return WS, WA

