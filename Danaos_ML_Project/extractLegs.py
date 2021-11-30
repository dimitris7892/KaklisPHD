from datetime import datetime
#import holoviews as hv
import pandas as pd
import numpy as np
import dataReading as dRead
import csv
import json
#hv.extension('bokeh', 'matplotlib')
dr = dRead.BaseSeriesReader()
import geopy.distance
import pyproj
from pyproj import Proj, transform
import cx_Oracle
import os
import glob
#from global_land_mask import globe
geodesic = pyproj.Geod(ellps='WGS84')

class vesselDetails:

    def __init__(self):
        self.vesselName = None
        self.imo = None

class connDetails:

    def __init__(self):

        self.server = None
        self.sid = None
        self.password = None
        self.usr = None


class extractLegs:

    def __init__(self, company, vessel , server, sid, password, usr):

        self.company = company
        self.vessel = vessel
        self.server = server
        self.sid = sid
        self.password = password
        self.usr = usr

    def extractImo(self, ):

        dsn = cx_Oracle.makedsn(
            self.server,
            '1521',
            service_name= self.sid
        )
        connection = cx_Oracle.connect(
            user= self.usr,
            password=self.password,
            dsn= dsn
        )
        cursor_myserver = connection.cursor()
        cursor_myserver.execute(
            'SELECT IMO_NO FROM VESSEL_DATA WHERE VESSEL_CODE =  (SELECT VESSEL_CODE FROM VESSEL_DATA WHERE VESSEL_NAME LIKE '"'%" + self.vessel + "%'"'  AND ROWNUM=1)  ')

        for row in cursor_myserver.fetchall():
            imo = row[0]
            print(str(imo))
        return imo

    def convertRawToRollingWindowAvg(self, rawData):
        #leg_data = pd.read_csv('./4_leg.csv')
        #leg_data.head()
        rawDataDt = rawData[:,8]
        rawDatalat = rawData[:,6]
        rawDatalon = rawData[:,7]

        rawDataWOdtLatLon = rawData[:,:6]
        print(rawDataWOdtLatLon[0,5])
        for i in range(0, len(rawDataWOdtLatLon)):
                rawDataWOdtLatLon[i] = np.mean(rawDataWOdtLatLon[i:i + 15], axis=0)

        raw1 = np.append(rawDataWOdtLatLon, rawDatalat.reshape(-1,1), axis=1)
        raw2 = np.append(raw1, rawDatalon.reshape(-1,1), axis=1)
        raw3 = np.append(raw2, rawDataDt.reshape(-1,1), axis=1)
        return raw3

    def convertRawDataLATLON(self,company, vessel):
        path = './data/GOLDENPORT/TRAMMO LAOURA/'
        dataSet = []
        for infile in sorted(glob.glob(path + '*.csv')):
            data = pd.read_csv(infile, sep=';', decimal='.', skiprows=1)
            dataSet.append(data.values)
            print(str(infile))
        # if len(dataSet)>1:
        dataSet = np.concatenate(dataSet)
        lats = []
        lons = []
        times =[]
        for i in range(0, len(dataSet)):
            lat = str(dataSet[i, 4])
            latDir = str(dataSet[i, 5])
            lon = str(dataSet[i, 6])
            lonDir = str(dataSet[i, 7])
            timestamp = dataSet[i,0]
            #####FIND BEARING
            LAT, LON = dr.convertLATLONfromDegMinSec(lat, lon, latDir, lonDir)
            lats.append(LAT)
            lons.append(LON)
            times.append(timestamp)

        with open('./data/' + company + '/' + vessel + '/mappedDataLATLON.csv', mode='w') as data:
            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow(
                ['lat','lon','timestamp'])
            for i in range(0, len(lats)):
                data_writer.writerow(
                    [lats[i],lons[i],times[i]])

    def getRawDataOOCEANGOLD(self,company, vessel):
        sFile = './data/' + company + '/' + vessel + '/trackRepMapped'+vessel+'_.csv'
        data = pd.read_csv(sFile, sep=',').values

        print(data.shape)
        data[:, 0] = np.round(data[:, 0].astype(float), 2)
        data[:, 5] = np.round(data[:, 5].astype(float), 2)
        ##knots to m/s
        wfS = data[:, 1].astype(float) #/ (1.944)
        data[:, 1] = np.round(wfS, 2)

        wfS = data[:, 1]
        for i in range(0, len(wfS)):
            wfS[i] = dr.ConvertMSToBeaufort(float(float(wfS[i])))
        data[:, 1] = wfS

        # lt/min => MT/day
        data[:, 15] = ((data[:, 15]) / 1000) * 1440
        ##Build training data array

        trData = np.array(np.append(data[:, 8].reshape(-1, 1),
                                    np.asmatrix([data[:, 10], data[:, 11], data[:, 12], data[:, 22], (data[:, 15]),
                                                 data[:, 26], data[:, 27], data[:, 28]]).T,
                                    axis=1))  # .astype(float)#data[:,26],data[:,27]
        trData = np.array([k for k in trData if
                           str(k[0]) != 'nan' and float(k[2]) >= 0 and float(k[4]) >= 0 and float(k[3]) >= 0 and float(
                               k[5]) > 0])

        # trData[:,8] = list(map(lambda s: s[:-3], trData[:,8]))
        d = {'draft': trData[:, 0], 'wd': trData[:, 1], 'ws': trData[:, 2], 'stw': trData[:, 3], 'swh': trData[:, 4],
             'foc': (trData[:, 5]), 'lat': trData[:, 6], 'lon': trData[:, 7], 'timestamp': trData[:, 8]}
        df = pd.DataFrame(d)
        print(trData.shape)
        return df

    def getRawData(self, company, vessel):

        sFile = './data/'+company+'/'+vessel+'/mappedData.csv'

        data = pd.read_csv(sFile,sep=',').values

        print(data.shape)
        data[:, 12] = np.round(data[:,12].astype(float),2)
        data[:, 8] = np.round(data[:,8].astype(float),2)

        ##replace wind dir values < 0
        for i in range(0, len(data[:, 11])):
            if float(data[i, 11]) < 0:
                data[i, 11] += 360
        ##knots to m/s
        '''wfS = data[:, 11].astype(float) / (1.944)
        data[:, 11] = np.round(wfS,2)
    
        wfS= data[:,11]
        for i in range(0, len(wfS)):
            wfS[i] = dr.ConvertMSToBeaufort(float(float(wfS[i])))
        data[:, 11] = wfS
    
        #lt/min => MT/day
        data[:, 15] = ((data[:, 15]) / 1000)* 1440'''
        ##Build training data array

        trData = np.array(np.append(data[:,8].reshape(-1,1),np.asmatrix([data[:,10],data[:,11],data[:,12],data[:,22],(data[:,15]),

                                                                         data[:,26],data[:,27],data[:,0], data[:,1], data[:,28],
                                                                        data[:,31], data[:,29], data[:,30] ]).T,axis=1))#.astype(float)#data[:,26],data[:,27]

        trData = np.array([k for k in trData if  str(k[0])!='nan' and float(k[2])>=0 and float(k[4])>=0 and float(k[3])>=0 and float(k[5])>0  ])

        trData[:,8] = list(map(lambda s: s[:-3], trData[:,8]))
        d = {'draft': trData[:,0], 'wd': trData[:,1],'ws':trData[:,2],'stw':trData[:,3],'swh':trData[:,4],

             'foc':(trData[:,5]),'lat':trData[:,6],'lon':trData[:,7],'timestamp':trData[:,8], 'vslHeading': trData[:,9], 'wsWS':trData[:,10],
             'sOvg':trData[:,11], 'currSp':trData[:,12], 'currDir':trData[:,13]}

        df  = pd.DataFrame(d)
        print(trData.shape)
        return df

    def filterRawWithTlgDates(self, company, vessel, numberOfLegs):

        df = self.getRawData(company,vessel)
        telegrams = pd.read_csv('./data/'+company+'/'+vessel+'/TELEGRAMS/'+vessel+'.csv',sep=';')
        originalTlgs = telegrams.values

        rawData = df.values
        rawDates = [str(k[8]).split(" ")[0] for k in rawData]
        firstRawTimestamp = str(rawData[0,8])
        #print(firstRawTimestamp )
        date = firstRawTimestamp.split(" ")[0]

        tlgsDates = list(map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').replace(hour=0,minute=0,second=0, microsecond=0),telegrams['TELEGRAM_DATE']))

        #telegrams['TELEGRAM_DATE'] = tlgsDates

        filteredRaw = np.array([k for k in rawData if
        datetime.strptime(str(k[8]), '%Y-%m-%d %H:%M').replace(hour=0, minute=0, second=0,
                                                                                                microsecond=0) in tlgsDates])

        dfFilt = pd.DataFrame(filteredRaw).head()

        tlgs = telegrams.values
        filteredTlgs = np.array([k for k in originalTlgs if k[0].split(" ")[0]
         in rawDates])

        candidateDt = filteredRaw[0, 8].split(" ")[0]

        canDidatesDts = []
        for i in range(0,len(filteredTlgs)):
            tlg_type = filteredTlgs[i][1]
            if tlg_type=='D':
                canDidatesDts.append(filteredTlgs[i][0])#.split(" ")[0])
            if len(canDidatesDts) > numberOfLegs : break

        # datetime.strptime(str(rawData[0,8]), '%Y-%m-%d %H:%M:%S').replace(hour=0,minute=0,second=0, microsecond=0)
        tlgs = telegrams.values

        return canDidatesDts , rawData , dfFilt , tlgs, telegrams

    def extractLeg(self, candidateDt, tlgs, telegrams):

            print("Candidate Date: " + str(candidateDt))
            nextTlg_Type = None
            # print(tlgs[0][0][:-3])
            tlgForRawDate = telegrams[telegrams['TELEGRAM_DATE'] == candidateDt]
            # np.array([k for k in tlgs if
            # datetime.strptime(k[0][:-3], '%Y-%m-%d %H:%M').replace(hour=0,minute=0,second=0)
            # ==datetime.strptime(candidateDt, '%Y-%m-%d %H:%M')])[0]

            row = tlgForRawDate.index.values[0]

            # print(row)
            tlg_type = tlgForRawDate['TELEGRAM_TYPE'].values[0]
            # tlg_type = tlgForRawDate[1]
            # print(tlg_type)
            #tlg_type='D'
            tlgRowArr = []
            if tlg_type == 'D':
                tlgRowDep = tlgForRawDate.values[0]
                k = row
                while nextTlg_Type != 'A' and k < len(tlgs):
                    nextTlg_Type = tlgs[k][1]
                    if nextTlg_Type=='A':
                        tlgRowArr = tlgs[k]
                    # print(prevTlg_Type)

                    k = k + 1

            return tlgRowDep, tlgRowArr, row, k - 1

    def extractJSONforLeg(self, rawDataLeg, legDepartureDate, legArrivalDate, portDeparure, portArrival, lat_lon_Dep, lat_lon_Arr, company, vessel):

        leg = {   'departure': str(legDepartureDate),
                        'arrival': str(legArrivalDate),
                        'depPort': portDeparure,
                        'arrPort': portArrival,
                        'lat_lon_Dep': lat_lon_Dep,
                        'lat_lon_Arr': lat_lon_Arr,
                        'data': []}

        '''rawDataLeg = np.array(
            [k for k in rawData if datetime.strptime(k[8], '%Y-%m-%d %H:%M') >= legDepartureDate and
             datetime.strptime(k[8], '%Y-%m-%d %H:%M') <= legArrivalDate])'''

        if os.path.isdir('./legs/'+vessel+'/') == False:
            os.mkdir('./legs/'+vessel+'/')
        with open('./legs/'+vessel+'/'+portDeparure+'_'+portArrival+'_leg.csv', mode='w') as dataw:
            data_writer = csv.writer(dataw, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            data_writer.writerow(['stw', 'draft', 'foc', 'ws', 'wd', 'swh', 'lat', 'lon', 'dt', 'vslHeading', 'wsWS', 'sOvg', 'currSp', 'currDir'])

            for i in range(0, len(rawDataLeg)):
                dt = str(rawDataLeg[i][8])
                # print(dt)
                draft = rawDataLeg[i][0]
                ws = rawDataLeg[i][2]
                wd = rawDataLeg[i][1]
                stw = rawDataLeg[i][3]
                foc = rawDataLeg[i][5]
                swh = rawDataLeg[i][4]
                lat = rawDataLeg[i][6]
                lon = rawDataLeg[i][7]
                vslDir = rawDataLeg[i][9]
                wsWS = rawDataLeg[i][10]
                sOvg = rawDataLeg[i][11]
                currSp = rawDataLeg[i][12]
                currDir = rawDataLeg[i][13]

                #inclX =rawDataLeg[i][11]


                data_writer.writerow([stw, draft, foc, ws, wd, swh, lat, lon, dt, vslDir, wsWS, sOvg, currSp, currDir])

                item = {'datetime': dt, 'ws': ws, 'swh'

                : swh, 'wd': wd, 'stw': stw, 'foc': foc, 'lat': lat, 'lon': lon, 'draft': draft, 'vslHeading':vslDir, 'wsWS': wsWS,
                        'sOvg': sOvg, 'currSp': currSp, 'currDir': currDir }

                leg['data'].append(item)
        return leg

    def randomFunction(self,):
        import seaborn as sns
        trData1 = pd.read_csv('./legs/HYUNDAI SMART/PUSAN_PANAMA_leg.csv').values
        trDataNew = trData1

        trData1Lon = trData1[:, 7]

        trData1Lat = trData1[:, 6]
        meanLon = np.mean(trData1Lon)
        stdLon = np.std(trData1Lon)
        print(stdLon)

        meanLat = np.mean(trData1Lat)
        stdLat = np.std(trData1Lat)
        # trData1Lon = trData1Lon[trData1Lon <= trData1Lon + 1.2 * meanLon]

        # print()
        # print(np.max(trData1[:,7]))
        newLons = []
        newLats = []
        vknots = []
        dists = []
        lons = []
        bears = []

        bearings = []
        indices = []
        diffs = []
        for i in range(1, len(trData1) - 1):

            lat = trData1[i][6]
            lon = trData1[i][7]

            prevLat = trData1[i - 1][6]
            prevLon = trData1[i - 1][7]

            nextLat = trData1[i + 1][6]
            nextLon = trData1[i + 1][7]
            lons.append(lon)
            dist = geopy.distance.distance((lat, lon), (prevLat, prevLon)).km

            elapsedTime = (datetime.strptime(trData1[i][8], '%Y-%m-%d %H:%M') -
                           datetime.strptime(trData1[i - 1][8], '%Y-%m-%d %H:%M'))
            totalSailingTimeHours = divmod(elapsedTime.total_seconds(), 60)[0] / 60
            v = dist / totalSailingTimeHours
            vknot = v / 1.852
            # print(vknot)
            bearing_fwd, back_bearing, distance = geodesic.inv(prevLon, prevLat, lon, lat)
            bearing_fwd_n, back_bearing_n, distance_n = geodesic.inv(nextLon, nextLat, lon, lat)
            # print(bearing_fwd)
            # print(bearing_fwd_n)

            # diffs.append(abs(abs(bearing_fwd) - abs(bearing_fwd_n)))
            if vknot < 25 and vknot > 0 and \
                    (trData1[i][7] <= meanLon + 2 * stdLon and trData1[i][7] >= meanLon - 2 * stdLon) and \
                    globe.is_ocean(lat, lon) and \
                    lon < 0 and prevLat > lat and lat > 8 and \
                    (bearing_fwd > 130 and bearing_fwd <= 150):
                dists.append(dist)
                vknots.append(vknot)
                newLons.append(trData1[i][7])
                newLats.append(trData1[i][6])
                bearings.append(bearing_fwd)
                diffs.append(bearing_fwd)
                indices.append(i)
        trDataNew = trData1[indices]
        # trDataNew[:,7] = newLons
        # df = pd.DataFrame({'lat':trDataNew[:,6],'lon':trDataNew[:,7]})
        # df.to_csv('./HYUNDAI SMART/newcoords.csv')
        # print(diffs)
        diff = pd.DataFrame({'diff': diffs})
        sns.displot(diff)
        print(trDataNew.shape)
        # print(vknots[3677])
        # print(dists[3677])
        # print(newLons[3677])
        # print(np.max(newLons))
        print(np.max(trData1[:, 7]))
        print("Total geodesic distance: " + str(np.sum(dists)) + ' km')
        print("Geodesic distance first - last: " + str(geopy.distance.distance((trData1[0][6], trData1[0][7]),
                                                                               (trData1[len(trData1) - 1][6],
                                                                                trData1[len(trData1) - 1][
                                                                                    7])).km) + ' km')
        # dfLats = pd.DataFrame({'lat':trData1Lat})
        # sns.displot(dfLons)
        # bearings

        # trData1 = pd.read_csv('./HYUNDAI SMART/PUSAN_PANAMA_leg.csv').values
        # sns.displot(dfLats)
        pd.DataFrame(trDataNew).to_csv('./PUSAN_PANAMA.csv')

    def preprocessCoordinates(self, rawDataLeg):
        newIndices = []
        newLons = []
        newLats = []
        vknots = []
        dists = []
        lons = []

        meanLon = np.mean(rawDataLeg[:, 7])
        stdLon = np.std(rawDataLeg[:, 7])

        lat = rawDataLeg[0][6]
        lon = rawDataLeg[0][7]

        prevLat = rawDataLeg[1][6]
        prevLon = rawDataLeg[1][7]
        initBearing_fwd, back_initBearing, distance = geodesic.inv(prevLon, prevLat, lon, lat)
        bearings = []

        for i in range(1, len(rawDataLeg)):
            lat = rawDataLeg[i][6]
            lon = rawDataLeg[i][7]

            prevLat = rawDataLeg[i - 1][6]
            prevLon = rawDataLeg[i - 1][7]
            lons.append(lon)
            dist = geopy.distance.distance((lat, lon), (prevLat, prevLon)).km

            elapsedTime = (
                        datetime.strptime(rawDataLeg[i][8], '%Y-%m-%d %H:%M') - datetime.strptime(rawDataLeg[i - 1][8],
                                                                                                  '%Y-%m-%d %H:%M'))
            totalSailingTimeHours = divmod(elapsedTime.total_seconds(), 60)[0] / 60
            v = dist / totalSailingTimeHours
            vknot = v / 1.852
            bearing_fwd, back_bearing, distance = geodesic.inv(prevLon, prevLat, lon, lat)
            # (bearing_fwd<60 and bearing_fwd>-60)
            if vknot < 30 and vknot > 0 and (
                    rawDataLeg[i][7] <= meanLon + 2 * stdLon and rawDataLeg[i][7] >= meanLon - 2 * stdLon) and \
                    globe.is_ocean(lat, lon) and \
                    (bearing_fwd > 130 and bearing_fwd <= 150):
                newIndices.append(i)
        return rawDataLeg[newIndices]

    def runExtractionProcess(self,):

        company = self.company
        vessel  = self.vessel

        print(vessel)

        imo = self.extractImo()

        candidateDts, rawData, dfFilt, tlgs, telegrams = self.filterRawWithTlgDates(company, vessel, 50)
        legsExtracted = []
        # candidateDts = ['2020-06-08 13:00:00']
        for candidateDt in candidateDts:
            tlgRowDep, tlgRowArr, row, rowArr = self.extractLeg(candidateDt, tlgs, telegrams)
            if len(tlgRowArr) == 0: continue
            originalDeptlgDate = \
            pd.read_csv('./data/' + company + '/' + vessel + '/TELEGRAMS/' + vessel + '.csv', sep=';').values[row][0]
            originalArrtlgDate = \
            pd.read_csv('./data/' + company + '/' + vessel + '/TELEGRAMS/' + vessel + '.csv', sep=';').values[rowArr][0]
            # tlgRowArr[1]
            legDepartureDate = tlgRowDep[0]
            legArrivalDate = tlgRowArr[0]
            portDeparure = tlgRowDep[22]
            portArrival = tlgRowArr[22]
            print("Departure Port: " + str(portDeparure) + " / " + str(legDepartureDate))
            print("Arrival Port: " + str(portArrival) + "  / " + str(legArrivalDate))
            # legDepartureDate

            originalDeptlgDate = datetime.strptime(originalDeptlgDate, '%Y-%m-%d %H:%M:%S')
            originalArrtlgDate = datetime.strptime(originalArrtlgDate, '%Y-%m-%d %H:%M:%S')

            print("Original Dep Date: " + str(originalDeptlgDate))
            print("Original Arr Date: " + str(originalArrtlgDate))
            rawDataLeg = np.array([k for k in rawData if
                                   datetime.strptime(k[8], '%Y-%m-%d %H:%M') >= originalDeptlgDate
                                   and datetime.strptime(k[8], '%Y-%m-%d %H:%M') <= originalArrtlgDate])

            if len(rawDataLeg) == 0: continue
            lat_lon_Dep = (rawDataLeg[0][6], rawDataLeg[0][7])
            lat_lon_Arr = (rawDataLeg[len(rawDataLeg) - 1][6], rawDataLeg[len(rawDataLeg) - 1][7])

            print(lat_lon_Dep)
            print(lat_lon_Arr)

            # rawDataLeg = preprocessCoordinates(rawDataLeg)
            # rollingAvgsRawData = convertRawToRollingWindowAvg(rawDataLeg)
            elapsedTime = (originalArrtlgDate - originalDeptlgDate)
            totalSailingTimeHours = np.round(divmod(elapsedTime.total_seconds(), 60)[0] / 60, 2)
            distTravelled = geopy.distance.distance(lat_lon_Dep, lat_lon_Arr).nm
            print("Total Sailing Time: " + str(totalSailingTimeHours) + " hours")
            print("Distance travelled: " + str(distTravelled) + " n mi")
            print("Mean Draft: " + str(np.mean(rawDataLeg[:, 0])) + " m\n")

            # if totalSailingTimeHours >= 24:
            leg = self.extractJSONforLeg(rawDataLeg, legDepartureDate, legArrivalDate, portDeparure, portArrival,
                                    lat_lon_Dep, lat_lon_Arr, company, vessel)
            legsExtracted.append(leg)
            # else:
            # continue

        jsonFile = {'vessel': vessel,
                    'imo': imo,
                    'legs': legsExtracted
                    }
        # pd.json_normalize(jsonFile['data']).head()
        if os.path.isdir('./legs/' + vessel ) == False:
            os.mkdir('./legs/' + vessel )
        with open('./legs/' + vessel + '/legs_' + vessel + '.json', 'w') as outfile:
            json.dump(jsonFile, outfile)

def main():

    #randomFunction()

    vessel = 'MELISANDE'

    company = 'DANAOS'
    conn = connDetails()
    conn.sid = 'OR12'
    conn.server = '10.2.5.80'
    conn.password = 'shipping'
    conn.usr = 'shipping'

    #convertRawDataLATLON(company, vessel)
    #return





# # ENTRY POINT
if __name__ == "__main__":
    main()