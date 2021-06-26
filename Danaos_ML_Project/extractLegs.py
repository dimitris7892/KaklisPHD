from datetime import datetime
import holoviews as hv
import pandas as pd
import numpy as np
import dataReading as dRead
import csv
import json
hv.extension('bokeh', 'matplotlib')
dr = dRead.BaseSeriesReader()
import geopy.distance
import pyproj
from pyproj import Proj, transform
import cx_Oracle
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


def extractImo(connDetails, vessel):

    dsn = cx_Oracle.makedsn(
        connDetails.server,
        '1521',
        service_name= connDetails.sid
    )
    connection = cx_Oracle.connect(
        user= connDetails.usr,
        password=connDetails.password,
        dsn= dsn
    )
    cursor_myserver = connection.cursor()
    cursor_myserver.execute(
        'SELECT IMO_NO FROM VESSEL_DATA WHERE VESSEL_CODE =  (SELECT VESSEL_CODE FROM VESSEL_DATA WHERE VESSEL_NAME LIKE '"'%" + vessel + "%'"'  AND ROWNUM=1)  ')
    # if cursor_myserver.fetchall().__len__() > 0:
    for row in cursor_myserver.fetchall():
        imo = row[0]
        print(str(imo))
    return imo

def convertRawToRollingWindowAvg(rawData):
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

def getRawData(vessel):
    sFile = './data/DANAOS/'+vessel+'/mappedData.csv'
    data = pd.read_csv(sFile,sep=',').values
    print(data.shape)
    data[:, 12] = np.round(data[:,12].astype(float),2)
    data[:, 8] = np.round(data[:,8].astype(float),2)
    ##knots to m/s
    wfS = data[:, 11].astype(float) / (1.944)
    data[:, 11] = np.round(wfS,2)

    wfS= data[:,11]
    for i in range(0, len(wfS)):
        wfS[i] = dr.ConvertMSToBeaufort(float(float(wfS[i])))
    data[:, 11] = wfS

    #lt/min => MT/day
    data[:, 15] = ((data[:, 15]) / 1000)* 1440
    ##Build training data array

    trData = np.array(np.append(data[:,8].reshape(-1,1),np.asmatrix([data[:,10],data[:,11],data[:,12],data[:,22],(data[:,15]),
                                                                     data[:,26],data[:,27],data[:,0]]).T,axis=1))#.astype(float)#data[:,26],data[:,27]
    trData = np.array([k for k in trData if  str(k[0])!='nan' and float(k[2])>=0 and float(k[4])>=0 and float(k[3])>=0 and float(k[5])>0  ])

    trData[:,8] = list(map(lambda s: s[:-3], trData[:,8]))
    d = {'draft': trData[:,0], 'wd': trData[:,1],'ws':trData[:,2],'stw':trData[:,3],'swh':trData[:,4],
         'foc':(trData[:,5]),'lat':trData[:,6],'lon':trData[:,7],'timestamp':trData[:,8] }
    df  = pd.DataFrame(d)
    print(trData.shape)
    return df

def filterRawWithTlgDates(vessel, numberOfLegs):

    df = getRawData(vessel)
    telegrams = pd.read_csv('./data/DANAOS/'+vessel+'/TELEGRAMS/'+vessel+'.csv',sep=';')
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

    print("Candidate Date: " + str(candidateDt))
    return canDidatesDts , rawData , dfFilt , tlgs, telegrams

def extractLeg(candidateDt, tlgs, telegrams):

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
        tlg_type='D'
        if tlg_type == 'D':
            tlgRowDep = tlgForRawDate.values[0]
            k = row
            while nextTlg_Type != 'A':
                tlgRowArr = tlgs[k]
                # print(prevTlg_Type)
                nextTlg_Type = tlgs[k][1]
                k = k + 1

        return tlgRowDep, tlgRowArr, row, k - 1

def extractJSONforLeg(rawDataLeg, legDepartureDate, legArrivalDate, portDeparure, portArrival, lat_lon_Dep, lat_lon_Arr, vessel):

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

    with open('./legs/'+vessel+'/'+portDeparure+'_'+portArrival+'_leg.csv', mode='w') as dataw:
        data_writer = csv.writer(dataw, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        data_writer.writerow(['stw', 'draft', 'foc', 'ws', 'wd', 'swh', 'lat', 'lon', 'dt'])
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

            data_writer.writerow([stw, draft, foc, ws, wd, swh, lat, lon, dt])

            item = {'datetime': dt, 'ws': ws, 'swh'
            : swh, 'wd': wd, 'stw': stw, 'foc': foc, 'lat': lat, 'lon': lon, 'draft': draft}
            leg['data'].append(item)
    return leg


def main():

    vessel = 'HYUNDAI SMART'
    conn = connDetails()
    conn.sid = 'OR12'
    conn.server = '10.2.5.80'
    conn.password = 'shipping'
    conn.usr = 'shipping'

    imo = extractImo(conn, vessel)

    candidateDts , rawData , dfFilt, tlgs, telegrams = filterRawWithTlgDates(vessel, 10)
    legsExtracted = []
    candidateDts = ['2020-06-08 13:00:00']
    for candidateDt in candidateDts:
        tlgRowDep , tlgRowArr , row, rowArr = extractLeg(candidateDt,tlgs, telegrams)

        originalDeptlgDate = pd.read_csv('./data/DANAOS/'+vessel+'/TELEGRAMS/'+vessel+'.csv',sep=';').values[row][0]
        originalArrtlgDate = pd.read_csv('./data/DANAOS/'+vessel+'/TELEGRAMS/'+vessel+'.csv',sep=';').values[rowArr][0]
        #tlgRowArr[1]
        legDepartureDate = tlgRowDep[0]
        legArrivalDate = tlgRowArr[0]
        portDeparure = tlgRowDep[22]
        portArrival = tlgRowArr[22]
        print("Departure Port: " + str(portDeparure) +" / "+ str(legDepartureDate))
        print("Arrival Port: " + str(portArrival) + "  / " +str(legArrivalDate))
        #legDepartureDate

        originalDeptlgDate = datetime.strptime(originalDeptlgDate, '%Y-%m-%d %H:%M:%S')
        originalArrtlgDate  = datetime.strptime(originalArrtlgDate, '%Y-%m-%d %H:%M:%S')


        print("Original Dep Date: " +str(originalDeptlgDate))
        print("Original Arr Date: " + str(originalArrtlgDate))
        rawDataLeg = np.array([k for k in rawData if
        datetime.strptime(k[8], '%Y-%m-%d %H:%M') >=originalDeptlgDate
                               and datetime.strptime(k[8], '%Y-%m-%d %H:%M')<=originalArrtlgDate])

        if len(rawDataLeg) == 0: continue
        lat_lon_Dep = (rawDataLeg[0][6],rawDataLeg[0][7])
        lat_lon_Arr = (rawDataLeg[len(rawDataLeg)-1][6],rawDataLeg[len(rawDataLeg)-1][7])

        print(lat_lon_Dep)
        print(lat_lon_Arr)

        #rawDataLeg = preprocessCoordinates(rawDataLeg)
        #rollingAvgsRawData = convertRawToRollingWindowAvg(rawDataLeg)
        if len(rawDataLeg)>1440:
            leg = extractJSONforLeg(rawDataLeg, legDepartureDate, legArrivalDate, portDeparure, portArrival, lat_lon_Dep, lat_lon_Arr, vessel)

            legsExtracted.append(leg)
        else:
            continue

    jsonFile = {'vessel': vessel,
                    'imo': imo,
                    'legs': legsExtracted
                    }
    #pd.json_normalize(jsonFile['data']).head()

    with open('./legs/'+vessel+'/legs_'+vessel+'.json', 'w') as outfile:
        json.dump(jsonFile, outfile)



def preprocessCoordinates(rawDataLeg):
    newIndices = []
    newLons = []
    newLats = []
    vknots = []
    dists = []
    lons = []

    meanLon = np.mean(rawDataLeg[:,7])
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

        elapsedTime = (datetime.strptime(rawDataLeg[i][8], '%Y-%m-%d %H:%M') - datetime.strptime(rawDataLeg[i - 1][8], '%Y-%m-%d %H:%M'))
        totalSailingTimeHours = divmod(elapsedTime.total_seconds(), 60)[0] / 60
        v = dist / totalSailingTimeHours
        vknot = v / 1.852
        bearing_fwd, back_bearing, distance = geodesic.inv(prevLon, prevLat, lon, lat)
        #(bearing_fwd<60 and bearing_fwd>-60)
        if  vknot < 30 and vknot > 0 and (rawDataLeg[i][7] <= meanLon + 3 * stdLon and rawDataLeg[i][7] >= meanLon -  3*stdLon) :
            newIndices.append(i)
    return rawDataLeg[newIndices]

# # ENTRY POINT
if __name__ == "__main__":
    main()