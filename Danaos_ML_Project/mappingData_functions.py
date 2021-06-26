from datetime import datetime
import pandas as pd
import glob
import numpy as np
import json
from datetime import date
import csv
from collections import Counter

class Mapping:

    def ConvertMSToBeaufort(self, ws):
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

    def calculateSTW(self, sovg, data,vslHeading, currentSpeedKnots, currentDir):
        '''vslHeading = data[:, 1]
        currentDir = data[:, 30]
        currentSpeedKnots = data[:, 29]  # * 1.943
        sovg = data[:, 31]'''
        stwFromCurrents = []
        vslHeadingNew = []

        for i in range(0, len(data)):
            xs = np.cos(vslHeading[i]) * sovg[i]
            ys = np.sin(vslHeading[i]) * sovg[i]

            xc = np.cos(currentDir[i]) * currentSpeedKnots[i]
            yc = np.sin(currentDir[i]) * currentSpeedKnots[i]

            xNew = xs + xc
            yNew = ys + yc

            stwC = np.round(np.sqrt(xNew ** 2 + yNew ** 2), 2)
            stwFromCurrents.append(stwC)

            vslHeadingNew.append(np.arctan(yNew / xNew))

        return stwFromCurrents, vslHeadingNew

    def calculate_stw_TEIDETECH(self, data, vslCode):

        stwTeidetech = pd.read_csv('./data/DANAOS/TEIDETECH/stwTEIDETECH_LEOC_EUROPE.csv')
        stwTeidetech_ = stwTeidetech['VESSEL_CODE'] == vslCode
        stwTeidetech_ = stwTeidetech[stwTeidetech_].values

        ttime = stwTeidetech_[:, 1]

        ttimeDt = list(map(lambda x: str(datetime.strptime(x, '%d/%m/%Y %H:%M:%S').replace(second=0, microsecond=0)),
                           ttime))

        stwDT = stwTeidetech_[:, 11]

        data[:, 0] = list(map(lambda x: str(datetime.strptime(x, '%Y-%m-%d %H:%M:%S').replace(second=0, microsecond=0)),
                              data[:, 0]))

        '''dataNew = data
        for i in range(0, len(stwDT)):
            if np.isnan(stwDT[i]):
                dt = ttimeDt[i]
                month = int(dt.split('-')[1])
                filteredDtsByMonth = np.array([d for d in dataNew if d[0].month==month])
                #filteredDtsByMonthStr = [str(k) for k in filteredDtsByMonth]
                #print(len(filteredDtsByMonth))
                dataDF_ = filteredDtsByMonth[filteredDtsByMonth[:,0]==datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')]
                #dataDF_ = data[data[:, 0] == dt]
                # print(dataDF_)
                if len(dataDF_) > 0:
                    # print("found")
                    stwDT[i] = dataDF_[0][12]

                dataNew = np.delete(dataNew, np.argwhere(dataNew[:,0] == datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')),axis=0)'''
        ttimeDt = np.array([k[:-3] for k in ttimeDt])
        ttimeDtDf = pd.DataFrame({'date': ttimeDt, 'vslHeading': stwTeidetech_[:, 4], 'sOvg': stwTeidetech_[:, 6],
                                  'stw': stwTeidetech_[:, 11], 'cs': stwTeidetech_[:, 10], 'cd': stwTeidetech_[:, 9]})

        data[:, 0] = np.array([k[:-3] for k in data[:, 0]])
        ###filtered teidech data
        ttimeNew = ttimeDtDf[ttimeDtDf['date'].isin(data[:, 0])]

        stw = np.nan_to_num(ttimeNew['stw'].values.astype(float))
        sOvg = np.nan_to_num(ttimeNew['sOvg'].values.astype(float))
        vslHeadingTD = np.nan_to_num(ttimeNew['vslHeading'].values.astype(float))

        currentSpeed = np.nan_to_num(ttimeNew['cs'].values.astype(float))
        currentDir = np.nan_to_num(ttimeNew['cd'].values.astype(float))

        # stwCalc  , vslHeadingNew = calculateSTW(sOvg, vslHeadingTD , currentSpeed , currentDir)

        return stw, vslHeadingTD, currentSpeed, currentDir

    def mapWithTrackReport(self):
      path = "./data/OCEAN_GOLD/PENELOPE/"
      telegrams = pd.read_csv(path+'/TELEGRAMS/PENELOPE.csv',sep=';').values
      dataSet = []
      for infile in sorted(glob.glob(path + '*.csv')):
              data = pd.read_csv(infile, sep=',', decimal='.', skiprows=1)
              dataSet.append(data.values)
              print(str(infile))
          # if len(dataSet)>1:
      dataSet = np.concatenate(dataSet)
      dataSet[:,0] = list(map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f').replace(second=0, microsecond=0), dataSet[:,0]))
      trackReport = pd.read_csv('./data/OCEAN_GOLD/Penelope TrackReport.csv').values


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
          latSec = float(latMinSec.split('"')[0].split("'")[1])
          dir = latMinSec.split('"')[1]
          decimalLat = latDegrees + latMin /60 + latSec/3600
          decimalLat = -decimalLat if dir =='S' else decimalLat

          lonDegrees = float(trackReport[i, 3].split("°")[0])
          lonMinSec = trackReport[i, 3].split("°")[1]
          lonMin = float(lonMinSec.split("'")[0])
          lonSec = float(lonMinSec.split('"')[0].split("'")[1])
          dir = lonMinSec.split('"')[1]
          decimalLon = lonDegrees + lonMin / 60 + lonSec / 3600
          decimalLon = -decimalLon if dir == 'W' else decimalLon

          timeItem = {'datetime':datetime.strptime(newDateTime, '%Y-%m-%d %H:%M'),'ws':windSpeed,'swh':swh,'wd':wd,'lat':decimalLat,'lon':decimalLon}
          json_decoded['dates'].append(timeItem)




      #with open('./data/OCEAN_GOLD/PENELOPE/' 'trackDates.json', 'w') as json_file:
          #json.dump(json_decoded, json_file)
      #return
      speeds = []
      trims = []
      focs = []
      windSpeeds = []
      swhs = []
      wds = []
      drafts=[]
      timestamps = []
      lats = []
      lons = []
      #dataSet = np.array(sorted( dataSet, key=lambda x: x[0] ))
      #sortedJson = sorted(json_decoded['dates'], key = lambda i: i['datetime'])

      print(len(dataSet))
      for k in range(0, len(dataSet)):
          foc = -1
          print(k)
          dateTimeRaw = str(dataSet[k, 0])[:-3]
          dateRaw = str(dataSet[k, 0]).split(" ")[0]
          dtraw = datetime.strptime(dateTimeRaw, '%Y-%m-%d %H:%M')
          month = int(dateRaw.split('-')[1])


              #if dateRaw == newDate:
                  #print("found")
          filteredDtsByMonth = [d for d in dts if d.month==month]
          keyDateTimeInTrackRep = min(filteredDtsByMonth, key=lambda x: abs(x - datetime.strptime(dateTimeRaw, '%Y-%m-%d %H:%M')))

          foc = dataSet[k, 11]
          stw=dataSet[k, 2]
          trackRepvalues = [l for l in json_decoded['dates'] if l['datetime'] == keyDateTimeInTrackRep][0]
          ws = trackRepvalues['ws']
          wd = trackRepvalues['wd']
          swh = trackRepvalues['swh']
          lat = trackRepvalues['lat']
          lon = trackRepvalues['lon']

          #break

          if foc == -1 : continue
          draft = -1
          trim = -1
          for n in range(0, len(telegrams)):
              dateTlg = str(telegrams[n, 0].split(" ")[0])
              if dateTlg == dateRaw:
                  #newDate.split(" ")[0]:
                  draft = telegrams[n, 8]
                  trim = telegrams[n, 16]
                  break

          speeds.append(stw)
          lats.append(lat)
          lons.append(lon)
          windSpeeds.append(ws)
          swhs.append(swh)
          wds.append(wd)
          timestamps.append(dateTimeRaw)
          focs.append(foc)
          trims.append(trim)
          ##map with tlgs for draft
          drafts.append(draft)


      df = {'speed':speeds,"ws":windSpeeds,"wd":wds,"swh":swhs,"foc":focs,'draft':drafts,'trim':trims,'lat':lats,'lon':lons,'timestamp':timestamps}
      df = pd.DataFrame(df).to_csv("./data/OCEAN_GOLD/PENELOPE/trackRepMapped.csv",index=False)
      return

    def extractJSON_TestData(self,imo,vessel,X_test,y_test):

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


    def writeTrainTestData(self,company,vessel,X_test,y_test,X_train,y_train):

        with open('./data/'+company+'/'+vessel+'/mappedDataTest.csv', mode='w') as data:
            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # data_writer.writerow(['Latitude', 'Longitude'])
            for i in range(0, len(X_test)):
                data_writer.writerow(
                    [X_test[i][0], X_test[i][1], X_test[i][2], X_test[i][3], X_test[i][4], y_test[i]])

        with open('./data/'+company+'/'+vessel+'/mappedDataTrain.csv', mode='w') as data:
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
                     X_train[i][6], X_train[i][7]])


    def extractLegsFromTelegrams(self,company,vessel):

        data = pd.read_csv('./data/'+company+"/"+vessel+"/"+vessel+"_PORTS.csv")

        data = data.values

        listLegs = []
        for i in range(0,len(data)-1,2):

            portFROM = data[i][4]
            portTO = data[i+1][4]

            latD_FROM  =data[i][0]
            lonD_FROM = data[i][1]
            latD_TO = data[i+1][0]
            lonD_TO = data[i+1][1]

            if str(portFROM).upper() == str(portTO).upper()  or (latD_FROM == latD_TO and  lonD_FROM == lonD_TO): continue
            listLeg = portFROM + " - " + portTO
            listLegs.append(listLeg)

        keys = list(Counter(listLegs).keys())  # equals to list(set(words))
        count = list(Counter(listLegs).values())  # counts the elements' frequency
        df = pd.DataFrame({'LEGS':keys,"COUNT":count})
        df = df.sort_values("COUNT",ignore_index=True,ascending=False)
        #df.to_csv('./data/'+company+"/"+vessel+'/'+vessel+'_LEGS.csv',index=False)

        with open('./data/'+company+"/"+vessel+'/'+vessel+'_LEGS.csv',mode='w') as data:
            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow(['STARTING PORT','DESTINATION PORT', 'COUNT'])
            array = df.values
            for i in range(0, len(array)):
                pFrom = array[i][0].split("-")[0]
                pTo = array[i][0].split("-")[1]
                data_writer.writerow([pFrom,pTo,array[i][1]])
