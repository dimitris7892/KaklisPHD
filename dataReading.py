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
        rpm={"data": [ ]}
        power={"data": [ ]}
        foc={"data": [ ]}

        with open('./MT_DELTA_MARIA.txt') as csv_file:
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
                #if id=='7' :
                    #drftA={}
                    #drftA["value"]=value
                    #drftA["dt"]=dt
                    #draftAFT[ "data" ].append(drftA)

                #if id=='8':
                    #drftF={}
                    #drftF["value"]=value
                    #drftF["dt"]=dt
                    #draftFORE[ "data" ].append(drftF)


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

                if id == '353' :
                    pow={}
                    pow[ "value" ] = value
                    pow[ "dt" ] = dt
                    power[ "data" ].append(pow)

                line_count+=1
                if int(id) >61 : break
            print('Processed '+str(line_count)+ ' lines.')



        with open('./MT_DELTA_MARIA_data.csv', mode='w') as data:
            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #data_writer.writerow(['Draft', 'WindSpeed', 'WindAngle', 'SpeedOvg','Rpm' ,'M/E FOC (kg/min)', 'Power','DateTime' ])
            data_writer.writerow(
                ['Draft','DateTime' ])

            for i in range(0,len(draft['data'])):
                value = draft['data'][i]['value']
                dt = draft['data'][i]['dt']
                data_writer.writerow([value,str(dt)])

            for dt in rrule(MINUTELY, dtstart=dtFrom, until=dtTo):
               dtTocompare = dt

               #ListfilteredDRFTaft=[i for i in range(0,len(draftAFT['data'])) if draftAFT['data'][i]['dt']==dtTocompare ]
               #filteredDRFTaft = "" if len(ListfilteredDRFTaft) == 0 else draftAFT['data'][ListfilteredDRFTaft[0]][ 'value' ]

               #ListfilteredDRFTfore = [ i for i in range(0, len(draftFORE[ 'data' ])) if draftFORE[ 'data' ][ i ][ 'dt' ] == dtTocompare ]
               #filteredDRFTfore = "" if len(ListfilteredDRFTfore) == 0 else draftFORE[ 'data' ][ ListfilteredDRFTfore[ 0 ] ]['value' ]

               ListfilteredDRFT=[i for i in range(0,len(draft['data'])) if draft['data'][i]['dt']==dtTocompare ]
               filteredDRFT="" if len(ListfilteredDRFT) == 0 else draftAFT['data'][ListfilteredDRFT[0]][ 'value' ]

               ListfilteredWA=[i for i in range(0,len(windAngle['data'])) if windAngle['data'][i]['dt']==dtTocompare ]
               filteredWA ="" if len(ListfilteredWA)==0 else windAngle["data"][ListfilteredWA[0]]['value']

               ListfilteredWS = [i for i in range(0,len(windSpeed['data'])) if windSpeed['data'][i]['dt']==dtTocompare ]
               filteredWS ="" if len(ListfilteredWS)==0 else windSpeed["data"][ListfilteredWS[0]]['value']

               ListfilteredSO = [i for i in range(0,len(SpeedOvg['data'])) if SpeedOvg['data'][i]['dt']==dtTocompare ]
               filteredSO ="" if len(ListfilteredSO)==0 else SpeedOvg["data"][ListfilteredSO[0]]['value']

               ListfilteredFO =[i for i in range(0,len(foc['data'])) if foc['data'][i]['dt']==dtTocompare ]
               filteredFO ="" if len(ListfilteredFO)==0 else foc["data"][ListfilteredFO[0]]['value']

               ListfilteredRPM =[i for i in range(0,len(rpm['data'])) if rpm['data'][i]['dt']==dtTocompare ]
               filteredRPM ="" if len(ListfilteredRPM)==0 else rpm["data"][ListfilteredRPM[0]]['value']


               ListfilteredPOW = [i for i in range(0,len(power['data'])) if power['data'][i]['dt']==dtTocompare ]
               filteredPOW ="" if len(ListfilteredPOW)==0 else power["data"][ListfilteredPOW[0]]['value']
               try:
                data_writer.writerow([ filteredDRFT, filteredWS, filteredWA , filteredSO ,filteredRPM,filteredFO ,
                                      filteredPOW,str(dtTocompare)])

               except:
                   x=0

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
        SO = np.asarray([ y for y in dt[ :, 2 ] ])
        RPM = np.asarray([ y for y in dt[ :, 3 ] ])
        FOC = np.asarray([ y for y in dt[ :, 4 ] ])
        POW = np.asarray([ y for y in dt[ :, 5 ] ])
        #DA = np.asarray([ np.nan_to_num(np.float(y)) for y in dt[ :, 13 ] ])
        #DF = np.asarray([ np.nan_to_num(np.float(y)) for y in dt[ :, 14 ] ])
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
            dt = data.values[ 0:, 2:23 ][ 41000:42000 ]
            #dt = data.sample(n=2880).values[ :90000:, 3:23 ]
            #dt = data.values[ 0:, 2:23 ]
        else:

            dt = data.values[ 0:, 2:23 ][k*10000:(k*10000 + 10000)]
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