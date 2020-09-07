#from pandas import Series
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp,chisquare,chi2_contingency
import math
import pyearth as sp
import matplotlib
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import pyodbc
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

    def insertDataAtDb(self):

        x=0
        conn = pyodbc.connect('DRIVER={SQL Server};SERVER=WEATHERSERVER_DEV;'
                              'DATABASE=millenia;'
                             'UID=sa;'
                              'PWD=sa1!')

        cursor = conn.cursor()
        cursor.execute('SELECT * FROM millenia.dbo.data')


        for row in cursor:
            print(row)

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
        data = np.array([k for k in data.values[0:, 2:23] if k[3] >8 and k[5]>15])
        #stw = data[:,3]
        #rpm = data[:,5]
        #stwRpm = np.array(np.append(stw.reshape(-1,1),np.asmatrix([rpm]).T,axis=1)).astype(float)
        c=0
        while c < len(data):
            if np.std(data[c:c+10,3]) > 1.2:
                data = np.delete(data,np.s_[c:c+10],axis=0)
            c=c+10
        #stw =data[:,0]
        #rpm = data[:,1]
        #plt.scatter(stw,rpm)
        #plt.show()
        #data[:,3]=stw
        #data[:,5]=rpm

        if self.__class__.__name__ == 'UnseenSeriesReader':
            dt = data[ 81000:82000 ]
            #dt = data.sample(n=2880).values[ :90000:, 3:23 ]
            #dt = data.values[ 0:, 2:23 ]
        else:


            dt = data[k*5000:(k*5000 + 5000)]

            #[0:5000]
            #dt=np.array(random.sample(dt,20000))
            #dt = data.values[0:,2:23]

        #PW = np.asarray([ np.nan_to_num(np.float(x)) for x in dt[ :, 0 ] ])
        X = np.asarray([ np.nan_to_num(np.float(x)) for x in dt[ :, 3 ] ])
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
                if dataDist[1] > 0:
                    flag = False
            if flag == False or k==0:
                return X,Y , WS , bearing
            else: return [],[],[],[]

class UnseenSeriesReader (BaseSeriesReader):
    # Imports data from time series file. Expected format: CSV with 2 fields (velocity, RPM)
   pass
