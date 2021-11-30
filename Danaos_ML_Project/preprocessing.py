import numpy as np
import pandas as pd
from scipy.stats import ks_2samp,chisquare,chi2_contingency
import math
#from pyproj import Proj, transform
import pyearth as sp
import matplotlib
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D
from decimal import Decimal
import random
from numpy import inf
#from coordinates.converter import CoordinateConverter, WGS84, L_Est97
import pyodbc
import csv
import pytz
import locale
#locale.setlocale(locale.LC_NUMERIC, "en_DK.UTF-8")
import datetime
import cx_Oracle
from dateutil.rrule import rrule, DAILY, MINUTELY
from sympy.solvers import solve
from sympy import Symbol
from pathlib import Path
import itertools
from sympy import cos, sin , tan , exp , sqrt , E
from openpyxl import load_workbook
import glob, os
from pathlib import Path
#from openpyxl.styles.colors import YELLOW
from openpyxl.styles import Font
from openpyxl.styles.borders import Border, Side
import shutil
from openpyxl.styles import PatternFill
from openpyxl.styles import Alignment
#import matplotlib.pyplot as plt
#import seaborn as sns
from openpyxl.drawing.image import Image
from scipy.stats import ttest_ind_from_stats
#pytz.timezone("Eastern European Time")
import geopy
from geopy import  geocoders
from tzwhere import tzwhere
import insertAtDB as dbIns


class preprocessing:

    def __init__(self, database):

        self.connData = dbIns.DBconnections()
        instanceDataDB = self.connData.DBcredentials("sa", "Man1f0ld", 'localhost, 1433', database)
        self.cnxn = self.connData.connectToDB(instanceDataDB)


    def extractData(self, cnxn, table):

        data = pd.read_sql("SELECT  * FROM  "+table+" ", cnxn)

        return data

    def processData(self, data):

        keys = data.keys().array
        try:
            for key in keys:
                if str(key).__contains__('id'): continue
                _key = data[key]
                isnan_keyindx = _key[np.isnan(_key.array) == True].index.array
                print(key+": "+ str(isnan_keyindx))


            ## stw >=8
            dataSTW8 = data[data['stw']>=8]
            ##cut outliers based on FOC
            dataSTW8val = dataSTW8.values
            meanFoc = np.mean(dataSTW8['me_foc' ], axis=0)
            stdFoc = np.std(dataSTW8['me_foc'], axis=0)
            dataSTW8filtered = np.array(
                [k for k in dataSTW8val if (float(k[8]) >= (meanFoc - (3 * stdFoc))).all() and (float(k[8]) <= (meanFoc + (3 * stdFoc))).all()])

            dataSTW8filteredAtSea =  dataSTW8filtered[dataSTW8filtered[:,13]==0]

            dataVesselClassListBulk = []
            for i in range(0,len(dataSTW8filteredAtSea)):
                dataVesselClassListBulk.append((
                    dataSTW8filteredAtSea[i][1],
                    dataSTW8filteredAtSea[i][2],
                    dataSTW8filteredAtSea[i][3],
                    dataSTW8filteredAtSea[i][4],
                    dataSTW8filteredAtSea[i][5],
                    dataSTW8filteredAtSea[i][6],
                    dataSTW8filteredAtSea[i][7],
                    dataSTW8filteredAtSea[i][8],
                    dataSTW8filteredAtSea[i][9],
                    dataSTW8filteredAtSea[i][10],
                    dataSTW8filteredAtSea[i][11],
                    dataSTW8filteredAtSea[i][12],
                    dataSTW8filteredAtSea[i][13],
                    dataSTW8filteredAtSea[i][14],
                    dataSTW8filteredAtSea[i][15],
                    dataSTW8filteredAtSea[i][16],
                    dataSTW8filteredAtSea[i][17],
                    dataSTW8filteredAtSea[i][18],
                    dataSTW8filteredAtSea[i][19],
                    dataSTW8filteredAtSea[i][20],
                    dataSTW8filteredAtSea[i][21],
                    dataSTW8filteredAtSea[i][22],
                    dataSTW8filteredAtSea[i][23],
                    dataSTW8filteredAtSea[i][24],
                    dataSTW8filteredAtSea[i][25],
                    dataSTW8filteredAtSea[i][26],
                    dataSTW8filteredAtSea[i][27],
                    dataSTW8filteredAtSea[i][28]
                ))  # append data

        except Exception as e:
            print(key)
            print(str(e))
        return dataVesselClassListBulk


    def createTableForProcessedData(self, tableName):

        pass

    def insertProcessedData(self, data):

        pass

def main():

    prepro = preprocessing("DANAOS")

    data = prepro.extractData(prepro.cnxn, "EXPRESS_ATHENS")

    dataCleaned = prepro.processData(data)

    print(data.shape)

    connInst = dbIns.DBconnections()
    instanceInfoInstallationsDB = connInst.DBcredentials("sa", "Man1f0ld", 'localhost, 1433', 'Installations')
    connInst.connectToDB(instanceInfoInstallationsDB)

    prepro.connData.createTables("DANAOS", "EXPRESS_ATHENS", 'processed')
    #instanceDataDB = prepro.connData.DBcredentials("sa", "Man1f0ld", 'localhost, 1433', 'DANAOS')
    #cnxn = prepro.connData.connectToDB(instanceDataDB)


    #dataVesselClassList, dataVesselClassListBulk = dbIns.fillDataVesselClass("DANAOS", "EXPRESS ATHENS", connInst, prepro.connData, dataCleaned)

    prepro.connData.insertIntoDataTableforVessel("DANAOS", "EXPRESS_ATHENS_PROCESSED",connInst , "raw", "bulk", None,
                                          dataCleaned, )

if __name__ == "__main__":
    main()