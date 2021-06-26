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

gmtTimezone = 9.00
gmtTimezone = "+" + str(int(gmtTimezone)) if gmtTimezone > 0 else "-" + str(int(gmtTimezone))
timezone = [k for k in list(pytz.all_timezones) if str(k).__contains__(gmtTimezone)][0]
local = pytz.timezone(timezone)
naive = datetime.datetime.strptime("2011-05-05 09:00:00", "%Y-%m-%d %H:%M:%S")
local_dt = local.localize(naive, is_dst=None)
utc_dt = local_dt.astimezone(pytz.utc)

class DBconnections:

    def __init__(self):
        self.connection = None
        self.cursor = None
        self.table = None

    class DBcredentials:

        def __init__(self, user, pwd, server, db):

            self.user = user
            self.pwd = pwd
            self.server = server
            self.db = db

    def connectToDB(self, creds):

        user = creds.user
        pwd = creds.pwd
        server = creds.server
        db = creds.db
        connINSTALLATIONS = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';'
                                            'DATABASE='+db+';'
                                            'UID='+user+';'
                                            'PWD='+pwd+'')

        cursor = connINSTALLATIONS.cursor()

        self.connection =  connINSTALLATIONS
        self.cursor  = cursor
        self.db =  db

    def insertInToTable(self,table,values):

        try:
            self.cursor.execute(
                "insert into [dbo].["+table+"]"
                "(companyName, dateCreated, lastUpdated )"
                + "values ("
                + "'" + str(values.compName) + "', "
                + "'"+ str(values.dateCreated) + "', "
                + "'"+ str(datetime.datetime.now().strftime("%Y%m%d %H:%M:%S")) + "' )")

            self.connection.commit()

        except Exception as e:
            print(str(e))

    def insertInToTableVessel(self, table,compName, values):


        self.cursor.execute(
            'SELECT companyId FROM COMPANIES WHERE companyName =  '"'" + compName + "'"'   ')
        # if cursor_myserver.fetchall().__len__() > 0:
        for row in self.cursor.fetchall():
            companyId = row[0]
            #print(str(imo))

        try:
            self.cursor.execute(
                "insert into [dbo].["+table+"]"
                "( companyId, imo, vesselName, vesselType, dateCreated, dataTypes )"
                + "values ("
                + "'" + str(companyId) + "', "
                + "'" + values.imo + "', "
                + "'"+ values.vslName + "', "
                + "'" + values.vslType + "', "
                + "'"+ str(datetime.datetime.now().strftime("%Y%m%d %H:%M:%S")) +"', "
                + str(values.dataTypes) +
                " )")

            self.connection.commit()

        except Exception as e:
            print(str(e))

    def insertIntoDataTableforVessel(self,company, vessel, values, valuesTlg ,connInst):

        connInst.cursor.execute(
            'SELECT vesselId FROM INSTALLATIONS WHERE vesselName =  '"'" + vessel + "'"'   ')
        # if cursor_myserver.fetchall().__len__() > 0:
        for row in connInst.cursor.fetchall():
            installationsId = row[0]


        for i in range(0,len(values)):
            self.cursor.execute(
                "insert into [dbo].["+vessel+"](installationsId, lat, lon, t_stamp , course , stw , s_ovg , me_foc , rpm , power , draft , trim , at_port ,daysFromLastPropClean, "
                + " daysFromLastDryDock , sensor_wind_speed , sensor_wind_dir , wind_speed ,wind_dir ,curr_speed ,curr_dir ,comb_waves_height ,comb_waves_dir , swell_height ,swell_dir,"
                + "wind_waves_height   , wind_waves_dir ) "

                + "values ("
                + str(installationsId) + ", "
                + str(values[i].OrigLAT) + ", "
                + str(values[i].OrigLON) + ", "
                + "'" + str(values[i].dt) + "', "
                + str(np.nan_to_num(values[i].Vcourse)) + ", "
                + str(np.nan_to_num(values[i].stw)) + ", "
                + str(np.nan_to_num(values[i].Sovg)) + " , "
                + str(np.nan_to_num(values[i].meFoc)) + " , "
                + str(values[i].rpm) + " , "
                + str(np.nan_to_num(values[i].power)) + " , "
                + str(np.nan_to_num(values[i].draft)) + " , "
                + str(np.nan_to_num(values[i].trim)) + " , "
                + str(values[i].atPort) + " , "
                + str(values[i].daysFromLastPropClean) + " , "
                + str(values[i].daysFromLastDryDock) + " , "
                + str(np.nan_to_num(values[i].SensorwindSpeed)) + " , "
                + str(np.nan_to_num(np.round(values[i].relSensorWindDir, 2))) + " , "
                + str(np.nan_to_num(values[i].windSpeed)) + " , "
                + str(np.nan_to_num(values[i].relWindDir)) + " , "
                + str(np.nan_to_num(values[i].currentsSpeed)) + ", "
                + str(np.nan_to_num(values[i].relCurrentsDir)) + " , "
                + str(np.nan_to_num(values[i].combSWH)) + " , "
                + str(np.nan_to_num(values[i].relCombDir)) + " , "
                + str(np.nan_to_num(values[i].swellSWH)) + " , "
                + str(np.nan_to_num(values[i].relSwellDir)) + " , "
                + str(np.nan_to_num(values[i].wavesSWH)) + " , "
                + str(np.nan_to_num(values[i].relWaveDir)) + " )")

        self.connection.commit()



        self.cursor.execute(
                "insert into [dbo].[" + vessel + '_TLG' + "](vdate, time_utc , type ,actl_spd  , avg_spd , slip , act_rpm , sail_time , weather_force , weather_wind_tv , sea_condition , swell_dir,"
                + "swell_lvl    , current_dir , current_speed , draft ) "
                + "values ("
                + "'" + str(vdate) + "', "
                + "'" + str(time_utc) + "', "
                + "'" + 'N' + "', "
                + str(np.nan_to_num(actl_speed)) + ", "
                + str(np.nan_to_num(avg_speed)) + ", "
                + str(np.nan_to_num(float(slip))) + ", "
                + str(np.nan_to_num(act_rpm)) + " , "
                + str(sail_time) + " , "
                + str(np.nan_to_num(weather_force)) + " , "
                + "'" + weather_wind_tv + "', "
                + "'" + sea_condition + "', "
                + "'" + swell_dir + "', "
                + "'" + str(swell_lvl) + "', "
                + "'" + str(current_dir) + "', "
                + str(np.nan_to_num(current_speed)) + ", "
                + str(0) + " )")

        self.connection.commit()

class Values:

    def __init__(self):
        pass

    def  company(self, compName, dateCreated):

        self.compName = compName
        self.dateCreated = dateCreated
        #self.lastUpd = lastUpd

    def vessel(self, vslName, companyId, vslType, dateCreated, imo, dataTypes):

        self.dateCreated = dateCreated
        self.companyId = companyId
        self.vslName = vslName
        self.vslType = vslType
        self.imo  = imo
        self.dataTypes = dataTypes

    def dataVessel(self):

        self.OrigLAT = -1
        self.OrigLON = -1
        self.dt = -1
        self.Vcourse = -1
        self.stw = -1
        self.Sovg = -1
        self.meFoc = -1
        self.rpm = -1
        self.power = -1
        self.draft = -1
        self.trim = -1
        self.atPort= -1
        self.daysFromLastPropClean= -1
        self.daysFromLastDryDock= -1
        self.SensorwindSpeed= -1
        self.roundrelSensorWindDir= -1
        self.windSpeed= -1
        self.relWindDir = -1
        self.relSensorWindDir = -1
        self.currentsSpeed= -1
        self.relCurrentsDir= -1
        self.combSWH= -1
        self.relCombDir= -1
        self.swellSWH= -1
        self.relSwellDir= -1
        self.wavesSWH= -1
        self.relWaveDir= -1

    def tlgDataVessel(self):

        self.vdate = -1
        self.time_utc = -1
        self.hours = -1
        self.minutes = -1
        self.actl_speed = -1
        self.avg_speed = -1
        self.slip = -1
        self.act_rpm = -1
        self.sail_time = -1
        self.weather_force = -1
        self.weather_wind_tv = -1
        self.sea_condition = -1
        self.swell_dir = -1
        self.swell_lvl = -1
        self.current_dir = -1
        self.current_speed = -1


def fillDataTlgVesselClass(company,vessel,values):

    tlgs = pd.read_csv('./data/' + company + '/' + vessel + '/TELEGRAMS/'+vessel+'.csv', sep=';').values
    tlgDataVesselClassList = []

    for i in range(0, len(tlgs)):

        values = Values()
        values.tlgDataVessel()

        values.vdate = tlgs[i][1]
        values.time_utc = tlgs[i][2]
        values.hours = str(values.time_utc).split(":")[0]
        values.minutes = str(values.time_utc).split(":")[1]
        values.actl_speed = tlgs[i][3]
        values.avg_speed = tlgs[i][4]
        values.slip = tlgs[i][5]
        values.act_rpm = tlgs[i][6]
        values.sail_time = tlgs[i][7]
        values.weather_force = tlgs[i][8]
        values.weather_wind_tv = tlgs[i][9]
        values.sea_condition = tlgs[i][10]
        values.swell_dir = tlgs[i][11]
        values.swell_lvl = tlgs[i][12]
        values.current_dir = tlgs[i][13]
        values.current_speed = tlgs[i][14]

        gmtTimezone = tlgs[i][15]
        gmtTimezone = "+"+str(gmtTimezone) if gmtTimezone > 0 else "-"+str(gmtTimezone)
        timezone = [k for k in list(pytz.all_timezones) if str(k).__contains__(gmtTimezone)][0]
        local = pytz.timezone(timezone)
        naive = datetime.strptime(vdate, "%Y-%m-%d %H:%M:%S")
        local_dt = local.localize(naive, is_dst=None)
        utc_dt = local_dt.astimezone(pytz.utc)

        values.swell_lvl = 'n/a' if str(values.swell_lvl) == 'nan' else str(values.swell_lvl)
        try:
            values.slip = str(values.slip).split(',')[0] + "." + str(values.slip).split(',')[1] if str(values.slip).split(
                ',').__len__() > 0 and str(values.slip) != 'nan' else values.slip
        except Exception as e:
            print(str(e))

        datetime1 = values.vdate + " " + values.time_utc
        datetime2 = datetime.datetime.strptime(datetime1, '%d/%m/%y %H:%M') - datetime.timedelta(hours=float(values.sail_time),
                                                                                                 minutes=float(0))
        datetime2 = str(datetime2)

        date = values.vdate
        month = date.split('/')[1]
        day = date.split('/')[0]
        year = '20' + date.split('/')[2]

        vdate = month + '/' + day + '/' + year

        hhMMss = values.time_utc
        newDate = year + "-" + month + "-" + day + " " + ":".join(hhMMss.split(":")[0:2])
        datetime1 = datetime.datetime.strptime(newDate, '%Y-%m-%d %H:%M')
        datetime1 = str(datetime1)
        tlgDataVesselClassList.append(values)

    return tlgDataVesselClassList

def fillDataVesselClass(company,vessel):

    data = pd.read_csv('./data/'+company+'/'+vessel+'/mappedData.csv').values
    dataVesselClassList = []
    for i in range(0, len(data)):

        values = Values()
        values.dataVessel()

        values.dt = data[i][0]
        values.Vcourse = data[i][1]
        values.blFlag = data[i][2]
        values.draft = data[i][8]
        values.relSensorWindDir = data[i][10]
        values.SensorwindSpeed = data[i][11]
        values.stw = data[i][12]

        values.meFoc = data[i][15]
        values.trim = data[i][17]
        values.swellSWH = data[i][20]
        values.relSwellDir = data[i][21]
        values.combSWH = data[i][22]
        values.relCombDir = data[i][23]
        values.wavesSWH = data[i][24]
        values.relWaveDir = data[i][25]
        values.OrigLAT = data[i][26]
        values.OrigLON = data[i][27]
        values.windSpeed = data[i][28]
        values.currentsSpeed = data[i][29]
        values.relCurrentsDir = data[i][30]
        values.Sovg = data[i][31]

        dataVesselClassList.append(values)

    return dataVesselClassList


def main():

    connInst = DBconnections()
    instanceInfoInstallationsDB = connInst.DBcredentials("sa", "Man1f0ld", 'localhost, 1433', 'Installations')
    connInst.connectToDB(instanceInfoInstallationsDB)

    connData = DBconnections()
    instanceDataDB = connData.DBcredentials("sa", "Man1f0ld", 'localhost, 1433', 'DANAOS')
    connData.connectToDB(instanceDataDB)
    values = Values()
    #values.company("DANAOS",datetime.datetime.now().strftime("%Y%m%d %H:%M:%S"))
    #values.vessel("EXPRESS ATHENS", "" ,"Container", datetime.datetime.now().strftime("%Y%m%d %H:%M:%S"), "9484948", 2)

    #values.dataVessel()
    #conn.insertInToTable("COMPANIES",values)

    #conn.insertInToTableVessel("INSTALLATIONS", "DANAOS" ,values)

    dataVesselClassList = fillDataVesselClass("DANAOS","EXPRESS ATHENS",values)

    connData.insertIntoDataTableforVessel("DANAOS","EXPRESS_ATHENS",dataVesselClassList, connInst)

if __name__ == "__main__":
    main()