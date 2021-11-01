import pandas as pd
from datetime import datetime, date
import numpy as np
import csv
from global_land_mask import globe
import Danaos_ML_Project.dataReading as dRead
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
import seaborn as sns
import cx_Oracle



class scriptForLAROSExtraction:


    def __init__(self, company, vessel, server, sid, password, usr):
        self.server = server
        self.sid = sid
        self.password = password
        self.usr = usr
        self.vessel = vessel
        self.company = company


    def extractDataForSiteID(self, vessel, siteId):


        dsn = cx_Oracle.makedsn(
            self.server,
            '1521',
            service_name = self.sid
        )
        connection = cx_Oracle.connect(
            user = self.usr,
            password = self.password,
            dsn=dsn
        )


        cursor_myserver = connection.cursor()
        cursor_myserver.execute(
            'select * from TBL_STATUS_PARAMETER_VAL where SITE_ID = '"'" + siteId + "'"' '
                'and DATETIME >  TO_DATE( '"'2020-01-01 12:00'"', '"'YYYY-MM-DD HH:MI'"') and '
                'DATETIME < TO_DATE('"'2020-01-30 12:00'"', '"'YYYY-MM-DD HH:MI'"') order by DATETIME')
        # if cursor_myserver.fetchall().__len__() > 0:

        rows = cursor_myserver.fetchall()
        fp = open('./'+vessel+'_'+self.sid+'_'+str(date.today())+'_.csv', 'w')
        myFile = csv.writer(fp)
        myFile.writerows(rows)
        fp.close()


def main():
    ext = scriptForLAROSExtraction("DANAOS", "CMA CGM TANCREDI", "10.2.5.80", "or19", "laros", "laros",)

    ext.extractDataForSiteID(ext.vessel, '20')

if __name__ == "__main__":
    main()