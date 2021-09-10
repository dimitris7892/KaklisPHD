import pandas as pd
from datetime import datetime
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

is_in_ocean = globe.is_ocean(23,54)
vessel = 'EXPRESS ATHENS'
def main():
    pl = ParseLaros()
    #pl.extractRAWLarosData(4, vessel+'.csv',vessel+'_data.csv')
    pl.extractFeaturesFromLarosData(vessel+'_data.csv', vessel)


class ParseLaros:

    def extractFeaturesFromLarosData(self, fileName, vessel):

        data = pd.read_csv('/home/dimitris/Desktop/' + fileName, delimiter=',', skiprows=0)
        '''lats, lons,names,altitude = [],[],[],[]

        # the asos_stations file can be found here:
        # https://engineersportal.com/s/asos_stations.csv

        df = pd.read_csv("./data/DANAOS/EXPRESS ATHENS/EXPRESS ATHENSCoor.csv", delimiter=',', skiprows=0, low_memory=False)

        geometry = [Point(xy) for xy in zip(df['Longitude'],df['Latitude'])]
        gdf = GeoDataFrame(df, geometry=geometry)

        #this is a simple map that goes with geopandas
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=15);'''
        # plt.show()
        # data = data.drop(["blFlags"], axis=1)
        # data = data.drop(["wind_speed", "wind_dir","trim"], axis=1)
        # x_train = data.drop(["blFlags","focs","tlgsFocs"], axis=1)

        # foc = np.array(np.mean(data.values[:,6:7],axis=1))
        # y_train = pd.DataFrame({
        # 'FOC': foc,
        # })
        '''dataR = dRead.BaseSeriesReader()
        keys = np.array([k for k in data.keys() if "draft" in k])
        keys.sort()
        notinOcean=[]
        latD = []
        lonD = []

        for i in range(0,len(data['Latitude'].values)):
            lat = str(data['Latitude'].values[i])
            latDir = 'S' if float(lat) < 0 else 'N'
            lat = lat.split('-')[1] if float(lat) < 0 else lat

            lon = str(data['Longitude'].values[i])
            lonDir = 'W' if float(lon) < 0 else 'E'
            lon = lon.split('-')[1] if float(lon) < 0 else lon

            LAT, LON = dataR.convertLATLONfromDegMinSec(lat, lon, latDir, lonDir)
            is_in_ocean = globe.is_ocean(LAT, LON)
            if is_in_ocean==False:

                fig = plt.figure(figsize=(8, 8))
                m = Basemap(projection='lcc', resolution=None,
                            width=8E6, height=8E6,
                            lat_0=LAT, lon_0=LON, )
                m.etopo(scale=0.7, alpha=0.7)

                # Map (long, lat) to (x, y) for plotting
                x = LAT
                Y = LON
                x, y = m(*np.meshgrid(LON,LAT))
                #m.drawcoastlines()
                plt.plot(x, y, 'ok', markersize=5)
                plt.text(x, y, '', fontsize=12)
                plt.savefig('/home/dimitris/Desktop/NoInOcean/LATLON_'+str(i))
                plt.show()
                latD.append(LAT)
                lonD.append(LON)
                notinOcean.append(i)
                #print("FALSE")
        company = 'DANAOS'
        vessel = 'EXPRESS ATHENS'
        with open('./data/' + company + '/' + vessel + '/EXPRESS ATHENSCoor.csv', mode='w') as data:
            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow(['Latitude','Longitude'])
            for i in range(0, len(latD)):
                data_writer.writerow(
                    [latD[i], lonD[i]])
        with open('./data/' + company + '/' + vessel + '/EXPRESS ATHENSSpeedNIO.csv', mode='w') as data:
            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for i in range(0, len(latD)):
                data_writer.writerow(
                    [notinOcean[i]])'''

        # plt.show()
        # print(len(notinOcean))
        vslHeading = data['VesselHeading'].values
        for i in range(0, len(vslHeading)):
            if vslHeading[i] < 0:
                vslHeading[i] = vslHeading[i] + 180

        foc = data['M/EFOFlow'].values
        focDf = pd.DataFrame({'foc':foc})
        #sns.displot(focDf, x="foc")
        #plt.show()
        # foc = (foc /1000) * 1440
        #draft = (data['AP_ FORWARDDRAFTLEVEL'].values + data['AP_ AFTERDRAFTLEVEL'].values)/2
        # draftMid = data['AP_DRAFT MIDP'].values
        # draftArray = np.array(np.append(draft.reshape(-1,1),np.asmatrix([draftMid]).T, axis=1))
        emptyColumn = np.array(['nan'] * len(vslHeading)).reshape(-1)
        #rpm = np.array(data['M/E RPM Terasaki'].values).reshape(-1)

        data = np.append(data['ttime'].values.reshape(-1, 1),
                         np.asmatrix([
                             vslHeading,
                             data['Latitude'].values,
                             data['Longitude'].values,
                             data['WindAngle'].values,
                             data['WindBF'].values,
                             data['STW'].values,
                             emptyColumn,
                             foc,
                             data['SpeedOverGround'].values,
                             data['M/ERPM'].values,
                             data['AP_ TOTALUSEDPOWER'].values
                         ]).T, axis=1)
        # data['DraftAfter'].values + data['DraftFore'].values) / 2
        # data = pd.read_csv(sFile, delimiter=';')
        # data = data.drop(["wind_speed", "wind_dir"], axis=1)
        # data = data[data['stw']>7].values
        data = np.array(data)  # .astype(float)

        ##################################################
        trData = data
        company = 'DANAOS'

        with open('./data/' + company + '/' + vessel + '/' + vessel + '_.csv', mode='w') as data:
            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in range(0, len(trData)):
                data_writer.writerow(
                    [trData[i][0], trData[i][1], trData[i][2], trData[i][3], trData[i][4], trData[i][5], trData[i][6],
                     trData[i][7], trData[i][8], trData[i][9],trData[i][10], trData[i][11]])

    def extractRAWLarosData(self, siteId, fileName, destinationFile):

        dictionary = pd.read_excel("/home/dimitris/Downloads/LarosMap.xlsx")
        dictionary = dictionary[dictionary["SITE_ID"] == siteId][["STATUS_PARAMETER_ID", "NAME"]]
        l = pd.read_csv("/home/dimitris/Desktop/" + fileName,
                        names=["id", "sensor", "ship_id", "txt", "bin", "value", "time"])[["sensor", "time", "value"]]
        # time = l['time'].array.to_numpy()
        # time.sort()
        # print(time)
        l["ttime"] = list(
            map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f').replace(second=0, microsecond=0), l["time"]))
        l = l.drop(columns=["time"])
        ll = l.groupby(["ttime", "sensor"])["value"].mean().reset_index()
        vector = ll.pivot(index='ttime', columns='sensor', values='value').reset_index()
        vector = vector.sort_values(by="ttime").reset_index().drop(columns=["index"])
        dates = vector["ttime"]
        del vector["ttime"]
        for i in vector.columns:
            vector[i] = vector[i].astype(float).fillna(method="ffill").fillna(method="bfill")
        helper = pd.DataFrame(vector.columns)
        helper["STATUS_PARAMETER_ID"] = helper["sensor"].apply(int)
        helper = helper.drop(columns=["sensor"])
        col_names = pd.merge(helper, dictionary, how='left', on=['STATUS_PARAMETER_ID'])
        # change of column names:
        vector.columns = list(col_names["NAME"])
        dvector = pd.concat([dates, vector], axis=1, sort=False)
        dvector.to_csv('/home/dimitris/Desktop/' + destinationFile, index=False)

        '''company = 'DANAOS'
            vessel = 'EXPRESS ATHENS'
            with open('/home/dimitris/Downloads/mappedData3.csv', 'rU') as myfile:
                filtered = (line.replace('\n', '') for line in myfile)
                with open('/home/dimitris/Downloads/mappedData2.csv', mode='w') as data:
                    for row in csv.reader(filtered):
                        if row!=[]:
                            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            data_writer.writerow(row)'''


if __name__ == "__main__":
    main()